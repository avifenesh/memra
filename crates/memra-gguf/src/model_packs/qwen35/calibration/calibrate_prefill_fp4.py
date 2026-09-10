#!/usr/bin/env python3
"""Offline BF16 calibration input producer for the Qwen35 prefill-A4 pack.

External Transformers/PyTorch are calibration instruments only. This script
does not qualify Memra execution or provide a serving backend. Source text is
private; the reproducibility bundle contains hashes, IDs, counts and this code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

PROGRAM = "qwen35-prefill-nvfp4-a4-v1"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def projections():
    names = {}
    for layer in range(64):
        pairs = {"mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
        if layer % 4 == 3:
            pairs.update({"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
                          "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"})
        else:
            pairs.update({"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
                          "linear_attn.out_proj": "ssm_out"})
        for hf, gguf in pairs.items():
            names[f"layers.{layer}.{hf}"] = f"blk.{layer}.{gguf}"
    assert len(names) == 400
    return names


def prepare(tokenizer, corpus, out):
    import numpy as np

    records = [json.loads(line) for line in corpus.open()]
    selected, used = [], set()
    # Fixed session-disjoint assignment, diagnostics reserved before fitting.
    def take(split, length, count, pool=None):
        got = 0
        for r in records:
            if r["id"] in used or (pool and r["pool"] != pool):
                continue
            if length > 4096 and r["stratum"] != "long":
                continue
            tokens = tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
            if len(tokens) < length:
                continue
            ids = np.array(tokens[:length], dtype="<u4")
            name = f"{split}-{length}-{r['id'][:16]}.bin"
            (out / name).write_bytes(ids.tobytes())
            selected.append({"split": split, "length": length, "id": r["id"], "pool": r["pool"],
                             "file": name, "tokens_sha256": sha(out / name), "available_tokens": len(tokens),
                             "hebrew_characters_in_source": r["hebrew_characters"],
                             "source_sha256": r["source_sha256"], "messages_sha256": r["messages_sha256"]})
            used.add(r["id"])
            got += 1
            if got == count:
                break
        if got != count:
            raise ValueError(f"insufficient real sessions for {split}/{length}/{pool}: {got}/{count}")

    take("diagnostic", 131070, 3)
    take("calibration", 131072, 3)
    take("calibration", 32768, 12)
    for pool, count in (("claude", 48), ("codex", 80), ("eigen", 80), ("hermes", 48)):
        take("calibration", 4096, count, pool)
    manifest = {"program": PROGRAM, "base_revision": BASE_REVISION, "seed": 20260909,
                "corpus_sha256": sha(corpus), "script_sha256": sha(__file__), "samples": selected,
                "tokenizer_files": {name: sha(Path(tokenizer.name_or_path) / name)
                                    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
                "calibration_tokens": sum(x["length"] for x in selected if x["split"] == "calibration"),
                "scope": "text-only, distinct SXC sessions, no synthetic padding/repetition"}
    atomic(out / "corpus.lock.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--deadline-epoch", type=float, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(20260909)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for name in tokenizer_files:
        if not (args.model / name).is_file():
            raise ValueError(f"incomplete source: missing {name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lockfile = args.out / "corpus.lock.json"
    manifest = json.loads(lockfile.read_text()) if lockfile.exists() else prepare(tokenizer, args.corpus, args.out)
    if manifest["corpus_sha256"] != sha(args.corpus) or manifest["script_sha256"] != sha(__file__):
        raise ValueError("resume corpus/script identity changed")
    if manifest["tokenizer_files"] != {name: sha(args.model/name) for name in tokenizer_files}:
        raise ValueError("tokenizer/template identity changed")
    if args.prepare_only:
        print(json.dumps({"event": "corpus_ready", "samples": len(manifest["samples"]),
                          "calibration_tokens": manifest["calibration_tokens"]}), flush=True)
        return

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    trunk = model.model.language_model
    if (trunk.config.num_hidden_layers, trunk.config.hidden_size, trunk.config.intermediate_size) != (64, 5120, 17408):
        raise ValueError("wrong Qwen geometry")
    names = projections()
    modules = dict(trunk.named_modules())
    if set(names) - set(modules):
        raise ValueError(f"missing projection modules: {set(names)-set(modules)}")
    counters = {n: torch.zeros(5, dtype=torch.float64, device="cuda") for n in names}
    maxima = {n: torch.zeros((), dtype=torch.float32, device="cuda") for n in names}
    phase = {"kind": "calibration", "scales": None}
    handles = []

    def hook(name):
        def capture(module, inputs):
            x = inputs[0].detach().float()
            if not x.shape[-1] % 16 == 0:
                raise ValueError(f"unaligned input: {name}")
            absolute = x.abs()
            maxima[name].copy_(torch.maximum(maxima[name], absolute.amax()))
            counters[name][0] += x.numel()
            counters[name][4] += (~torch.isfinite(x)).sum()
            if phase["kind"] == "diagnostic":
                s = phase["scales"][name]
                blocks = absolute.reshape(-1, 16)
                raw = blocks.amax(dim=-1, keepdim=True) / (6 * s)
                micro = raw.clamp(max=448).to(torch.float8_e4m3fn).float()
                normalized = blocks / (micro * s).clamp_min(torch.finfo(torch.float32).tiny)
                # E2M1 RN: 5 is tied to 4 (even), so only >5 rounds to max 6.
                counters[name][1] += (normalized > 5).sum()
                counters[name][2] += (absolute > (6 * 448 * s)).sum()
                counters[name][3] += (raw > 448).sum()
        return capture

    for n in names:
        handles.append(modules[n].register_forward_pre_hook(hook(n)))

    def forward(row, length):
        file = args.out / row["file"]
        if sha(file) != row["tokens_sha256"]:
            raise ValueError("token input changed")
        ids = np.fromfile(file, dtype="<u4")[:length]
        cache = None
        started = time.time()
        with torch.inference_mode():
            for pos in range(0, length, args.chunk):
                if time.time() > args.deadline_epoch:
                    raise TimeoutError("rental budget deadline reached")
                inputs = torch.tensor(ids[pos:pos+args.chunk].astype(np.int64), device="cuda")[None]
                output = trunk(input_ids=inputs, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                del output
                if pos % 8192 == 0:
                    torch.cuda.synchronize()
                    print(json.dumps({"event": "progress", "kind": phase["kind"], "sample": row["id"],
                                      "position": pos+len(inputs[0]), "length": length,
                                      "elapsed_s": time.time()-started,
                                      "allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
        del cache
        torch.cuda.synchronize()
        return time.time()-started

    checkpoint = args.out / "calibration-progress.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"completed": [], "maxima": {}}
    for n, v in state["maxima"].items():
        maxima[n].fill_(v)
    runs = state.get("runs", [])
    for row in manifest["samples"]:
        if row["split"] != "calibration" or row["id"] in state["completed"]:
            continue
        elapsed = forward(row, row["length"])
        values = {n: float(v.cpu()) for n, v in maxima.items()}
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("invalid calibration maxima")
        runs.append({"id": row["id"], "tokens": row["length"], "elapsed_s": elapsed})
        state = {"completed": state["completed"]+[row["id"]], "maxima": values, "runs": runs}
        atomic(checkpoint, state)
    scales = {n: float(np.float32(v / 2688.0)) for n, v in state["maxima"].items()}
    record = {"program": PROGRAM, "source": "BF16 target; calibration only", "base_revision": BASE_REVISION,
              "corpus_lock_sha256": sha(lockfile), "script_sha256": sha(__file__),
              "torch": torch.__version__, "transformers": transformers.__version__, "chunk": args.chunk,
              "linear_count": len(scales), "calibration_runs": runs,
              "scales": {names[n]+".input_scale": {"multiplier": s, "amax": state["maxima"][n], "hf_name": n}
                         for n, s in scales.items()}}
    atomic(args.out / "activation-scales.json", record)
    phase.update(kind="diagnostic", scales=scales)
    diagnostic = {"program": PROGRAM, "source_program": "BF16 activations, calibrated quantizer observed without feedback",
                  "definition": "fp4_max_fraction counts RN E2M1 +/-6; global_clip_fraction counts abs(x)>2688*s; block_scale_clip_fraction counts raw UE4M3 scale>448",
                  "native_mint_diagnostic_required": True, "runs": []}
    for row in manifest["samples"]:
        if row["split"] != "diagnostic":
            continue
        for length in (8192, 32768, 131070):
            for n in names:
                counters[n].zero_()
                maxima[n].zero_()
            elapsed = forward(row, length)
            per = {}
            for n in names:
                count, fpmax, clipped, blocks, nonfinite = counters[n].cpu().tolist()
                per[names[n]] = {"values": int(count), "fp4_max_values": int(fpmax),
                                 "global_clipped_values": int(clipped), "clipped_blocks": int(blocks),
                                 "nonfinite_values": int(nonfinite), "amax": float(maxima[n].cpu()),
                                 "fp4_max_fraction": fpmax/count, "global_clip_fraction": clipped/count,
                                 "block_scale_clip_fraction": blocks/(count/16)}
            diagnostic["runs"].append({"id": row["id"], "tokens": length, "elapsed_s": elapsed, "projections": per})
            atomic(args.out / "clipping-bf16.json", diagnostic)
    for h in handles:
        h.remove()
    print(json.dumps({"event": "complete", "scales": len(scales), "diagnostic_runs": len(diagnostic["runs"])}), flush=True)


if __name__ == "__main__":
    main()
