#!/usr/bin/env python3
"""Lane E: tiny random Inkling 2 fixture in the CHECKPOINT layout (names, dtypes, quantized experts) for TinyParity.

The fixture is a small checkpoint directory: config.json (checkpoint-style keys), model.safetensors
(model.llm.* names; BF16 / F32 / U8 / F8_E4M3 as in the real file; NVFP4 and MXFP8 experts),
hf_quant_config.json (the real groups applied to the tiny names), prompt.json. Every float value is exactly
representable in its storage dtype, so an f64 run of it is the exact program.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ink2_oracle as O

VARIANTS = {
    # 4 blocks of 2 layers: prelude block, 2 looped core blocks, coda block -> 8 + 2*4 = 16 cache slots
    "tiny4": {"num_hidden_layers": 8, "core": (2, 6), "mxfp8_layers": [6, 7]},
    # 2 blocks of 2 layers: prelude block, 1 looped core block, no coda -> 4 + 2*2 = 8 cache slots
    "tiny2": {"num_hidden_layers": 4, "core": (2, 4), "mxfp8_layers": [3]},
}


def tiny_config(v):
    n = v["num_hidden_layers"]
    block = 2
    return {
        "dtype": "bfloat16",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "eos_token_id": 290,
        "text_config": {
            "model_type": "inkling2_loop_text", "model_max_length": 4096, "torch_dtype": "bfloat16",
            "hidden_size": 64, "num_hidden_layers": n, "vocab_size": 320, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "d_rel": 4, "rel_extent": 12, "q_bias": False, "o_bias": False,
            "log_scaling_n_floor": 24, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
            "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 4.0, "rel_l2_clip_threshold": 2.0,
            "local_layer_ids": [i for i in range(n) if (i + 1) % block != 0],
            "sconv_kernel_size": 4, "unpadded_vocab_size": 300, "final_logit_softcapping": 3.0,
            "swa_head_dim": 16, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "sliding_window_size": 8, "n_layers_per_attn_res_block": block, "attn_res_normalize_values": True,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": 32,
            "mlp_gate_softcap": 2.0, "mlp_up_softcap": 3.0, "route_scale": 2.0, "use_gate_bias": True,
            "loop_config": {"core_start": v["core"][0], "core_end": v["core"][1], "num_passes": 3, "gate_rank": 8,
                            "is_varloop": False},
        },
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def rng_for(seed, name):
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def value_std(name, shape, H):
    """(mean, std) of the normal a tensor is drawn from, by checkpoint name."""
    if name.endswith("embed.weight") and "unembed" not in name:
        return 0.0, 1.0
    if "unembed.weight" in name:
        return 0.0, 0.25
    if re.search(r"(norm|_norm)\.weight$", name):
        return 1.0, 0.2
    if "l2_clip.scale" in name:
        return 1.0, 0.3
    if "rel_logits_proj.proj" in name:
        return 0.0, 0.5
    if "sconv.weight_proj" in name:
        return 0.0, 0.5 / H ** 0.5
    if "sconv.static_w" in name or name.endswith("sconv.weight"):
        return 0.0, 0.5
    if "attn_sink_proj" in name:
        return 0.0, 1.5 / H ** 0.5
    if ".attn.wk_dv" in name:
        return 0.0, 2.0 / H ** 0.5
    if ".attn.wo_ud" in name:
        return 0.0, 1.0 / shape[1] ** 0.5
    if ".attn." in name:
        return 0.0, 1.0 / H ** 0.5
    if "controllers" in name:
        return 0.0, 1.0 / shape[1] ** 0.5
    if "mlp.gate.weight" in name:
        return 0.0, 2.0 / H ** 0.5
    if "mlp.gate.bias" in name:
        return 0.0, 0.3
    if "queries" in name or "output_query" in name:
        return 0.0, 1.0
    raise KeyError(name)


def quant_experts(seed, layer, kind, E, H, FF):
    """Random quantized expert tensors in the checkpoint layout."""
    out = {}
    p = f"model.llm.layers.{layer}.mlp.experts."
    for nm, (rows, cols) in (("w13", (2 * FF, H)), ("w2", (H, FF))):
        r = rng_for(seed, p + nm)
        if kind == "nvfp4":
            codes = r.integers(0, 256, size=(E, rows, cols // 2), dtype=np.uint8)
            sc = r.integers(0x70, 0x7F, size=(E, rows, cols // 16), dtype=np.uint8)  # 128 .. 448
            u = r.random(sc.shape)
            sc = np.where(u < 0.02, r.integers(0x01, 0x31, size=sc.shape, dtype=np.uint8), sc)  # subnormal .. tiny
            sc = np.where(u > 0.99, np.uint8(0), sc).astype(np.uint8)  # zero block
            amax = r.uniform(0.4, 1.0, size=(E,)).astype(np.float32)
            out[p + nm + "_weight"] = ("U8", codes)
            out[p + nm + "_weight_scale"] = ("F8_E4M3", sc)
            out[p + nm + "_weight_amax"] = ("F32", amax)
            out[p + nm + "_weight_outer_scale_denominator"] = ("F32", np.array([2688.0], dtype=np.float32))
        else:
            ex = r.integers(3, 7, size=(E, rows, cols), dtype=np.uint8)
            w = (r.integers(0, 2, size=ex.shape, dtype=np.uint8) << 7) | (ex << 3) | r.integers(0, 8, size=ex.shape, dtype=np.uint8)
            u = r.random(ex.shape)
            w = np.where(u < 0.02, np.where(u < 0.01, np.uint8(0x00), np.uint8(0x80)), w)  # +0 and -0
            w = np.where(u > 0.99, r.integers(0x01, 0x08, size=ex.shape, dtype=np.uint8), w).astype(np.uint8)  # subnormal
            sc = r.integers(124, 127, size=(E, rows, cols // 32), dtype=np.uint8)
            out[p + nm + "_weight"] = ("F8_E4M3", w)
            out[p + nm + "_weight_scale"] = ("U8", sc)
    return out


def write_safetensors(path, tensors):
    """tensors: name -> (dtype string, numpy array of the raw storage type)."""
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        dt, arr = tensors[name]
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def norm_name(k):
    return re.sub(r"\.\d+(\.|$)", r".N\1", k)


def cmd_gen(args):
    v = VARIANTS[args.variant]
    cfg_json = tiny_config(v)
    tc = cfg_json["text_config"]
    H, FF, E = tc["hidden_size"], tc["intermediate_size"], tc["n_routed_experts"]
    os.makedirs(args.out, exist_ok=True)
    # 1. discover the checkpoint names and shapes of the non-expert tensors through the PR's own reverse mapping
    m = O.load_modeling(False)
    cfg = O.build_text_config(m, cfg_json, {})
    model = m.Inkling2LoopForCausalLM(cfg)
    with tempfile.TemporaryDirectory(dir=args.out) as td:
        model.save_pretrained(td)
        saved = O.STFile(os.path.join(td, "model.safetensors"))
        shapes = {k: saved.info(k)[1] for k in saved.keys()}
    real = json.load(open(args.real_names))  # normalized checkpoint name -> dtype, from the real header
    tensors = {}
    for k, shape in sorted(shapes.items()):
        if O.is_expert_key(k):
            continue
        nk = norm_name(k)
        assert nk in real, f"PR reverse mapping produced a name the real checkpoint does not have: {k}"
        dt = real[nk]
        mean, std = value_std(k, shape, H)
        x = rng_for(args.seed, k).normal(mean, std, size=shape).astype(np.float32)
        if dt == "BF16":
            x16 = torch.from_numpy(x).to(torch.bfloat16)
            tensors[k] = ("BF16", x16.view(torch.int16).numpy().view(np.uint16))
        else:
            assert dt == "F32", (k, dt)
            tensors[k] = ("F32", x)
    for layer in range(tc["num_hidden_layers"]):
        kind = "mxfp8" if layer in v["mxfp8_layers"] else "nvfp4"
        tensors.update(quant_experts(args.seed, layer, kind, E, H, FF))
    have = {norm_name(k) for k in tensors}
    missing = sorted(set(real) - have)
    assert not missing, f"real checkpoint name classes the tiny fixture lacks: {missing}"
    write_safetensors(os.path.join(args.out, "model.safetensors"), tensors)
    json.dump(cfg_json, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    with open(os.path.join(args.out, "generation_config.json"), "w") as g:
        g.write("{}")
    # 2. quant config: the real groups, applied to the tiny names
    rq = json.load(open(args.real_quant))
    groups = []
    for g in rq["model"]["parameter_groups"]:
        pats = {norm_name(t) for t in g["tensors"]}
        mine = []
        for k in sorted(tensors):
            if norm_name(k) not in pats:
                continue
            if O.is_expert_key(k):
                if not k.endswith("_weight"):
                    continue
                layer = int(k.split(".")[3])
                kind = "mxfp8" if layer in v["mxfp8_layers"] else "nvfp4"
                if g["compute_dtype"] != kind:
                    continue
            elif g["compute_dtype"] in ("nvfp4", "mxfp8"):
                continue
            mine.append(k)
        gg = {kk: vv for kk, vv in g.items() if kk != "tensors"}
        gg["tensors"] = mine
        groups.append(gg)
    json.dump({"model": {"parameter_groups": groups}, "quant_method": rq.get("quant_method")},
              open(os.path.join(args.out, "hf_quant_config.json"), "w"), indent=1)
    # 3. prompt: random ids, long enough to pass the window (8), the relative extent (12) and the log floor (24)
    ids = rng_for(args.seed, "prompt").integers(0, tc["unpadded_vocab_size"], size=args.prompt_len).tolist()
    wsha = O.sha256_file(os.path.join(args.out, "model.safetensors"))
    json.dump({"name": args.variant + "_random", "note": "random token ids", "input_ids": ids, "weights_sha256": wsha},
              open(os.path.join(args.out, "prompt.json"), "w"))
    fx = {
        "variant": args.variant, "seed": args.seed, "pr_head": O.PR_HEAD,
        "files": {fn: O.sha256_file(os.path.join(args.out, fn)) for fn in ("model.safetensors", "config.json", "hf_quant_config.json", "prompt.json")},
        "tensors": {k: {"dtype": dt, "shape": list(a.shape), "sha256": O.sha256_bytes(np.ascontiguousarray(a).tobytes())} for k, (dt, a) in sorted(tensors.items())},
        "expert_formats": {str(l): ("mxfp8" if l in v["mxfp8_layers"] else "nvfp4") for l in range(tc["num_hidden_layers"])},
        "notes": [
            "checkpoint w13 rows are interleaved (gate_0, up_0, gate_1, up_1, ...)",
            "NVFP4: low nibble = even element; value = e2m1(code) * e4m3(scale) * amax / outer_scale_denominator; some scale blocks are zero or subnormal on purpose",
            "MXFP8: value = e4m3(byte) * 2^(scale - 127); some bytes are +0, -0 or subnormal on purpose",
            "unembed rows >= unpadded_vocab_size are random and must be dropped",
        ],
    }
    json.dump(fx, open(os.path.join(args.out, "fixture.json"), "w"), indent=1)
    print("tiny fixture", args.variant, len(tensors), "tensors", wsha)
    print("TINY_GEN_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="tiny4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--prompt-len", type=int, default=40)
    ap.add_argument("--real-names", required=True)
    ap.add_argument("--real-quant", required=True)
    args = ap.parse_args()
    cmd_gen(args)


if __name__ == "__main__":
    main()
