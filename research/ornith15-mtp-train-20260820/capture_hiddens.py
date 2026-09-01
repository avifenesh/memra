#!/usr/bin/env python3
"""Teacher-forced trunk hidden capture for the Ornith-1.5 MTP-head trainer.

Replays each own-gen corpus row (prompt via chat template, response re-rendered
exactly as served: think rows re-wrap reasoning + "\\n</think>\\n\\n" + content,
finish=length rows stay mid-stream, finish=stop rows get <|im_end|>) through the
pinned BF16 checkpoint and stores the PRE-final-norm trunk hidden h_i for every
position (captured by a pre-forward hook on the text model's final norm — the
exact seed memra's MTP glue consumes, hybrid.rs `hnorm(h_i)`).

Right-padded batches are safe here by causality: pads sit after the real tokens,
and both the causal full-attn mask and the GDN left-to-right scan mean a pad can
never influence an earlier real position — and only real positions are stored.

Shard output: mtp-train/hiddens/shard-NNNNN.pt = list of
{id, ids int32 [L], prompt_len, h bf16 [L-1, hidden]} (h[j] = h_j for j in
0..L-2 — row j+1 of the draft pairs (emb(t_{j+1}), h_j) and predicts t_{j+2}).
Resumable via manifest.jsonl; --follow trails a live corpus file.
"""
import argparse
import json
import pathlib
import time

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

IM_END = "<|im_end|>"


def render_ids(tok, row):
    think = row["mode"] != "nothink"
    enc = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        add_generation_prompt=True,
        enable_thinking=think,
    )
    # transformers 5.8.1 returns a BatchEncoding here, not a plain id list
    prompt_ids = enc if isinstance(enc, list) else enc["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    reasoning = row.get("reasoning") or ""
    content = row.get("content") or ""
    if think:
        if content:
            resp = reasoning + "\n</think>\n\n" + content
        else:
            resp = reasoning  # cut mid-think (finish=length)
    else:
        resp = content
    if row.get("finish_reason") == "stop":
        resp += IM_END
    resp_ids = tok(resp, add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-dir", required=True)
    ap.add_argument("--corpus", required=True, type=pathlib.Path)
    ap.add_argument("--corpus-log", type=pathlib.Path, help="driver log; DONE line ends --follow")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--shard-rows", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    done_ids = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    shard_idx = len(sorted(args.out_dir.glob("shard-*.pt")))

    tok = AutoTokenizer.from_pretrained(args.bf16_dir)
    print("loading model (bf16)...", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.bf16_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()

    # Pre-forward hook on the text model's final norm: its INPUT is the
    # pre-output_norm trunk hidden — memra's h_i seed (hybrid.rs).
    norm_mods = [m for n, m in model.named_modules() if n.endswith("language_model.norm")]
    assert len(norm_mods) == 1, f"final norm ambiguous: {len(norm_mods)}"
    captured = {}

    def hook(_mod, inputs):
        captured["h"] = inputs[0].detach()

    norm_mods[0].register_forward_pre_hook(hook)

    mf = open(manifest, "a", encoding="utf-8")
    shard, total_rows, t0 = [], 0, time.time()

    def flush_shard():
        nonlocal shard, shard_idx
        if not shard:
            return
        path = args.out_dir / f"shard-{shard_idx:05d}.pt"
        torch.save(shard, path)
        for rec in shard:
            mf.write(json.dumps({"id": rec["id"], "shard": path.name}) + "\n")
        mf.flush()
        shard_idx += 1
        shard = []

    def process_batch(batch):
        nonlocal total_rows
        maxlen = max(len(b["ids"]) for b in batch)
        input_ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["ids"])
            input_ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
            mask[i, :L] = 1
        with torch.no_grad():
            model(input_ids=input_ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        h = captured.pop("h")  # [B, maxlen, hidden]
        for i, b in enumerate(batch):
            L = len(b["ids"])
            shard.append({
                "id": b["id"],
                "ids": torch.tensor(b["ids"], dtype=torch.int32),
                "prompt_len": b["prompt_len"],
                "h": h[i, : L - 1].to(torch.bfloat16).cpu(),
            })
            total_rows += 1
            if len(shard) >= args.shard_rows:
                flush_shard()
        if total_rows % 256 < len(batch):
            print(f"captured {total_rows} rows, {time.time()-t0:.0f}s", flush=True)

    seen_offset = 0  # byte offset — corpus is read in binary so unicode never skews seeks
    final_pass = False
    while True:
        new = []
        with open(args.corpus, "rb") as f:
            f.seek(seen_offset)
            chunk = f.read()
            # only consume complete lines; a partial tail is re-read next pass
            last_nl = chunk.rfind(b"\n")
            if last_nl >= 0:
                seen_offset += last_nl + 1
                for line in chunk[: last_nl + 1].decode("utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row["id"] in done_ids:
                        continue
                    done_ids.add(row["id"])
                    new.append(row)
        if new:
            prepped = []
            for row in new:
                p_ids, r_ids = render_ids(tok, row)
                ids = (p_ids + r_ids)[: args.max_len]
                if len(ids) < len(p_ids) + 8:
                    continue  # response fully clipped — useless row
                prepped.append({"id": row["id"], "ids": ids, "prompt_len": len(p_ids)})
            prepped.sort(key=lambda b: len(b["ids"]))
            batch = []
            for b in prepped:
                cand_max = max(len(b["ids"]), max((len(x["ids"]) for x in batch), default=0))
                if batch and cand_max * (len(batch) + 1) > args.batch_tokens:
                    process_batch(batch)
                    batch = []
                batch.append(b)
            if batch:
                process_batch(batch)
        elif args.follow:
            log_done = args.corpus_log and args.corpus_log.exists() and \
                "DONE ok=" in args.corpus_log.read_text()[-2000:]
            if log_done:
                # driver flushes all rows before printing DONE — one more read
                # pass picks up anything written between our read and this check
                if final_pass:
                    break
                final_pass = True
                continue
            time.sleep(60)
            continue
        else:
            break
    flush_shard()
    mf.close()
    print(f"CAPTURE DONE rows={total_rows} shards={shard_idx} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
