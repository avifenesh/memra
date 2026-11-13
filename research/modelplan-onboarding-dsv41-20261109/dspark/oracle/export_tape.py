"""Export a DSpark draft tape (spec_oracle.py, <id>.pt) as raw little-endian arrays for `dsv41-spec` (draft-tape job)
and for draft_cmp.py: the inputs forward_spec consumed and what it produced, one file per tensor.

usage: export_tape.py TAPE_DIR OUT_DIR ID [ID ...]
Writes OUT_DIR/<id>/: meta.json, prompt.i32, gen.i32, prompt_mh.bf16 [s][3 dim], and per step (n steps stacked):
tok_in.i32, pos.i32, mh.bf16 [n][3 dim], main_x.bf16 [n][dim], h<k>.bf16 [n][block][hc][dim], pre<k>.f32 [n][block][hc],
attn_in<k>/attn_out<k>/ffn_out<k>.bf16 [n][block][dim], route_i<k>.i32 [n][block][topk], head_x.bf16 [n][block][dim],
out_ids.i32 [n][block] (the drafts), conf.f32 [n][block], lse.f32 [n][block], top_ids.i32 / top_vals.f32 [n][block][32],
logits_full.f32 / bias_full.f32 [f][block][vocab] (the first f steps), ring0.bf16 / ring1.bf16 [stages][window][head_dim]
(after the prefill seed, at the end)."""
import json, os, sys
import numpy as np
import torch

src, dst, ids = sys.argv[1], sys.argv[2], sys.argv[3:]


def raw(t):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype in (torch.int64, torch.int32):
        return t.to(torch.int32).numpy().astype("<i4").tobytes()
    return t.to(torch.float32).numpy().astype("<f4").tobytes()


for cid in ids:
    r = torch.load(os.path.join(src, f"{cid}.pt"))
    out = os.path.join(dst, cid)
    os.makedirs(out, exist_ok=True)
    st = r["steps"]

    def put(name, t):
        open(os.path.join(out, name), "wb").write(raw(t))

    put("prompt.i32", torch.tensor(r["prompt_ids"]))
    put("gen.i32", torch.tensor(r["gen_ids"]))
    put("prompt_mh.bf16", r["prompt_main_hidden"])
    put("ring0.bf16", r["ring_after_prefill"])
    put("ring1.bf16", r["ring_final"])
    n = len(st)
    if n:
        put("tok_in.i32", torch.tensor([s["tok_in"] for s in st]))
        put("pos.i32", torch.tensor([s["pos"] for s in st]))
        put("mh.bf16", torch.stack([s["main_hidden"] for s in st]))
        put("main_x.bf16", torch.stack([s["main_x"][0] for s in st]))
        stages = sum(1 for k in st[0] if k.startswith("pre"))
        for k in range(stages):
            put(f"h{k}.bf16", torch.stack([s[f"h{k}"] for s in st]))
            put(f"pre{k}.f32", torch.stack([s[f"pre{k}"] for s in st]))
            put(f"attn_in{k}.bf16", torch.stack([s[f"attn_in{k}"] for s in st]))
            put(f"attn_out{k}.bf16", torch.stack([s[f"attn_out{k}"] for s in st]))
            put(f"ffn_out{k}.bf16", torch.stack([s[f"ffn_out{k}"] for s in st]))
            put(f"route_i{k}.i32", torch.stack([s[f"route_i{k}"] for s in st]))
        put("head_x.bf16", torch.stack([s["head_x"] for s in st]))
        put("out_ids.i32", torch.stack([s["out_ids"][1:] for s in st]))
        put("conf.f32", torch.stack([s["conf"] for s in st]))
        put("lse.f32", torch.stack([s["lse"] for s in st]))
        put("top_ids.i32", torch.stack([s["top_ids"] for s in st]))
        put("top_vals.f32", torch.stack([s["top_vals"] for s in st]))
        full = [s["full_logits"] for s in st if s["full_logits"] is not None]
        if full:
            put("logits_full.f32", torch.stack(full))
            put("bias_full.f32", torch.stack([s["full_bias"] for s in st if s["full_bias"] is not None]))
    else:
        stages, full = 0, []
    meta = {"id": cid, "prompt_len": len(r["prompt_ids"]), "gen": len(r["gen_ids"]), "steps": n, "stages": stages,
            "full_steps": len(full), "config": r["config"], "greedy_prefix": r["greedy_prefix"]}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print(json.dumps(meta)[:300])
