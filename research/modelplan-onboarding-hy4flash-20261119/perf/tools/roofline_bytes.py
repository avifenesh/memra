#!/usr/bin/env python3
"""Bytes a c=1 decode step of the hy4f-gpu route reads, from the checkpoint's safetensors headers.

usage: roofline_bytes.py CKPT_DIR [CTX ...]
Counts what the route holds resident and what one decode token touches: every non-expert weight
once, 16 routed experts per MoE layer (NVFP4 codes + E4M3 group scales + scale_2), one embedding
row, the head, Engram rows (heads x dim FP8 + scale per Engram layer), and the KV cache read at
context CTX (F32 576 a key; sliding layers <= 512 keys; indexed layers <= topk keys plus the
indexer's pooled keys, 128 F32 a group).
"""
import json, struct, sys, os, re
from collections import defaultdict

d = sys.argv[1]
ctxs = [int(x) for x in sys.argv[2:]] or [36, 2104]
cfg = json.load(open(os.path.join(d, "config.json")))
tc = cfg.get("text_config", cfg)
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
tensors = {}
for f in files:
    with open(os.path.join(d, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        tensors[k] = (v["dtype"], v["shape"], b - a)

cat = defaultdict(int)
expert_layers = defaultdict(int)
nexp = {}
for k, (dt, sh, nb) in tensors.items():
    if not k.startswith("model.layers.") and not k.startswith("lm_head") and not k.startswith("model."):
        cat["other(not text)"] += nb
        continue
    if ".engram_layer.multi_head_embedding" in k:
        cat["engram_table(disk)"] += nb
        continue
    if "mtp" in k or "visual" in k or "vision" in k:
        cat["unused(" + k.split(".")[1] + ")"] += nb
        continue
    m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.", k)
    if m:
        L = int(m.group(1))
        expert_layers[L] += nb
        nexp[L] = sh[0]
        continue
    if k == "model.embed_tokens.weight":
        cat["embed"] += nb
        continue
    if k == "lm_head.weight":
        cat["head"] += nb
        continue
    part = "attn" if ".self_attn." in k else ("mlp" if ".mlp." in k else ("engram_proj" if "engram" in k else "misc"))
    if ".compressed_indexer." in k:
        part = "indexer"
    if ".shared_experts." in k:
        part = "shared"
    if ".mlp.gate." in k:
        part = "router"
    if "routed_expert_" in k:
        part = "latent_proj"
    cat[part] += nb

topk = tc.get("num_experts_per_tok", 16)
exp_total = sum(expert_layers.values())
exp_active = sum(v * topk / nexp[L] for L, v in expert_layers.items())
G = 1e9
print("tensors", len(tensors))
for k in sorted(cat):
    print(f"  {k:24s} {cat[k]/G:9.3f} GB")
print(f"  experts total            {exp_total/G:9.3f} GB over {len(expert_layers)} layers ({topk} of {set(nexp.values())})")
print(f"  experts active / token   {exp_active/G:9.3f} GB")
# kv bytes; the route keeps the tied kv_b both as stored and transposed nope copy (reads value rows + kb_t)
nonexp = sum(v for k, v in cat.items() if k in ("attn", "indexer", "shared", "router", "latent_proj", "mlp", "engram_proj", "misc"))
print(f"  non-expert per token     {nonexp/G:9.3f} GB (+head {cat['head']/G:.3f})")
L = tc["num_hidden_layers"]
print(json.dumps({k: tc.get(k) for k in ("sliding_window", "index_topk", "engram_layer_ids", "layer_types") if k in tc})[:600])
