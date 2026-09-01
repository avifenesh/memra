#!/usr/bin/env python3
"""Independent NumPy reference for the gemma-4 vision tower (gemma4v mmproj).

Implements the law derived from llama.cpp clip_graph_gemma4v from the SAME mmproj
weights, with a code path fully independent of the Rust implementation. Compares
per-token cosine at every dumped stage.

Usage: gemma_vision_ref.py <mmproj.gguf> <dump_dir>   (dump_dir from gemma_vision_oracle)
"""
import sys

import numpy as np

# point GGUF_PY at a llama.cpp checkout's gguf-py if the package is not installed
import os

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader

HIDDEN, HEADS, HEAD_DIM, INTER, DEPTH = 1152, 16, 72, 4304, 27
MERGE, EPS, THETA = 3, 1e-6, 100.0

mmproj, dump = sys.argv[1], sys.argv[2]
r = GGUFReader(mmproj)
T = {t.name: t for t in r.tensors}


def w(name):
    t = T[name]
    d = t.data
    if d.dtype == np.uint8:  # raw bytes (BF16): pair to u16 then shift to f32
        d = d.view(np.uint16)
    if d.dtype == np.uint16:
        return (d.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return d.astype(np.float64)


def rms(x, weight=None, eps=EPS):
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = x * inv
    return y * weight if weight is not None else y


gw, gh = map(int, open(f"{dump}/grid.txt").read().split())
n = gw * gh
patches = np.fromfile(f"{dump}/patches.bin", dtype=np.float32).astype(np.float64).reshape(n, 768)

# patch embed (conv-as-linear, no bias) + factored additive x/y tables
pe = w("v.patch_embd.weight").reshape(HIDDEN, 768)
x = patches @ pe.T
pos = w("v.position_embd.weight").reshape(2, 10240, HIDDEN)
cols = np.arange(n) % gw
rows = np.arange(n) // gw
x = x + pos[0][cols] + pos[1][rows]

stages = {"pre_blocks": x.copy()}

# rope tables: first 36 dims by x, last 36 by y; neox pairs (i, i+18) inside each half
quarter = HEAD_DIM // 4  # 18
inv_freq = THETA ** (-2.0 * np.arange(quarter) / (HEAD_DIM / 2))
ang_x = cols[:, None] * inv_freq[None, :]
ang_y = rows[:, None] * inv_freq[None, :]


def rope_half(h, ang):
    # h: [n, heads, 36]; pairs (i, i+18)
    a, b = h[..., :quarter], h[..., quarter:]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


for il in range(DEPTH):
    p = f"v.blk.{il}"
    h = rms(x, w(f"{p}.ln1.weight"))
    q = (h @ w(f"{p}.attn_q.weight").reshape(HIDDEN, HIDDEN).T).reshape(n, HEADS, HEAD_DIM)
    k = (h @ w(f"{p}.attn_k.weight").reshape(HIDDEN, HIDDEN).T).reshape(n, HEADS, HEAD_DIM)
    v = (h @ w(f"{p}.attn_v.weight").reshape(HIDDEN, HIDDEN).T).reshape(n, HEADS, HEAD_DIM)
    q = rms(q, w(f"{p}.attn_q_norm.weight"))
    k = rms(k, w(f"{p}.attn_k_norm.weight"))
    v = rms(v)  # weightless V norm (gemma4v-only)
    q = np.concatenate([rope_half(q[..., :36], ang_x), rope_half(q[..., 36:], ang_y)], -1)
    k = np.concatenate([rope_half(k[..., :36], ang_x), rope_half(k[..., 36:], ang_y)], -1)
    # UNSCALED full attention (kq_scale = 1.0)
    att = np.einsum("qhd,khd->hqk", q, k)
    att = att - att.max(-1, keepdims=True)
    att = np.exp(att)
    att /= att.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", att, v).reshape(n, HIDDEN)
    o = o @ w(f"{p}.attn_out.weight").reshape(HIDDEN, HIDDEN).T
    x = x + rms(o, w(f"{p}.attn_post_norm.weight"))
    h = rms(x, w(f"{p}.ln2.weight"))
    gate = h @ w(f"{p}.ffn_gate.weight").reshape(INTER, HIDDEN).T
    up = h @ w(f"{p}.ffn_up.weight").reshape(INTER, HIDDEN).T
    act = gate / (1.0 + np.exp(-1.702 * gate)) * up  # gelu_quick(gate) * up
    d = act @ w(f"{p}.ffn_down.weight").reshape(HIDDEN, INTER).T
    x = x + rms(d, w(f"{p}.ffn_post_norm.weight"))
    if il == 0:
        stages["blk0"] = x.copy()

stages["post_blocks"] = x.copy()

# head: 3x3 avg-pool, *sqrt(1152), std affine, weightless RMS, project
g = x.reshape(gh, gw, HIDDEN).reshape(gh // MERGE, MERGE, gw // MERGE, MERGE, HIDDEN)
pooled = g.mean((1, 3)).reshape(-1, HIDDEN) * np.sqrt(HIDDEN)
pooled = (pooled - w("v.std_bias")) * w("v.std_scale")
pooled = rms(pooled)
stages["pre_proj"] = pooled.copy()
proj = pooled @ w("mm.input_projection.weight").reshape(5376, HIDDEN).T
stages["projected"] = proj


def cos_stats(a, b):
    a = a.reshape(b.shape)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    c = num / den
    return c.min(), c.mean()


fail = False
for tag, ref in stages.items():
    try:
        got = np.fromfile(f"{dump}/rust_{tag}.bin", dtype=np.float32).astype(np.float64)
    except FileNotFoundError:
        print(f"{tag:12s} (no rust dump)")
        continue
    mn, mean = cos_stats(got, ref)
    ok = mn > 0.9995
    fail |= not ok
    print(f"{tag:12s} min_cos {mn:.6f} mean_cos {mean:.6f} {'OK' if ok else 'DIVERGED'}")

sys.exit(1 if fail else 0)
