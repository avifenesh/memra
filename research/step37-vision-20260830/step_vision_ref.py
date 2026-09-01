#!/usr/bin/env python3
"""Independent NumPy reference for the step37 (Step-3.7-Flash) vision tower.

Implements the law derived from the checkpoint's own vendor code (vision_encoder.py /
modeling_step3p7.py at the pinned rev) from the SAME safetensors weights, with a code
path fully independent of the Rust implementation (and of torch). Compares per-token
cosine at every stage the memra oracle dumps.

Usage: step_vision_ref.py <model_dir> <dump_dir>   (dump_dir from step_vision_oracle)

Weights are read straight from the safetensors shards (manual header parse, no
dependency beyond numpy); model.safetensors.index.json routes tensor names.
"""
import json
import os
import struct
import sys

import numpy as np

HIDDEN, HEADS, HEAD_DIM, INTER, DEPTH = 1536, 16, 96, 8960, 47
PATCH, POS_GRID, EPS, THETA = 14, 52, 1e-5, 10000.0

model_dir, dump = sys.argv[1], sys.argv[2]

# ---- safetensors reader (BF16 -> f64) ----
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
weight_map = idx["weight_map"]
_shards = {}


def shard(fname):
    if fname not in _shards:
        f = open(os.path.join(model_dir, fname), "rb")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        _shards[fname] = (f, 8 + hlen, header)
    return _shards[fname]


def w(name):
    f, base, header = shard(weight_map[name])
    info = header[name]
    lo, hi = info["data_offsets"]
    f.seek(base + lo)
    raw = f.read(hi - lo)
    if info["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    elif info["dtype"] == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    else:
        raise SystemExit(f"{name}: unsupported dtype {info['dtype']}")
    return arr.reshape(info["shape"])


P = "model.vision_model"
g = int(open(f"{dump}/grid.txt").read().split()[0])
n = g * g
patches = np.fromfile(f"{dump}/patches.bin", dtype=np.float32).astype(np.float64).reshape(n, 588)

# ---- patch embed (conv14-as-linear, no bias) + abs posemb + ln_pre ----
conv1 = w(f"{P}.conv1.weight").reshape(HIDDEN, 588)
x = patches @ conv1.T

pos = w(f"{P}.positional_embedding")  # [2704, 1536]
if g != POS_GRID:
    # F.interpolate bilinear align_corners=False from 52x52 to gxg
    src = pos.reshape(POS_GRID, POS_GRID, HIDDEN)
    out = np.zeros((g, g, HIDDEN))
    scale = POS_GRID / g
    for y in range(g):
        sy = min(max((y + 0.5) * scale - 0.5, 0.0), POS_GRID - 1)
        y0 = int(np.floor(sy))
        y1 = min(y0 + 1, POS_GRID - 1)
        fy = sy - y0
        for xg in range(g):
            sx = min(max((xg + 0.5) * scale - 0.5, 0.0), POS_GRID - 1)
            x0 = int(np.floor(sx))
            x1 = min(x0 + 1, POS_GRID - 1)
            fx = sx - x0
            out[y, xg] = (
                src[y0, x0] * (1 - fy) * (1 - fx)
                + src[y0, x1] * (1 - fy) * fx
                + src[y1, x0] * fy * (1 - fx)
                + src[y1, x1] * fy * fx
            )
    pos = out.reshape(g * g, HIDDEN)
x = x + pos


def layernorm(v, weight, bias, eps=EPS):
    mu = v.mean(-1, keepdims=True)
    var = ((v - mu) ** 2).mean(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * weight + bias


x = layernorm(x, w(f"{P}.ln_pre.weight"), w(f"{P}.ln_pre.bias"))


def cos_report(stage, ours):
    ref = np.fromfile(f"{dump}/rust_{stage}.bin", dtype=np.float32).astype(np.float64)
    ref = ref.reshape(ours.shape)
    dot = (ours * ref).sum(-1)
    denom = np.sqrt((ours * ours).sum(-1) * (ref * ref).sum(-1))
    cos = dot / np.maximum(denom, 1e-30)
    print(f"{stage:14s} min_cos {cos.min():.6f}  mean_cos {cos.mean():.6f}")
    return cos.min()


results = {}
results["pre_blocks"] = cos_report("pre_blocks", x)

# ---- 2D rope tables: col angles on dims 0..48, row angles on 48..96, interleaved
#      pairs (2i, 2i+1) sharing one angle; inv_freq[i] = theta^(-2i/48) ----
quarter = HEAD_DIM // 4  # 24
inv_freq = THETA ** (-2.0 * np.arange(quarter) / (HEAD_DIM // 2))
rows_idx = np.arange(n) // g
cols_idx = np.arange(n) % g
ang_col = cols_idx[:, None] * inv_freq[None, :]  # [n, 24]
ang_row = rows_idx[:, None] * inv_freq[None, :]
# per-dim angle vector [n, 96]: repeat_interleave(2) inside each half
ang = np.concatenate([np.repeat(ang_col, 2, axis=1), np.repeat(ang_row, 2, axis=1)], axis=1)
COS, SIN = np.cos(ang), np.sin(ang)


def rope(t):  # t: [n, heads, 96]
    # rotate_half: pairs (2i, 2i+1) -> (-b, a)
    rot = np.empty_like(t)
    rot[..., 0::2] = -t[..., 1::2]
    rot[..., 1::2] = t[..., 0::2]
    return t * COS[:, None, :] + rot * SIN[:, None, :]


def softmax(v):
    m = v.max(-1, keepdims=True)
    e = np.exp(v - m)
    return e / e.sum(-1, keepdims=True)


scale = 1.0 / np.sqrt(HEAD_DIM)
for il in range(DEPTH):
    bp = f"{P}.transformer.resblocks.{il}"
    res = x
    h = layernorm(x, w(f"{bp}.ln_1.weight"), w(f"{bp}.ln_1.bias"))
    qkv = h @ w(f"{bp}.attn.in_proj_weight").T + w(f"{bp}.attn.in_proj_bias")
    q, k, v = np.split(qkv, 3, axis=-1)
    q = rope(q.reshape(n, HEADS, HEAD_DIM))
    k = rope(k.reshape(n, HEADS, HEAD_DIM))
    v = v.reshape(n, HEADS, HEAD_DIM)
    att = softmax(np.einsum("qhd,khd->hqk", q, k) * scale)
    o = np.einsum("hqk,khd->qhd", att, v).reshape(n, HIDDEN)
    o = o @ w(f"{bp}.attn.out_proj.weight").T + w(f"{bp}.attn.out_proj.bias")
    x = res + o * w(f"{bp}.ls_1.gamma")
    res = x
    h = layernorm(x, w(f"{bp}.ln_2.weight"), w(f"{bp}.ln_2.bias"))
    f1 = h @ w(f"{bp}.mlp.c_fc.weight").T + w(f"{bp}.mlp.c_fc.bias")
    f1 = f1 / (1.0 + np.exp(-1.702 * f1))  # quick_gelu = x * sigmoid(1.702 x)
    f2 = f1 @ w(f"{bp}.mlp.c_proj.weight").T + w(f"{bp}.mlp.c_proj.bias")
    x = res + f2 * w(f"{bp}.ls_2.gamma")
    if il == 0:
        results["blk0"] = cos_report("blk0", x)

# NO ln_post (use_ln_post false)
results["post_blocks"] = cos_report("post_blocks", x)


def conv3x3s2(feat, g_in, weight, bias):
    """feat token-major [g*g, C_in]; PyTorch Conv2d k3 s2 p1 semantics."""
    c_out, c_in = weight.shape[0], weight.shape[1]
    og = (g_in - 1) // 2 + 1
    fmap = feat.reshape(g_in, g_in, c_in)
    out = np.zeros((og, og, c_out))
    wf = weight.reshape(c_out, c_in * 9)
    for oy in range(og):
        for ox in range(og):
            win = np.zeros((c_in, 3, 3))
            for ky in range(3):
                for kx in range(3):
                    iy, ix = 2 * oy + ky - 1, 2 * ox + kx - 1
                    if 0 <= iy < g_in and 0 <= ix < g_in:
                        win[:, ky, kx] = fmap[iy, ix]
            out[oy, ox] = wf @ win.reshape(-1) + bias
    return out.reshape(og * og, c_out), og


d1, g1 = conv3x3s2(x, g, w(f"{P}.vit_downsampler1.weight"), w(f"{P}.vit_downsampler1.bias"))
d2, g2 = conv3x3s2(d1, g1, w(f"{P}.vit_downsampler2.weight"), w(f"{P}.vit_downsampler2.bias"))
results["downsampled"] = cos_report("downsampled", d2)

proj = d2 @ w("model.vit_large_projector.weight").T
results["projected"] = cos_report("projected", proj)

bar = 0.9997
verdict = "PASS" if results["projected"] >= bar else "FAIL"
print(f"projected min_cos {results['projected']:.6f} vs bar {bar} -> {verdict}")
sys.exit(0 if verdict == "PASS" else 1)
