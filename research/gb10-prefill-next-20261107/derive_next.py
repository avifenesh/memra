#!/usr/bin/env python3
"""derive_next.py: DRAM bytes of the non-attention, non-GEMM prime blocks of Qwen3.8-27B on one
4,096-row prime chunk, before and after this lane's levers, and the time each takes at the GB10's
measured stream ceilings (read 232.5 GB/s, copy 215.3 GB/s; research/gb10-prefill-20261106).

Shapes (the GGUF's qwen35 config): n_embd 5,120; n_ff 17,408; 48 GDN layers with 16 key heads and
48 value heads of 128 (conv_dim 10,240, value_dim 6,144), conv kernel 4, chunk C = 32; 16 full
attention layers with 24 query and 4 KV heads of 256 (q projection 12,288 with the output gate).
Bytes count each tensor once per pass that touches DRAM (a write plus every later read); the
L2 (24 MB) holds no 4,096-row tensor of these widths, so passes do not share bytes."""

T, E, FF = 4096, 5120, 17408
NV, NK, D, C = 48, 16, 128, 32
NC = T // C
CONV = 2 * NK * D + NV * D
F4 = 4
MB = 1e6
COPY = 215.3e9


def gb(x):
    return x / 1e9


rows = []


def row(block, before, after, note):
    rows.append((block, before, after, note))


act = T * E * F4                      # one [T, n_embd] f32 tensor
e4 = T * E + (E // 32) * T * 4        # its e4m3 v2 scratch (values + scales)
q8 = T * E + (E // 32) * T * 4        # its q8_1 pair

# residual add + norm + the activation quantize, two sites per layer (64 layers)
before = (3 * act) + (2 * act) + (act + e4)            # add, rms_norm, quantize
after_z = 2 * act + act + e4                           # fused: read a, b; write res, e4m3
after_h = 2 * act + 2 * act + e4                       # ... and the f32 norm output
row("add + RMSNorm + e4m3, FFN site (x64)", before, after_z, "z f32 not written")
row("add + RMSNorm + e4m3, mixer site (x64)", before, after_h, "h f32 kept for small projections")
# GDN layers also quantize h to q8_1 twice (beta, alpha)
row("q8_1 of h for beta and alpha (x48)", 2 * (act + q8), q8, "once, from the fused norm")

# GDN q/k: broadcast [T, 48, 128] vs compact [T, 16, 128]
qk_b, qk_c = T * NV * D * F4, T * NK * D * F4
row("conv: q/k writes (x48)", 2 * qk_b, 2 * qk_c, "compact layout")
row("l2 norm q and k + k bf16 twin (x48)", 4 * qk_b + qk_b // 2, 4 * qk_c + qk_c // 2, "compact layout")
row("K2 q/k reads (x48)", 2 * qk_b, 2 * qk_c, "heads-fastest grid: L2 shares a key head")
row("K3 k read (x48)", qk_b, qk_c, "heads-fastest grid")
row("K3 f32 W store (x48)", NC * NV * C * D * F4, 0, "only the bf16 twin has a reader")
row("K4 k bf16 read (x48)", qk_b // 2, qk_c // 2, "compact layout")
row("K5 q read (x48)", qk_b, qk_c, "heads-fastest grid")

tot_b = tot_a = 0.0
mult = {"(x64)": 64, "(x48)": 48}
print(f"{'block':44s} {'before MB':>10s} {'after MB':>10s} {'saved ms/chunk':>15s}  note")
for block, b, a, note in rows:
    k = next(v for s, v in mult.items() if s in block)
    tot_b += k * b
    tot_a += k * a
    print(f"{block:44s} {b / MB:10.1f} {a / MB:10.1f} {k * (b - a) / COPY * 1e3:15.2f}  {note}")
saved = (tot_b - tot_a) / COPY
print(f"\nper 4,096-row chunk: {gb(tot_b):.2f} GB -> {gb(tot_a):.2f} GB, {saved * 1e3:.1f} ms at 215.3 GB/s")
for n in (32_784, 131_051, 259_951):
    print(f"  {n:7d} tokens ({n / T:5.2f} chunks): {saved * n / T:6.2f} s")
