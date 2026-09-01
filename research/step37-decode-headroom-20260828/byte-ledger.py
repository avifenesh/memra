#!/usr/bin/env python3
"""Weight bytes streamed PER CARD PER TOKEN at the step37 TP2 serving config.

This is the DENOMINATOR of the question "can a weight-precision door reach 90 tok/s".
Dims are read off the checkpoint's own config.json (hidden 4096, moe_intermediate 1280,
top_k 8, 288 experts, 42 MoE + 3 dense layers, dense intermediate 11264, share_expert_dim
1280, vocab 128896, 45 layers) under MEMRA_STEP_TP=0-44@0,1, so every term is the shape the
serving box actually streams, not an estimate.

Run: python3 byte-ledger.py
"""
H, MOEI, TOPK = 4096, 1280, 8
NL, NMOE, NDENSE = 45, 42, 3
DENSEI, SHEXP, VOCAB = 11264, 1280, 128896
Q8, BF16, NVFP4 = 34 / 32, 2.0, 0.5 + 1 / 16   # bytes per weight (q8_0 block = 34 B per 32)
TP = 2


def gb(x):
    return x / 1e9


attn = H * H + 2 * H * 512 + H * H              # q + k + v + o, per card under TP2
rows = [
    ("attention q/k/v/o (q8, BANKED)", NL * attn * Q8),
    ("routed experts top-8 (NVFP4)", NMOE * (TOPK * 3 * H * MOEI // TP) * NVFP4),
    ("shared expert (q8 hi half / bf16 lo half)", NL * (3 * H * SHEXP // TP) * ((Q8 + BF16) / 2)),
    ("dense FFN layers 0-2 (q8)", NDENSE * (3 * H * DENSEI // TP) * Q8),
    ("lm head lo half (bf16, DARK)", (VOCAB // 2) * H * BF16),
]
tot = sum(v for _, v in rows)
print("PER CARD PER TOKEN")
for k, v in rows:
    print("  %-42s %7.3f GB" % (k, gb(v)))
print("  %-42s %7.3f GB" % ("TOTAL", gb(tot)))
for label, tbs in (("device peak 1.79 TB/s", 1.79e12), ("blended achieved ~1.25 TB/s", 1.25e12)):
    print("  floor at %-30s %6.2f ms/token -> %6.1f tok/s" % (label, 1e3 * tot / tbs, tbs / tot))
print()
for tps in (78.6, 90.0):
    print("  %.1f tok/s = %.2f ms/token" % (tps, 1000.0 / tps))
print()
print("READING: weight streaming is a MINORITY of the measured token. The remainder is the")
print("attention walk, routing, per-launch latency, host issue and the cross-device joins,")
print("and NO weight-precision door (W8, W4, BF16_MMV) can touch any of it.")
