# Lever ledger: GB10 prime attention, 2026-11-07

One row per arm tried: what changed, hardware, shape, n, baseline, result, commit and receipt.
Model: the owner GGUF Qwen3.8-27B (NVFP4 + Q5_K) with the masked MTP draft; attention shape 24 query
/ 4 KV heads x 256, causal, BF16 K/V workspace, f32 Q/O. Hardware: DGX Spark (GB10, sm_121a, 48 SMs,
2,177 MHz under load), CUDA 13.0 on the device. Continues `research/gb10-prefill-20261106/` (its
rows 0-15).

Conventions:

- **Probe:** `probe/probe_attn.cu` on unit 2, the real kernels compiled from `flash_attn.cu` with
  the fatbin build's flags, T = 4,096 rows at depths 4,096 / 16,384 / 65,536 / 131,072, median of 3
  after one warmup, TFLOPS on the causal work, every candidate's O compared with h2's bit for bit.
- **Served:** cold TTFT of one fresh prompt per length per boot (`harness/pf_long_memra.py`), fresh
  boot per cell, arms interleaved in the order written.
- Every arm below is exact unless the row says otherwise (0 mismatching floats at every depth).

| # | arm | change | n | baseline (h2) | result | sha | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|
| A0 | ceilings | bf16 m16n8k16 forms at 4 and 8 warps/SM; h2 body without staging/barriers and without softmax | 5 reps | | 8 warps/SM: 8 acc 101.4, 4 chains 106.8, 4 chains + ldmatrix 102.1 TFLOPS; h2 body 67.8 -> 77.5 (no staging) -> 96.4 (no softmax) | | ceilings/, probe-runs/probe-r1.log | measurement |
| A1 | h2c | h2's body copied into the probe | 3 | 67.78 | 67.86 TFLOPS at 128k | | probe-r1 | control |
| A2 | sb2 | one CTA barrier per tile (wait, barrier, prefetch next) | 3 | 67.78 | 68.17 (+0.6%) | | probe-r1 | rejected (flat) |
| A3 | ad | precomputed ldmatrix offsets, K lanes ordered for consecutive B registers (generic address) | 3 | 67.78 | 69.25 (+2.2%) | | probe-r1 | folded into A14 |
| A4 | sbad | A2 + A3 | 3 | 67.78 | 69.54 (+2.6%) | | probe-r1 | superseded |
| A5 | qh8 | A2 + Q k-tiles 8..15 from smem (32 registers freed, 96 KB smem) | 3 | 67.78 | 70.14 (+3.5%) | | probe-r1 | superseded |
| A6 | qh12 | A2 + Q k-tiles 12..15 from smem | 3 | 67.78 | 71.29 (+5.2%) | | probe-r1 | superseded |
| A7 | qh8db | h2 barriers + A5 | 3 | 67.78 | 68.28 (+0.7%) | | probe-r1 | rejected |
| A8 | fa2s | all of Q in smem, FA2's single K and V planes and stagger | 3 | 67.78 | 65.35 (-3.6%) | | probe-r1 | rejected |
| A9 | pp | three-stage ring, per-stage full/empty mbarriers, head B one QK behind head A | 3 | 67.78 | 73.94 (+9.1%) | | probe-r1 | superseded by A14 |
| A10 | pp2 | A9 with a precomputed-address producer | 3 | 68.00 | 71.78 (+5.6%) | | probe-r2 | superseded |
| A11 | ppad | A10 + A3 | 3 | 68.00 | 76.64 (+12.7%) | | probe-r2 | superseded by A14 |
| A12 | ppadm / ppm | A11 / A10 + no mask compares on interior tiles | 3 | 68.00 | 76.63 / 70.70 | | probe-r2 | rejected (flat / worse) |
| A13 | ad, sbad with `ldmatrix.shared` | 32-bit shared offsets, immediates fold, no 64-bit address pairs | 3 | 67.89 | 70.78 / 71.04 (+4.3% / +4.6%) | | probe-r3 | folded into A14 |
| A14 | **h2p** (ppad with `ldmatrix.shared`) | A11 + A13 | 3 | 67.89 | **85.40 (+25.8%)**; 4k 62.70, 16k 79.19, 64k 84.74; 262,144 depth 305.7 vs 384.5 ms | b6604e1b90 (probe copy `cand_ppad` in `probe/attn_h2p.cuh`), landed body `fa_prefill_qw_h2p` | probe-r3, edge.log | **adopted** (served, row A23) |
| A15 | ppadm with `.shared` | A14 + no interior mask compares | 3 | 67.89 | 82.64 (-3.2% vs A14) | | probe-r3 | rejected |
| A16 | ppadz | A14 + explicit zero C operand on each chain's first MMA | 3 | 67.89 | 85.06 (flat vs A14; ptxas already uses RZ) | | probe-r3 | rejected |
| A19 | h2p, shipped symbol | `fa_prefill_qw_h2p` from `flash_attn.cu`, round 4 on unit 1, two passes | 3 + 3 | 67.92 / 67.99 | 85.04 / 85.14 (+25.2%); 4k 62.56 / 62.31, 16k 78.35 / 78.33, 64k 84.51 / 83.84; 21 ragged cells exact, 262,144 depth 305.7 vs 384.4 ms | 6f59cac641 (flash_attn.cu sha256 e759f4ac) | probe-r4-unit1, edge-h2p-unit1 | **adopted** (served, row A23) |
| A20 | sk1 | A14 with head B released after head A's first softmax | 3 | 67.92 | 84.94 (flat vs A19) | | probe-r4-unit1 | rejected (flat) |
| A21 | sk2 | A14 with head B released after head A's first tile | 3 | 67.92 | 78.10 (-8.2% vs A19) | | probe-r4-unit1 | rejected |
| A22 | pqk | A14 + QK(i+1) issued before softmax(i) in each warp | 3 | 67.92 | 56.48 (-33.6% vs A19); 32 B stack | | probe-r4-unit1 | rejected |
| A23 | h2p served | GB10 default, base main `6f6d816757` | B H H B B H, unit 1, 32k/131k/260k per boot | 32.574 / 167.899 / 432.708 s (medians of 3) | 32.056 / 158.237 / 393.874 s (-1.59% / -5.75% / -8.97%), every h2p boot below every main boot, text identical, MemAvailable floor 66.6 GiB | 6f59cac641 | served-unit1-final/, gates-unit1-final/ | **adopted** (GB10 default) |
| A17 | split-KV | partial softmax states merged by rescaling | | | not built: a different float program | | README | rejected (not exact) |
| A18 | persistent BF16 workspace | write each row's dequantized K/V once instead of rewriting the prefix every chunk | | | not built: about 1.4 s at 131k, 5 s at 260k, 17 GB of workspace at 260k | | README | open |
