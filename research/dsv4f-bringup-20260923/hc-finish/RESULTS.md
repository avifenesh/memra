# DSv4 fused HC finish (memra #693)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 2026-09-23. Served program: PP-2, matrix expert
program, host sampler; DSpark rows add `MEMRA_DSV4_DRAFTER=dspark`. Lane `lane/dsv4-mhc-20260923`.

## What changed

Every HC entry site (the attention entry and the FFN entry of each layer, 86 per step on the
served program) ran, after the small-kernel diet (#675, #690): the split-dot partials, the slice
reduce, the one-CTA small HC kernel (rowsq, Sinkhorn, collapse), the entry RMSNorm and, at the
attention entry, a bf16 pack for the Q projection. `dsv4_hc_finish_f32_fixed_order_kernel<S>`
takes the slice sum, rowsq, Sinkhorn, collapse, RMSNorm and the pack in one CTA per position, so
a site is the partial kernel plus this one: 2 launches. Thread t owns `x[t + 128j]`, which is the
rowsq leaf order, the collapse column order and the register RMSNorm's column order, so x is read
once and y never leaves registers. Every sum keeps the tree of the kernel it replaces.

`hc_pre_norm_batch_dev` dispatches it on the diet shape with the split class on, at every row
count. The attention entry's bf16 pack feeds wq_a, wkv and the indexer weights projection
directly (`gemv_m_dev` on the packed row), which drops their three separate activation packs.

## Correctness

`crates/memra-engine/tests/dsv4_hc_finish_gpu.rs`, `NVIDIA_TF32_OVERRIDE=0`:

| run | result | log |
|---|---|---|
| one RTX PRO 6000 WS (600 W), first build | 144 cases exact, then the fn_w red arm FAILED: one weight of a 16384-term dot moved the sum by less than its ulp | `raw/card1/gate-r1.log`, `gate-r2.log` |
| same card, red arm bumps the whole weight row | `DSV4_HC_FINISH EXACT cases=144 outputs=7 slices=8,16,32 rows=1,2,5,6 red_arms=3`, twice | `raw/card1/gate-r3.log`, `gate-r4.log` |
| PRO 6000 pair, card 0, lane merged with main `6978f5fac` | same line | `raw/pair/component/gate.log` |
| served DSpark identity gate (`dsv4-gpu-dspark-gate --served`, tape 416) | `GPU DSPARK GATE [PASS]` | `raw/pair/dspark-served/gate.log` |

The first red-arm failure is kept as a record: the arm was too weak, not the kernel wrong.

## Component timing

`dsv4_hc_finish_timing`: device time per HC entry site as a CUDA event chain of 2000
back-to-back sites, the unfused chain, main's diet and the fused pair, 5 reps (spread under 0.1%).

| card | rows | chain us | main's diet us | fused us | fused vs diet |
|---|---|---|---|---|---|
| PRO 6000 WS, 600 W (pod) | 1 | 22.80 | 13.94 | 7.14 | -48.8% |
| PRO 6000 WS, pair card 0 (420 to 450 W) | 1 | 30.72 | 18.44 | 8.20 | -55.5% |
| same | 6 | 32.79 | 24.63 | 14.34 | -41.8% |

The first card's rows=6 diet column predates the timing fix that runs the diet at 6 rows.

## Served A/B

Shared-control campaign `../kernel-ab-short/` (protocol, all 40 rows, the queue script and the
binaries' hashes): one boot per row, base (main `6978f5fac`) and three lanes in a Williams order,
N=5 per arm per mode, 2x RTX PRO 6000 WS at a 450 W limit, PP-2, host sampler. The rows below are
this lane's and base's, in campaign order; `lane_tab.py short hc2` prints them.

### DSpark (`MEMRA_DSV4_DRAFTER=dspark`)

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 66.87 | 14.95 / 17.17 / 17.37 | 0.02 / 54.30 | 267 / 290 | 4,103 | 62.31 | 54.31 | 66.75 |
| r4 | hc2 | 68.30 | 14.64 / 16.79 / 16.98 | 0.02 / 53.18 | 267 / 284 | 4,008 | 63.62 | 55.26 | 68.21 |
| r6 | hc2 | 68.32 | 14.64 / 16.80 / 16.99 | 0.02 / 53.20 | 263 / 281 | 4,008 | 63.67 | 55.24 | 68.19 |
| r7 | base | 66.86 | 14.96 / 17.17 / 17.37 | 0.02 / 54.27 | 270 / 292 | 4,095 | 62.27 | 54.06 | 66.78 |
| r9 | hc2 | 68.32 | 14.64 / 16.79 / 16.98 | 0.02 / 53.16 | 263 / 281 | 4,007 | 63.70 | 55.17 | 68.23 |
| r12 | base | 66.88 | 14.95 / 17.18 / 17.37 | 0.02 / 54.30 | 267 / 284 | 4,091 | 62.34 | 54.25 | 66.73 |
| r14 | base | 66.90 | 14.95 / 17.17 / 17.37 | 0.02 / 54.32 | 270 / 286 | 4,096 | 62.29 | 53.99 | 66.74 |
| r15 | hc2 | 68.32 | 14.64 / 16.79 / 16.98 | 0.02 / 53.16 | 264 / 281 | 4,007 | 63.68 | 55.24 | 68.20 |
| r17 | base | 66.85 | 14.96 / 17.19 / 17.39 | 0.02 / 54.34 | 266 / 285 | 4,093 | 62.29 | 54.09 | 66.70 |
| r20 | hc2 | 68.33 | 14.64 / 16.79 / 16.98 | 0.02 / 53.15 | 263 / 281 | 4,008 | 63.69 | 55.43 | 68.22 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| hc2 | 5 | 68.32 | 55.24 | 68.21 | 263 |
| base | 5 | 66.87 | 54.09 | 66.74 | 267 |

Delta, lane vs base: greedy +2.16%, sampled +2.14%, ignore-eos +2.19%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: identical across every row of both arms; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 249..251 W, peak 482 W, SM clock 2610..2850 MHz, max temp 55 C, 3469 samples.

Thermal hc2: power median 252..254 W, peak 473 W, SM clock 2632..2850 MHz, max temp 57 C, 3376 samples.

### Plain

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 214 / 231 | 5,220 | 49.04 | 46.20 | 50.88 |
| r4 | hc2 | 53.87 | 18.56 / 18.57 / 18.57 | 18.31 / 19.52 | 211 / 237 | 4,945 | 51.74 | 48.59 | 53.84 |
| r6 | hc2 | 53.86 | 18.57 / 18.57 / 18.57 | 18.32 / 19.53 | 215 / 255 | 4,949 | 51.67 | 48.45 | 53.83 |
| r7 | base | 50.93 | 19.64 / 19.64 / 19.64 | 19.39 / 20.60 | 217 / 267 | 5,226 | 48.94 | 45.95 | 50.88 |
| r9 | hc2 | 53.86 | 18.57 / 18.57 / 18.57 | 18.32 / 19.53 | 215 / 237 | 4,949 | 51.67 | 48.45 | 53.83 |
| r12 | base | 50.93 | 19.63 / 19.64 / 19.65 | 19.39 / 20.58 | 212 / 230 | 5,217 | 49.07 | 46.16 | 50.88 |
| r14 | base | 50.94 | 19.63 / 19.64 / 19.64 | 19.38 / 20.58 | 217 / 229 | 5,220 | 49.06 | 46.04 | 50.87 |
| r15 | hc2 | 53.87 | 18.56 / 18.57 / 18.57 | 18.31 / 19.53 | 211 / 227 | 4,945 | 51.76 | 48.25 | 53.83 |
| r17 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 212 / 233 | 5,220 | 49.05 | 46.02 | 50.87 |
| r20 | hc2 | 53.87 | 18.56 / 18.57 / 18.57 | 18.31 / 19.53 | 211 / 241 | 4,945 | 51.71 | 48.50 | 53.83 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| hc2 | 5 | 53.87 | 48.45 | 53.83 | 211 |
| base | 5 | 50.93 | 46.04 | 50.88 | 214 |

Delta, lane vs base: greedy +5.76%, sampled +5.25%, ignore-eos +5.81%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: identical across every row of both arms; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 216..218 W, peak 472 W, SM clock 2610..2850 MHz, max temp 54 C, 4290 samples.

Thermal hc2: power median 223..225 W, peak 431 W, SM clock 2610..2857 MHz, max temp 54 C, 4046 samples.

## Verdict

**HC2 wins on both served modes, bit-identical.** Plain greedy decode moves 50.93 -> 53.87 tok/s
(+5.8%, N=5 per arm, TPOT p50 19.63 -> 18.56 ms), sampled +5.3%; DSpark greedy 66.87 -> 68.32
(+2.2%), sampled +2.1%. Every request of every row of both arms hashed the same, including the
ignore-eos cell, and the served DSpark identity gate passed on the lane binary. Row spreads are
under 0.1% within each arm, so the deltas are far outside the noise.

The plain step saves 1.07 ms of TPOT. The kernel-level saving at 86 HC entry sites per step is
0.88 ms (18.44 -> 8.20 us at one row). The remaining 0.19 ms is consistent with the activation
packs the attention entry no longer launches, because wq_a, wkv (43 layers each) and the indexer
weights (21 layers) read the fused kernel's bf16 row: 107 packs, about 1.8 us each. This lane did
not time those packs alone. DSpark
gains less, as expected from its wider verify rows (24.63 -> 14.34 us per site at 6 rows) in a
round the MoE and attention dominate; the round was not profiled here.

The fused kernel is the default at every row count on the diet shape; there is no door. The lane
lands.
