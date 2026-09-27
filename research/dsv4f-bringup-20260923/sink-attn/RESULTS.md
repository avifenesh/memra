# DSv4 two-launch sink attention (memra #683)

Status: complete. The kernel-boundary bit gate on one RTX PRO 6000 Blackwell, on the target pair
and on the local RTX 5090, the component timing, the served DSpark identity gate and the served
A/B on 2x RTX PRO 6000 are banked here.

Scope: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`, device
f32x chains, sink attention at 64 heads x 512 (main attention) and 32 x 512 geometry. Lane
`lane/dsv4-sink-attn-20260923`, base main `649d96210`.

## What changed

The f32x sink attention was three launches per layer, plus a q transpose on the batched path:
scores (`dsv4_sink_scores_mq_f32acc_kernel`, q `[nq][hd][heads]`), soft (max, `ev`, den into a
workspace) and out. It is now two launches of `memra_dsv4_sink_attn_st_f32acc`:

- `dsv4_sink_scores_st_f32acc_kernel`: one 64-thread block per (8-slot tile, 8-head tile, query).
  The block stages its q rows and its selected kv rows once by cp.async; each thread is one score,
  a single f32 accumulator over `x` ascending, then `* scale`, `-INF` for a `-1` slot.
- `dsv4_sink_softout_st_f32acc_kernel`: one 256-thread block per (16-column tile, 16-head tile,
  query). Max (`fmaxf`, floored at `-1e30`), `ev = expf(s - m)` or 0 for `-INF`, den summed in
  ascending slot order then `+ expf(sink - m)`, and `o = (ascending sum of ev * kv over ev != 0) /
  den` over 128-slot kv tiles, with no workspace round trip.

Replay (graph) launches read the position on device: live slots are
`win + (ratio ? min((pos + 1) / ratio, topk) : 0)`, the score row stride is the live count, and
grid blocks past it exit.

Prior art: the tile and grid split follow FlashInfer's `sparse_mla_sm120` decode kernels (a score
grid whose operands one CTA stages once, and a soft/out CTA that finishes in shared memory).
FlashInfer splits the slots across CTAs and merges partial softmaxes in the exp2 domain; both
change the order of the sums, so neither is taken. The numeric program is the three-kernel one.

`MEMRA_DSV4_SINK_SCORE` and the 2026-09-06 tiled scorer are deleted (`docs/FLAGS.md`, "Removed
doors, 2026-09-23 (the tiled sink scorer ...)").

## Correctness

`crates/memra-engine/tests/dsv4_sink_attn_st_gpu.rs` (ignored GPU tests) compares every output
bit of the two-launch program against the three former entry points on the same inputs:

| test | coverage |
|---|---|
| `sink_attn_st_matches_three_kernel_batched` | 208 cases: shapes 64x512, 32x512, 16x256, 48x64; nq 1 and 3; slots 1, 7, 8, 9, 127, 128, 129, 187, 255, 256, 257, 640, 1100; magnitudes (-2, 1) and (-1, 4); 0% or 30% padded indices, one all-pad query at nq 3 |
| `sink_attn_st_matches_three_kernel_replay` | 86 cases against `memra_dsv4_replay_attention`: heads 64 and 32, (ratio, topk) in (0, max), (4, 512) at limits 1024 and 4096, (128, max) at 65536, positions across each boundary |
| `sink_attn_st_matches_three_kernel_single_query` | 26 cases against the eager single-query entry |
| `sink_attn_st_red_arm` | a 2^-10 change to one live kv element is caught |

- One RTX PRO 6000 Blackwell Workstation Edition, 600 W limit, committed tree `2f8e23c22`, test
  binary `5eae971f380b3417` (`raw/pro6000/summary`, `raw/pro6000/gate-r{1,2,3}.log`, script
  `raw/pro6000/sink-pro.sh`): **5 passed in each of 3 runs, 320 cases bit-identical, red arm
  caught.**
- 2x RTX PRO 6000 WS pair (450 W), card 0, lane `764aca9f1` (merged with main `6978f5fac`),
  test binary `cc7bf00d2839e082` (`raw/pair/component/gate.log`): 5 passed, 320 cases
  bit-identical, red arm caught.
- `dsv4-gpu-dspark-gate --served` (tape 416) on the pair's lane binary: `GPU DSPARK GATE [PASS]`
  (`raw/pair/dspark-served/gate.log`).
- Local RTX 5090 Laptop (`raw/rtx5090/gate-worktree.log`, the working tree before the kernels
  were committed, st kernels as committed): 5 passed, 320 cases bit-identical, red arm caught.

## Component timing

`sink_attn_st_timing`: one layer's sink attention on the graph replay entry, old program (q
transpose plus the three kernels) against the two launches, 2000 launches per rep after 64 warmup
launches, us per layer.

RTX PRO 6000, 3 runs x 5 reps, medians of 15 (two-launch range across the 15 in brackets):

| shape | old | two-launch | change |
|---|---|---|---|
| 64 heads, ratio 0, 128 live | 26.578 | 8.639 (8.601..8.641) | -67.5% |
| 64 heads, ratio 4, 187 live | 32.250 | 11.753 (11.720..11.760) | -63.6% |
| 64 heads, ratio 4, 640 live | 65.260 | 28.785 (28.776..28.816) | -55.9% |
| 32 heads, ratio 0, 128 live | 26.115 | 8.355 (8.353..8.357) | -68.0% |
| 32 heads, ratio 4, 187 live | 31.176 | 11.006 (11.002..11.012) | -64.7% |
| 32 heads, ratio 4, 640 live | 64.600 | 23.777 (23.771..23.783) | -63.2% |

Local RTX 5090 Laptop, 5 reps, medians of 5:

| shape | old | two-launch | change |
|---|---|---|---|
| 64 heads, ratio 0, 128 live | 27.235 | 11.930 | -56.2% |
| 64 heads, ratio 4, 187 live | 31.960 | 19.768 | -38.1% |
| 64 heads, ratio 4, 640 live | 70.396 | 55.262 | -21.5% |
| 32 heads, ratio 0, 128 live | 25.806 | 8.787 | -65.9% |
| 32 heads, ratio 4, 187 live | 30.489 | 11.759 | -61.4% |
| 32 heads, ratio 4, 640 live | 65.425 | 30.342 | -53.6% |

These are isolated launches in a loop. They bound the change per layer; they are not the served
gain.

## Served A/B

Shared-control campaign `../kernel-ab-short/` (protocol, all 40 rows, the queue script and the
binaries' hashes): one boot per row, base (main `6978f5fac`) and three lanes in a Williams order,
N=5 per arm per mode, 2x RTX PRO 6000 WS at a 450 W limit, PP-2, host sampler. The rows below are
this lane's and base's, in campaign order; `lane_tab.py short sink` prints them. The order is the
campaign's four-arm Williams order, not the `l b b l l b b l l b` this file planned.

### DSpark (`MEMRA_DSV4_DRAFTER=dspark`)

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 66.87 | 14.95 / 17.17 / 17.37 | 0.02 / 54.30 | 267 / 290 | 4,103 | 62.31 | 54.31 | 66.75 |
| r2 | sink | 67.65 | 14.78 / 16.98 / 17.17 | 0.02 / 53.67 | 265 / 297 | 4,054 | 63.00 | 54.70 | 67.53 |
| r5 | sink | 67.68 | 14.77 / 16.97 / 17.17 | 0.02 / 53.71 | 265 / 296 | 4,051 | 63.00 | 54.85 | 67.55 |
| r7 | base | 66.86 | 14.96 / 17.17 / 17.37 | 0.02 / 54.27 | 270 / 292 | 4,095 | 62.27 | 54.06 | 66.78 |
| r11 | sink | 67.67 | 14.78 / 16.97 / 17.17 | 0.02 / 53.71 | 265 / 285 | 4,046 | 63.06 | 54.66 | 67.57 |
| r12 | base | 66.88 | 14.95 / 17.18 / 17.37 | 0.02 / 54.30 | 267 / 284 | 4,091 | 62.34 | 54.25 | 66.73 |
| r14 | base | 66.90 | 14.95 / 17.17 / 17.37 | 0.02 / 54.32 | 270 / 286 | 4,096 | 62.29 | 53.99 | 66.74 |
| r16 | sink | 67.68 | 14.77 / 16.97 / 17.16 | 0.02 / 53.72 | 266 / 284 | 4,046 | 63.02 | 54.78 | 67.55 |
| r17 | base | 66.85 | 14.96 / 17.19 / 17.39 | 0.02 / 54.34 | 266 / 285 | 4,093 | 62.29 | 54.09 | 66.70 |
| r18 | sink | 67.64 | 14.78 / 16.98 / 17.18 | 0.02 / 53.79 | 265 / 284 | 4,047 | 62.99 | 54.60 | 67.47 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| sink | 5 | 67.67 | 54.70 | 67.55 | 265 |
| base | 5 | 66.87 | 54.09 | 66.74 | 267 |

Delta, lane vs base: greedy +1.18%, sampled +1.12%, ignore-eos +1.20%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: identical across every row of both arms; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 249..251 W, peak 482 W, SM clock 2610..2850 MHz, max temp 55 C, 3469 samples.

Thermal sink: power median 246..251 W, peak 479 W, SM clock 2632..2865 MHz, max temp 55 C, 3404 samples.

### Plain

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 214 / 231 | 5,220 | 49.04 | 46.20 | 50.88 |
| r2 | sink | 53.55 | 18.68 / 18.69 / 18.69 | 18.41 / 19.57 | 212 / 243 | 4,974 | 51.45 | 48.07 | 53.49 |
| r5 | sink | 53.52 | 18.68 / 18.69 / 18.69 | 18.41 / 19.58 | 212 / 241 | 4,975 | 51.41 | 48.32 | 53.48 |
| r7 | base | 50.93 | 19.64 / 19.64 / 19.64 | 19.39 / 20.60 | 217 / 267 | 5,226 | 48.94 | 45.95 | 50.88 |
| r11 | sink | 53.46 | 18.70 / 18.74 / 18.74 | 18.44 / 19.87 | 211 / 229 | 4,981 | 51.37 | 47.64 | 53.48 |
| r12 | base | 50.93 | 19.63 / 19.64 / 19.65 | 19.39 / 20.58 | 212 / 230 | 5,217 | 49.07 | 46.16 | 50.88 |
| r14 | base | 50.94 | 19.63 / 19.64 / 19.64 | 19.38 / 20.58 | 217 / 229 | 5,220 | 49.06 | 46.04 | 50.87 |
| r16 | sink | 53.55 | 18.67 / 18.68 / 18.69 | 18.41 / 19.57 | 211 / 239 | 4,973 | 51.45 | 48.25 | 53.50 |
| r17 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 212 / 233 | 5,220 | 49.05 | 46.02 | 50.87 |
| r18 | sink | 53.53 | 18.68 / 18.69 / 18.69 | 18.41 / 19.57 | 219 / 232 | 4,979 | 51.42 | 48.05 | 53.49 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| sink | 5 | 53.53 | 48.07 | 53.49 | 212 |
| base | 5 | 50.93 | 46.04 | 50.88 | 214 |

Delta, lane vs base: greedy +5.10%, sampled +4.41%, ignore-eos +5.13%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: identical across every row of both arms; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 216..218 W, peak 472 W, SM clock 2610..2850 MHz, max temp 54 C, 4290 samples.

Thermal sink: power median 221..222 W, peak 516 W, SM clock 2700..2850 MHz, max temp 55 C, 4087 samples.

## Verdict

**The two-launch sink attention wins on both served modes, bit-identical.** Plain greedy decode
moves 50.93 -> 53.53 tok/s (+5.1%, N=5 per arm, TPOT p50 19.63 -> 18.68 ms), sampled +4.4%;
DSpark greedy 66.87 -> 67.67 (+1.2%), sampled +1.1%. Every request of every row of both arms
hashed the same, and the served DSpark identity gate passed on the lane binary. Row spreads are
under 0.1% within each arm.

The plain step saves 0.95 ms of TPOT over 43 layers, about 22 us per layer. The one-card
component timing measured 18 to 20.5 us per layer at 128 and 187 live slots (the range these short
prompts run in); the served step was not profiled, so the extra 2 us per layer is not attributed.
DSpark gains less, as expected in a round where the verify rows' MoE and attention dominate; not
profiled either.

The two launches are the code on every f32x device chain; there is no door, and
`MEMRA_DSV4_SINK_SCORE` with its tiled scorer is deleted. The lane lands.
