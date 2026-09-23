# DSv4 latency kernels: +13.3% served plain decode, +3.9% DSpark, same bits

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (a second pair, not the pair of
`../moe-defer-670/`), 500 W limit, 2026-09-23. Served program: PP-2, matrix expert program, host
sampler, memra-server otherwise naked. One scored campaign at a time under `/tmp/memra-gpu.lock`,
250 ms telemetry per row. Base: main `7029cd67c` (the small-kernel diet of `../small-diet/`,
memra #675). Lane: `2a41f68bd`.

## What changed

Four batch-1 kernels from the adoption list of `../PRIOR-ART.md`, each rebuilt so its bits equal
the kernel it replaces. Prior art supplied the launch shapes; the numeric program stays memra's
(the eight divergences that list records are not taken).

- **RMSNorm, register resident.** One 128-thread row loads every owned `x` and `w` element
  before the first add, so the row pays one load latency where the loop form paid one per
  8-load batch and one per output. Same per-thread ascending element order, same
  `dsv4_block_sum_f32` pairing (levels 64 and 32 through shared memory, 16..1 by shuffle in
  warp 0), same expressions. Rows up to 4096 columns take it. FlashInfer's `norm.cuh` uses
  512 threads and an xor butterfly, which is a different tree, so only the load structure is
  borrowed.
- **One-warp router (`route_m`).** Warp 0 selects and finishes with no block barrier. Each of the
  6 rounds is a value-desc/index-asc argmax: a lane scans its columns ascending with strict `>`,
  the butterfly takes the partner on a larger value or an equal value at a lower index. On
  NaN-free scores that order has one maximum, so every lane holds the pick the former
  128-thread tree made. The tail sorts on shared copies and writes each output once instead of
  reading `sel`/`selw` back from global memory. The shape is the TileKernels
  `topk_gate_kernel.py` / vLLM `dsv4_topk.py` one-warp router.
- **Warp expert prefix (`grouped_routes`).** One warp writes `expert_ids` and
  `offsets[1..experts]`: lane `l` owns a contiguous run of `ceil(experts / 32) <= 16` experts and
  a shuffle scan adds the runs before it. Integer sums, so the offsets equal a serial loop's. The
  block prefix scan (vLLM `moe_align_sum_kernels.cu` uses `cub::BlockScan`) was the whole cost at
  6 slots.
- **Grouped `wo_a` on the dense-fast program, default ON.** One t=1 FP8 `wo_a` launch over all
  eight groups replaces eight per-group launches. When the slices would take dense fast (exact
  tail and dense fast on, operands admitted), the grouped launch takes
  `dsv4_dense_fast_fp8_kernel<2,true>` over all groups' rows, so the 128 leaf sequences and the
  halving tree are the slices' own. vLLM and b12x group `wo_a` too, but on FP8/MXFP8 activations;
  this keeps the checkpoint's activation program. The `set_dense_wo_a_grouped_for_gate` seam stays
  (`false` selects the eight launches), decide-by 2026-10-07 in `docs/FLAGS.md`.

## Correctness

`crates/memra-engine/tests/dsv4_latency_kernels_gpu.rs` (new, ignored GPU tests), run on both
trees:

| test | result |
|---|---|
| `rmsnorm_f32acc_matches_exact_oracle` | 240 cases bit-identical to the exact CPU oracle |
| `rmsnorm_f32acc_red_arm` | a perturbed element is caught |
| `grouped_routes_match_cpu_oracle` | 1755 cases identical |
| `grouped_routes_red_arm` | caught |
| `route_m_matches_golden_and_structure` | 360 cases, output hash `0xa883c1c05022dc87` equals the `31dd11455` capture; picks in range, no duplicates, value-desc/index-asc, no unpicked expert above the last pick, weights sum to `route_scale`, hash layers take the `tid2eid` row |
| `route_m_red_arm` | a six-pick case with one weight changed is caught |

- PRO 6000, `raw/lat2/correct-{base,lane}/`: 6 passed on both trees (test binaries `3c44c6ce...`
  base, `bbc294e3...` lane).
- Local RTX 5090 Laptop, `raw/rtx5090/correct-lane-pinned.log` (binary `a237dd8b...`, lane
  `2a41f68bd`): 6 passed. The earlier `correct-{base,lane}-prepin.log` ran before the golden was
  pinned (`c6e61bdf8`): both trees already produced `0xa883c1c05022dc87` against a placeholder 0,
  and the old one-pick red arm was blind because one weight always normalizes to `route_scale`.
  The six-pick arm replaced it.
- Grouped `wo_a`, `raw/lat2/woa-lane/gate.log` and `raw/rtx5090/woa.log`:
  `cuda_gemv_fp8_grouped_dense_fast_matches_every_slice_program` passes all 27 cells (nine
  slice-program x grouped-program pairs at 8x1024, 2x128 and 3x256, k=4096) bit-exact, with one
  dense-fast enqueue only on the dense-fast arm; `cuda_gemv_fp8_grouped_m1_matches_eight_slices_and_counts_one_enqueue`
  passes (bit-exact, padding guarded, `dispatch_delta=1`, malformed strides refused).
- DSpark served identity gate on the lane, `raw/lat2/dspark-served/`: `GPU DSPARK GATE [PASS]`.

## Component timing

`latency_kernel_timing`, 5 rounds in alternating order (base lane, lane base, ...) x 5 reps, us
per launch, medians of 25 (`raw/lat2/timing-r*-*.log`, `raw/rtx5090/timing-r*-*.log`):

| kernel | shape | PRO 6000 base | PRO 6000 lane | change | 5090 base | 5090 lane | change |
|---|---|---|---|---|---|---|---|
| grouped_routes | 6 slots, 256 experts | 11.606 | 5.168 | -55.5% | 11.597 | 4.298 | -62.9% |
| grouped_routes (partition) | 6 slots, 128 experts | 7.422 | 5.172 | -30.3% | 7.331 | 4.290 | -41.5% |
| rmsnorm | 1x4096 | 6.011 | 1.733 | -71.2% | 5.875 | 1.605 | -72.7% |
| rmsnorm | 1x1024 | 2.401 | 1.724 | -28.2% | 2.339 | 1.400 | -40.1% |
| rmsnorm | 1x512 | 1.792 | 1.726 | -3.7% | 1.799 | 1.393 | -22.6% |
| route_m | 1 row, 256 experts, top 6 | 9.373 | 4.401 | -53.0% | 9.477 | 4.444 | -53.1% |
| wo_a grouped | 8x1024x4096 | 15.059 | 13.268 | -11.9% | | | |
| wo_a eight slices | 8x1024x4096 | 24.931 | 24.886 | -0.2% | | | |

Main served eight slices, so the served `wo_a` goes from 24.9 to 13.3 us per layer. The 5090 run
predates the grouped dense-fast commit, so it has no `wo_a` rows.

## Served A/B

One boot per row, order `b l l b b l l b` (N=4 per arm), binaries `22397e5bc486d092` (base) and
`6cab3c388cb020fa` (lane). Queue `raw/q-lat2.sh`, summary `raw/q-lat2.summary`, receipts
`raw/lat2/plain/r<N>-<arm>/` and `raw/lat2/spec/s<N>-<arm>/` (cells.jsonl, controller.log,
serve.log, /metrics, /healthz, /readyz, 250 ms telemetry, binary sha256, env). Cells: a 32-token
warmup, greedy c1 on 8 prompts, sampled c1 on 8 prompts, greedy c1 ignore-eos on 4 prompts, 256
max tokens, the fixed prompts of `../raw/bench.py`.

Plain:

| row | arm | greedy decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 55.94 | 17.88 / 17.91 / 17.91 | 17.75 / 18.50 | 197 / 213 | 4,755 | 53.80 | 51.27 | 19.50 | 55.86 |
| r2 | lane | 63.29 | 15.80 / 15.86 / 15.88 | 15.68 / 16.55 | 193 / 210 | 4,223 | 60.60 | 57.35 | 17.44 | 63.30 |
| r3 | lane | 63.36 | 15.78 / 15.81 / 15.81 | 15.67 / 16.40 | 192 / 209 | 4,215 | 60.67 | 57.36 | 17.43 | 63.28 |
| r4 | base | 55.92 | 17.88 / 17.91 / 17.91 | 17.75 / 18.51 | 197 / 214 | 4,755 | 53.80 | 51.28 | 19.50 | 55.88 |
| r5 | base | 55.91 | 17.88 / 17.91 / 17.91 | 17.76 / 18.52 | 196 / 214 | 4,757 | 53.78 | 51.24 | 19.52 | 55.83 |
| r6 | lane | 63.35 | 15.78 / 15.80 / 15.81 | 15.67 / 16.37 | 192 / 209 | 4,216 | 60.67 | 57.44 | 17.41 | 63.30 |
| r7 | lane | 63.32 | 15.79 / 15.81 / 15.81 | 15.67 / 16.37 | 192 / 210 | 4,218 | 60.66 | 57.41 | 17.42 | 63.30 |
| r8 | base | 55.92 | 17.88 / 17.91 / 17.92 | 17.76 / 18.50 | 196 / 213 | 4,756 | 53.79 | 51.24 | 19.51 | 55.83 |

**Plain greedy: lane 63.34 tok/s median (N=4, 63.29..63.36) against base 55.92 (N=4,
55.91..55.94), +13.3%, TPOT p50 17.88 -> 15.79 ms. Sampled: 57.38 against 51.26, +11.9%.
Ignore-eos: 63.30 against 55.85.** Disjoint arms: a 7.4 tok/s gap against a within-arm spread
under 0.08 tok/s. TTFT p50 197 -> 192 ms, E2E p50 4,756 -> 4,217 ms.

DSpark (`MEMRA_DSV4_DRAFTER=dspark`):

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | base | 73.26 | 13.65 / 15.69 / 15.88 | 0.02 / 49.76 | 245 / 263 | 3,738 | 68.29 | 60.44 | 16.54 | 73.03 |
| s2 | lane | 76.21 | 13.12 / 15.11 / 15.28 | 0.02 / 47.94 | 240 / 258 | 3,603 | 70.89 | 62.50 | 16.00 | 76.04 |
| s3 | lane | 75.75 | 13.20 / 15.15 / 15.31 | 0.02 / 48.17 | 242 / 259 | 3,618 | 70.58 | 62.38 | 16.03 | 75.96 |
| s4 | base | 73.23 | 13.66 / 15.72 / 15.91 | 0.02 / 49.74 | 244 / 264 | 3,739 | 68.25 | 60.53 | 16.52 | 72.91 |
| s5 | base | 73.22 | 13.66 / 15.71 / 15.90 | 0.02 / 49.80 | 245 / 263 | 3,740 | 68.22 | 60.34 | 16.57 | 73.12 |
| s6 | lane | 76.14 | 13.13 / 15.12 / 15.31 | 0.02 / 47.96 | 239 / 258 | 3,600 | 70.89 | 62.46 | 16.01 | 76.05 |
| s7 | lane | 76.01 | 13.16 / 15.12 / 15.30 | 0.02 / 48.01 | 240 / 258 | 3,606 | 70.82 | 62.47 | 16.01 | 76.00 |
| s8 | base | 73.17 | 13.67 / 15.73 / 15.90 | 0.02 / 49.84 | 245 / 263 | 3,742 | 68.16 | 60.38 | 16.56 | 73.07 |

**DSpark greedy: lane 76.07 tok/s median (N=4, 75.75..76.21) against base 73.23 (N=4,
73.17..73.26), +3.9%. Sampled: 62.47 against 60.41, +3.4%. Ignore-eos: 76.02 against 73.05,
+4.1%.** Disjoint arms. DSpark gains less for two reasons. The rmsnorm, router and expert-prefix
kernels run on verify rows too, but a verify round pays its 43 layers once per 3.56 committed
tokens (the gate's tokens per round), so each saved microsecond counts about a third as much per
emitted token. Grouped `wo_a` is one-token only: t>1 rows keep the eight launches.

**Identity.** Every row of both campaigns produced the same text on every request: greedy
`aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled `53944095
73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80`, ignore-eos the first four
greedy hashes, warmup `e78fb458`. These are the hashes of `../moe-defer-670/RESULTS.md`, so spec
output still equals plain output.

**Thermal regime.** Pooled over both cards, 250 ms samples at 20% utilization or more: plain
power median 236..237 W per base row and 253..255 W per lane row, peak 291 W; DSpark 269..271 W
base and 274..277 W lane, peak 304 W, all against the 500 W limit. SM clock 2497..2850 MHz,
with single low samples in r3 (180 MHz), r7 (1800 MHz), s1 (847 MHz), s2 (180 MHz) and s6
(2100 MHz) that did not move their medians. GPU temperature at most 73 C.

## What this lane did not measure

- The served step was not profiled, so the 2.09 ms per plain token is not split between the four
  kernels. The component rows are isolated launches in a loop; they bound the size of each
  change but are not an attribution of the served gain.
- PDL, the last item of the adoption list, is not in this lane.
