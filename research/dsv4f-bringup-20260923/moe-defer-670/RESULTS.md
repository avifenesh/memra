# DSv4 deferred MoE checks (memra #670): +6.8% served plain decode, +2.7% DSpark, same bits

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (the healthy pair of `../REBASELINE.md`), 500 W
limit, 2026-09-23. Served program: PP-2, matrix expert program, host sampler, memra-server
otherwise naked. One scored campaign at a time under `/tmp/memra-gpu.lock`, 250 ms telemetry per
row. Base: the multi-row stream visitor of `../mrow-stream/` (memra #669, PR #674).

## What changed

The `../m1-stream-664/` profile counted 131 pageable D2H copies per served plain step. 129 of
them were the MoE transaction's own checks, three per MoE layer on 43 layers: the grouped route
status (every slot names a live expert), the routed input FP8/half mirror status and the
intermediate mirror status. Each one read a word back and synchronized the stream, so the host
stopped queuing work until the GPU drained.

The checks now write instead of read. The route prefix kernel and both mirror kernels OR a fault
bit into a per-layer device word when the matrix transaction armed it (`memra_dsv4_grouped_routes_fault`,
`memra_dsv4_fp8_gather_half_fault`). `Dsv4Gpu::take_moe_faults` reads every armed stage once,
before a verify transaction commits and before any token leaves the engine, and fails the
transaction with the refusal text the synchronous arm would have produced. The launches and every
written bit are unchanged on a valid route; on an invalid one the count kernel never counts the
bad id and the `-1` tail stays inert in gather and scatter, so nothing writes out of bounds before
the refusal. Partitioned (EP) routes keep their synchronized live count because the scatter needs
it. Gate setters that turn the checks off keep the unchecked path.

## Correctness

`cuda_deferred_moe_faults_match_the_synchronous_checks` (ignored GPU test in `src/dsv4_grouped.rs`),
box run `raw/component/gate.log`, test binary `9b1b2fd0...`:

- `EXACT` gate, up, H and down contribution against the synchronous arm at 1, 2, 5 and 16 rows,
  fault word 0.
- Three red arms at 1 and 5 rows. Each sets its own bit while the synchronous arm refuses with
  the text the served error carries: route `0x1` "grouped route contains an invalid expert id",
  input mirror `0x2` "FP8-QAT half mirror is not lossless at gathered row 0, cols=4096",
  intermediate mirror `0x4` "... cols=2048".
- The rest of the grouped suite on the same binary: 8 passed, 0 failed.

DSpark served identity gate on the shipped code (`raw/dspark-served/gate.log`, binary
`f285f01a...`): `GPU DSPARK GATE [PASS]`: 160/160 greedy spec==plain tokens literal on the
sequential and batched arms, 14 cells / 77 logit rows / 3206 cache classes batched==sequential
bit for bit, drafter ring writes bit-identical, deterministic across two runs.

## Served A/B

Lane binary `be465f0d...` (tree `31dd11455` plus `raw/defer-lane.patch`, a lane-only
`MEMRA_DSV4_MOE_DEFER` read that was never merged: `0` allocates no fault word, so every check
reads back and synchronizes as before). One boot per row, order `A B B A A B B A A B` (A = deferred,
the shipped code). Queue `raw/q-defer.sh`, summary `raw/q-defer.summary`, receipts
`raw/ab/r<N>-<arm>/` and `raw/spec/s<N>-<arm>/` (cells.jsonl, controller.log, serve.log,
/metrics, /healthz, /readyz, 250 ms telemetry, binary sha256, env).

Plain, cells `raw/defer-cells.txt` (a 32-token warmup, greedy c1 on 8 prompts, sampled c1 on 8
prompts, 256 max tokens, the fixed prompts of `../raw/bench.py`):

| row | arm | greedy decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| r1 | deferred | 53.51 | 18.69 / 18.72 / 18.72 | 18.56 / 19.31 | 197 / 213 | 4,961 | 51.57 | 49.23 | 20.31 |
| r2 | sync | 50.09 | 19.96 / 19.99 / 20.00 | 19.85 / 20.66 | 200 / 217 | 5,288 | 48.36 | 46.31 | 21.59 |
| r3 | sync | 50.07 | 19.97 / 20.00 / 20.00 | 19.86 / 20.61 | 200 / 217 | 5,292 | 48.35 | 46.27 | 21.61 |
| r4 | deferred | 53.48 | 18.70 / 18.72 / 18.72 | 18.57 / 19.34 | 198 / 214 | 4,969 | 51.53 | 49.25 | 20.30 |
| r5 | deferred | 53.44 | 18.71 / 18.73 / 18.73 | 18.58 / 19.35 | 197 / 214 | 4,968 | 51.52 | 49.20 | 20.32 |
| r6 | sync | 50.07 | 19.97 / 20.00 / 20.00 | 19.86 / 20.58 | 201 / 218 | 5,292 | 48.34 | 46.36 | 21.57 |
| r7 | sync | 50.05 | 19.98 / 20.00 / 20.01 | 19.87 / 20.59 | 200 / 217 | 5,294 | 48.34 | 46.30 | 21.60 |
| r8 | deferred | 53.48 | 18.70 / 18.72 / 18.72 | 18.58 / 19.32 | 197 / 215 | 4,964 | 51.54 | 49.18 | 20.33 |
| r9 | deferred | 53.45 | 18.71 / 18.72 / 18.73 | 18.58 / 19.32 | 197 / 215 | 4,966 | 51.53 | 49.24 | 20.31 |
| r10 | sync | 50.06 | 19.98 / 19.99 / 19.99 | 19.87 / 20.62 | 200 / 217 | 5,294 | 48.34 | 46.35 | 21.57 |

**Plain greedy: deferred 53.48 tok/s median (N=5, 53.44..53.51) against synchronous 50.07 (N=5,
50.05..50.09), +6.8%, TPOT p50 19.97 -> 18.70 ms. Sampled: 49.23 against 46.31, +6.3%.** The arms
are disjoint: a 3.4 tok/s gap against a within-arm spread under 0.07 tok/s. TTFT p50 moves
200 -> 197 ms, E2E p50 5,292 -> 4,966 ms. The synchronous arm reproduces the 50.06..50.07 plain rows of
`../mrow-stream/RESULTS.md`.

DSpark (`MEMRA_DSV4_DRAFTER=dspark`), cells `raw/cells-spec.txt` (warmup, greedy c1 on 8 prompts,
sampled c1 on 8 prompts, greedy c1 ignore-eos on 4 prompts):

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | deferred | 73.11 | 13.68 / 15.72 / 15.91 | 0.02 / 49.85 | 245 / 264 | 3,745 | 68.13 | 60.18 | 16.62 | 72.96 |
| s2 | sync | 71.18 | 14.05 / 16.20 / 16.39 | 0.02 / 51.16 | 251 / 268 | 3,844 | 66.28 | 58.84 | 17.00 | 71.02 |
| s3 | sync | 71.20 | 14.05 / 16.20 / 16.39 | 0.02 / 51.19 | 249 / 268 | 3,843 | 66.32 | 58.83 | 17.00 | 71.06 |
| s4 | deferred | 73.10 | 13.68 / 15.73 / 15.92 | 0.02 / 49.86 | 245 / 264 | 3,746 | 68.14 | 60.41 | 16.55 | 72.91 |
| s5 | deferred | 73.19 | 13.66 / 15.71 / 15.89 | 0.02 / 49.81 | 245 / 264 | 3,742 | 68.21 | 60.46 | 16.54 | 73.06 |
| s6 | sync | 71.20 | 14.04 / 16.20 / 16.39 | 0.02 / 51.15 | 249 / 268 | 3,843 | 66.32 | 58.90 | 16.98 | 71.09 |
| s7 | sync | 71.10 | 14.06 / 16.19 / 16.38 | 0.02 / 51.23 | 250 / 269 | 3,849 | 66.27 | 58.74 | 17.02 | 71.02 |
| s8 | deferred | 73.08 | 13.68 / 15.73 / 15.91 | 0.02 / 49.85 | 246 / 264 | 3,747 | 68.13 | 60.29 | 16.59 | 72.99 |
| s9 | deferred | 73.12 | 13.68 / 15.72 / 15.91 | 0.02 / 49.88 | 246 / 264 | 3,745 | 68.13 | 60.33 | 16.57 | 72.98 |
| s10 | sync | 71.01 | 14.08 / 16.21 / 16.40 | 0.02 / 51.16 | 249 / 270 | 3,854 | 66.23 | 58.79 | 17.01 | 70.92 |

**DSpark greedy: deferred 73.11 tok/s median (N=5, 73.08..73.19) against 71.18 (N=5,
71.01..71.20), +2.7%. Sampled: 60.33 against 58.83, +2.5%. Ignore-eos: 72.98 against 71.02,
+2.8%.** Disjoint arms again. The gain is smaller than plain because a verify round already
amortizes its 43 layers of checks over 3.56 committed tokens (the gate's tokens per round), so
there were fewer syncs per emitted token to remove.

**Identity.** Every row of both campaigns produced the same text on every request: greedy
`aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled `53944095
73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80`, ignore-eos the first four
greedy hashes, warmup `e78fb458`. These equal the plain and DSpark hashes of
`../mrow-stream/RESULTS.md`, so spec output still equals plain output.

**Thermal regime.** Pooled over both cards, 250 ms samples at 20% utilization or more (a PP-2
card at batch 1 idles while the other runs, so the 50% cut of earlier lanes kept too few
synchronous-arm samples): plain power median 214..224 W per row, peak 246 W; DSpark 255..263 W,
peak 303 W, all against the 500 W limit. SM clock 2587..2850 MHz, with single low samples in r9
(1297 MHz), s7 (1762 MHz) and s10 (697 MHz) that did not move their medians. GPU temperature at
most 66 C.

## Where the 1.7 ms went

One nsys capture per arm on the plain server, 510 steps each (`raw/prof/served-plain-{on,off}/ana.txt`;
the captures themselves are not committed):

| | deferred | synchronous |
|---|---|---|
| step span | 20.39 ms | 22.08 ms |
| dev0 kernels / kernel-sum / busy-union | 1711 / 9.06 / 9.26 ms | 1711 / 9.06 / 9.29 ms |
| dev1 kernels / kernel-sum / busy-union | 1662 / 9.41 / 9.62 ms | 1662 / 9.41 / 9.65 ms |
| `cuMemcpyDtoHAsync` per step (host ms) | 3 (5.94) | 131 (10.11) |
| `cuStreamSynchronize` per step | 5 | 133 |
| memcpys per step dev0 / dev1 | 229 / 237 | 294 / 300 |

Kernel count and kernel time are identical, so the saving is all gap: 22.08 - 20.39 = 1.69 ms
per step, and the GPU busy share of the step rose from 85.8% to 92.6% (dev0 + dev1 busy-union
over the span; PP-2 at batch 1 runs the two cards one after the other). The three D2H copies
left carry 5.94 ms of host time: that is the host waiting for the GPU at the end of a step,
which is where it should wait.

## What is left on this path

- Host launch is not the bound now: 3373 `cudaLaunchKernel` per step take 10.4 ms of host time
  under nsys (which inflates each launch) and queue ahead of 18.9 ms of GPU work.
  Launch-count work (graphs, PDL, fusion) buys device-side gaps between dependent kernels, not
  host headroom.
- 414 `cuMemcpyDtoDAsync` per step move 13.4 MB. They are not in the kernel top list but they
  are launches on the same stream.
- The bound is the kernels: 18.9 ms of busy time per token against the 6.38 ms PP-2 roofline
  of `../BASELINE.md` (11.44 GB/token at the card's 1.792 TB/s nominal). The next levers are the
  dense FP8 GEMV (26% of dev0), the M1 stream (20%), and the small kernels (rmsnorm, sinkhorn,
  dots: 22% together), then TP/EP so both cards read weights at the same time.
