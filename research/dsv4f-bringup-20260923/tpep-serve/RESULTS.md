# DSv4 served TP/EP against PP-2 (memra #454, #679)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 500 W limit, 2026-09-23. One scored campaign at
a time under `/tmp/memra-gpu.lock`, 250 ms telemetry per row, one boot per row. Lane
`lane/dsv4-tpep-serve-20260923`.

**Verdict.** After the #679 deferral, TP/EP beats PP-2 on plain decode. `tp_ep_attn` (split
attention heads) serves 69.11 tok/s greedy against 63.38 (N=5 each, +9.0%) with its own text.
`tp_ep` (replicated attention) serves 65.02 (N=3, +2.6%) with the PP-2 text on every request.
DSpark does not boot on either TP/EP arm: both run out of memory at load. PP-2 DSpark (76.12)
therefore stays the fastest served program, and the server still selects no TP/EP program.

## What the lane adds

- TP/EP multi-row verify (width 1..=6) and chunked prefill run the same walk as t=1 decode.
- The DSpark drafter is resident on the TP/EP head rank, and the eager TP/EP step writes its taps.
- Park/restore writes both rank planes.
- #679: EP-partitioned routes no longer sync a live count per layer. The partition launch
  ORs an out-of-range id into a device fault word (`memra_dsv4_grouped_routes_partition_fault`).
  The live count stays on the device, and the intermediate mirror takes
  `offsets[local_expert_count]` as its bound. Each rank reads its words once, before either cache
  plane commits. That is the same step-end read the PP-2 program got in #670.

The server has no topology selector. `raw/topology-lane.patch` adds the lane-only
`MEMRA_DSV4_TOPOLOGY` read (`pp`, `tp_ep`, `tp_ep_attn`) that served these A/Bs. It is never merged.

## Correctness

Run 2 tree (`raw/tpep3/`, lane `874d3667e` merged with main `c3eb41d12`):

- Component tests, `memra_engine` lib filtered to the grouped suite: 9 passed, including
  `cuda_deferred_partition_faults_match_the_synchronous_checks` ("EXACT deferred partition MoE
  checks part=0 rows=1 live=3 launch=6 word=0") and `cuda_deferred_moe_faults_match_the_synchronous_checks`.
- `dsv4_tp_ep_verify_gate` on three arms (`raw/tpep3/verify/{pp,tpep-at0,tpep-at1}/gate.log`):
  "PASS topology=pp rows chunk=65 verify=70 restore=22 bit-equal to sequential", and the same
  line with `topology=tpep` for attention TP off and on.
- `dsv4-gpu-dspark-gate --tpep` (`raw/tpep3/dspark-tpep/`): `GPU DSPARK GATE [PASS]`, "greedy
  spec==plain: 160/160 tokens LITERAL identity, sequential AND batched arms".

Run 1 (`raw/serve1/verify/`) passed the same verify gate on tree `979ef9cab`.

## Served A/B, run 1: tree `979ef9cab` (before #679)

Binary `0080b00c7a82c4da`. Queue `raw/serve1/q-tpep-serve.sh`, summary `raw/serve1/q-tpep-serve.summary`.
Cells: a 32-token warmup, greedy c1 on 8 prompts, sampled c1 on 8 prompts, greedy c1 ignore-eos
on 4 prompts, 256 max tokens.

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | pp | 50.05 | 19.98 / 20.00 / 20.00 | 19.86 / 20.66 | 200 / 217 | 5,295 | 48.33 | 46.32 | 21.59 | 50.04 |
| r2 | tp_ep_attn | 36.48 | 27.41 / 27.50 / 27.50 | 27.10 / 28.90 | 195 / 211 | 7,181 | 35.62 | 34.53 | 28.96 | 36.43 |
| r3 | tp_ep_attn | 36.38 | 27.49 / 27.53 / 27.54 | 27.19 / 28.87 | 195 / 212 | 7,209 | 35.52 | 34.56 | 28.94 | 36.41 |
| r4 | pp | 50.04 | 19.98 / 20.01 / 20.02 | 19.87 / 20.80 | 201 / 217 | 5,295 | 48.31 | 46.29 | 21.60 | 49.96 |
| r5 | pp | 50.06 | 19.98 / 19.99 / 20.00 | 19.86 / 20.71 | 200 / 217 | 5,293 | 48.34 | 46.31 | 21.59 | 50.03 |
| r6 | tp_ep_attn | 36.62 | 27.31 / 27.32 / 27.33 | 26.98 / 28.80 | 195 / 210 | 7,157 | 35.76 | 34.64 | 28.87 | 36.65 |
| r7 | tp_ep_attn | 36.54 | 27.37 / 27.45 / 27.48 | 27.04 / 29.01 | 195 / 211 | 7,174 | 35.68 | 34.70 | 28.82 | 36.67 |
| r8 | pp | 50.07 | 19.97 / 19.99 / 19.99 | 19.86 / 20.59 | 200 / 217 | 5,291 | 48.35 | 46.35 | 21.58 | 50.06 |
| r9 | pp | 50.04 | 19.98 / 20.00 / 20.00 | 19.87 / 20.62 | 200 / 219 | 5,295 | 48.32 | 46.29 | 21.60 | 50.04 |
| r10 | tp_ep_attn | 36.56 | 27.35 / 27.38 / 27.38 | 27.02 / 28.70 | 195 / 212 | 7,169 | 35.71 | 34.69 | 28.82 | 36.56 |
| r11 | tp_ep | 28.65 | 34.91 / 34.94 / 34.95 | 34.64 / 36.32 | 290 / 309 | 9,191 | 27.85 | 27.46 | 36.42 | 28.65 |
| r12 | tp_ep | 28.65 | 34.90 / 34.94 / 34.94 | 34.62 / 36.17 | 290 / 309 | 9,189 | 27.86 | 27.39 | 36.51 | 28.59 |

Plain greedy: PP-2 50.05 (N=5, 50.04..50.07), `tp_ep_attn` 36.54 (N=5, 36.38..36.62, -27%),
`tp_ep` 28.65 (N=2, -43%). DSpark: PP-2 70.76..71.18 (N=5, `raw/serve1/spec/`); every
`tp_ep_attn` DSpark boot exited rc=71.

The nsys capture of run 1 (`raw/prof1/{pp,tpa}/ana.txt`, 510 steps per arm) put the loss on the
host: 6548 launches per TP step against 3373 for PP, 347 D2H copies per TP step from the
per-layer partition live counts, and rank-replicated small kernels at full size on both cards.
memra #679 lists the fixes. Thermal regime of run 1: power median 220..225 W per PP-2 row,
224..239 W per `tp_ep_attn` row, 213..215 W per `tp_ep` row, SM clock 2587..2850 MHz, GPU
temperature at most 69 C.

## Served A/B, run 2: lane `874d3667e` + main `c3eb41d12` (after #679)

Server binary `8dd0b632ffd72adf`, verify gate `83207722665dcc65`, DSpark gate `23c7d463fa1e952f`.
Queue `raw/q-tpep3.sh`, summary `raw/q-tpep3.summary`, receipts `raw/tpep3/`. Same cells.

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | sampled TPOT p50 ms | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | pp | 63.36 | 15.78 / 15.86 / 15.87 | 15.68 / 16.94 | 192 / 208 | 4,220 | 60.64 | 57.73 | 17.32 | 63.31 |
| r2 | tp_ep_attn | 69.11 | 14.47 / 14.49 / 14.49 | 14.26 / 15.32 | 142 / 156 | 3,831 | 66.79 | 62.63 | 15.97 | 68.97 |
| r3 | tp_ep_attn | 69.33 | 14.42 / 14.45 / 14.45 | 14.25 / 15.24 | 142 / 157 | 3,820 | 66.98 | 62.73 | 15.94 | 69.28 |
| r4 | pp | 63.38 | 15.78 / 15.79 / 15.79 | 15.66 / 16.36 | 192 / 209 | 4,215 | 60.71 | 57.56 | 17.37 | 63.31 |
| r5 | pp | 63.38 | 15.78 / 15.79 / 15.79 | 15.66 / 16.41 | 192 / 208 | 4,216 | 60.71 | 57.55 | 17.38 | 63.31 |
| r6 | tp_ep_attn | 68.88 | 14.52 / 14.65 / 14.69 | 14.29 / 15.98 | 143 / 166 | 3,845 | 66.39 | 62.25 | 16.06 | 68.87 |
| r7 | tp_ep_attn | 69.11 | 14.47 / 14.50 / 14.50 | 14.26 / 15.51 | 142 / 156 | 3,831 | 66.77 | 62.52 | 15.99 | 68.92 |
| r8 | pp | 63.39 | 15.78 / 15.78 / 15.78 | 15.66 / 16.35 | 192 / 209 | 4,214 | 60.72 | 57.49 | 17.39 | 63.33 |
| r9 | pp | 63.38 | 15.78 / 15.78 / 15.79 | 15.66 / 16.34 | 192 / 213 | 4,214 | 60.69 | 57.59 | 17.37 | 63.32 |
| r10 | tp_ep_attn | 69.05 | 14.48 / 14.58 / 14.62 | 14.27 / 15.89 | 142 / 157 | 3,836 | 66.65 | 62.49 | 16.00 | 69.01 |
| r11 | tp_ep | 65.02 | 15.38 / 15.40 / 15.40 | 15.26 / 16.04 | 163 / 179 | 4,087 | 62.65 | 59.09 | 16.92 | 64.93 |
| r12 | tp_ep | 64.99 | 15.39 / 15.40 / 15.40 | 15.26 / 16.04 | 164 / 179 | 4,087 | 62.63 | 59.13 | 16.91 | 64.90 |
| r13 | tp_ep | 65.02 | 15.38 / 15.41 / 15.41 | 15.25 / 16.08 | 164 / 179 | 4,084 | 62.66 | 59.06 | 16.93 | 64.95 |

**Plain greedy: `tp_ep_attn` 69.11 tok/s median (N=5, 68.88..69.33) against PP-2 63.38 (N=5,
63.36..63.39), +9.0%; TPOT p50 15.78 -> 14.47 ms. Sampled 62.52 against 57.56 (+8.6%),
ignore-eos 68.97 against 63.31. `tp_ep` 65.02 (N=3, 64.99..65.02), +2.6%.** Disjoint arms. TTFT
p50 falls 192 -> 142 ms on `tp_ep_attn` and to 164 ms on `tp_ep`. From run 1 to run 2, PP-2 gained
26.6% (the #675 diet and the #678 latency kernels landed on main in between), `tp_ep_attn`
gained 89%, and `tp_ep` 127%.

**Identity.** Every PP-2 and `tp_ep` row produced the fixed text: greedy `aea6e69e 26ac8df7
850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled `53944095 73fe7f91 6329ee94
827c0b56 437a4427 07de3a97 8a59518d b424bd80`, ignore-eos the first four greedy hashes, warmup
`e78fb458`. `tp_ep` is the PP-2 numeric program on the served surface. `tp_ep_attn` produced
its own text, identical across its 5 rows (and across run 1): greedy `b2ed1de4 2ab1467d 4f99b252
1b292b73 237458e6 364f95f1 8e0036d0 8493e730`, sampled `ad19f053 3b4d00f1 77c07f9b 723ba155
4bffd6ba 4f68e735 36a221bf 2b978739`, warmup `e78fb458`. Its numeric class is
`dsv4_attention_wo_b_input_split_f32_rank_reduce`: each rank's `wo_b` takes half the K columns,
and the halves meet in an f32 rank sum. That is a different accumulation order from the one-card
`wo_b` row.

**DSpark.** PP-2 s2: 76.12 tok/s greedy (`raw/tpep3/spec/s2-pp/`). Both TP/EP boots failed at load
(`raw/tpep3/spec/s{1,3}-*/serve.log`):

- `tp_ep_attn`: "FATAL: worker init failed: load dsv4f: FP8 pack code allocation:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")".
- `tp_ep`: "FATAL: worker init failed: load dsv4f: dsv4 memory calibration: vws f32:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")".

Both logs show "drafter: DSpark ... resident on stage 1" and "per-rank-resident=all" at
`max_seq 1048576`. The plain TP/EP verify load reports `pool_used_gib=83.60` and
`effective_free_gib=10.82` on each card (`raw/tpep3/verify/tpep-at1/gate.log`).

**Thermal regime.** Per row, 250 ms samples at 20% utilization or more, pooled over both cards:
power median 244 W on PP-2, 316..319 W on `tp_ep_attn`, 346..348 W on `tp_ep`, peak 392 W, all
against the 500 W limit. SM clock 2587..2850 MHz, with brief low samples on PP-2 rows r1 (270 MHz),
r4 (180), r5 (637) and r8 (480, 1702) that did not move their medians. GPU temperature at most 71 C.

## Step anatomy, run 2

nsys over 510 steps per arm (`raw/tpep3/prof/{pp,tpa}/ana.txt`). The capture inflates the step
(17.24 ms PP, 23.90 ms `tp_ep_attn`), so these are shares, not served times.

| | PP-2 dev0 / dev1 | `tp_ep_attn` dev0 / dev1 |
|---|---|---|
| kernels per step | 1446 / 1409 | 3062 / 3069 |
| kernel sum ms per step | 7.58 / 7.96 | 14.93 / 12.97 |
| `dsv4_dense_fast_fp8_kernel` ms (launches) | 2.19 (187) / 2.09 (180) | 3.13 (496) / 3.12 (496) |
| `moe_kq_m1_stream_kernel` ms | 1.79 / 1.69 | 1.80 / 1.78 |
| `dsv4_dense_fast_dots_kernel` ms | 0.54 / 1.22 | 1.10 / 1.75 |
| `memra_tp_ar_1stage_kernel` ms (launches) | | 3.18 (87) / 0.69 (87) |
| host `cudaLaunchKernel` per step | 2855, 7.75 ms | 6131, 14.92 ms |

PP-2 is GPU bound: the two cards' kernel sums add to 15.54 ms against a served step of 15.78 ms.
`tp_ep_attn` is close to host bound. The host issues 6131 launches per step for both cards, and
the served step (14.47 ms) is about 2.4 us per launch. The all-reduce kernel takes 3.18 ms per step on
rank 0 and 0.69 ms on rank 1, for the same 87 launches. The asymmetry reads as rank 0 waiting on
rank 1, whose launches the host issues second; this capture does not separate that wait from the
transfer. The routed experts split as intended: `moe_kq_m1_stream_kernel` takes 1.80 ms per rank
over 43 layers at half the experts, against 1.79 ms for one PP-2 stage's 22 layers at all of
them. Replicated work does not split: the FP8 dense projections take 3.13 ms per rank against
4.28 ms for the whole PP-2 step, and the dense dots and small kernels run full size on both
ranks.

## Decision

The TP/EP engine program merges with the lane. The server selects no TP/EP program yet, for two
reasons:

1. DSpark does not boot on either TP/EP arm, and PP-2 DSpark (76.12) beats every plain arm.
2. The larger win (`tp_ep_attn`, +9.0%) changes the text. The exact arm (`tp_ep`) gains 2.6%.

Missing gates before a TP/EP default, tracked in memra #679:

- DSpark fits under TP/EP at the served context.
- Attention TP keeps the PP-2 bits: gather the local `wo_a` outputs, then split `wo_b` by output
  rows, which keeps each row's accumulation program.
- The TP step stops being launch bound: a captured step, or fewer launches.

The gate setters keep their 2026-10-07 decide-by in `docs/FLAGS.md`.
