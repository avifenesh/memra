# DSv4 exact attention TP2 on the TP/EP program (memra #679)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition (600 W limit, SM clock max 2430 MHz), 2026-09-23/24.
Lane `lane/dsv4-attn-tp-exact-20260923` `75c55dfd4` (served rows and gates), `f33ee547c` (replay
gate census fix). The served rows select the topology through `raw/topology-lane.patch`, a
lane-only server selector that is not part of this lane's merge.

## What changed

The TP/EP program replicates every trunk layer on both ranks and splits the experts. With
attention TP2 each rank takes 32 of the 64 heads. Before this lane the two ranks' attention
outputs met in an all-reduce, whose sum order differs from the one-rank program, so `tp_ep_attn`
served its own text (`../tpep-serve/RESULTS.md`).

- wo_b is split by output rows, not by input columns. Each rank computes its heads' wo_a
  groups, the ranks gather those rows (`memra_tp_ar_gather_rows_f32_kernel`), then each rank
  computes its half of the wo_b output rows over the full wo_a rows, and a second gather joins
  them. No value is ever summed across ranks, so each output element has the one-rank program's
  arithmetic.
- The full Q_b/wo_a/wo_b planes are freed after packing the rank-local rows (8.06 GiB per rank).
- The attention TP2 ranks take the grouped wo_a launch for their four groups.
- The full-token replay census counts 129 collectives per graph: the expert reduction and two
  row gathers per layer.

Numeric class `dsv4_attention_head_split_row_gather_exact`.

## Correctness

| gate | result | log |
|---|---|---|
| `dsv4_tp_ep_gate`, attention TP off vs on | `DIGEST_EQUAL yes`: output and state sha256 identical (`c1e0448d...`, `c391bfaa...`) | `raw/tpep-gate-at0/`, `raw/tpep-gate-at1/` |
| multi-row verify on the exact program | `PASS topology=tpep rows chunk=65 verify=70 restore=22 bit-equal to sequential` | `raw/verify-at1/` |
| DSpark gate `--tpep` on the exact program | `GPU DSPARK GATE [PASS]` | `raw/dspark-tpa/` |
| full-token replay, eager vs graph | first run: panicked on the AR epoch census (it still counted an attention all-reduce; block 0 ticked 258 for an expected 172, which is the two gathers). Census fixed in `f33ee547c`; rerun pending | `raw/replay-at1/` |
| served text | every request of every row hashed the same as PP-2 in all three cells | below |

## Served A/B (plain)

One boot per row, order `pp tpa tpa pp pp tpa tpa pp pp tpa`, N=5 per arm, then two `tp` rows
(the TP/EP program without attention TP) at the end. Those are not interleaved, so they are a
reading, not an A/B. Cells: greedy c1 (8 requests), sampled c1 (8), greedy c1 ignore-eos (4),
256 tokens.

### greedy-c1

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | pp | 46.25 | 21.62 / 21.67 / 21.69 | 235 / 250 | 5,749 | 8/8 |
| r2 | tpa | 48.54 | 20.60 / 20.63 / 20.64 | 179 / 191 | 5,431 | 8/8 |
| r3 | tpa | 48.53 | 20.61 / 20.64 / 20.64 | 177 / 190 | 5,431 | 8/8 |
| r4 | pp | 46.22 | 21.64 / 21.68 / 21.70 | 235 / 249 | 5,753 | 8/8 |
| r5 | pp | 46.20 | 21.64 / 21.69 / 21.70 | 235 / 250 | 5,755 | 8/8 |
| r6 | tpa | 48.53 | 20.60 / 20.63 / 20.64 | 178 / 190 | 5,431 | 8/8 |
| r7 | tpa | 48.52 | 20.61 / 20.64 / 20.65 | 178 / 190 | 5,432 | 8/8 |
| r8 | pp | 46.22 | 21.64 / 21.69 / 21.70 | 235 / 250 | 5,753 | 8/8 |
| r9 | pp | 46.21 | 21.64 / 21.69 / 21.70 | 235 / 250 | 5,754 | 8/8 |
| r10 | tpa | 48.55 | 20.60 / 20.63 / 20.63 | 178 / 190 | 5,429 | 8/8 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| tpa | 5 | 48.53 | 20.60 | 178 |
| pp | 5 | 46.22 | 21.64 | 235 |

Delta, lane vs base: decode +5.02%, TTFT -24.34%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### sampled-c1

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | pp | 43.45 | 23.02 / 23.09 / 23.10 | 235 / 250 | 6,105 | 8/8 |
| r2 | tpa | 45.53 | 21.96 / 21.99 / 21.99 | 177 / 190 | 5,771 | 8/8 |
| r3 | tpa | 45.51 | 21.97 / 22.00 / 22.01 | 177 / 189 | 5,771 | 8/8 |
| r4 | pp | 43.47 | 23.01 / 23.07 / 23.09 | 235 / 250 | 6,101 | 8/8 |
| r5 | pp | 43.41 | 23.03 / 23.10 / 23.12 | 235 / 250 | 6,109 | 8/8 |
| r6 | tpa | 45.59 | 21.94 / 21.97 / 21.97 | 176 / 189 | 5,766 | 8/8 |
| r7 | tpa | 45.49 | 21.98 / 22.02 / 22.02 | 176 / 189 | 5,777 | 8/8 |
| r8 | pp | 43.39 | 23.05 / 23.12 / 23.14 | 235 / 250 | 6,113 | 8/8 |
| r9 | pp | 43.38 | 23.05 / 23.12 / 23.13 | 236 / 250 | 6,114 | 8/8 |
| r10 | tpa | 45.50 | 21.98 / 22.01 / 22.01 | 176 / 190 | 5,774 | 8/8 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| tpa | 5 | 45.51 | 21.97 | 176 |
| pp | 5 | 43.41 | 23.03 | 235 |

Delta, lane vs base: decode +4.83%, TTFT -24.96%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

### greedy-c1-ignore-eos

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | pp | 46.23 | 21.63 / 21.63 / 21.63 | 233 / 249 | 5,749 | 4/4 |
| r2 | tpa | 48.60 | 20.57 / 20.61 / 20.62 | 175 / 189 | 5,420 | 4/4 |
| r3 | tpa | 48.96 | 20.42 / 20.52 / 20.54 | 174 / 187 | 5,390 | 4/4 |
| r4 | pp | 46.21 | 21.64 / 21.64 / 21.64 | 233 / 249 | 5,751 | 4/4 |
| r5 | pp | 46.20 | 21.65 / 21.65 / 21.65 | 233 / 249 | 5,753 | 4/4 |
| r6 | tpa | 48.63 | 20.56 / 20.62 / 20.63 | 175 / 189 | 5,418 | 4/4 |
| r7 | tpa | 48.64 | 20.56 / 20.62 / 20.62 | 175 / 188 | 5,416 | 4/4 |
| r8 | pp | 46.21 | 21.64 / 21.64 / 21.64 | 233 / 249 | 5,751 | 4/4 |
| r9 | pp | 46.20 | 21.65 / 21.65 / 21.65 | 233 / 250 | 5,753 | 4/4 |
| r10 | tpa | 48.67 | 20.55 / 20.61 / 20.62 | 175 / 188 | 5,414 | 4/4 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| tpa | 5 | 48.64 | 20.56 | 175 |
| pp | 5 | 46.21 | 21.64 | 233 |

Delta, lane vs base: decode +5.25%, TTFT -25.18%.

Hashes: identical across every row of both arms; 4 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a.

Thermal pp: power median 177..178 W, peak 202 W, SM clock 2280..2370 MHz, max temp 46 C, 4818 samples.

Thermal tpa: power median 211..213 W, peak 234 W, SM clock 2272..2415 MHz, max temp 47 C, 4565 samples.

`tp` (TP/EP, attention replicated), rows 11 and 12, greedy c1:

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r11 | tp | 46.74 | 21.39 / 21.41 / 21.41 | 201 / 217 | 5,656 | 8/8 |
| r12 | tp | 46.76 | 21.39 / 21.42 / 21.42 | 200 / 217 | 5,649 | 8/8 |
| r11 | tp | 44.79 | 22.32 / 22.41 / 22.42 | 200 / 214 | 5,891 | 8/8 |
| r12 | tp | 44.69 | 22.38 / 22.62 / 22.66 | 199 / 215 | 5,904 | 8/8 |
| r11 | tp | 48.12 | 20.78 / 20.92 / 20.94 | 199 / 214 | 5,498 | 4/4 |
| r12 | tp | 47.94 | 20.86 / 20.98 / 20.99 | 199 / 213 | 5,515 | 4/4 |

## DSpark on the exact program

The drafter now loads resident on stage 1. Then memory calibration fails allocating the
verify workspace: `dsv4 memory calibration: vws f32: CUDA_ERROR_OUT_OF_MEMORY` at the served
`max_seq` 1,048,576 (`raw/spec/fit-tpa/serve.log`). DSpark does not fit on the TP/EP program at
the served context. The DSpark served rows were not run.

## Step anatomy

nsys over the served exact-TP route, 510 greedy steps (`raw/prof/tpa/ana.txt`; the capture
slows the step from 20.6 to 23.4 ms):

```
capture span 11938.3 ms over 510 steps = 23.41 ms/step
dev0: kernels 1517482 (2975/step) kernel-sum 16.36 ms/step busy-union 16.75 ms/step memcpys 462/step
dev1: kernels 1520570 (2982/step) kernel-sum 15.50 ms/step busy-union 15.91 ms/step memcpys 463/step
--- dev0 top kernels (ms/step, count/step, share)
    3.184    366  19.5%  dsv4_dense_fast_fp8_kernel
    2.688     87  16.4%  memra_tp_ar_gather_rows_f32_kernel
    2.004    130  12.2%  moe_kq_m1_stream_kernel
    1.188    168   7.3%  dsv4_dense_fast_dots_kernel
    1.051     87   6.4%  dsv4_small_hc_f32_fixed_order_kernel
    0.904     43   5.5%  memra_tp_ar_1stage_kernel
    0.749     43   4.6%  dsv4_sink_scores_mq_f32acc_kernel
    0.557     43   3.4%  dsv4_sink_out_mq_f32acc_kernel
    0.491     87   3.0%  dsv4_fp8_gather_half_kernel
    0.241     21   1.5%  dsv4_indexer_score_f32acc_kernel
    0.231     43   1.4%  dsv4_route_m_kernel
    0.223    142   1.4%  dsv4_rmsnorm_f32acc_kernel
    0.207    281   1.3%  dsv4_cvt_bf16_kernel
    0.206     87   1.3%  dsv4_hc_dot_split_partial_kernel
    0.205     43   1.3%  dsv4_small_norm_pack_f32_fixed_order_kernel
--- dev1 top kernels (ms/step, count/step, share)
    3.208    366  20.7%  dsv4_dense_fast_fp8_kernel
    2.015    130  13.0%  moe_kq_m1_stream_kernel
    1.921    170  12.4%  dsv4_dense_fast_dots_kernel
    1.081     87   7.0%  dsv4_small_hc_f32_fixed_order_kernel
    0.983     87   6.3%  memra_tp_ar_gather_rows_f32_kernel
    0.816     43   5.3%  memra_tp_ar_1stage_kernel
    0.769     43   5.0%  dsv4_sink_scores_mq_f32acc_kernel
    0.575     43   3.7%  dsv4_sink_out_mq_f32acc_kernel
    0.499     87   3.2%  dsv4_fp8_gather_half_kernel
    0.244     21   1.6%  dsv4_indexer_score_f32acc_kernel
    0.236     43   1.5%  dsv4_route_m_kernel
    0.228    143   1.5%  dsv4_rmsnorm_f32acc_kernel
    0.213    281   1.4%  dsv4_cvt_bf16_kernel
    0.209     87   1.3%  dsv4_hc_dot_split_partial_kernel
    0.207     43   1.3%  dsv4_small_norm_pack_f32_fixed_order_kernel
memcpy kind: count/step bytes/step ms/step {1: (91.1921568627451, 48280, 0.031), 8: (828.8, 26707192, 0.671), 2: (5.03921568627451, 594535, 0.015)}
```

```
--- runtime API (count/step, ms/step)
      5957   15.010  cudaLaunchKernel_v7000
       829    2.737  cuMemcpyDtoDAsync_v2
         5    0.959  cuMemcpyDtoHAsync_v2
      5957    0.461  cuKernelGetName
        94    0.237  cuMemsetD8Async
        91    0.159  cuMemcpyHtoDAsync_v2
       444    0.038  cuCtxSetCurrent
         7    0.017  cuStreamSynchronize
         0    0.005  cuMemHostAlloc
         7    0.004  cuMemAllocAsync
         7    0.004  cuMemFreeAsync
         0    0.003  cuMemFreeHost
         0    0.000  cuMemGetInfo_v2
         0    0.000  cuEventDestroy_v2
```

The host issues 5,957 kernel launches per step, and they take 15.0 ms of host time. Each card is
busy 15.9 to 16.8 ms of the 23.4 ms step. The row gathers cost 2.7 ms per step on card 0
against 1.0 ms on card 1. A gather moves a few KiB per layer, so the difference is most likely
card 0 arriving first and waiting at the join; this profile does not separate wait from copy.

## Verdict

**Exact attention TP2 is bit-identical to the PP-2 program and 5.0% faster than it at one
request.** Greedy decode 46.22 -> 48.53 tok/s (N=5 per arm), sampled +4.8%, TTFT p50 235 -> 178
ms (-24%). Every request's text equals PP-2's, where the all-reduce form served its own text.

It does not change which route is fastest for one stream. DSpark does not fit on TP/EP at the
served context. On the WS pair, PP-2 DSpark served 66.87 tok/s against 50.93 plain
(`../kernel-ab-short/`), a 31% margin that the 5% here does not close. That comparison is from a
different pair; DSpark was not measured on this one. The TP/EP step is
launch bound: removing the host's share of it (captured steps) comes before any kernel work
there. The server-side topology selector stays out of main until TP/EP beats PP-2 DSpark.
