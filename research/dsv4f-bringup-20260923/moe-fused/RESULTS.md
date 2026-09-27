# DSv4 fused one-token MoE (memra #694)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 2026-09-23. Served program: PP-2, matrix expert
program, host sampler; DSpark rows add `MEMRA_DSV4_DRAFTER=dspark`. Lane
`lane/dsv4-moe-fused-20260923`.

## What changed

The served plain step (t=1) ran the routed MoE of every layer as the grouped chain: act_quant of
x, route count, prefix and scatter, the x FP8-QAT half mirror, the gate and up stream visitors,
two scale_rows, the weighted SwiGLU, act_quant of h, the h mirror, the down visitor, scale_rows,
the slot scatter and combine_rows_m. Sixteen launches per layer. Two kernels now compute the same
program:

- `dsv4_moe_fused_gu_kernel`: the x mirror inline, then gate and up of every selected slot with
  the one-token stream visitor's body op for op, the macro scales, and SwiGLU with the slot's route
  weight into `h[slot]`.
- `dsv4_moe_fused_down_kernel`: the h mirror, down, the macro scale into `contribution[slot]`, and
  the last CTA to finish each column tile sums the slots in combine_rows_m's order into `y`.

Every f32 op is the chain's op in the chain's order under `-fmad=false`, so H, each contribution
and the combined row are bit-identical to the chain. The deferred fault words (#670) carry the
same bits: a slot naming no live expert, a lossy x mirror and a lossy h mirror. The pair engages
only on t=1 matrix steps with device routes and armed deferred faults (`moe_fused_engages`);
verify rounds, prefill, EP and TP/EP keep the chain.

## Correctness

| run | result | log |
|---|---|---|
| PRO 6000 pair, card 0, grouped suite on the lane merged with main `6978f5fac` | 10 passed, including `cuda_fused_one_token_moe_is_the_grouped_chain_bit_for_bit` (EXACT H/contribution/y per case, each red fixture's bit equal: route 0x1, input 0x2, intermediate 0x4) | `raw/pair/component/gate.log` |
| served DSpark identity gate (`dsv4-gpu-dspark-gate --served`, tape 416) | `GPU DSPARK GATE [PASS]`; the plain arm counts the fused dispatches | `raw/pair/dspark-served/gate.log` |

## Served A/B

### First pair: shared-control campaign

`../kernel-ab-short/` (protocol, all 40 rows, queue script, binary hashes): 2x RTX PRO 6000 WS at
450 W, base (main `6978f5fac`) and three lanes in a Williams order, N=5 per arm per mode. The
rows below are this lane's and base's, in campaign order; `lane_tab.py short fused2` prints them.

#### DSpark (`MEMRA_DSV4_DRAFTER=dspark`)

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 66.87 | 14.95 / 17.17 / 17.37 | 0.02 / 54.30 | 267 / 290 | 4,103 | 62.31 | 54.31 | 66.75 |
| r3 | fused2 | 67.19 | 14.88 / 17.19 / 17.39 | 0.02 / 54.24 | 264 / 283 | 4,076 | 62.52 | 54.33 | 67.06 |
| r7 | base | 66.86 | 14.96 / 17.17 / 17.37 | 0.02 / 54.27 | 270 / 292 | 4,095 | 62.27 | 54.06 | 66.78 |
| r8 | fused2 | 67.21 | 14.88 / 17.19 / 17.39 | 0.02 / 54.22 | 264 / 292 | 4,070 | 62.50 | 54.42 | 67.06 |
| r10 | fused2 | 67.17 | 14.89 / 17.19 / 17.39 | 0.02 / 54.24 | 265 / 282 | 4,072 | 62.50 | 54.26 | 67.05 |
| r12 | base | 66.88 | 14.95 / 17.18 / 17.37 | 0.02 / 54.30 | 267 / 284 | 4,091 | 62.34 | 54.25 | 66.73 |
| r13 | fused2 | 67.20 | 14.88 / 17.20 / 17.39 | 0.02 / 54.24 | 264 / 284 | 4,070 | 62.50 | 54.51 | 67.03 |
| r14 | base | 66.90 | 14.95 / 17.17 / 17.37 | 0.02 / 54.32 | 270 / 286 | 4,096 | 62.29 | 53.99 | 66.74 |
| r17 | base | 66.85 | 14.96 / 17.19 / 17.39 | 0.02 / 54.34 | 266 / 285 | 4,093 | 62.29 | 54.09 | 66.70 |
| r19 | fused2 | 67.18 | 14.88 / 17.20 / 17.39 | 0.02 / 54.22 | 265 / 284 | 4,072 | 62.48 | 54.20 | 67.08 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| fused2 | 5 | 67.19 | 54.33 | 67.06 | 264 |
| base | 5 | 66.87 | 54.09 | 66.74 | 267 |

Delta, lane vs base: greedy +0.46%, sampled +0.45%, ignore-eos +0.47%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: identical across every row of both arms; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 249..251 W, peak 482 W, SM clock 2610..2850 MHz, max temp 55 C, 3469 samples.

Thermal fused2: power median 249..251 W, peak 468 W, SM clock 2685..2857 MHz, max temp 57 C, 3428 samples.

#### Plain

| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 214 / 231 | 5,220 | 49.04 | 46.20 | 50.88 |
| r3 | fused2 | 56.62 | 17.66 / 17.68 / 17.68 | 17.42 / 18.64 | 210 / 232 | 4,712 | 54.28 | 50.81 | 56.57 |
| r7 | base | 50.93 | 19.64 / 19.64 / 19.64 | 19.39 / 20.60 | 217 / 267 | 5,226 | 48.94 | 45.95 | 50.88 |
| r8 | fused2 | 56.62 | 17.66 / 17.68 / 17.68 | 17.42 / 18.62 | 212 / 228 | 4,715 | 54.28 | 50.59 | 56.56 |
| r10 | fused2 | 56.62 | 17.66 / 17.68 / 17.68 | 17.42 / 18.63 | 211 / 237 | 4,716 | 54.24 | 50.01 | 56.57 |
| r12 | base | 50.93 | 19.63 / 19.64 / 19.65 | 19.39 / 20.58 | 212 / 230 | 5,217 | 49.07 | 46.16 | 50.88 |
| r13 | fused2 | 56.62 | 17.66 / 17.67 / 17.68 | 17.42 / 18.62 | 210 / 228 | 4,713 | 54.28 | 50.48 | 56.56 |
| r14 | base | 50.94 | 19.63 / 19.64 / 19.64 | 19.38 / 20.58 | 217 / 229 | 5,220 | 49.06 | 46.04 | 50.87 |
| r17 | base | 50.93 | 19.63 / 19.64 / 19.64 | 19.39 / 20.58 | 212 / 233 | 5,220 | 49.05 | 46.02 | 50.87 |
| r19 | fused2 | 56.65 | 17.65 / 17.67 / 17.67 | 17.41 / 18.61 | 213 / 235 | 4,715 | 54.27 | 50.35 | 56.58 |

| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |
|---|---|---|---|---|---|
| fused2 | 5 | 56.62 | 50.48 | 56.57 | 211 |
| base | 5 | 50.93 | 46.04 | 50.88 | 214 |

Delta, lane vs base: greedy +11.17%, sampled +9.66%, ignore-eos +11.18%.

Hash identity per cell (each arm's set of per-request sha tuples across its rows):

- `greedy-c1`: identical across every row of both arms; 8 requests, first hashes aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8
- `greedy-c1-ignore-eos`: DIFFERS; 4 requests, first hashes aea6e69e 26ac8df7 850f75ed 96bbce86
- `sampled-c1`: identical across every row of both arms; 8 requests, first hashes 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80

Thermal base: power median 216..218 W, peak 472 W, SM clock 2610..2850 MHz, max temp 54 C, 4290 samples.

Thermal fused2: power median 224..226 W, peak 487 W, SM clock 2610..2857 MHz, max temp 55 C, 3880 samples.

The plain `greedy-c1-ignore-eos` DIFFERS line is row r13 alone: its fourth request stalled after
181 tokens and was cut when the server was stopped (next section). Those 181 tokens are an exact
prefix of base's text for that request.

### Second pair: stress campaign, also an A/B

2x RTX PRO 6000 Server Edition, lane `899aac4d0` against main `6978f5fac`, one boot per row in
the order `F M M F F M M F`, N=4 per arm. Each boot runs 64 greedy ignore-eos requests and 8
sampled, 256 tokens (`raw/server-pair/`). A watchdog would have captured a gdb stack of any stall
past 90 s; none happened in 288 fused2 requests or 288 main requests.

### plain


### ieos-a

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | fused2 | 51.19 | 19.54 / 19.54 / 19.54 | 231 / 244 | 5,214 | 16/16 |
| r2 | main | 46.21 | 21.64 / 21.70 / 21.71 | 234 / 246 | 5,752 | 16/16 |
| r3 | main | 46.21 | 21.64 / 21.71 / 21.71 | 234 / 246 | 5,753 | 16/16 |
| r4 | fused2 | 51.16 | 19.55 / 19.55 / 19.55 | 233 / 245 | 5,217 | 16/16 |
| r5 | fused2 | 51.19 | 19.54 / 19.54 / 19.54 | 232 / 244 | 5,214 | 16/16 |
| r6 | main | 46.21 | 21.64 / 21.70 / 21.70 | 234 / 246 | 5,753 | 16/16 |
| r7 | main | 46.21 | 21.64 / 21.70 / 21.71 | 234 / 245 | 5,753 | 16/16 |
| r8 | fused2 | 51.17 | 19.54 / 19.55 / 19.55 | 232 / 244 | 5,214 | 16/16 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| fused2 | 4 | 51.18 | 19.54 | 232 |
| main | 4 | 46.21 | 21.64 | 234 |

Delta, lane vs base: decode +10.76%, TTFT -0.88%.

Hashes: identical across every row of both arms; 16 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8 aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### ieos-b

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | fused2 | 52.37 | 19.10 / 19.53 / 19.54 | 228 / 240 | 5,101 | 16/16 |
| r2 | main | 46.21 | 21.64 / 21.70 / 21.70 | 233 / 245 | 5,751 | 16/16 |
| r3 | main | 46.21 | 21.64 / 21.71 / 21.71 | 233 / 245 | 5,752 | 16/16 |
| r4 | fused2 | 51.16 | 19.54 / 19.55 / 19.55 | 231 / 242 | 5,215 | 16/16 |
| r5 | fused2 | 51.19 | 19.54 / 19.54 / 19.54 | 231 / 242 | 5,212 | 16/16 |
| r6 | main | 46.20 | 21.64 / 21.70 / 21.70 | 233 / 244 | 5,752 | 16/16 |
| r7 | main | 46.20 | 21.65 / 21.70 / 21.70 | 233 / 245 | 5,752 | 16/16 |
| r8 | fused2 | 51.18 | 19.54 / 19.54 / 19.54 | 231 / 243 | 5,213 | 16/16 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| fused2 | 4 | 51.18 | 19.54 | 231 |
| main | 4 | 46.21 | 21.64 | 233 |

Delta, lane vs base: decode +10.77%, TTFT -1.00%.

Hashes: identical across every row of both arms; 16 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8 aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### ieos-c

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | fused2 | 51.20 | 19.53 / 19.54 / 19.54 | 229 / 241 | 5,204 | 16/16 |
| r2 | main | 46.21 | 21.64 / 21.70 / 21.70 | 233 / 245 | 5,751 | 16/16 |
| r3 | main | 46.21 | 21.64 / 21.71 / 21.71 | 233 / 244 | 5,752 | 16/16 |
| r4 | fused2 | 51.17 | 19.54 / 19.55 / 19.55 | 231 / 243 | 5,215 | 16/16 |
| r5 | fused2 | 51.19 | 19.53 / 19.54 / 19.54 | 230 / 241 | 5,212 | 16/16 |
| r6 | main | 46.20 | 21.64 / 21.70 / 21.70 | 233 / 245 | 5,752 | 16/16 |
| r7 | main | 46.20 | 21.65 / 21.70 / 21.70 | 233 / 245 | 5,753 | 16/16 |
| r8 | fused2 | 51.18 | 19.54 / 19.54 / 19.54 | 231 / 243 | 5,213 | 16/16 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| fused2 | 4 | 51.19 | 19.54 | 230 |
| main | 4 | 46.21 | 21.64 | 233 |

Delta, lane vs base: decode +10.78%, TTFT -1.12%.

Hashes: identical across every row of both arms; 16 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8 aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### ieos-d

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | fused2 | 51.19 | 19.54 / 19.54 / 19.54 | 230 / 242 | 5,212 | 16/16 |
| r2 | main | 46.21 | 21.64 / 21.71 / 21.71 | 234 / 245 | 5,752 | 16/16 |
| r3 | main | 46.21 | 21.64 / 21.71 / 21.71 | 233 / 245 | 5,751 | 16/16 |
| r4 | fused2 | 51.17 | 19.54 / 19.55 / 19.55 | 231 / 242 | 5,215 | 16/16 |
| r5 | fused2 | 51.19 | 19.53 / 19.54 / 19.54 | 231 / 242 | 5,211 | 16/16 |
| r6 | main | 46.20 | 21.64 / 21.70 / 21.70 | 233 / 244 | 5,752 | 16/16 |
| r7 | main | 46.20 | 21.64 / 21.70 / 21.70 | 233 / 245 | 5,752 | 16/16 |
| r8 | fused2 | 51.18 | 19.54 / 19.54 / 19.54 | 231 / 243 | 5,213 | 16/16 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| fused2 | 4 | 51.18 | 19.54 | 231 |
| main | 4 | 46.21 | 21.64 | 233 |

Delta, lane vs base: decode +10.77%, TTFT -1.00%.

Hashes: identical across every row of both arms; 16 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8 aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### sampled

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | fused2 | 47.82 | 20.91 / 20.93 / 20.93 | 232 / 247 | 5,563 | 8/8 |
| r2 | main | 43.38 | 23.05 / 23.12 / 23.14 | 235 / 250 | 6,112 | 8/8 |
| r3 | main | 43.37 | 23.06 / 23.13 / 23.15 | 235 / 249 | 6,114 | 8/8 |
| r4 | fused2 | 47.71 | 20.96 / 20.99 / 20.99 | 233 / 248 | 5,576 | 8/8 |
| r5 | fused2 | 47.85 | 20.90 / 20.92 / 20.92 | 233 / 247 | 5,561 | 8/8 |
| r6 | main | 43.51 | 22.98 / 23.04 / 23.06 | 234 / 250 | 6,094 | 8/8 |
| r7 | main | 43.51 | 22.98 / 23.04 / 23.06 | 235 / 250 | 6,095 | 8/8 |
| r8 | fused2 | 47.73 | 20.95 / 20.97 / 20.97 | 232 / 247 | 5,574 | 8/8 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| fused2 | 4 | 47.77 | 20.93 | 233 |
| main | 4 | 43.45 | 23.02 | 235 |

Delta, lane vs base: decode +9.96%, TTFT -1.03%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

Thermal fused2: power median 189..190 W, peak 216 W, SM clock 2272..2415 MHz, max temp 49 C, 12063 samples.

Thermal main: power median 181..181 W, peak 203 W, SM clock 2280..2370 MHz, max temp 46 C, 13301 samples.


## Verdict

**The fused one-token MoE wins plain decode by about 11% on both pairs, bit-identical.** Plain greedy decode moves 50.93 -> 56.62 tok/s on the WS pair (+11.2%, N=5 per
arm) and 46.21 -> 51.18 on the Server Edition pair (+10.8%, N=4 per arm), sampled +9.7% and
+10.0%. DSpark moves +0.5%: its verify rows keep the chain. Every completed request of every row
matched main's text on both pairs.

**One stall, unattributed.** WS-pair row `short/plain/r13-fused2` stalled in its last request (the fourth
greedy ignore-eos request), after 181 tokens:

- The `dsv4-serve` thread was running on the host: state R, 421 s of user time.
- Both GPUs sat idle at 180 MHz.
- The server was stopped by hand after about 440 s.
- The pod refuses ptrace, so there is no stack (`../kernel-ab-short/README.md`).

Nothing in the two kernels can hold a thread on the host. They have no spin and no early exit;
the last CTA of a column tile combines and resets its counter. The counters live in each
request's per-stage workspace, not across cards. The Server Edition pair then served 288 fused2
requests under a gdb watchdog without a stall. A stalled cross-card copy would also leave the
SMs idle while a host sync spins, and would also fit.

The same stress runs again on the WS pair, the pod where the stall happened, with a
thread-state watchdog (`raw/q-pair7.sh`). The lane merges after that stress: a stall there under
fused2 and not under main blocks it; none keeps the verdict.

**WS-pair stress, 2026-09-24 (`raw/pair-stress/`):** 8 boots in the order F M M F F M M F
(fused2 `899aac4d0`, main `6978f5fac`). Each boot served 64 greedy ignore-eos requests plus 8
sampled ones. All 8 rows returned rc=0 with every request ok. No row stalled under either
binary, and the watchdog never fired. The stall stays a one-off with its cause unknown. It did
not reproduce under fused2 in 256 requests on the pod where it happened, so the verdict holds.
