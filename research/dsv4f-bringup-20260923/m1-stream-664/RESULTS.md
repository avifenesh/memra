# DSv4 one-token MoE stream visitor (memra #664): +29% served plain decode, same bits

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (the healthy pair of `../REBASELINE.md`, effective
SM clock 2839..2861 MHz, no power brake), 500 W limit, 2026-09-23. Served program: PP-2, matrix
expert program, host sampler, memra-server otherwise naked. One scored campaign at a time under
`/tmp/memra-gpu.lock`, 250 ms telemetry per row.

## What changed

The served plain step runs each expert projection of a routed token as one CSR row (`m_e = 1`).
Before this lane those rows rode the sktail visitor, a 32x64x64 tensor-core tile that pads the one
row to 32 and walks one 64-column tile per CTA. The stream visitor (`moe_kq_m1_stream_kernel<4>` in
`cu/moe_f16_grouped.cu`) keeps the numeric program and changes the schedule:

- The same ModelOpt NVFP4 dequant to f16, the same `mma.sync m16n8k16` f16 x f16 -> f32, and the
  same k order, so every output element sees the same products summed in the same order.
- One warp per n8 column tile, four warps per CTA. The activation row is staged in shared memory
  once per CTA; each warp then streams its 8 weight rows through a private `cp.async` ring and
  dequantizes them in registers into the MMA B fragment, with no block barrier after the stage.
- Gate and up take it on the one-token plain step; down takes it unless a gate armed the M1
  tensor-core or half2 down tail, which keeps precedence (the TP/EP bench pins that program and
  counts its enqueues).

Bit identity is proven at the kernel boundary by
`cuda_m1_stream_matches_sktail_bit_for_bit` (ignored GPU test in `src/dsv4_grouped.rs`): an
exhaustive dequant probe over all 256 FP4 codes and every UE4M3 scale, then four route patterns
through `gate_up` and `down` with the outputs compared byte for byte, the engagement counter
required to move, and the gate-armed down precedence case. `docs/TESTING.md` has the command.

## Served A/B

Lane binary `a5c644c3...` with a lane-only `MEMRA_DSV4_MOE_M1_STREAM` read (the shipped code has
no read; the stream is the default). One boot per row, order `A B B A A B B A A B` (A = stream),
each row a 32-token warmup and then the 8-prompt greedy c1 cell, 256 max tokens, the fixed prompts
of `../raw/bench.py`. Queue script `raw/q-m1ab.sh`, summary `raw/q-m1ab.summary`, per-row receipts
`raw/ab/r<N>-<arm>/` (cells.jsonl, controller.log, serve.log, /metrics, /healthz, /readyz, 250 ms
telemetry, binary sha256, env).

| row | arm | decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s |
|---|---|---|---|---|---|---|---|
| r1 | stream | 50.06 | 19.98 / 20.00 / 20.00 | 19.86 / 20.60 | 201 / 216 | 5,294 | 48.34 |
| r2 | sktail | 38.77 | 25.79 / 25.99 / 26.07 | 25.66 / 26.68 | 205 / 223 | 6,782 | 37.82 |
| r3 | sktail | 38.80 | 25.77 / 25.97 / 26.05 | 25.64 / 26.67 | 206 / 224 | 6,780 | 37.84 |
| r4 | stream | 50.13 | 19.95 / 19.98 / 19.98 | 19.84 / 20.58 | 199 / 218 | 5,284 | 48.40 |
| r5 | stream | 50.11 | 19.96 / 19.99 / 19.99 | 19.84 / 20.64 | 200 / 216 | 5,288 | 48.38 |
| r6 | sktail | 38.79 | 25.78 / 25.96 / 26.04 | 25.65 / 26.65 | 205 / 222 | 6,781 | 37.85 |
| r7 | sktail | 38.77 | 25.79 / 25.99 / 26.06 | 25.66 / 26.65 | 205 / 222 | 6,781 | 37.83 |
| r8 | stream | 50.10 | 19.96 / 19.99 / 20.00 | 19.85 / 20.57 | 200 / 217 | 5,288 | 48.38 |
| r9 | stream | 50.09 | 19.96 / 19.99 / 20.00 | 19.84 / 20.64 | 200 / 216 | 5,287 | 48.37 |
| r10 | sktail | 38.83 | 25.75 / 25.96 / 26.04 | 25.63 / 26.66 | 205 / 223 | 6,775 | 37.86 |

**Stream 50.10 tok/s median (N=5, 50.06..50.13) against sktail 38.79 (N=5, 38.77..38.83): +29.2%,
TPOT 25.78 -> 19.96 ms (-5.82 ms per token).** The arms are disjoint: an 11.2 tok/s gap against a within-arm
spread under 0.1 tok/s. TTFT p50 moves 205 -> 200 ms; this lane did not isolate where in the prime those 5 ms go.

**Identity.** Every row produced the same text on all 8 prompts (sha256 prefixes `aea6e69e
26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, and `e78fb458` for the warmup),
which are also the plain and DSpark hashes of `../REBASELINE.md`.

**Thermal regime.** Decode power 214..220 W median per card, peak 241 W against the 500 W limit,
SM clock under load 2595..2850 MHz, GPU temperature at most 60 C in every row.

The queue also held four TP/EP full-token replay rows. They were skipped (`raw/w-skip.sh`) before
any row measured: the lane binary on the box put the stream ahead of the gate-armed half2 down tail,
which would trip the bench's `down_half2 == local_steps` engagement assertion, and the shipped order keeps
the gate arm first, so those rows measured a program that no longer exists. (The watcher was
started twice, so the skip line is in the summary twice.)

## DSpark on top

Same lane binary, `MEMRA_DSV4_DRAFTER=dspark`, one boot per row, order `A B B A` (A = stream),
cells `raw/cells-spec.txt` (32-token warmup, greedy c1 8 prompts, sampled c1 8 prompts, greedy c1
ignore-eos 4 prompts). Queue `raw/q-m1spec.sh`, summary `raw/q-m1spec.summary`, receipts
`raw/spec/s<N>-<arm>/`.

| row | arm | greedy c1 tok/s | sampled c1 tok/s | ignore-eos tok/s | greedy TPOT p50 / p95 / p99 ms | greedy TTFT p50 ms |
|---|---|---|---|---|---|---|
| s1 | stream | 56.07 | 47.47 | 56.00 | 17.83 / 20.60 / 20.90 | 261 |
| s2 | sktail | 55.98 | 47.49 | 55.94 | 17.86 / 20.60 / 20.89 | 267 |
| s3 | sktail | 56.07 | 47.50 | 55.94 | 17.84 / 20.58 / 20.88 | 267 |
| s4 | stream | 56.12 | 47.48 | 56.01 | 17.82 / 20.59 / 20.89 | 261 |

**Flat: greedy 56.10 vs 56.03 (+0.1%), sampled 47.48 vs 47.50, ignore-eos 56.01 vs 55.94.** Both
arms are N=2, so this is a no-effect reading, not a measured delta. It is the expected shape: a
DSpark step is a T=k+1 verify transaction plus drafter rows, and the stream only takes the
one-token plain step, so the verify rows still ride sktail. The multi-row visitor lane
(`lane/dsv4-moe-mrow-20260923`) is the one that moves the spec step. Every row returned the same
text as sktail on all 20 requests (greedy `aea6e69e ... 937f04d8`, sampled `53944095 73fe7f91
6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80`), so spec output equals plain output under
both arms.

## Final tree

Tree `093f96627` (the shipped code: no read, the stream is the default), binary `3f587f24...`,
receipts `raw/fin/`, summary `raw/q-fin.summary`.

- Component: `cuda_m1_stream_matches_sktail_bit_for_bit` passes on the box (`raw/fin/component/`).
- Naked plain server: greedy c1 50.08 tok/s (TPOT p50 19.97 ms), sampled c1 46.47, the same
  hashes as every A/B row. Naked DSpark server: greedy c1 56.11, sampled c1 47.45, ignore-eos
  55.95.
- DSpark identity gate (`dsv4-gpu-dspark-gate`, `MEMRA_DSV4_DRAFTER=dspark
  MEMRA_DSV4_DECODE_PATH=device`): PASS on served defaults and on historical pins, binary
  `633d3422...` (`raw/gates/fin-served/`, `raw/gates/fin-hist/`, `raw/q-gates.summary`). Greedy
  spec equals plain on 160/160 tokens in the sequential and batched arms, batched verify equals
  sequential bit for bit on 14 cells (77 logit rows, 3206 cache classes), accepted-position ring
  writes are bit-identical, two batched runs are identical. Arm P logs 20640 stream dispatches,
  so the stream visitor is the kernel under test. The first attempt (`raw/fin/dspark-served/`,
  `raw/fin/dspark-hist/`) refused at rc=2 because the queue did not set the drafter; the gate
  refuses rather than running the wrong program.

The plain server cell also recorded the serve route's concurrency shape on this tree:

| cell | ok | agg tok/s | TTFT p50 / p95 ms | TPOT p50 ms |
|---|---|---|---|---|
| sampled c1 | 8/8 | 45.00 | 201 / 218 | 21.52 |
| sampled c4 | 16/16 | 44.93 | 17,270 / 17,315 | 21.53 |
| sampled c16 | 9/32 (23 got 429) | 44.63 | 17,272 / 21,084 | 21.56 |

The route is still one request at a time, so aggregate throughput is flat in concurrency and the
queue shows up as TTFT. That is memra #667, not this lane.

## Where the step goes now

nsys over 510 served plain steps on the final binary, same prompts (`raw/prof/served-plain-m1/ana.txt`;
the 1.1 GB captures stay off the repo). The capture runs 22.06 ms per step under the profiler.

| | kernels / step | kernel-sum ms | busy-union ms |
|---|---|---|---|
| dev0 (layers 0..22) | 1,711 | 9.06 | 9.29 |
| dev1 (layers 23..42 + head) | 1,662 | 9.41 | 9.64 |

Before the stream (`raw/prof/served-plain/ana.txt`) the two cards were 12.41 and 12.20 ms busy.
The two cards' busy time adds up to the whole step, because PP-2 at one request in flight leaves
each card idle while the other runs. Top kernels on dev0 now: `dsv4_dense_fast_fp8_kernel` 2.43 ms
(341 launches), `moe_kq_m1_stream_kernel` 1.79 ms (66), `dsv4_rmsnorm_f32acc_kernel` 0.73 ms (94),
`dsv4_hc_sinkhorn_m_kernel` 0.69 ms (44, one 32-thread CTA each), `dsv4_dense_fast_dots_kernel`
0.54 ms. The host issues 3,373 launches and 131 D2H copies per step. The small-kernel diet (#339)
targets the rmsnorm, Sinkhorn, rowsq and bf16 pack share.

## Against the bound

Served PP-2 plain greedy is now 19.96 ms per token, 32.0% of the 6.38 ms PP-2 bandwidth bound
(was 24.7%). The public vLLM TP=2 floor on this card shape is 9.15 ms per token (109.3 tok/s);
memra served plain is at 46% of it (was 35%).
