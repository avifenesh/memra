# DSv4 multi-row MoE stream visitor (memra #669): +26.8% served DSpark decode, same bits

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (the healthy pair of `../REBASELINE.md`), 500 W
limit, 2026-09-23. Served program: PP-2, matrix expert program, host sampler, DSpark drafter
(`MEMRA_DSV4_DRAFTER=dspark`), memra-server otherwise naked. One scored campaign at a time under
`/tmp/memra-gpu.lock`, 250 ms telemetry per row. Base: the one-token stream visitor of
`../m1-stream-664/` (memra #664, PR #672).

## What changed

A DSpark verify round runs the target on T = k+1 = 5 tokens, so each routed expert's CSR group
holds a handful of rows (one to five on a verify step, often one or two). The one-token visitor
only takes `m = 1` steps, so these rows still rode sktail, which pads each group to a 32-row tile.
The multi-row visitor (`moe_kq_mrow_stream_kernel<4>` in `cu/moe_f16_grouped.cu`) takes steps of
2..=16 rows under the same switch:

- Each group is cut into 16-row chunks. The rows of a chunk fill the `mma.sync m16n8k16` M
  dimension (unused rows are zero), so one expert weight stream feeds up to 16 tokens.
- The weight path is the one-token visitor's: one warp per n8 column tile, four warps per CTA, a
  private `cp.async` ring per warp, ModelOpt NVFP4 dequantized to f16 in registers straight into
  the B fragment, f32 accumulate, the same k order.
- Identity by construction: an MMA output row depends only on its own A row, B and C, so every
  row runs the exact chain sktail runs for it. Padding rows never mix into a live row.

Bit identity is proven at the kernel boundary by `cuda_mrow_stream_matches_sktail_bit_for_bit`
(ignored GPU test in `src/dsv4_grouped.rs`): gate, up, H and the down contribution compared byte
for byte on 2..17 rows, three bank partitions, five route patterns including groups past one
chunk (widest 86 rows) and seeded random top-6, with the dispatch counter required to move.
Box run: `raw/component/gate.log`, test binary `1f701b58...`, every case `EXACT`, 2 passed.

## Served A/B

Lane binary `da60b46c...` (tree `0d333aa2c` plus `raw/mrow-lane.patch`, a lane-only
`MEMRA_DSV4_MOE_MROW_STREAM` read that was never merged: `0` puts the verify rows back on sktail,
`1` is the shipped dispatch, `2` also routes the one-token step through the multi-row kernel).
One boot per row, order `A B B A A B B A A B` (A = mrow), cells `raw/cells-spec.txt`: a 32-token
warmup, greedy c1 on 8 prompts, sampled c1 on 8 prompts, greedy c1 ignore-eos on 4 prompts, 256
max tokens, the fixed prompts of `../raw/bench.py`. Queue `raw/q-mrow.sh`, summary
`raw/q-mrow.summary`, receipts `raw/ab/r<N>-m<arm>/` (cells.jsonl, controller.log, serve.log,
/metrics, /healthz, /readyz, 250 ms telemetry, binary sha256, env).

Greedy c1, 8 prompts:

| row | arm | decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled c1 tok/s | ignore-eos tok/s |
|---|---|---|---|---|---|---|---|---|---|
| r1 | mrow | 71.20 | 14.05 / 16.19 / 16.38 | 0.02 / 51.11 | 250 / 268 | 3,843 | 66.32 | 58.90 | 71.05 |
| r2 | sktail | 56.06 | 17.84 / 20.61 / 20.88 | 0.02 / 65.49 | 263 / 282 | 4,826 | 52.84 | 47.47 | 55.98 |
| r3 | sktail | 56.12 | 17.82 / 20.60 / 20.89 | 0.02 / 65.44 | 261 / 281 | 4,818 | 52.89 | 47.52 | 56.02 |
| r4 | mrow | 71.16 | 14.05 / 16.19 / 16.38 | 0.02 / 51.20 | 249 / 270 | 3,846 | 66.29 | 58.88 | 70.99 |
| r5 | mrow | 71.10 | 14.07 / 16.19 / 16.38 | 0.02 / 51.25 | 250 / 268 | 3,848 | 66.28 | 58.86 | 70.99 |
| r6 | sktail | 56.08 | 17.83 / 20.61 / 20.90 | 0.02 / 65.40 | 261 / 282 | 4,822 | 52.85 | 47.43 | 55.94 |
| r7 | sktail | 56.07 | 17.83 / 20.59 / 20.88 | 0.02 / 65.40 | 262 / 281 | 4,822 | 52.88 | 47.48 | 55.95 |
| r8 | mrow | 71.08 | 14.07 / 16.20 / 16.39 | 0.02 / 51.23 | 250 / 268 | 3,849 | 66.25 | 58.80 | 70.91 |
| r9 | mrow | 71.13 | 14.06 / 16.19 / 16.37 | 0.02 / 51.17 | 249 / 268 | 3,846 | 66.31 | 59.11 | 71.00 |
| r10 | sktail | 56.09 | 17.83 / 20.58 / 20.87 | 0.02 / 65.48 | 261 / 282 | 4,821 | 52.89 | 47.54 | 56.00 |

**Greedy: mrow 71.13 tok/s median (N=5, 71.08..71.20) against sktail 56.08 (N=5, 56.06..56.12),
+26.8%, TPOT p50 17.83 -> 14.06 ms. Sampled: 58.88 against 47.48, +24.0%. Ignore-eos: 70.99
against 55.98, +26.8%.** The arms are disjoint: a 15 tok/s gap against a within-arm spread under
0.15 tok/s. TTFT p50 moves 261 -> 250 ms; this lane did not isolate where those 11 ms go. ITL
p50 is 0.02 ms in both arms because a verify round emits its accepted tokens together; ITL p99 (the round period) drops 65.4 -> 51.2 ms.

**Identity.** Every row produced the same text on all 20 requests: greedy `aea6e69e 26ac8df7
850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled `53944095 73fe7f91 6329ee94
827c0b56 437a4427 07de3a97 8a59518d b424bd80`, ignore-eos the first four greedy hashes, warmup
`e78fb458`. These equal the plain rows below and the plain and DSpark hashes of `../REBASELINE.md`
and `../m1-stream-664/RESULTS.md`, so spec output still equals plain output.

**Thermal regime.** Pooled over both cards, samples at 50% utilization or more: power median
266..275 W per row, peak 297 W against the 500 W limit, SM clock 2617..2842 MHz (one 1687 MHz
sample in r8, which did not move its median), GPU temperature at most 66 C.

## Should the one-token step take the multi-row kernel too?

Same lane binary without the drafter, cells `raw/cells-base.txt` (warmup, greedy c1, sampled c1,
sampled c4, sampled c16). Order `1 2 0 0 2 1`: arm 1 is the shipped dispatch (one-token visitor on
t=1), arm 2 routes the t=1 step through the multi-row kernel, arm 0 is arm 1 with verify rows on
sktail (a plain server has no verify rows, so 0 and 1 run the same program).

| row | arm | greedy c1 tok/s | TPOT p50 ms | sampled c1 tok/s |
|---|---|---|---|---|
| p1 | 1 | 50.06 | 19.98 | 46.23 |
| p2 | 2 | 49.42 | 20.24 | 45.77 |
| p3 | 0 | 50.06 | 19.97 | 46.31 |
| p4 | 0 | 50.07 | 19.97 | 46.36 |
| p5 | 2 | 49.41 | 20.24 | 45.85 |
| p6 | 1 | 50.09 | 19.96 | 46.38 |

**No: arm 2 loses 1.3% (49.42 vs 50.07 greedy, N=2 each, disjoint).** The dedicated one-token
visitor stays the t=1 program, and the shipped dispatch keeps the split at `rows == 1`. All six
rows returned the same greedy and sampled hashes as the DSpark rows above.

## Gates

- Component: `cuda_mrow_stream_matches_sktail_bit_for_bit` EXACT on the box (above).
- DSpark identity gate (`dsv4-gpu-dspark-gate`, `MEMRA_DSV4_DRAFTER=dspark
  MEMRA_DSV4_DECODE_PATH=device`) on the lane tree: PASS on served defaults (`raw/gates/mrow-served/`,
  binary `0c521c7e...`). The first attempt (`raw/dspark-served/`) refused at rc=2 because the queue
  did not set the drafter.
- MROWGATE_RECEIPTS

## Against the bound

Served DSpark greedy is now 14.06 ms per emitted token (71.13 tok/s). The prior-art survey
(`../PRIOR-ART.md`) records public speculative-decode figures of 212 tok/s on this card shape (TP=2,
DSpark K5); memra served DSpark is at 34% of that. The next steps are the ones that survey names:
fewer, larger kernels around mHC and the router, PDL between launches, and the TP/EP served path
(memra #454) so both cards work on every token.
