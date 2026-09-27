# DSv4 small-kernel diet on every row count (S1): +1.6% served DSpark decode, plain flat, same bits

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (the dev pair), 500 W limit, 2026-09-23. Served
program: PP-2, matrix expert program, host sampler, memra-server otherwise naked; DSpark rows add
`MEMRA_DSV4_DRAFTER=dspark`. One scored campaign at a time under `/tmp/memra-gpu.lock`, 250 ms
telemetry per row. Lane `lane/dsv4-diet-mrow-20260923`, change `b2da4b66b`.

## What changed

The small-kernel diet (`../small-diet/RESULTS.md`, #675) fused the HC finish
(`rowsq_scale_f32acc` + `hc_sinkhorn_m` + `hc_collapse`) and the Q-LoRA norm/pack
(`rmsnorm_f32acc` + `cvt_bf16`) on t=1 steps only. A DSpark verify round runs the target on
T=6 rows, so every verify step kept the unfused launches at each of the 86 HC sites and 43 Q
sites. Both fused kernels now index their row by `blockIdx.x` and launch one CTA per row, which
is exactly how the unfused kernels lay rows out, and the two dispatch conditions drop `t == 1`.
A verify row therefore runs the chain the same row runs when decoded alone.

## Kernel boundary

`crates/memra-engine/tests/dsv4_small_diet_gpu.rs` adds multi-row cases (2, 6, 7, 16, 17 and 64
rows, one magnitude range per row). Each row must equal the unfused multi-row chain and the same
row launched alone, bit for bit, on every output.

| run | result | log |
|---|---|---|
| PRO 6000 dev pair, card 0, lane merged with main `c3eb41d12` | 4 passed: `DSV4_SMALL_HC_DIET EXACT cases=96 outputs=5 red_arms=3`, the multi-row HC and norm/pack cases, the red arms | `raw/main3/dmrow3/component/gate.log` |
| same card, the latency kernel suite on the same tree | 7 passed | `raw/main3/dmrow3/latency/gate.log` |
| served DSpark identity gate (`dsv4-gpu-dspark-gate --served`, tape 416, determinism x2) | `GPU DSPARK GATE [PASS]` | `raw/main3/dmrow3/dspark-served/gate.log` |

## Served A/B

Two binaries, no lane-only read. Lane = `a90851bff` (the change merged with main `c3eb41d12`),
base = main `c3eb41d12`. One boot per row, order `l b b l l b b l l b`, cells
`raw/main3/cells-spec.txt` (a 32-token warmup, then greedy c1 on 8 prompts, sampled c1 on 8,
greedy ignore-EOS on 4, 256 max tokens). Queue `raw/main3/q-dmrow3.sh`, summary
`raw/main3/q-dmrow3.summary`, receipts `raw/main3/dmrow3/{spec,plain}/<row>-<arm>/`
(cells.jsonl, controller.log, serve.log, /metrics, /healthz, /readyz, 250 ms telemetry, binary
sha256, env). Server binaries: lane `5b82e4e86e54851c`, base `eed10fd0792c18ce`.

### DSpark

| row | arm | greedy decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | greedy agg tok/s | sampled decode tok/s |
|---|---|---|---|---|---|---|---|
| s1 | lane | 77.58 | 12.89 / 14.83 / 15.01 | 237 / 255 | 3,536 | 72.20 | 63.95 |
| s2 | base | 76.32 | 13.10 / 15.07 / 15.25 | 239 / 261 | 3,602 | 71.00 | 62.85 |
| s3 | base | 76.20 | 13.12 / 15.08 / 15.26 | 239 / 257 | 3,597 | 71.01 | 62.94 |
| s4 | lane | 77.52 | 12.90 / 14.83 / 15.01 | 237 / 261 | 3,539 | 72.13 | 63.62 |
| s5 | lane | 77.56 | 12.89 / 14.86 / 15.02 | 237 / 255 | 3,536 | 72.09 | 63.60 |
| s6 | base | 76.33 | 13.10 / 15.07 / 15.25 | 238 / 256 | 3,591 | 71.07 | 62.98 |
| s7 | base | 76.33 | 13.10 / 15.07 / 15.26 | 238 / 257 | 3,591 | 71.07 | 62.84 |
| s8 | lane | 77.43 | 12.91 / 14.84 / 15.03 | 238 / 255 | 3,546 | 72.09 | 63.64 |
| s9 | lane | 77.45 | 12.91 / 14.88 / 15.04 | 237 / 256 | 3,542 | 71.97 | 63.82 |
| s10 | base | 76.28 | 13.11 / 15.09 / 15.27 | 238 / 259 | 3,593 | 70.95 | 62.70 |

**Lane 77.52 tok/s greedy median (N=5, 77.43..77.58) against base 76.32 (N=5, 76.20..76.33),
+1.6%, TPOT p50 13.10 -> 12.90 ms.** Sampled 63.64 against 62.85 (N=5 each), +1.3%. The arms
are disjoint in every row. TTFT moves 238..239 -> 237..238 ms.

### Plain

Lane 63.37 tok/s greedy median (N=5, 63.30..63.42) against base 63.40 (N=5, 63.31..63.44):
flat, as expected, since a plain step is t=1 and already took the diet. TTFT 191..193 against
192..196 ms: the prime's multi-row sites are a small share of a 190 ms prime.

**Identity.** All twenty rows returned the fixed texts: greedy `aea6e69e 26ac8df7 850f75ed
a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled `53944095 73fe7f91 6329ee94 827c0b56
437a4427 07de3a97 8a59518d b424bd80`, warmup `e78fb458`, the same tuples as plain.

**Thermal regime.** Pooled over both cards: DSpark rows power median 265..273 W, peak 299 W,
plain rows 245..252 W, peak 282 W, against the 500 W limit; SM clock up to 2880 MHz; GPU
temperature at most 67 C.

### Earlier runs on the pre-#678 tree

- `raw/main3/q-dmrow2.sh` repeated the same two-binary design on lane `58348d94d` (main plus
  `b7e837993`) against main `7029cd67c`, N=4 per arm: DSpark 74.63 against 73.27 tok/s (+1.9%,
  sampled 61.45 against 60.37), plain 55.96 against 55.98 (flat), same hashes. Receipts
  `raw/main3/dmrow2/`, summary `raw/main3/q-dmrow2.summary`. Both absolute levels are lower
  than dmrow3 because main moved in between (#678).
- `raw/q-dev2.sh` (receipts `raw/{spec,plain,component,dspark-served}/`, summary
  `raw/q-dev2.summary`) is **void as an A/B**: its lane-only read (`raw/dmrow-lane.patch`,
  `MEMRA_DSV4_DIET_AB=t1`) did not apply, so both arms ran the lane binary. It reads 74.58
  against 74.62, which is the lane measured twice, not a flat result. Its component gate and
  served identity gate are valid for that tree and kept as such.

## Verdict

Winner, no door: the diet takes every row count by default on the device f32x HC4 hidden-4096
programs. The gain is DSpark-only, where verify rows now skip three launches per HC site and one
per Q site.
