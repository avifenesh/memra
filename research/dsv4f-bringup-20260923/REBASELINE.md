# DSv4-Flash served baseline on a healthy pair, 2026-09-23

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, 500 W power limit. Rented
box. Source main `f69119ae0`. One scored campaign at a time under `/tmp/memra-gpu.lock`, 250 ms
GPU telemetry per cell, single runs unless a row says otherwise.

This replaces `BASELINE.md`, which was measured on a pair with the hardware power brake latched
(effective SM clock 722 MHz behind a reported 2865 MHz). Finding and proof:
`power-brake/POWER-BRAKE.md`. Card acceptance on this pair:
effective SM clock 2839..2861 MHz on both cards, HW Slowdown and HW Power Brake Not Active,
braking counter 0 us (`power-brake/raw/healthy/`).

## TP/EP full-token replay (the 2026-09-08 anchor protocol)

`dsv4_tp_ep_sampled_perf_gate --full-token-replay`, attention TP-2, expert-id EP, device
sampler, 256-token prime, 256 sampled tokens, 20 rows in blocks `AAAAA BBBBB BBBBB AAAAA`
(A eager, B graph), one load. Binary `8056c984...`. Receipt `rebaseline/raw/tpep-replay-main-r1/`.

| arm | rows | tok/s | historical (2026-09-08, `bd30a57`, Max-Q pair) |
|---|---|---|---|
| eager | 10 | 50.04 | 42.80 |
| graph | 10 | 50.68 | 44.01 |

Prime wall 5.046 s (historical 5.88 s). Generated sha256 `35e9e69e...`, final logits sha256
`37eb73d8...`, cache digests `5673480229060882075`: the same bits as the anchor and as the
braked-pair replay. Current main on a healthy pair is 16.9% above the anchor on eager and 15.1%
on graph; the TP/EP regression was the brake, not the code.

## Served cells

memra-server naked (binary `721d738f...`), streaming chat completions, 256 max tokens, 8 fixed
prompts (`raw/bench.py`), usage-authoritative token counts. Loaded program: PP-2, matrix expert
program, EP off, host sampler. DSpark arm: `MEMRA_DSV4_DRAFTER=dspark`, nothing else.

| arm | cell | ok | decode tok/s (1/TPOT p50) | TPOT p50/p95/p99 ms | ITL p50/p99 ms | TTFT p50/p95 ms | E2E p50 ms | agg tok/s | braked decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| plain | greedy c1 | 8/8 | 38.77 | 25.79 / 25.98 / 26.05 | 25.66 / 26.64 | 206 / 221 | 6,785 | 37.83 | 15.86 |
| plain | sampled c1 | 8/8 | 36.54 | 27.37 / 27.59 / 27.67 | 27.25 / 28.31 | 207 / 224 | 7,188 | 35.67 | 15.28 |
| plain | sampled c4 | 16/16 | 36.51 | 27.39 / 27.68 / 27.69 | 27.27 / 28.32 | 21,744 / 21,942 | 28,713 | 35.64 | 15.28 |
| plain | sampled c16 | 9/32 | 36.48 | 27.41 / 27.60 / 27.68 | 27.27 / 28.32 | 21,684 / 26,376 | 28,684 | 35.53 | 15.28 |
| DSpark | greedy c1 | 8/8 | 56.01 | 17.85 / 20.62 / 20.91 | | 267 / 287 | 4,832 | 52.76 | 22.00 |
| DSpark | sampled c1 | 8/8 | 47.47 | 21.07 / 22.70 / 22.73 | | 279 / 298 | 5,662 | 46.19 | 18.23 |
| DSpark | greedy c1 ignore-eos | 4/4 | 55.90 | 17.89 / 18.98 / 19.10 | | 268 / 289 | 4,843 | 52.39 | |

DSpark ITL is per streamed chunk (one spec round per chunk), so its p50 is near zero and its
p99 is the round wall; TPOT is the comparable figure.

Boot to `/readyz`: 122 s plain, 131 s DSpark. Decode power: 215..221 W median per card plain,
257..266 W DSpark, peak 287 W, against the 500 W limit. SM clock under load 2602..2842 MHz.

**Identity.** DSpark greedy text sha256 equals plain greedy on all 8 prompts (and on the 4
ignore-eos prompts), so the #660 fix holds on the healthy pair: the served spec number is a
servable speedup of the same output.

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| plain = DSpark | aea6e69e | 26ac8df7 | 850f75ed | a7784b9a | 7568f9b3 | 9dd1aedd | 4fd72390 | 937f04d8 |

**Acceptance** (`[dspark-acc]` lines in `raw/spec-dspark-r1/serve.log`, per cell): greedy c1
(serve.log lines 57..78, the 8 cell requests) 1437 of 2961 drafted over 603 rounds (0.485, 4.91
drafted per round, 3.38 committed per round); sampled c1 1403 of 3140 over 637 rounds (0.447).
The log also holds the warmup request (6 rounds) and the ignore-eos cell (304 rounds), which are
not in either figure. A greedy round costs 3.38 x 17.85 = 60.4 ms, 2.34 plain steps, so DSpark
buys 1.44x plain greedy and 1.30x plain sampled.

**Concurrency.** The route is serial (`capacity=serial`): c4 and c16 aggregate equal c1 and the
queue shows up as TTFT, and at c16 23 of 32 requests got 429 from the admission book. This part
of `BASELINE.md` is unchanged by the clock.

Receipts: `rebaseline/raw/base-plain-r1/`, `rebaseline/raw/spec-dspark-r1/` (cells.jsonl,
controller.log, serve.log, /metrics, /healthz and /readyz snapshots between cells, 250 ms
telemetry, binary sha256). Queue script: `rebaseline/raw/q-rebase.sh`.

## Against the roofline and the public anchor

Roofline from `BASELINE.md` (unchanged, it is arithmetic on the checkpoint): 11.44 GB/token,
PP-2 bound 6.38 ms/token (157 tok/s), TP-2 bound 3.19 ms/token (313 tok/s).

| program | ms/token | share of its topology bound |
|---|---|---|
| served PP-2 plain greedy | 25.79 | 24.7% of PP-2 |
| served PP-2 DSpark greedy | 17.85 | 35.7% of PP-2 (per committed token) |
| TP/EP replay, graph | 19.73 | 16.2% of TP-2 |

Public floor on the same card shape (vLLM TP=2, from `BASELINE.md`): 109.3 tok/s c1 plain
(9.15 ms/token), 193.2 with a k=3 drafter. memra served plain is 35% of that plain figure and
served DSpark is 29% of that drafter figure.
