# DSv4 B-row decode across requests (memra #667 lever 2)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (450 W), 2026-09-24. Lane
`lane/dsv4-brow-serve-20260924`, stacked on the request-pipelining lane (#699).

## Why

One plain step reads the whole 11.44 GB/token working set, 7.79 GB of it dense and attention
weights that every request reads identically (`BASELINE.md`). A step over B requests reads the
dense weights once for all of them. The routed experts are the union of each row's top-6.
Aggregate tokens per step should grow with B for about the dense share.

## What changed

Engine (`dsv4_gpu.rs`):

- `attention_verify_dev` is a one-group call of `attention_rows_dev`. That runs the attention
  sub-block over row groups: a group is consecutive positions of one request at workspace rows
  `row0..row0 + t`.
  - Everything that reads weights runs once over all rows: the HC entry, the q, kv and
    indexer projections (the indexer ones hoisted out of the per-request section), the
    output de-rotation and the output projections.
  - Each group's ring write, compressors, index lists, indexer scoring and sink attention
    read and write only its own cache, checkpoint and rows.
  - A verify round or prefill chunk is one group, so its launches are the same kernels in
    the same stream order, less the reordering of independent projections.
- `decode_rows_greedy`, `decode_rows_logits` and `decode_rows_full`: one decode step for B
  states. It uses the row groups, the row-batched MoE tail and the head. Each state commits its
  row through the same commit as its one-row step.
- `decode_rows_enqueue`, `_wait` and `_complete`: the same step split like #699's pipelined
  one-row step. Readbacks land pinned behind per-stage events, and each request commits
  through its own one-row workspace. Two groups can be in flight on two workspaces.

Server (`dsv4_serve.rs`):

- `MEMRA_DSV4_ROWS=B` (default off) coalesces the lanes' plain steps. A lane deposits its
  row, and the lane that completes the batch (or waits out 500 us) runs up to B rows.
- Greedy, penalized-greedy and host-sampled rows share a batch. Greedy rows always take the
  device argmax.
- With `2 * B <= MEMRA_DSV4_SESSIONS`, two groups can be in flight: queue under the launch
  turn, wait without it, commit.

## Correctness

`dsv4_rows_gate`: four sessions with prompts of 256, 197, 311 and 150 tokens decode 24 steps
alone, and each step's full logits row is hashed bit for bit. Fresh copies then decode as B-row
steps. The sessions join at steps 0, 3, 7 and 12 and leave after their 24 tokens, so the width
runs 1 to 4 and back, and the row order rotates every step:

```
PROTOCOL {"sessions":4,"prompt_tokens":[256, 197, 311, 150],"join_steps":[0, 3, 7, 12],"steps":24,"timing_steps":64,"greedy":true,"compare":"full logits bits per step","source_sha256":"11e4bd80352f4a24504ffdee519b33bbc527b3b9bcfeb531815e5731b190cb8c"}
WIDTHS [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1]
SESSION 0 len=256 join=0 steps=24 IDENTICAL head=[337, 18598, 2838, 2366, 75374, 2366]
SESSION 1 len=197 join=3 steps=24 IDENTICAL head=[1706, 337, 65, 2023, 2967, 2366]
SESSION 2 len=311 join=7 steps=24 IDENTICAL head=[106657, 1200, 12774, 3103, 6849, 2366]
SESSION 3 len=150 join=12 steps=24 IDENTICAL head=[595, 1387, 17555, 510, 3103, 1706]
PASS: every row bit-identical to its solo step across join, leave and row moves
```

Served: the text hash of every request is the same across every row of every arm in all five
cells, greedy and sampled (tables below).

## Serial B-row timing (gate, one-row step vs one B-row step, 64 steps, two reps)

```
TIME rep=0 B=1 one_row_ms_per_token=17.603 rows_ms_per_step=17.607 one_row_tok_s=56.81 rows_tok_s=56.80 speedup=1.000
TIME rep=0 B=2 one_row_ms_per_token=17.595 rows_ms_per_step=25.624 one_row_tok_s=56.84 rows_tok_s=78.05 speedup=1.373
TIME rep=0 B=4 one_row_ms_per_token=17.596 rows_ms_per_step=38.537 one_row_tok_s=56.83 rows_tok_s=103.80 speedup=1.826
TIME rep=1 B=1 one_row_ms_per_token=17.614 rows_ms_per_step=17.606 one_row_tok_s=56.77 rows_tok_s=56.80 speedup=1.000
TIME rep=1 B=2 one_row_ms_per_token=17.598 rows_ms_per_step=25.635 one_row_tok_s=56.83 rows_tok_s=78.02 speedup=1.373
TIME rep=1 B=4 one_row_ms_per_token=17.603 rows_ms_per_step=38.588 one_row_tok_s=56.81 rows_tok_s=103.66 speedup=1.825
```

Each added row costs about 7 ms on a 17.6 ms step. The routed experts account for part of that:
each row adds six experts per layer, about 3.7 GB. Each row's per-request launches account for
the rest: its compressors (whose projections this lane still runs per group), index lists,
indexer and sink attention.

## Served A/B (serial B-row)

One binary, one boot per row, order `A B C C A B B C A`, N=3 per arm. A = defaults (two
pipelined lanes, #699), B = `MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=4`, C =
`MEMRA_DSV4_SESSIONS=2 MEMRA_DSV4_ROWS=2`. Cells `raw/serial/served/*/cells.jsonl`, script
`raw/q-pairR.sh`.

#### greedy-c1

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | A | 54.49 | 56.91 | 17.57 / 17.59 / 17.59 | 17.31 / 18.50 | 210 / 246 | 4,691 / 4,727 | 8/8 |
| r2 | B | 54.50 | 56.87 | 17.59 / 17.59 / 17.60 | 17.32 / 18.49 | 210 / 237 | 4,695 / 4,722 | 8/8 |
| r3 | C | 54.57 | 56.91 | 17.57 / 17.60 / 17.60 | 17.31 / 18.50 | 207 / 229 | 4,688 / 4,710 | 8/8 |
| r4 | C | 54.59 | 56.94 | 17.56 / 17.60 / 17.60 | 17.31 / 18.48 | 207 / 229 | 4,686 / 4,707 | 8/8 |
| r5 | A | 54.52 | 56.90 | 17.58 / 17.59 / 17.59 | 17.32 / 18.50 | 214 / 232 | 4,695 / 4,713 | 8/8 |
| r6 | B | 54.53 | 56.89 | 17.58 / 17.59 / 17.60 | 17.32 / 18.50 | 211 / 232 | 4,694 / 4,717 | 8/8 |
| r7 | B | 54.54 | 56.89 | 17.58 / 17.59 / 17.59 | 17.31 / 18.49 | 210 / 233 | 4,693 / 4,715 | 8/8 |
| r8 | C | 54.58 | 56.93 | 17.56 / 17.60 / 17.60 | 17.31 / 18.50 | 207 / 230 | 4,686 / 4,707 | 8/8 |
| r9 | A | 54.52 | 56.89 | 17.58 / 17.60 / 17.60 | 17.32 / 18.50 | 210 / 232 | 4,692 / 4,714 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| A | 3 | 54.52 | 56.90 | 17.58 | 210 |
| B | 3 | 54.53 | 56.89 | 17.58 | 210 |
| C | 3 | 54.58 | 56.93 | 17.56 | 207 |

Aggregate B vs A: +0.0%; per-request decode -0.0%.

Aggregate C vs A: +0.1%; per-request decode +0.1%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | A | 85.05 | 45.03 | 22.21 / 22.65 / 22.66 | 21.13 / 24.27 | 350 / 470 | 6,024 / 6,043 | 8/8 |
| r2 | B | 72.17 | 37.70 | 26.53 / 26.95 / 26.96 | 25.79 / 28.38 | 339 / 444 | 7,085 / 7,144 | 8/8 |
| r3 | C | 72.18 | 37.70 | 26.52 / 26.95 / 26.96 | 25.79 / 28.37 | 336 / 439 | 7,086 / 7,136 | 8/8 |
| r4 | C | 72.19 | 37.72 | 26.51 / 26.96 / 26.97 | 25.79 / 28.36 | 327 / 439 | 7,085 / 7,136 | 8/8 |
| r5 | A | 85.16 | 45.04 | 22.20 / 22.65 / 22.65 | 21.13 / 24.26 | 350 / 459 | 6,009 / 6,035 | 8/8 |
| r6 | B | 72.19 | 37.71 | 26.52 / 26.96 / 26.98 | 25.79 / 28.38 | 325 / 441 | 7,088 / 7,134 | 8/8 |
| r7 | B | 72.18 | 37.71 | 26.52 / 26.94 / 26.96 | 25.78 / 28.35 | 339 / 444 | 7,084 / 7,139 | 8/8 |
| r8 | C | 72.19 | 37.72 | 26.51 / 26.96 / 26.97 | 25.79 / 28.41 | 328 / 440 | 7,084 / 7,136 | 8/8 |
| r9 | A | 85.04 | 44.96 | 22.24 / 22.65 / 22.65 | 21.14 / 24.28 | 351 / 475 | 6,011 / 6,066 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| A | 3 | 85.05 | 45.03 | 22.21 | 350 |
| B | 3 | 72.18 | 37.71 | 26.52 | 339 |
| C | 3 | 72.19 | 37.72 | 26.51 | 328 |

Aggregate B vs A: -15.1%; per-request decode -16.3%.

Aggregate C vs A: -15.1%; per-request decode -16.2%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | A | 84.93 | 45.02 | 22.21 / 22.65 / 22.66 | 21.14 / 24.24 | 6,287 / 6,501 | 12,016 / 12,100 | 8/8 |
| r2 | B | 91.29 | 23.99 | 41.68 / 42.80 / 42.83 | 40.56 / 43.83 | 585 / 879 | 11,217 / 11,279 | 8/8 |
| r3 | C | 72.14 | 37.73 | 26.51 / 26.97 / 27.01 | 25.83 / 28.04 | 7,355 / 7,555 | 14,142 / 14,272 | 8/8 |
| r4 | C | 72.11 | 37.73 | 26.50 / 26.98 / 27.01 | 25.83 / 28.08 | 7,369 / 7,569 | 14,128 / 14,285 | 8/8 |
| r5 | A | 84.97 | 45.01 | 22.22 / 22.65 / 22.65 | 22.29 / 23.61 | 6,283 / 6,495 | 11,990 / 12,102 | 8/8 |
| r6 | B | 91.43 | 23.99 | 41.69 / 42.76 / 42.80 | 40.58 / 43.93 | 572 / 863 | 11,199 / 11,248 | 8/8 |
| r7 | B | 91.32 | 24.00 | 41.67 / 42.79 / 42.81 | 40.53 / 43.83 | 579 / 881 | 11,212 / 11,271 | 8/8 |
| r8 | C | 72.06 | 37.70 | 26.53 / 26.95 / 26.96 | 25.79 / 28.47 | 7,367 / 7,572 | 14,149 / 14,285 | 8/8 |
| r9 | A | 84.90 | 45.00 | 22.22 / 22.65 / 22.66 | 22.30 / 23.61 | 6,298 / 6,510 | 12,036 / 12,100 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| A | 3 | 84.93 | 45.01 | 22.22 | 6,287 |
| B | 3 | 91.32 | 23.99 | 41.68 | 579 |
| C | 3 | 72.11 | 37.73 | 26.51 | 7,367 |

Aggregate B vs A: +7.5%; per-request decode -46.7%.

Aggregate C vs A: -15.1%; per-request decode -16.2%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### sampled-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | A | 72.51 | 37.98 | 26.33 / 26.76 / 26.77 | 25.32 / 28.48 | 340 / 462 | 7,053 / 7,091 | 8/8 |
| r2 | B | 48.38 | 24.88 | 40.19 / 40.59 / 40.59 | 39.39 / 41.84 | 330 / 443 | 10,575 / 10,611 | 8/8 |
| r3 | C | 48.26 | 24.81 | 40.31 / 40.80 / 40.81 | 39.52 / 41.82 | 338 / 444 | 10,600 / 10,682 | 8/8 |
| r4 | C | 48.34 | 24.85 | 40.25 / 40.63 / 40.63 | 39.45 / 41.89 | 337 / 442 | 10,589 / 10,612 | 8/8 |
| r5 | A | 72.29 | 37.90 | 26.38 / 26.78 / 26.78 | 25.37 / 28.54 | 361 / 466 | 7,080 / 7,093 | 8/8 |
| r6 | B | 48.35 | 24.85 | 40.25 / 40.69 / 40.74 | 39.40 / 41.87 | 330 / 446 | 10,572 / 10,652 | 8/8 |
| r7 | B | 48.37 | 24.86 | 40.22 / 40.62 / 40.63 | 39.42 / 41.86 | 330 / 442 | 10,579 / 10,614 | 8/8 |
| r8 | C | 48.47 | 24.91 | 40.14 / 40.49 / 40.50 | 39.34 / 41.71 | 337 / 442 | 10,554 / 10,592 | 8/8 |
| r9 | A | 72.46 | 37.96 | 26.35 / 26.78 / 26.80 | 25.34 / 28.52 | 349 / 463 | 7,061 / 7,097 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| A | 3 | 72.46 | 37.96 | 26.35 | 349 |
| B | 3 | 48.37 | 24.86 | 40.22 | 330 |
| C | 3 | 48.34 | 24.85 | 40.25 | 337 |

Aggregate B vs A: -33.2%; per-request decode -34.5%.

Aggregate C vs A: -33.3%; per-request decode -34.5%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

### sampled-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | A | 72.35 | 37.95 | 26.35 / 26.78 / 26.80 | 25.32 / 28.49 | 7,332 / 7,549 | 14,108 / 14,203 | 8/8 |
| r2 | B | 48.27 | 12.36 | 80.88 / 81.96 / 81.99 | 79.79 / 83.14 | 593 / 887 | 21,211 / 21,274 | 8/8 |
| r3 | C | 48.21 | 24.82 | 40.29 / 40.79 / 40.79 | 39.51 / 41.96 | 10,890 / 11,091 | 21,170 / 21,336 | 8/8 |
| r4 | C | 48.42 | 24.93 | 40.12 / 40.57 / 40.58 | 39.91 / 41.53 | 10,828 / 11,029 | 21,129 / 21,193 | 8/8 |
| r5 | A | 72.19 | 37.86 | 26.41 / 26.79 / 26.79 | 25.35 / 28.63 | 7,357 / 7,570 | 14,146 / 14,207 | 8/8 |
| r6 | B | 48.25 | 12.36 | 80.91 / 82.14 / 82.34 | 79.91 / 83.47 | 581 / 877 | 21,217 / 21,347 | 8/8 |
| r7 | B | 48.27 | 12.37 | 80.86 / 81.97 / 81.98 | 79.81 / 83.19 | 584 / 885 | 21,208 / 21,275 | 8/8 |
| r8 | C | 48.38 | 24.90 | 40.16 / 40.58 / 40.59 | 39.36 / 41.79 | 10,835 / 11,038 | 21,119 / 21,203 | 8/8 |
| r9 | A | 72.27 | 37.92 | 26.37 / 26.84 / 26.86 | 26.42 / 27.77 | 7,355 / 7,570 | 14,108 / 14,221 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| A | 3 | 72.27 | 37.92 | 26.37 | 7,355 |
| B | 3 | 48.27 | 12.36 | 80.88 | 584 |
| C | 3 | 48.38 | 24.90 | 40.16 | 10,835 |

Aggregate B vs A: -33.2%; per-request decode -67.4%.

Aggregate C vs A: -33.1%; per-request decode -34.3%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

Thermal A: power median 290..292 W, peak 474 W, SM clock 2610..2865 MHz, max temp 58 C, 3329 samples.

Thermal B: power median 228..229 W, peak 474 W, SM clock 2610..2850 MHz, max temp 56 C, 4052 samples.

Thermal C: power median 229..229 W, peak 424 W, SM clock 2610..2857 MHz, max temp 56 C, 4196 samples.

## Reading

**B-row steps are bit-identical to one-row steps, but a serial B-row step does not beat two
pipelined lanes.**

- At greedy c4, four rows in one step serve 91.3 tok/s against 84.9 (+7.5%).
- At c2 a two-row step (72.2) loses to two pipelined one-row steps (85.1). The serial step
  runs its stages one after the other, so each card idles half the step. Pipelining keeps both
  cards busy.
- Sampled traffic loses a third (72.5 -> 48.4 at c2). Every lane samples on the host after the
  batch returns, and the next batch waits for all of them. Pipelining hides that sampling
  behind the other session's GPU step.

Both levers compose: two groups of B rows in flight, one group's stage 0 on card 0 while the
other's stage 1 runs on card 1, and one group sampling while the other computes. That is the
pipelined arm below. `MEMRA_DSV4_ROWS` stays off by default until it measures.

## Pipelined groups

Pending (`raw/q-pairP.sh`): the gate's pipelined identity arm and a two-groups-of-two timing row,
then served A (defaults) against D (`MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=2`).
