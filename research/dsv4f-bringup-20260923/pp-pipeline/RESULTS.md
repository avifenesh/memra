# DSv4 PP-2 request pipelining (memra #667, lever 1)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition (600 W limit, SM clock max 2430 MHz, EPYC 9555 host), 2026-09-23/24. Lane `lane/dsv4-pp-pipeline-20260923`.

## Why

At one request in flight, PP-2 runs its two cards one after the other: card 0 is idle while card 1
runs stage 1, and the reverse. The served plain step is GPU bound (the two cards' kernel sums add up
to the step, `../moe-defer-670/RESULTS.md`), so the serial route leaves about half of each card
unused, and the route serves one request at a time (#667: c16 aggregate equals c1).

With two sessions in flight and their steps skewed by one stage, card 0 runs one session's stage 0
while card 1 runs the other's stage 1. The bound is aggregate about 1/max(stage) instead of
1/sum(stages), close to 2x at c >= 2, with no kernel change.

## What changed

Engine (`dsv4_gpu.rs`):

- `decode_step_greedy_enqueue` queues the whole PP-2 plain step (every stage, the device argmax,
  the MoE fault words) with no host wait. The readbacks land in portable pinned memory behind one
  event per stage (`StepLanding`, `VerifyOutput::ArgmaxDeferred`).
- `decode_step_greedy_wait` blocks on this step's own events only, never on a whole stream, so it
  does not wait for another session's work queued behind it. It queues nothing.
- `decode_step_greedy_complete` applies the MoE fault words exactly as `take_moe_faults` does
  (a nonzero word fails the step before anything commits and clears the words), commits the row
  and returns the token.
- `decode_step_logits_enqueue` / `_complete` are the same with the full logits row
  (`VerifyOutput::FullDeferred`), for the host sampler and penalized greedy.
- `prefill_with_cache_chunked_yielding` / `continue_prefix_chunked_yielding` call a hook after
  every committed chunk but the last.

The kernels, their order and each request's buffers are unchanged, so each request's bits are
the serial program's. What changes is only which host waits exist.

Server (`dsv4_serve.rs`):

- `MEMRA_DSV4_SESSIONS=N` (1..=4, default 1) spawns N serving lanes on one queue and one launch
  turn (`Turn`, a mutex that also owns the parked-prefix cache). Every engine call runs holding
  the turn, so one session's step is queued whole on the stage streams before another's, and the
  per-stage shared scratch is used in stream order.
- A plain step gives the turn up only while it waits for its own readbacks; a chunked plain
  prefill gives it up between chunks. Restored-prefix continuation, the device sampler and DSpark
  hold it for their whole run in this slice.
- Route health counts requests in flight (BUSY until the last one ends), and the route contract
  declares `Sessions(N)`. Memory admission runs under the turn, so a second session's charge sees
  the first session's allocations.
- One lane keeps the serial route's one-call step.

## Correctness

- `dsv4_pipeline_gate`: two sessions (256-token prompts, 256 greedy tokens each), serial then
  pipelined, every token of both sessions equal to its serial stream on every repeat.
- `dsv4_pipeline_gate` on the pair, lane `4049a5833`: every token of both sessions equal to its
  serial stream on all 5 repeats, `PASS`; serial 45.80 tok/s, pipelined 69.26 tok/s aggregate on
  the first repeat (`raw/server-pair/gate/gate.log`).
- Served: `MEMRA_DSV4_SESSIONS` 1, 2 and 4 on one binary. Every request's text hash is the same
  in every row of every arm, in all five cells (greedy c1, c2, c4; sampled c2, c4).

## Results

One binary (lane `4049a5833`), one boot per row, `MEMRA_DSV4_SESSIONS` 2 vs 1 in the order
`2 1 1 2 2 1 1 2 2 1` (N=5 per arm), then two rows at 4. Cells (`raw/server-pair/cells-pipe.txt`):
8 requests each, 256 output tokens, at 1, 2 and 4 concurrent clients, greedy with text kept and
sampled. Raw rows under `raw/server-pair/served/`; `pipe_tab.py raw/server-pair/served` prints
the tables below.

### greedy-c1

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | s2 | 44.62 | 46.34 | 21.58 / 21.62 / 21.63 | 21.36 / 22.60 | 234 / 253 | 5,738 / 5,762 | 8/8 |
| r2 | s1 | 44.50 | 46.22 | 21.64 / 21.68 / 21.70 | 21.41 / 22.67 | 235 / 249 | 5,752 / 5,775 | 8/8 |
| r3 | s1 | 44.53 | 46.24 | 21.63 / 21.67 / 21.69 | 21.40 / 22.67 | 235 / 250 | 5,750 / 5,772 | 8/8 |
| r4 | s2 | 44.52 | 46.24 | 21.63 / 21.67 / 21.68 | 21.40 / 22.65 | 235 / 253 | 5,750 / 5,773 | 8/8 |
| r5 | s2 | 44.53 | 46.25 | 21.62 / 21.67 / 21.69 | 21.40 / 22.64 | 235 / 253 | 5,749 / 5,773 | 8/8 |
| r6 | s1 | 44.52 | 46.24 | 21.63 / 21.67 / 21.69 | 21.40 / 22.65 | 235 / 250 | 5,751 / 5,772 | 8/8 |
| r7 | s1 | 44.54 | 46.25 | 21.62 / 21.67 / 21.68 | 21.40 / 22.66 | 233 / 249 | 5,747 / 5,769 | 8/8 |
| r8 | s2 | 44.55 | 46.27 | 21.61 / 21.65 / 21.66 | 21.39 / 22.62 | 235 / 253 | 5,747 / 5,769 | 8/8 |
| r9 | s2 | 44.58 | 46.31 | 21.59 / 21.63 / 21.65 | 21.37 / 22.62 | 235 / 253 | 5,742 / 5,764 | 8/8 |
| r10 | s1 | 44.52 | 46.24 | 21.63 / 21.67 / 21.69 | 21.41 / 22.66 | 235 / 249 | 5,750 / 5,772 | 8/8 |
| r11 | s4 | 44.60 | 46.33 | 21.58 / 21.63 / 21.64 | 21.36 / 22.61 | 235 / 253 | 5,740 / 5,763 | 8/8 |
| r12 | s4 | 44.51 | 46.25 | 21.62 / 21.67 / 21.69 | 21.40 / 22.65 | 236 / 255 | 5,751 / 5,776 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| s1 | 5 | 44.52 | 46.24 | 21.63 | 235 |
| s2 | 5 | 44.55 | 46.27 | 21.61 | 235 |
| s4 | 2 | 44.56 | 46.29 | 21.60 | 236 |

Aggregate s2 vs s1: +0.1%; per-request decode +0.1%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | s2 | 67.61 | 35.31 | 28.32 / 28.90 / 28.99 | 27.26 / 30.52 | 353 / 494 | 7,560 / 7,641 | 8/8 |
| r2 | s1 | 44.50 | 46.20 | 21.64 / 21.68 / 21.70 | 21.42 / 22.65 | 5,987 / 6,005 | 11,507 / 11,529 | 8/8 |
| r3 | s1 | 44.51 | 46.23 | 21.63 / 21.67 / 21.68 | 21.41 / 22.65 | 5,982 / 6,010 | 11,498 / 11,521 | 8/8 |
| r4 | s2 | 65.91 | 34.46 | 29.02 / 29.55 / 29.56 | 27.95 / 31.17 | 366 / 500 | 7,775 / 7,794 | 8/8 |
| r5 | s2 | 66.38 | 34.74 | 28.79 / 29.51 / 29.52 | 27.86 / 31.06 | 364 / 499 | 7,724 / 7,793 | 8/8 |
| r6 | s1 | 44.53 | 46.23 | 21.63 / 21.67 / 21.68 | 21.41 / 22.64 | 5,981 / 6,004 | 11,497 / 11,521 | 8/8 |
| r7 | s1 | 44.51 | 46.21 | 21.64 / 21.68 / 21.70 | 21.42 / 22.66 | 5,985 / 6,003 | 11,504 / 11,528 | 8/8 |
| r8 | s2 | 65.92 | 34.52 | 28.97 / 29.57 / 29.57 | 27.93 / 31.16 | 359 / 503 | 7,778 / 7,791 | 8/8 |
| r9 | s2 | 66.02 | 34.59 | 28.91 / 29.53 / 29.54 | 27.91 / 31.15 | 373 / 509 | 7,770 / 7,797 | 8/8 |
| r10 | s1 | 44.50 | 46.23 | 21.63 / 21.70 / 21.70 | 21.41 / 22.66 | 5,983 / 6,010 | 11,501 / 11,535 | 8/8 |
| r11 | s4 | 66.96 | 34.43 | 29.05 / 29.14 / 29.14 | 27.46 / 30.65 | 236 / 417 | 7,648 / 7,846 | 8/8 |
| r12 | s4 | 66.11 | 33.98 | 29.43 / 29.57 / 29.57 | 27.89 / 31.16 | 238 / 413 | 7,749 / 7,947 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| s1 | 5 | 44.51 | 46.23 | 21.63 | 5,983 |
| s2 | 5 | 66.02 | 34.59 | 28.91 | 364 |
| s4 | 2 | 66.54 | 34.20 | 29.24 | 237 |

Aggregate s2 vs s1: +48.3%; per-request decode -25.2%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | s2 | 67.54 | 34.93 | 28.63 / 28.84 / 28.87 | 27.25 / 30.24 | 7,786 / 8,068 | 15,100 / 15,374 | 8/8 |
| r2 | s1 | 44.48 | 46.21 | 21.64 / 21.68 / 21.70 | 21.42 / 22.66 | 17,461 / 17,532 | 22,975 / 23,051 | 8/8 |
| r3 | s1 | 44.50 | 46.23 | 21.63 / 21.67 / 21.68 | 21.41 / 22.66 | 17,478 / 17,518 | 22,990 / 23,043 | 8/8 |
| r4 | s2 | 66.27 | 34.70 | 28.82 / 29.50 / 29.54 | 27.89 / 31.14 | 7,990 / 8,259 | 15,386 / 15,478 | 8/8 |
| r5 | s2 | 67.71 | 35.47 | 28.19 / 28.72 / 28.73 | 27.07 / 30.22 | 7,810 / 8,077 | 15,083 / 15,175 | 8/8 |
| r6 | s1 | 44.50 | 46.23 | 21.63 / 21.67 / 21.69 | 21.41 / 22.65 | 17,435 / 17,513 | 22,951 / 23,023 | 8/8 |
| r7 | s1 | 44.48 | 46.21 | 21.64 / 21.69 / 21.70 | 21.42 / 22.66 | 17,462 / 17,522 | 22,976 / 23,041 | 8/8 |
| r8 | s2 | 66.39 | 34.70 | 28.82 / 29.45 / 29.53 | 27.93 / 31.12 | 7,988 / 8,246 | 15,366 / 15,522 | 8/8 |
| r9 | s2 | 66.80 | 34.95 | 28.61 / 29.17 / 29.18 | 27.54 / 30.72 | 7,915 / 8,181 | 15,281 / 15,381 | 8/8 |
| r10 | s1 | 44.48 | 46.23 | 21.63 / 21.67 / 21.68 | 21.40 / 22.65 | 17,476 / 17,534 | 22,987 / 23,059 | 8/8 |
| r11 | s4 | 66.80 | 17.10 | 58.48 / 60.39 / 60.95 | 56.38 / 60.11 | 251 / 938 | 15,276 / 16,002 | 8/8 |
| r12 | s4 | 66.32 | 17.12 | 58.42 / 61.20 / 61.75 | 56.60 / 60.66 | 270 / 967 | 15,389 / 16,252 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| s1 | 5 | 44.48 | 46.23 | 21.63 | 17,462 |
| s2 | 5 | 66.80 | 34.93 | 28.63 | 7,915 |
| s4 | 2 | 66.56 | 17.11 | 58.45 | 261 |

Aggregate s2 vs s1: +50.2%; per-request decode -24.4%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### sampled-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | s2 | 60.99 | 31.78 | 31.47 / 32.10 / 32.12 | 30.56 / 33.80 | 358 / 498 | 8,401 / 8,431 | 8/8 |
| r2 | s1 | 41.93 | 43.47 | 23.00 / 23.07 / 23.08 | 22.79 / 24.06 | 6,334 / 6,369 | 12,199 / 12,233 | 8/8 |
| r3 | s1 | 41.91 | 43.46 | 23.01 / 23.08 / 23.09 | 22.80 / 24.09 | 6,337 / 6,367 | 12,209 / 12,238 | 8/8 |
| r4 | s2 | 60.56 | 31.53 | 31.72 / 32.29 / 32.30 | 30.68 / 33.88 | 358 / 498 | 8,452 / 8,482 | 8/8 |
| r5 | s2 | 62.16 | 32.39 | 30.87 / 31.41 / 31.44 | 29.77 / 33.04 | 370 / 503 | 8,237 / 8,260 | 8/8 |
| r6 | s1 | 41.90 | 43.42 | 23.03 / 23.09 / 23.10 | 22.82 / 24.11 | 6,341 / 6,367 | 12,213 / 12,243 | 8/8 |
| r7 | s1 | 41.97 | 43.50 | 22.99 / 23.06 / 23.07 | 22.78 / 24.06 | 6,332 / 6,357 | 12,196 / 12,225 | 8/8 |
| r8 | s2 | 60.74 | 31.59 | 31.66 / 32.18 / 32.22 | 30.62 / 33.86 | 366 / 498 | 8,433 / 8,448 | 8/8 |
| r9 | s2 | 60.74 | 31.69 | 31.56 / 32.24 / 32.27 | 30.64 / 33.85 | 365 / 498 | 8,444 / 8,475 | 8/8 |
| r10 | s1 | 41.89 | 43.42 | 23.03 / 23.10 / 23.11 | 22.82 / 24.12 | 6,343 / 6,368 | 12,218 / 12,247 | 8/8 |
| r11 | s4 | 60.73 | 31.18 | 32.07 / 32.19 / 32.19 | 30.62 / 33.88 | 237 / 414 | 8,413 / 8,621 | 8/8 |
| r12 | s4 | 60.77 | 31.23 | 32.02 / 32.26 / 32.27 | 30.61 / 33.77 | 237 / 413 | 8,403 / 8,639 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| s1 | 5 | 41.91 | 43.46 | 23.01 | 6,337 |
| s2 | 5 | 60.74 | 31.69 | 31.56 | 365 |
| s4 | 2 | 60.75 | 31.20 | 32.05 | 237 |

Aggregate s2 vs s1: +44.9%; per-request decode -27.1%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

### sampled-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | s2 | 60.52 | 31.52 | 31.72 / 32.26 / 32.28 | 30.67 / 33.80 | 8,704 / 8,973 | 16,884 / 16,945 | 8/8 |
| r2 | s1 | 41.93 | 43.47 | 23.00 / 23.07 / 23.09 | 22.79 / 24.08 | 18,530 / 18,574 | 24,395 / 24,454 | 8/8 |
| r3 | s1 | 41.83 | 43.36 | 23.06 / 23.13 / 23.14 | 22.85 / 24.18 | 18,564 / 18,608 | 24,458 / 24,484 | 8/8 |
| r4 | s2 | 60.57 | 31.21 | 32.04 / 32.29 / 32.30 | 30.69 / 33.83 | 8,721 / 8,984 | 16,870 / 17,107 | 8/8 |
| r5 | s2 | 61.55 | 32.10 | 31.15 / 31.67 / 31.67 | 30.09 / 33.32 | 8,572 / 8,821 | 16,609 / 16,646 | 8/8 |
| r6 | s1 | 41.86 | 43.41 | 23.03 / 23.10 / 23.11 | 22.83 / 24.12 | 18,540 / 18,623 | 24,414 / 24,496 | 8/8 |
| r7 | s1 | 41.94 | 43.50 | 22.99 / 23.06 / 23.07 | 22.78 / 24.06 | 18,504 / 18,585 | 24,367 / 24,447 | 8/8 |
| r8 | s2 | 60.80 | 31.69 | 31.55 / 32.28 / 32.30 | 30.65 / 33.88 | 8,718 / 8,976 | 16,803 / 16,892 | 8/8 |
| r9 | s2 | 60.62 | 31.58 | 31.67 / 32.24 / 32.29 | 30.64 / 33.82 | 8,716 / 8,975 | 16,857 / 16,957 | 8/8 |
| r10 | s1 | 41.85 | 43.42 | 23.03 / 23.10 / 23.11 | 22.82 / 24.15 | 18,563 / 18,622 | 24,435 / 24,509 | 8/8 |
| r11 | s4 | 60.80 | 15.64 | 63.94 / 66.73 / 67.29 | 62.49 / 66.49 | 256 / 949 | 16,786 / 17,641 | 8/8 |
| r12 | s4 | 60.80 | 15.59 | 64.14 / 66.43 / 66.99 | 62.18 / 65.18 | 275 / 974 | 16,784 / 17,594 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| s1 | 5 | 41.86 | 43.42 | 23.03 | 18,540 |
| s2 | 5 | 60.62 | 31.58 | 31.67 | 8,716 |
| s4 | 2 | 60.80 | 15.62 | 64.04 | 265 |

Aggregate s2 vs s1: +44.8%; per-request decode -27.3%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

Thermal s1: power median 178..178 W, peak 201 W, SM clock 2280..2385 MHz, max temp 46 C, 9542 samples.

Thermal s2: power median 205..214 W, peak 239 W, SM clock 2265..2430 MHz, max temp 50 C, 7091 samples.

Thermal s4: power median 207..210 W, peak 233 W, SM clock 2265..2407 MHz, max temp 48 C, 2835 samples.

## Verdict

**Two sessions serve 48% more greedy tokens per second at c2 and 50% more at c4, token for token
the serial text.** Aggregate greedy 44.51 -> 66.02 tok/s at c2 and 44.48 -> 66.80 at c4, sampled
+44.9% and +44.8% (N=5 per arm). c1 is flat (44.52 vs 44.55), so the second lane costs a lone
request nothing. At c2 the second request no longer waits for the first: TTFT p50 5,983 -> 364
ms. Each request decodes 25% slower while two share the cards (TPOT p50 21.63 -> 28.91 ms at
greedy c2), which is the trade the aggregate buys.

Four lanes add nothing to the aggregate (66.56 greedy c4, N=2). A two-stage pipeline overlaps two
steps at most, one per card, so a third and fourth session only add turns to wait for. What four
lanes change is who waits. At c4 every request starts at
once (TTFT p50 261 ms against 7,915) and each decodes at half the two-lane rate (TPOT 58.45 ms).
That is an operator choice, not a throughput one.

The bound for two stages is 2x, and the gate reaches 1.51x. Two things can hold it below 2x: an
uneven split of the step's work between the cards, and the host time to queue a whole step under
the launch turn, which is thousands of launches per plain step. This lane did not profile which
one binds. A captured step per stage (CUDA graph) would shrink the host's share of the turn, so
it is the next lever to test.

**Default.** Two lanes are the default on the plain PP matrix device program over two or more
stages (`Dsv4Gpu::pipelined_steps_supported`, no drafter armed); every other load keeps one.
DSpark holds the launch turn for a whole request in this slice, so a second lane there would
wait, and it was not measured. `MEMRA_DSV4_SESSIONS=1` is the rollback seam (decide-by
2026-10-08); `2..=4` stays settable.

## Default-selection receipt

Binary `beae5264a` (the default at two lanes, the FIFO turn, the door sleeping without it), one
boot per row: unset twice, `MEMRA_DSV4_SESSIONS=1` twice, then unset with DSpark armed. Each
boot's log names its lane count and source:

- unset, plain: `2 serving lane(s) (default on the plain PP matrix program; MEMRA_DSV4_SESSIONS=1 is the serial route)`
- `=1`: `1 serving lane(s) (MEMRA_DSV4_SESSIONS)`
- unset, DSpark armed: `1 serving lane(s) (default)`

`cells-pipe2.txt` adds c2 at about 2k-token prompts, so a prefill of four chunks yields the FIFO
turn between chunks while the other lane decodes (`raw/server-pair/default/`):

### greedy-c1

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 44.53 | 46.26 | 21.62 / 21.67 / 21.68 | 21.40 / 22.64 | 236 / 253 | 5,749 / 5,773 | 8/8 |
| r2 | one | 44.50 | 46.22 | 21.64 / 21.69 / 21.70 | 21.42 / 22.71 | 235 / 250 | 5,754 / 5,775 | 8/8 |
| r3 | unset | 44.59 | 46.32 | 21.59 / 21.63 / 21.64 | 21.37 / 22.62 | 235 / 253 | 5,740 / 5,764 | 8/8 |
| r4 | one | 44.49 | 46.20 | 21.64 / 21.68 / 21.70 | 21.42 / 22.66 | 235 / 250 | 5,754 / 5,776 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 44.50 | 46.21 | 21.64 | 235 |
| unset | 2 | 44.56 | 46.29 | 21.60 | 235 |

Aggregate unset vs one: +0.1%; per-request decode +0.2%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 65.82 | 34.52 | 28.97 / 29.51 / 29.53 | 27.95 / 31.19 | 379 / 506 | 7,786 / 7,806 | 8/8 |
| r2 | one | 44.51 | 46.21 | 21.64 / 21.69 / 21.70 | 21.41 / 22.66 | 5,984 / 6,006 | 11,503 / 11,529 | 8/8 |
| r3 | unset | 66.75 | 34.94 | 28.62 / 29.13 / 29.17 | 27.57 / 30.67 | 373 / 497 | 7,662 / 7,728 | 8/8 |
| r4 | one | 44.50 | 46.20 | 21.65 / 21.69 / 21.70 | 21.42 / 22.65 | 5,988 / 6,006 | 11,508 / 11,530 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 44.50 | 46.20 | 21.64 | 5,986 |
| unset | 2 | 66.28 | 34.73 | 28.79 | 376 |

Aggregate unset vs one: +48.9%; per-request decode -24.8%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### greedy-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 66.35 | 34.75 | 28.78 / 29.33 / 29.38 | 27.78 / 30.94 | 8,017 / 8,242 | 15,416 / 15,506 | 8/8 |
| r2 | one | 44.46 | 46.21 | 21.64 / 21.68 / 21.70 | 21.42 / 22.65 | 17,478 / 17,539 | 22,991 / 23,068 | 8/8 |
| r3 | unset | 66.99 | 35.14 | 28.45 / 28.91 / 28.91 | 27.40 / 30.61 | 7,924 / 8,166 | 15,233 / 15,335 | 8/8 |
| r4 | one | 44.46 | 46.20 | 21.65 / 21.68 / 21.70 | 21.42 / 22.67 | 17,483 / 17,536 | 22,997 / 23,066 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 44.46 | 46.21 | 21.64 | 17,480 |
| unset | 2 | 66.67 | 34.95 | 28.62 | 7,971 |

Aggregate unset vs one: +50.0%; per-request decode -24.4%.

Hashes: identical across every row of both arms; 8 requests, first aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8.

### sampled-c2

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 60.51 | 31.55 | 31.69 / 32.18 / 32.19 | 30.68 / 33.89 | 374 / 497 | 8,459 / 8,474 | 8/8 |
| r2 | one | 41.85 | 43.39 | 23.05 / 23.11 / 23.13 | 22.84 / 24.12 | 6,346 / 6,378 | 12,223 / 12,257 | 8/8 |
| r3 | unset | 60.72 | 31.63 | 31.62 / 32.12 / 32.18 | 30.60 / 33.85 | 375 / 498 | 8,421 / 8,472 | 8/8 |
| r4 | one | 41.89 | 43.41 | 23.04 / 23.10 / 23.11 | 22.83 / 24.11 | 6,343 / 6,369 | 12,216 / 12,249 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 41.87 | 43.40 | 23.04 | 6,344 |
| unset | 2 | 60.61 | 31.59 | 31.65 | 374 |

Aggregate unset vs one: +44.8%; per-request decode -27.2%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

### sampled-c4

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 60.56 | 31.65 | 31.59 / 32.15 / 32.17 | 30.70 / 33.79 | 8,744 / 8,984 | 16,867 / 16,983 | 8/8 |
| r2 | one | 41.84 | 43.39 | 23.05 / 23.11 / 23.13 | 22.84 / 24.10 | 18,541 / 18,625 | 24,417 / 24,497 | 8/8 |
| r3 | unset | 61.05 | 31.82 | 31.42 / 31.98 / 32.03 | 30.70 / 33.44 | 8,690 / 8,938 | 16,683 / 16,876 | 8/8 |
| r4 | one | 41.85 | 43.40 | 23.04 / 23.10 / 23.12 | 22.83 / 24.12 | 18,535 / 18,606 | 24,409 / 24,481 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 41.85 | 43.40 | 23.04 | 18,538 |
| unset | 2 | 60.81 | 31.74 | 31.51 | 8,717 |

Aggregate unset vs one: +45.3%; per-request decode -26.9%.

Hashes: identical across every row of both arms; 8 requests, first 53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80.

### greedy-c2-2k

| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |
|---|---|---|---|---|---|---|---|---|
| r1 | unset | 13.33 | 30.96 | 32.30 / 35.86 / 35.87 | 30.27 / 31.76 | 15,039 / 15,501 | 19,189 / 19,342 | 8/8 |
| r2 | one | 12.07 | 43.85 | 22.81 / 22.87 / 22.88 | 22.59 / 23.80 | 18,301 / 18,465 | 21,194 / 21,364 | 8/8 |
| r3 | unset | 13.35 | 30.98 | 32.28 / 35.79 / 35.99 | 29.22 / 32.14 | 15,015 / 15,491 | 19,162 / 19,316 | 8/8 |
| r4 | one | 12.06 | 43.76 | 22.85 / 22.89 / 22.89 | 22.60 / 23.80 | 18,307 / 18,470 | 21,211 / 21,374 | 8/8 |

| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|---|
| one | 2 | 12.07 | 43.80 | 22.83 | 18,304 |
| unset | 2 | 13.34 | 30.97 | 32.29 | 15,027 |

Aggregate unset vs one: +10.6%; per-request decode -29.3%.

Hashes: identical across every row of both arms; 8 requests, first a031af1a 27abea10 5a190b71 386c38fe 0c449626 1494b4fc fb699e2c 78f53cd0.

Thermal one: power median 178..179 W, peak 352 W, SM clock 2280..2422 MHz, max temp 52 C, 4707 samples.

Thermal unset: power median 208..215 W, peak 354 W, SM clock 2265..2422 MHz, max temp 54 C, 3605 samples.

Every cell's text is identical across both arms, the yielding prefill included. The unset arm
repeats the A/B above: +48.9% greedy c2, +50.0% c4, +44.8% sampled c2. At 2k-token prompts the
second request's first token comes 18% sooner (TTFT p50 18.3 -> 15.0 s).
