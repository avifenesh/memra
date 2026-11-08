# c8-serving-20261107: scored A/B

Main (A, `memra-server-base-46172c1bac`, sha256 `41741f5da6cc67b8`) against the lane (B,
`memra-server-ec935b6f99`, sha256 `7fba3e856b8b9257`). `ec935b6f99` is the lane on main `46172c1bac`; it
differs from the landed code (`8b84ed2913` and `b70b55484b` on main `c347e782d2`) by one comment
(`git range-diff`). One fresh boot per run, interleaved, on one RTX 5090 Laptop GPU (24 GB, balanced
platform profile). Model and context as in [README.md](README.md). Naked defaults: the cache is on, and the
first-token deadline is 90 s.

How to read the tables:
- "Wait" is the time to a request's first byte. A 408 counts as a wait that ended when its error arrived (90 s), not as a missing row.
- "First token" counts served requests only, as the lane's earlier tables did.
- `tokens` is the completion tokens served in the pass, and `tok/s` is those tokens over the pass's wall time.
- The free floor is the lowest device free memory `nvidia-smi` saw in the pass, sampled every 250 ms.

Counters, from the server log:
- defers: VRAM defer lines.
- deadline-order ticks: times the earliest-deadline order turned on.
- reserve evictions: the cache yielding after an insert.
- OOM: `CUDA_ERROR_OUT_OF_MEMORY` lines.

Per-run tables: `harness/tables.py runs/PHASE`. Pooled percentiles and the trace split:
`harness/split_t.py runs/PHASE`.

## Agentic c8 (`runs/phaseS8/`, A B B A A B)

The full shape: a prime pass (`max_tokens=1`), then the measured pass (T=1, 768 new tokens, thinking
on), 29 turns each, 8 closed-loop users.

| run | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB | defers | deadline-order ticks | reserve evictions | OOM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-A | measure | 26/29 | 3 | 18.8 | 88.3 | 18.8 | 90.0 | 9910 | 41.44 | 2099 | 1805 | 0 | 0 | 0 |
| 01-A | prime | 26/29 | 3 | 22.0 | 76.2 | 22.0 | 90.0 | - | - | 2291 |  | | |  |
| 02-B | measure | 29/29 | 0 | 34.8 | 62.9 | 34.8 | 62.9 | 10623 | 40.35 | 2163 | 1693 | 8 | 0 | 0 |
| 02-B | prime | 29/29 | 0 | 38.3 | 81.5 | 38.3 | 81.5 | - | - | 1747 |  | | |  |
| 03-B | measure | 29/29 | 0 | 36.4 | 71.7 | 36.4 | 71.7 | 10623 | 39.81 | 2067 | 1661 | 8 | 2 | 0 |
| 03-B | prime | 29/29 | 0 | 41.3 | 79.0 | 41.3 | 79.0 | - | - | 1203 |  | | |  |
| 04-A | measure | 26/29 | 3 | 17.4 | 83.3 | 19.5 | 90.0 | 9889 | 41.04 | 2195 | 1820 | 0 | 0 | 0 |
| 04-A | prime | 27/29 | 2 | 25.8 | 80.0 | 26.7 | 90.0 | - | - | 691 |  | | |  |
| 05-A | measure | 27/29 | 2 | 19.7 | 80.5 | 19.7 | 90.0 | 10294 | 39.86 | 1939 | 1950 | 0 | 0 | 0 |
| 05-A | prime | 27/29 | 2 | 21.3 | 83.6 | 28.8 | 90.0 | - | - | 2419 |  | | |  |
| 06-B | measure | 29/29 | 0 | 35.7 | 65.0 | 35.7 | 65.0 | 10623 | 39.95 | 2131 | 1727 | 10 | 1 | 0 |
| 06-B | prime | 29/29 | 0 | 42.5 | 83.8 | 42.5 | 83.8 | - | - | 1491 |  | | |  |

| arm | N | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | measure | 26/29 | 3 | 18.8 | 83.3 | 19.5 | 90.0 | 9910 | 41.04 | 2099 |
| A | 3 | prime | 27/29 | 2 | 22.0 | 80.0 | 26.7 | 90.0 | 27 | 0.17 | 2291 |
| B | 3 | measure | 29/29 | 0 | 35.7 | 65.0 | 35.7 | 65.0 | 10623 | 39.95 | 2131 |
| B | 3 | prime | 29/29 | 0 | 41.3 | 81.5 | 41.3 | 81.5 | 29 | 0.16 | 1491 |

- Main failed 15 of its 174 requests across both passes, all of them 408. The lane failed none.
- All 15 failures are turns of 12,327-17,101 prompt tokens (workload rows 8, 14, 15 and 22).
- No run hit an OOM. The lowest free floor was 691 MiB on main and 1,203 MiB on the lane.
- The sampling seed is fixed (1234 + turn), so the lane's measured pass served exactly 10,623 tokens in all three runs.

Measured pass, median wait by prompt size (all requests, three runs pooled):

| turns per run | prompt tokens | main | lane |
|---|---|---|---|
| 14 | under 4k | 8.7 s | 20.5 s |
| 7 | 4k-8k | 17.1 s | 37.4 s |
| 2 | 8k-12k | 45.5 s | 49.4 s |
| 6 | 12k and up | 85.8 s | 48.2 s |

On main, short turns overtake a deferred long one, and the long one times out. On the lane they wait
for it, and it is served. The median wait is set by the short turns, so it roughly doubles: 19.5 s
against 35.7 s, medians of the three runs. The p99 wait falls from the deadline (90.0 s) to 65.0 s.

The lane serves 7% more tokens per pass (10,623 against 9,910), because the three turns that failed on
main are now answered. Over the pass wall that is 2.7% lower tok/s (40.0 against 41.0). Turn total
(E2E, a 408 counted at 90 s), medians of the runs: p50 66.7 s against 55.7 s, p99 136.4 s against
138.6 s.

Server-log timeline: in the measured pass, memory defers were active for about 97 s of 265 s in both
arms, and the deadline order for 0-3.5 s. In the prime pass the deadline order was on for 39-82 s of
about 175 s. So the measured-pass wait comes from the memory hold, and the prime-pass wait from both.

## Equal work, traced (`runs/phaseT/`, A900 B8 B B8 A900 B B8)

Is the median cost lost work (main sheds its three largest turns, 23% of the pass's prompt tokens) or
order? A900 is main with `MEMRA_STREAM_TTFT_MS_MAX=900000` (and `MEMRA_SSE_PREFILL_COMMIT_MS=60000`,
which that setting requires), so it serves every turn. B8 is the refinement `cand9`, which was dropped
(below). `MEMRA_TTFT_TRACE=1` was on in every run.

| run | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB | defers | deadline-order ticks | reserve evictions | OOM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-A900 | measure | 29/29 | 0 | 19.0 | 152.1 | 19.0 | 152.1 | 10623 | 35.21 | 2131 | 1928 | 0 | 0 | 0 |
| 01-A900 | prime | 29/29 | 0 | 26.1 | 112.4 | 26.1 | 112.4 | - | - | 1363 |  | | |  |
| 02-B8 | measure | 29/29 | 0 | 26.7 | 86.2 | 26.7 | 86.2 | 10623 | 36.26 | 2099 | 1946 | 5 | 1 | 0 |
| 02-B8 | prime | 29/29 | 0 | 27.8 | 89.7 | 27.8 | 89.7 | - | - | 1203 |  | | |  |
| 03-B | measure | 29/29 | 0 | 37.4 | 69.6 | 37.4 | 69.6 | 10623 | 39.85 | 2131 | 1663 | 7 | 1 | 0 |
| 03-B | prime | 29/29 | 0 | 42.5 | 78.1 | 42.5 | 78.1 | - | - | 1235 |  | | |  |
| 04-B8 | measure | 29/29 | 0 | 35.7 | 85.9 | 35.7 | 85.9 | 10623 | 38.1 | 1235 | 1626 | 4 | 3 | 0 |
| 04-B8 | prime | 29/29 | 0 | 29.4 | 89.3 | 29.4 | 89.3 | - | - | 2259 |  | | |  |
| 05-A900 | measure | 29/29 | 0 | 22.4 | 118.6 | 22.4 | 118.6 | 10623 | 36.02 | 371 | 1815 | 0 | 0 | 0 |
| 05-A900 | prime | 29/29 | 0 | 24.3 | 117.6 | 24.3 | 117.6 | - | - | 1683 |  | | |  |
| 06-B | measure | 29/29 | 0 | 35.5 | 64.7 | 35.5 | 64.7 | 10623 | 40.24 | 2323 | 1699 | 4 | 1 | 0 |
| 06-B | prime | 29/29 | 0 | 35.7 | 74.3 | 35.7 | 74.3 | - | - | 1459 |  | | |  |
| 07-B8 | measure | 29/29 | 0 | 21.6 | 75.4 | 21.6 | 75.4 | 10623 | 36.95 | 1203 | 1621 | 4 | 3 | 0 |
| 07-B8 | prime | 29/29 | 0 | 29.7 | 89.0 | 29.7 | 89.0 | - | - | 1875 |  | | |  |

| arm | N | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A900 | 2 | measure | 29/29 | 0 | 20.7 | 135.4 | 20.7 | 135.4 | 10623 | 35.62 | 1251 |
| A900 | 2 | prime | 29/29 | 0 | 25.2 | 115.0 | 25.2 | 115.0 | 29 | 0.17 | 1523 |
| B | 2 | measure | 29/29 | 0 | 36.4 | 67.1 | 36.4 | 67.1 | 10623 | 40.05 | 2227 |
| B | 2 | prime | 29/29 | 0 | 39.1 | 76.2 | 39.1 | 76.2 | 29 | 0.17 | 1347 |
| B8 | 3 | measure | 29/29 | 0 | 26.7 | 85.9 | 26.7 | 85.9 | 10623 | 36.95 | 1235 |
| B8 | 3 | prime | 29/29 | 0 | 29.4 | 89.3 | 29.4 | 89.3 | 29 | 0.16 | 1875 |

Measured pass, pooled over the runs (58 turns for A900 and B, 87 for B8):

| arm | p10 | p25 | p50 | p75 | p90 | p99 | mean | whole turn mean / p99 | tok/s (median run) |
|---|---|---|---|---|---|---|---|---|---|
| A900 (main, every turn served) | 5.1 s | 9.9 s | 19.4 s | 40.0 s | 111.4 s | 152.1 s | 34.5 s | 75.0 s / 166.2 s | 35.6 |
| B (the lane) | 12.7 s | 19.3 s | 37.4 s | 41.5 s | 53.9 s | 69.6 s | 33.3 s | 68.5 s / 141.4 s | 40.0 |
| B8 (dropped refinement) | 6.7 s | 17.2 s | 26.7 s | 56.3 s | 69.7 s | 86.2 s | 34.8 s | 72.1 s / 163.9 s | 37.0 |

Prime pass, pooled: A900 p50 26.0 s, p99 117.6 s; B p50 40.0 s, p99 78.1 s; B8 p50 29.4 s, p99 89.7 s.

It is order. At equal work main keeps its 19.4 s median, so the hold delays turns that would have been
fine: turns under 4k wait 10.3 s on main and 20.7 s on the lane. The mean wait is the same, so the order
moves waiting from the long turns to the short ones and loses none. Main's p90 and p99 are the 408s at
its real deadline. The lane finishes the same work 12% faster, and whole turns end sooner on average.

The trace split of the served first tokens (`split-T.txt`) shows where each arm waits. Main's long turns
wait in `prime_wait` (76 s median for 12k and up in the measured pass: admission starvation). The lane's
short turns wait there too (11.0 s under 4k, against 1.3 s on main), behind the deferred head.

### The dropped refinement (B8, `cand9`)

B8 let a later request past the deferred head while an upper bound said the head still made its
deadline. It also gave a rescued prefill only the share of the device it needed, keeping the shortest
other rows while every deadline in the set fit.

- It cut the median to 26.7 s with no failure.
- It raised everything above the median, cost 8% tok/s, and put the slowest first token at 86-90 s in both passes against the 90 s deadline.
- Short turns gained nothing (20.6 s under 4k); only mid-size turns moved.

An independent review then found optimistic terms in its bound (the posted review on memra#1099). A
corrected bound is more conservative, so it lies between B and B8 on the same trade line. Dropped; the
decision record lists it under Rejected.

## Greedy c8 exactness on the final code (`runs/phaseX/x7-final-c8-g1`)

The greedy cell (T=0, 256 new tokens, a prime pass then a measured pass at c8) with the lane binary. It
served 58 of 58 requests with no OOM, and its lowest free floor was 1,843 MiB. Main failed 11 of 58 in
each of two runs (`x2-base-c8-g1`, `x3-base-c8-g2`). On the 47 rows both served, the generated bytes are
identical to both main runs: 47/47 against each (`harness/exact_cmp.py`; `runs/phaseX/exact-x7-*.txt`).
The tail merge in the batch path can change the program of a prompt whose batched last chunk left a 1-15
token tail; none of the compared rows changed.

## Device records on the stack top

`research/release-qualification/cells/`, for `1877d4661a` (the code top after the rebase onto main
`3a7d48a27c`). Both perf A/Bs ran against `f68bf32282`, whose source inputs differ from
`3a7d48a27c` only in README.md, a docs path no crate reads:
- **sm_120a** (RTX 5090, record `1877d4661a4f-rtx5090-battery-perf-serving`): battery, serve-smoke and perf pass. qwen3.8-27b decode 45.96 against 45.03 tok/s, ornith-1.5 213.86 against 217.84: not regressed (the tool needs complete separation of the runs plus a minimum drop; this change does not touch decode).
- **sm_121a** (GB10, record `1877d4661a4f-gb10-battery-perf-serving`): battery, serve-smoke and perf pass. qwen3.8-27b 13.65 against 13.64 tok/s, ornith-1.5 80.55 against 80.64: not regressed.

The records on the earlier tops are not banked, because two README commits on main voided them
before they landed. They read the same: 5090 q27 +3.6% and -1.4%, GB10 within 0.3%.

## Not measured here

c4 ABBA, c1 AB and the mixed shape on the final code are queued behind other lanes' landing records on
the 5090. They follow as an addendum to this file.
- At c1 none of the four changes can engage: the hold needs a memory defer, which needs a peer; seed batching and the deadline order need two rows; the reserve yield needs effective free under the reserve.
- The mixed shape's single runs are in [README.md](README.md): batch TTFT p50 1,304 ms on main, 935 ms with seed batching, 736 ms with the cache off.

A PRO 6000 measurement of the effect there is owed (decision record).
