# Winning the DFlash multi-turn speed back exactly (memra#1093 follow-up, 2026-11-08)

Verdict: (a), the DSpark-pool rewind, wins the speed back exactly with no memory regression, and
ships (PR #1103). On the RTX 5090 Laptop at its device default the long-decode DFlash continuation's
turn-3 TTFT goes from 2.462 s (main, a cold re-prime) to 0.275 s, byte-equal to cold in 9 of 9 pair
cells (greedy, T=1 and penalized; chat, raw and second-generation chains), and the agentic memory
A/B against main refuses and times out in the same cells with the same output. (b) is rejected:
turning `MEMRA_DSPARK_PREFIX_RESTORE` on for the 5090 wins where the card has room and runs out of
room one concurrency step sooner, as the 2026-11-05 decision cell found.

Since memra#1093 (PR #1098) a recurrent DFlash session is freed at retire: its rows past the prompt
were written by verify and decode, off the prime grid, so the pool resume at the parked end left the
cold text. On the RTX 5090, where the restore switch is off, a long-decode continuation then
re-primes the whole conversation: turn 3 TTFT 0.163 s (inexact pool resume) became 2.745 s
(`../pool-resume-cost-20261107/run-3-dspark-long/`). Two ways back were on the table:

- (a) a DSpark-pool rewind: park the session rewound to its prime-boundary capture;
- (b) the restore switch on for the 5090, re-measuring the memory floor and c1..c8.

The rule: take whichever restores most of that path with exact output and no memory regression.

## (b) The restore switch on the RTX 5090, re-measured on current main (`switch-ab/`)

Binary: main c347e782d2 (code b30d0b92ad, sm_120a, sha256 bddf1f1a5d19). The 2026-11-05 decision
cell's harness and config (`../dflash-restorable-hit-20261103/rig5090-flip/`): Qwen3.8-27B
NVFP4+Q5_K GGUF + DFlash2 q4 drafter + FR-Spec trim, `MEMRA_MTP_SKIP=1`, context 32,768, 8
sessions, prefix cache 4,000 MB, no host tier, `MEMRA_PREFIX_STABLE_BOUNDARY=0`, T=1 seed 1234,
thinking on, `max_tokens` 768. ON adds `MEMRA_DSPARK_PREFIX_RESTORE=1`; OFF is the device default.
Fresh boot per arm, run order as listed (`switch_ab.sh`, `chain.log`), 2026-11-08 03:04-04:31.
Host load average 13-36 from other sessions throughout (a busy box, the same for every arm).
Workloads as in the decision cell: **full** (29 agentic bodies, sha256 5fe1afe2afdc, prompts 1.7k
to 17.9k tokens) and **le8k** (the 22 shortest, 331f269100d2). The workload files and the generated
texts stay out of the repo (derived from private agent sessions).

| cell | ON | OFF (default) |
|---|---:|---:|
| c1, full, order ON / OFF / OFF / ON | 31.17 / 31.48 tok/s | 28.58 / 29.09 tok/s |
| c2, full | refused: 2 requests HTTP 400 `context_length_exceeded` (17,933 and 14,770 contexts; largest admissible 10,685 and 6,782) | 29.93 tok/s |
| c4, full | refused (HTTP 400) | refused (HTTP 400: 17,166 against 16,002-16,162 admissible) |
| c8, full | refused (HTTP 400) | refused (HTTP 400) |
| c2, le8k | 42.46 tok/s | 36.05 tok/s |
| c4, le8k | 2 requests HTTP 408 (queued behind VRAM defers) | 37.28 tok/s |
| c8, le8k | failed | 1 request HTTP 408 |

Aggregate generated tok/s. The arms do not generate the same tokens at T=1 (ON 10,818 at c1, OFF
11,549): ON serves its DFlash turns by DSPARK restore (14 and 11 per boot), OFF re-primes them or
takes a plain prefix hit under the `cache-preferred` veto (4 and 2), and a different program draws a
different sample. The learned fixed residual reads 485-493 MB with ON against 340 MB with OFF, which
is what pushes the long contexts over the idle-box limit. That is the 11-05 cell's finding on
today's main: +8.6% at c1 and +18% at c2 le8k, and the first refusal one concurrency step sooner on
both workloads. (b) fails the no-memory-regression bar; the device default stays OFF.

## (a) The DSpark-pool rewind (PR #1103)

With the restore switch off, a retiring recurrent DFlash session parks rewound to a prime-boundary
capture it published itself: trunk KV lengths cut to `g`, the recurrent state copied from the
prefix-cache entry that publish inserted (matched by id), the drafter cut to `g` with its window
below `g` held by a soft keep that never grows the ring. The exact token tier resumes it, seated for
the new request as a cold prime seats a session (sampler and seed, fresh Philox counters, penalty
window, verify ladder, the request's round guard), and primes the rest from `g`. A resumed prime
captures the new prompt's grid cut, so the next turn has a target. Binary 608404e003 (sm_120a,
sha256 a0c652bc8656), base main c347e782d2 (bddf1f1a5d19).

### Exactness (`pairs/`, 2026-11-08 08:47-09:11)

`pool_pair_cells.py` (this directory's copy adds sampled and penalized requests and a path check):
one warm boot serves a three-turn chat in order, one cold boot per turn 2 and turn 3. Qwen3.8-27B
NVFP4 + Q5_K GGUF + DFlash2, context 8,192, two sessions, prefix cache at its derived size, device
defaults; thinking off. A case passes only if turns 2 and 3 are byte-equal to cold AND the warm log
shows the expected path (`_EXPECT=rewind:P:R`: at least P rewound parks, R prime-boundary resumes,
every boundary on the 32-token grid, no grid-law decline; `_EXPECT=skip`: every park skipped, no
pool resume).

| case | route | max_tokens | sampling / env | turn 3 warm / cold chars | bytes | rewound parks (fed -> g) | prime-boundary resumes (g) | skips |
|---|---|---|---|---:|---|---|---|---:|
| dspark-1024-rw | chat | 96 / 1024 / 512 | greedy | 371 / 371 | pass | 3082->2976, 3114->3072, 3219->3104 | 3072 (parked 3114) | 0 |
| dspark-1024-t1-rw | chat | 96 / 1024 / 512 | T=1 seed 1234 | 651 / 651 | pass | 3073->2976, 3105->3072, 3281->3104 | 3072 (parked 3105) | 0 |
| dspark-chain-rw | chat | 1536 / 448 / 448 | greedy | 371 / 371 | pass | 3082->2976, 3114->3072, 3219->3104 | 2976 (parked 3082), 3072 (parked 3114) | 0 |
| dspark-chain-t1-rw | chat | 1536 / 448 / 448 | T=1 seed 1234 | 651 / 651 | pass | 3073->2976, 3105->3072, 3281->3104 | 2976 (parked 3073), 3072 (parked 3105) | 0 |
| dspark-1024-t1-pen-rw | chat | 96 / 1024 / 512 | T=1, frequency 0.3, presence 0.2 | 1372 / 1372 | pass | 3088->2976, 3131->3072, 3371->3136 | 3072 (parked 3131) | 0 |
| dspark-chain-t1-pen-rw | chat | 1536 / 448 / 448 | T=1, frequency 0.3, presence 0.2 | 330 / 330 | pass | 3101->2976, 3143->3104, 3228->3136 | 2976 (parked 3101), 3104 (parked 3143) | 0 |
| dspark-raw-1024-rw | raw | 96 / 1024 / 512 | greedy | 157 / 157 | pass | 3068->2944, 3099->3040, 3144->3072 | 3040 (parked 3099) | 0 |
| dspark-raw-chain-t1-rw | raw | 1536 / 448 / 448 | T=1 seed 1234 | 131 / 131 | pass | 3003->2944, 3067->2976, 3097->3040 | 2944 (parked 3003), 2976 (parked 3067) | 0 |
| dspark-256-nopc-aff0-rw | chat | 96 / 256 / 96 | prefix cache off, MEMRA_AFFINITY=0 | 371 / 371 | pass | - | - | 3 |

Turn 2 is byte-equal in every case too. In the chain cases turn 3 resumes the session turn 2 itself
resumed and parked again (rewound to its resume-capture cut): the second-generation path. The
96 / 1024 / 512 shape is the one that left the cold text before memra#1093 (383 vs 371 characters at
3,114 of 3,144); it now resumes at 3,072 and primes 72 rows. With the prefix cache off no entry
exists to rewind to, so the session is freed (`dspark-park SKIPPED`), as on main.

### What it wins back (`ttft/`, 09:01-09:06)

`../pool-resume-cost-20261107/cost_cell.py`, shape `chat-dspark-long` (the chat at 96 / 1024 / 512,
3 distinct conversations per boot, greedy, thinking off, streamed), base main against the fix, ABBA,
one boot per call. Host load 25 from other sessions.

| turn 3 | TTFT median (range), 6 requests | path |
|---|---:|---|
| main (c347e782d2) | 2.462 s (2.335-2.754) | cold DFlash prime of the 3,006-3,017-token prompt |
| fix (608404e003) | 0.275 s (0.265-0.284) | `dspark-reuse: 2944 committed tokens resumed at the prime boundary` |

Completion lengths are the same in both arms (69, 117 and 79 tokens per conversation). Turn 2 is a
cold prime in both arms (turn 1's park is too small for its cap): 2.354 s against 2.571 s, ranges
overlapping (2.27-2.70 and 2.35-2.74). The inexact parked-end resume this replaces measured 0.163 s
(`../pool-resume-cost-20261107/run-3-dspark-long/`): the exact rewind primes from the turn's grid cut
(2,944) instead of the parked end, about 70 rows against 30. With the switch on the same turn
measured 0.277 s by DSPARK restore (`../pool-resume-cost-20261107/run-4-switch/`), so (a) wins back
the same speed as (b) on this shape, on the switch-off default.

### Memory (`mem/`, 09:21-12:18)

`mem_ab.sh`: the (b) cell's harness and config (above), main (b30d0b92ad code, sha256 bddf1f1a5d19)
against the fix (608404e003, a0c652bc8656), both at the device default (switch off). One run per arm
and workload; the arms yielded the card twice between arms to landing records (`window-*.log`), so
they ran in the order fix full, main full, main le8k, fix le8k. Host load average at each arm's start:
40-46, 6-7, 13 and 6.

| cell | main | fix |
|---|---:|---:|
| c1, full | 31.83 tok/s | 30.02 tok/s |
| c2, full | 32.13 tok/s | 31.84 tok/s |
| c4, full | 1 request HTTP 400: 17,166 context, largest admissible 15,961 | 1 request HTTP 400: 17,166 context, largest admissible 16,162 |
| c8, full | 1 request HTTP 408 | 1 request HTTP 408 |
| c2, le8k | 36.30 tok/s | 35.62 tok/s |
| c4, le8k | 38.20 tok/s | 37.83 tok/s |
| c8, le8k | 1 request HTTP 408 | 41.67 tok/s (completed) |

- The output is the same: every completed cell generated the same tokens in both arms (11,549 at c1
  and c2 full, 7,742 and 7,721 at c2 and c4 le8k), and the generated texts are identical, 29 of 29
  and 22 of 22 per cell (compared before the texts were dropped from the receipts).
- The memory floor is the same: the learned fixed residual reads 340 MB on the full workload and
  323-324 MB on le8k in both arms, the idle-box capacity check refuses the same 17,166-token context
  at c4 (the fix's admissible ceiling is 201 tokens higher), and no `out of memory` line appears.
- The fix parked 43 sessions rewound on the full workload and 19 on le8k (11 and 3 freed: the
  boundary's prefix entry had been evicted under the 4,000 MB budget), and
  resumed none: the agentic client's caps grow every turn, so no parked session covers the next
  turn's cap. Reclaim never evicted a DSpark session (0 in both arms); prefix-entry evictions were
  157 against 193 (full) and 140 against 139 (le8k).
- The le8k c8 HTTP 408 in main and the fix's completed c8 are one request's queue deadline, at
  different host load; one run each, so the fix's c8 is not claimed as a gain.
- The throughput differences are under 6%, from single runs under shifting host load (the fix's
  full arm started at load 40-46, main's at 6-7); they are not a measured effect of the change.

## Scope

- One card (RTX 5090 Laptop), one model (Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter), the
  device default. Switch-on devices (GB10, the Max-Q) free a recurrent session at retire exactly as
  main does, and record no rewind target, so their ring compaction is main's too.
- The rewind serves exact token extensions whose cap fits the parked allocation. A client whose turn
  caps grow past the parked session's (the agentic shape above) gets no resume; growing a parked
  DFlash session before the rewind is memra#1093's open grow-order item.
- The prefix cache must be on, and the boundary's entry must still be resident at park: with the
  cache off (`dspark-256-nopc-aff0-rw`) or the entry evicted, the session is freed as on main.
