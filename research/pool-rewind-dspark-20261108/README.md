# Winning the DFlash multi-turn speed back exactly (memra#1093 follow-up, 2026-11-08)

Verdict: PENDING the (a) cells. (b) is rejected: turning `MEMRA_DSPARK_PREFIX_RESTORE` on for the
RTX 5090 Laptop again wins where the card has room and runs out of room one concurrency step
sooner.

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

## (a) The DSpark-pool rewind

PENDING.
