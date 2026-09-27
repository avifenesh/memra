# WP-B day 50: O11's overlap revision, making the exact resume fast where DAY44's E2 failed

OWED.md O11. The owner's direction (2026-09-26): "resume vs rewind - i think its not or or question, but more of we
didnt make it right yet". DAY44 made the resume exact (E1: 0 flips against cold on every `exact` boot on the target
card, against 24 of 60 on `keep`). It is not yet fast everywhere: E2 fails at 8 of 24 cells (DAY44 2.1), all on TTFT.
DAY44 1.4 named the next revision. This day registers it, stage 0 first, before any code.

## 1. Pre-registration

Committed and pushed before any day-50 code and before any day-50 run. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 What the receipts place (DAY44 2.1, the target card, the 27B)

- **M1, zero gap on the plain route:** no settle lands before the next turn, so the resume primes from the in-call
  checkpoint: 64 extra rows at G=32 and 288 at G=256. The prime phase grows by that work: 68.3 to 114.9 ms at 6,144
  G=256, 94.3 to 140.6 at 30,720, 188.7 to 235.4 at 122,880.
- **M2, an arrival waiting for a settle call:** on spec RX at G=256 the resume's own prime phase is faster than `keep`'s
  (84.7 against 149.7 ms at 6,144), but the turn waits for its entry's settle call to land (race case 2). The TTFT excess
  over the prime phase (about 105 ms) matches the settle call's p50 (108 ms). At plain RXg 122,880 G=256 some arrivals
  meet a settle in flight: p50 +25 ms, p95 340 against 196 ms.
- **The cost both pay is a short prime call.** A call of 32 rows costs about 68, 94 and 188 ms at 6,144, 30,720 and
  122,880 context, and each extra row about 0.16 to 0.2 ms (DAY44 2.1's prime phases). The fixed part dominates. What that
  fixed part is spent on (GPU kernels, host launch, syncs) is not yet measured, and it decides the design.

### 1.2 Stage 0 (before the design is chosen): what a short prime call spends its time on

`concat-prime-probe` gains a `callcost` mode: prime `[0, L)` once; then for each R in {32, 64, 288} and N = 5 reps,
take a snapshot at L, prime `[L, L + R)` as one call (the settle and resume shape), time the call's wall around a
device sync, and restore the snapshot. L in {6,144, 30,720, 122,880} on the target card (the 27B) and {6,144, 30,720} on
the 5090 (the 9B). Each (L, R) runs once more under `nsys profile` for the kernel timeline. Readings per (L, R): wall
p50; the GPU-busy time (the union of kernel intervals); the host gaps between kernels; the time in full-attention
kernels, in the GDN scan and in the GEMMs. No clause; stage 0 selects the arm by 1.3's rule.

### 1.3 The arms and the rule that selects them (stated before stage 0)

- **Arm S (short-call overhead).** If the GPU is busy for less than 60% of a 32-row call's wall at any L on the target
  card, the fixed cost is mostly host-side. The revision then removes those gaps from the prime path. That makes every
  short prime faster, `keep`'s resume included, so E2's ratio moves only by the share the gaps took of the extra work.
  Stage 0 decides this first because it is the cheaper change and helps every route.
- **Arm O (overlap on a second stream).** If the GPU is busy for at least 60% of the call, or E2 still fails after arm S,
  settles run on a second compute stream and never block the worker:
  - A settle starts at park time on the side stream. An arrival for another entry runs at once on the main stream,
    which removes M2 for other entries.
  - An arrival for the entry being settled resumes from the checkpoint at `g` on the main stream and drops the
    settle's result, so it never waits (that turn pays M1's cost, bounded by one call).
  - For zero-gap turns (M1), the settle of a turn's own decoded rows starts once the decode crosses the last grid
    point that `max_tokens` allows, so it overlaps the decode's tail.
  - Precondition, as its own census: the engine's shared prime scratch (the prime slabs, the CUTLASS NVFP4 scratch,
    the q8_1 activation scratch, the GDN scan buffers, the cuBLASLt handle) is safe today only because all compute
    serializes on one stream. The side stream gets its own set, and a census test names every buffer the prime path
    touches.
- **Weighed and not chosen:** settling in segments during decode on the same stream puts G / 32 short calls on the
  decode's critical path (about 8 x 68 ms at 6,144 G=256 against a 3,154 ms E2E, +17%). Settling at the turn's end before
  the final frame adds one call to E2E (+15% at 6,144 G=32). Both fail E2's E2E half. A mixed forward that carries
  prime-program rows beside decode rows does not exist in the engine.

### 1.4 Clauses (DAY44's, unchanged, plus E7)

DAY44 1.7's E1 to E6 on DAY44 1.6's cells, both cards, both orders, plus:

- **E7 decode.** On every `exact` boot, the resumed turns' TPOT p50 is at most 1.02 x `keep`'s in the same cell (arm O's
  side stream must not slow the decode it overlaps).

Readings: the settle calls' wall and count, the side stream's busy share (arm O), idle driver-free and pool-cached bytes.

### 1.5 What the reading decides

Nothing moves a default. If E1 to E7 pass on a card class, the exact resume is PROMOTE-ELIGIBLE there and the grid
rewind door's deletion is owed under door hygiene. `MEMRA_RESUME_EXACT` keeps its decide-by (2026-10-10), and the
owner decides at that date.

### 1.6 Price

Stage 0: about 0.3 agent-day (the probe mode and its reader) plus about 1 h on each card. Arm S: about 1 agent-day plus
the E1 to E7 cells (about 12 h on the target card, 5 h on the 5090). Arm O: about 3 to 4 agent-days (the side stream,
its scratch set and census, the settle launcher and the arrival rule, GPU tests of bit-identity against the one-stream
settle) plus the same cells.

## 2. Results

Written after the runs. Section 1 is unchanged.
