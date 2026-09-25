# WP-B day 40: O3, `MEMRA_ADMIT_BY_MEMORY` OFF against ON on the final booking (decide-by 2026-10-07)

OWED.md O3. Day 36 ran the owner's decision cell (DAY34's cell) on `809c16444`, which booked every armed session's
full prefix seed and day 33's per-session prime workspace. Since then the seed booking is capped at the prefix cache's
remaining budget (`34a4b7f23`, on main) and the prime booking is day 39's revised term (`be2177ead`: the slab's owed
growth once, the call's returned rows, the walker stacks, the owed checkpoint snapshots), which reads GREEN on the
target card (DAY39 2.2). This day reruns day 36's cell on that final booking, so the ON concurrency rows the owner reads
are the booking that would ship. Every cell is `executed-not-qualified`; no default moves; no open-output value is
picked.

## 1. Pre-registration

Committed and pushed before any day-40 code and before any day-40 boot. Nothing in section 1 changes after a number is
seen.

### 1.1 The program and the binary

The lane tip at the first boot, `S40` = `b46ae200e` (the tip with 1.4's print): main through integ60 (`7bcb6364d`
merged it), the capped seed booking, the day-39 revised prime booking, and every lane door unset. One binary per card,
built in a detached worktree. Door OFF is day 36's OFF program plus whatever main and the lane changed outside the
door; V-OFF is not a clause here (day 36 has none), and the ON rows are read against this day's own OFF boots.

### 1.2 The cell

DAY34.md 1.3 to 1.8 unchanged, as DAY36.md 1.3 to 1.8 ran them: arms `off`, `on2048`, `on8192`, `on32768`; orders O1
and O2 (one collector hold per order, `day31-order.sh`, burst 32 on the 5090 and 64 on the target card, as days 34 and
36 ran them); the day-31 workload through the unchanged drivers (their sha256 match DAY34.md 1.2's);
`day34-compare.py` unchanged (sha256 `c6013822...deaea5e9`) with DAY34.md 1.5's command lines on the directory names
`rtx5090-day40/boots` and `pro-single-day40/box/boots`; every DAY34.md 1.6 term (the day-32 terms, V-ALLOC on the
engine's form, V-OOM, G-BOOK, PARK and V-DOOR) and DAY32.md 1.7's selection rules (SELECT R1 to R4, stated, not chosen).
Cards: the local RTX 5090 (the 9B at `MEMRA_CTX=65536`) and one RTX PRO 6000 Blackwell Workstation Edition (the 27B at
the checkpoint's context).

### 1.3 The added readings (no bound)

- **Whether the seed cap bound.** Every `[admit-mem]` line gains `pending_seed_uncapped=` (the sum before the cap,
  printed beside the capped `pending_seed=`; a print only, under the door). Per ON boot: the lines in the burst window
  where the capped value is below the uncapped one, and the largest difference.
- **The prime booking.** Per ON boot, the burst's peak `pending_prime` against `pending_prime_v1`.
- **Against day 36.** Per card and arm, the burst's 200 and 429 counts beside day 36's (a reading across days, not a
  comparison of timing).

### 1.4 Code before the boots

`pending_seed_uncapped=` in `MemoryLine` and both constructions, with the unit tests that pin the line's fields.

### 1.5 Failures

Causes are quoted from captured stderr. A rerun happens only as a whole order under a new name, with the reason in
section 2.

### 1.6 Price

0.1 agent-day of code; about 4.5 h on the 5090 (after its reset) and about 6 h on the target card (day 36 took 5.8 h).

## 2. Results

Written after the runs. Section 1 is unchanged.
