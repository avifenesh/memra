# WP-B day 36: `MEMRA_ADMIT_BY_MEMORY` OFF against ON on the day-35 tree, the owner's decision cell

The door's decide-by is 2026-10-07. Day 34 ran the cell on `9f335ac48` and banked its 5090 half as a record of that
tree (V-DOOR PASS). Its target-card half did not run, because memra#680's seed term was still open (DAY34.md 2.2).
Day 35 booked that term (`809c16444`). This is the clean rerun the lead named: day 34's cell with its binary updated
to the day-35 tree. Every cell is `executed-not-qualified`; no qualification is claimed, no default moves, and no
open-output value is picked.

## 1. Pre-registration

Committed and pushed before the first boot of this day. Nothing in section 1 changes after a number is seen.

### 1.1 Start condition

Both cards start only if DAY35.md's BOX4 legs read card verdict GREEN under DAY35.md 1.4: R-OOM RED on `red-R64`,
and G-NOOM, G-BOOK, V-ID-FIX, V-ID and V-OFF PASS on both green runs. If they do not, neither card starts, and
section 2 says why.

### 1.2 The program and the binaries

The day-35 fix `809c16444`: main `25bbb91f5` (#668 and the memra#680 workspace term) plus the seed term. The door
books `pending_prime + pending_seed` on every headroom reading of the admission block, and its lines carry
`pending_seed=`. Door OFF is day 34's program; DAY35.md's V-OFF checks that row for row on both cards.

One binary per card, the day-35 green build, the same bytes day 35 judged:

- local `0a940566626b6d62b9adf0e8031df809abe1f3655761de5b966f52a850165f4c` (`rtx5090-day35/build-green/`), copied to
  `target/day36/memra-server`, recorded in `rtx5090-day36/binary.sha256`;
- BOX4: day 35's `bins/green` build, whose source must read `809c16444` and whose sha256 must match its record
  (`pro-single-day36/chain.sh` refuses otherwise). It is copied to the receipt root's `bins/tip/`.

The cards and models are DAY34.md 1.2's, except that the target card is BOX4 (one RTX PRO 6000 Blackwell Workstation
Edition at 600 W, model sha256 `1facf36c...1e024a`). Nothing is compared against BOX3's receipts, and timing is never
compared across cards or boxes.

### 1.3 to 1.8

DAY34.md 1.3 to 1.8 unchanged: arms, orders, one collector hold per order; the day-31 workload through the unchanged
drivers (hashes in DAY34.md 1.2); `day34-compare.py` unchanged (sha256 `c6013822...deaea5e9`), run with DAY34.md 1.5's
command lines and the directory names `rtx5090-day36/boots` and `pro-single-day36/box/boots`; every DAY34.md 1.6 term
(the day-32 terms, V-ALLOC on the engine's form, V-OOM, G-BOOK, PARK and V-DOOR); DAY32.md 1.7's selection rules.
Chains: `rtx5090-day36/chain.sh` (day 34's local chain with day-36 paths) and `pro-single-day36/chain.sh` (day 34's
part B with the binary above).

Expected readings, stated before any number:

- **5090:** day 34's readings (DAY34.md 2.3 and 2.4): V-DOOR PASS, no OOM, no park; the 32768 burst about 19 x 200
  and 13 typed 429s. SELECT R1 = 8192, R2 = 8192, R3 = 32768, R4 = `context`.
- **BOX4:** V-OOM PASS with no park on every boot; the 32768 burst about 49 x 200 and 15 typed 429s, as DAY35's
  green R64 is expected to read. SELECT R1 = none (the long class stops at 18,443 to 193,178 under OFF), R2 = none of
  [2048, 8192, 32768] (max_natural_G 193,178), R3 = 32768, R4 = `context`. Every other term as on day 32 with the
  engine's V-ALLOC form.

### 1.9 Failures

Causes are quoted from captured stderr, never inferred. A rerun happens only as a whole order under a new name, with
the reason in section 2.

## 2. Results

Written after the runs. Section 1 is unchanged.
