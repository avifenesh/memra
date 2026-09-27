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
Every number below is read from the committed receipts named next to it.

### 2.1 Commits and timeline (UTC)

| commit | time | what |
|---|---|---|
| `88bb3b1e2` | 2026-09-23 22:19:00 | section 1, both chains, the local binary record |
| (start condition) | 22:37:48 | DAY35.md's BOX4 legs read card verdict GREEN (DAY35.md 2.3) |
| (first boots) | 22:39:14 BOX4, 22:39:46 5090 | O1 collector holds (each card's `order.log`) |
| `207c1ef28` | 2026-09-24 00:54:00 | local O1 receipts |
| `9cf8aee9d` | 01:35:38 | BOX4 O1 receipts |
| `7fd32a4c4` | 03:38:46 | local O2 receipts and the 5090 `SUMMARY.txt` |
| this commit | after 04:29:07 | BOX4 O2 receipts and `SUMMARY.txt`, this section |

5090: O1 22:39:46 to 00:53:07, O2 01:21:08 to 03:35:56 (the lead's integ54 and integ55 batteries held the card between
the orders). BOX4: O1 22:39:14 to 01:34:09, O2 01:34:10 to 04:29:07. Binaries: local `0a940566...`, BOX4
`573482b6...` (day 35's green builds from `809c16444`).

### 2.2 Notes (no rule, arm, value or reader changed)

- **This cell measured the uncapped seed booking.** After these runs started, the review on #705 found that
  booking every armed session's full seed over-books a warm prefix cache, because `prepare_snapshot` evicts or demotes
  older entries to fit a seed inside the cache's budget. The fix caps the booking at the budget left
  (`34a4b7f23`, DAY35.md section 3). Every ON row here ran the uncapped booking. Where the prefix cache was warm, its
  admissions may be over-booked (extra defers or 429s), and the owner should read the ON concurrency rows that way.
  Identity and truncation do not depend on the booking.
- The target card is BOX4 (DAY34.md 2.2). Nothing is compared against BOX3's receipts.

### 2.3 Verdict lines, verbatim

On both cards V-BOOT, V-CRASH, V-OOM, G-BOOK, PARK, V-ALLOC and V-RETRY are PASS on all 8 boots; the per-boot lines
are in each `SUMMARY.txt`. No boot has a panic, respawn, FATAL, OOM line or park (`FAULTS.txt`).

`rtx5090-day36/SUMMARY.txt`:

```
== V-ID / V-TRUNC
DAY34 V-ID card=rtx5090 order=O1 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O1 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O1 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
== SELECT (stated, not chosen)
DAY34 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:66']
DAY34 SELECT card=rtx5090 R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405)
DAY34 SELECT card=rtx5090 R3_registry=32768 R4_survey=context
DAY34 V-DOOR card=rtx5090 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS
```

`pro-single-day36/box/SUMMARY.txt`:

```
== V-ID / V-TRUNC
DAY34 V-ID card=pro6000 order=O1 arm=on2048 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O1 arm=on2048 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=pro6000 order=O1 arm=on8192 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O1 arm=on8192 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=pro6000 order=O1 arm=on32768 vs=off eligible=47 equal=47 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O1 arm=on32768 on_length_rows=5 truncated_with_twin=4 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=pro6000 order=O2 arm=on2048 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O2 arm=on2048 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=pro6000 order=O2 arm=on8192 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O2 arm=on8192 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=pro6000 order=O2 arm=on32768 vs=off eligible=47 equal=47 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=pro6000 order=O2 arm=on32768 on_length_rows=5 truncated_with_twin=4 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
== SELECT (stated, not chosen)
DAY34 SELECT card=pro6000 R1_smallest_zero_truncation=none admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:10', '8192:10', '32768:8']
DAY34 SELECT card=pro6000 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768] (max_natural_G=193178)
DAY34 SELECT card=pro6000 R3_registry=32768 R4_survey=context
DAY34 V-DOOR card=pro6000 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS
```

### 2.4 Truncation and concurrency per value, per card (both orders read the same)

`tw` = `truncated_with_twin`, `lb` = `length_both`, `pd` = `length_prompt_differs`; every other V-TRUNC term,
`open_non200` and every OOM and park count are 0. Every ON `length` row has G = v.

Local RTX 5090, Qwen3.5-9B, B = 32:

| v | V-ID | tw | lb | pd | burst | active max | arith | booked `pending_prime` max |
|---|---|---|---|---|---|---|---|---|
| off | | | | | 32 x 200 (27 `stop`, 5 `length` at the cap) | 11 | 7 | (no admit line) |
| 2048 | 16/16 | 18 | 3 | 15 | 32 x 200, all `length` at 2048 | 32 | 19 | 9.23 GB |
| 8192 | 48/48 | 0 | 4 | 0 | 32 x 200 (27 `stop`, 5 `length`) | 32 | 17 | 7.79 GB |
| 32768 | 48/48 | 0 | 4 | 0 | 19 x 200, 13 x 429 (Retry-After 60) | 19 | 11 | 4.91 GB |

BOX4, one RTX PRO 6000 Blackwell WS, Qwen3.8-27B, B = 64:

| v | V-ID | tw | lb | burst | active max | arith | booked `pending_prime` max |
|---|---|---|---|---|---|---|---|
| off | | | | 64 x 200 `stop` | 9 | 7 | (no admit line) |
| 2048 | 46/46 | 5 | 1 | 64 x 200 (61 `stop`, 3 `length` at 2048) | 64 | 60 | 40.83 GB |
| 8192 | 46/46 | 5 | 1 | 64 x 200 `stop` | 64 | 51 | 35.57 GB |
| 32768 | 47/47 | 4 | 1 | 44 x 200, 20 x 429 (Retry-After 60) | 44 | 32 | 22.40 GB |

- On the 27B, 2048 and 8192 cut 5 requests per order that stop under OFF at 18,443 to 193,178 tokens (the long
  class). 32768 cuts 4, all but `iv-b-r0`. The `lb` row is `iv-a-r1`, which runs to 262,143 under OFF.
- On the 9B, 8192 and 32768 cut no request that stops under OFF. 2048 cuts every (i) request.
- Against day 32 on the pre-#680 tree: every 503 is gone on both cards (day 32: 2 per order on the 5090 and 46/47 per
  order on BOX3 at 32768). The target-card comparison is across boxes, so it is a program difference only, never a
  timing one.

### 2.5 Selection rules, as printed (not chosen)

- 5090: R1 = 8192, R2 = 8192 (largest natural stop 6405), R3 = 32768, R4 = `context`.
- BOX4: R1 = none (all three admissible; they truncate 10, 10 and 8 rows over both orders), R2 = none of [2048,
  8192, 32768] (largest natural stop 193,178), R3 = 32768, R4 = `context`.

All match section 1's expectations. Which rule applies is the owner's call. The door's decide-by is 2026-10-07.

### 2.6 Owed

| item | why | price |
|---|---|---|
| The ON concurrency rows on the capped booking (`34a4b7f23`) | 2.2: these rows ran the uncapped booking; DAY35.md section 3 checks the cap on the cold R64 burst only | a rerun of this cell on the integ55 tree, 0.1 agent-day plus about 4.5 h local and 6 h on a target card, if the owner wants the capped rows |
| The door decision | the owner, on these receipts | owner |
