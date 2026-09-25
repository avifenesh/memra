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

### 2.1 The target card (the third sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-25 06:53 to 12:06Z)

`S40` = `b46ae200e`, sha256 `c3387e1b...b51d20a0`, built on the box; receipts at `pro-single-day40/box/` (the box
manifest checked, the binary by hash only). `day34-compare.py` and `day40-read.py` ran on the box; the verdict lines
are in `SUMMARY.txt` and `READINGS.txt`. Verbatim, the door line and the readings:

```
DAY34 V-DOOR card=pro6000 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS
== SELECT (stated, not chosen)
DAY34 SELECT card=pro6000 R1_smallest_zero_truncation=none admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:10', '8192:10', '32768:8']
DAY34 SELECT card=pro6000 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768] (max_natural_G=193178)
DAY34 SELECT card=pro6000 R3_registry=32768 R4_survey=context
== V-CONC / V-RETRY
DAY34 V-CONC card=pro6000 order=O1 arm=off window_ms=152236 B=64 status={200: 64} finish={'stop': 64}
DAY34 V-CONC card=pro6000 order=O1 arm=on2048 window_ms=120070 B=64 status={200: 64} finish={'stop': 61, 'length': 3}
DAY34 V-CONC card=pro6000 order=O1 arm=on8192 window_ms=125170 B=64 status={200: 64} finish={'stop': 64}
DAY34 V-CONC card=pro6000 order=O1 arm=on32768 window_ms=92288 B=64 status={429: 18, 200: 46} finish={'stop': 46}
DAY34 V-CONC card=pro6000 order=O2 arm=off window_ms=150557 B=64 status={200: 64} finish={'stop': 64}
DAY34 V-CONC card=pro6000 order=O2 arm=on2048 window_ms=117524 B=64 status={200: 64} finish={'stop': 61, 'length': 3}
DAY34 V-CONC card=pro6000 order=O2 arm=on8192 window_ms=125281 B=64 status={200: 64} finish={'stop': 64}
DAY34 V-CONC card=pro6000 order=O2 arm=on32768 window_ms=79668 B=64 status={429: 18, 200: 46} finish={'stop': 46}
DAY40 READING card=pro6000 boot=O1-off burst_status={200: 64} admit_mem_lines_in_burst=0 seed_cap_bound_lines=0 max_cap_cut_bytes=0 peak_pending_prime=None
DAY40 READING card=pro6000 boot=O1-on2048 burst_status={200: 64} admit_mem_lines_in_burst=64 seed_cap_bound_lines=49 max_cap_cut_bytes=9568841728 peak_pending_prime=(9755701248, '40816668672', 'admit')
DAY40 READING card=pro6000 boot=O1-on32768 burst_status={200: 46, 429: 18} admit_mem_lines_in_burst=82 seed_cap_bound_lines=26 max_cap_cut_bytes=5179748352 peak_pending_prime=(7088517120, '29637623808', 'defer')
DAY40 READING card=pro6000 boot=O1-on8192 burst_status={200: 64} admit_mem_lines_in_burst=64 seed_cap_bound_lines=49 max_cap_cut_bytes=9766656000 peak_pending_prime=(9755701248, '40829607936', 'admit')
DAY40 READING card=pro6000 boot=O2-off burst_status={200: 64} admit_mem_lines_in_burst=0 seed_cap_bound_lines=0 max_cap_cut_bytes=0 peak_pending_prime=None
DAY40 READING card=pro6000 boot=O2-on2048 burst_status={200: 64} admit_mem_lines_in_burst=64 seed_cap_bound_lines=49 max_cap_cut_bytes=9766656000 peak_pending_prime=(9755701248, '40831524864', 'admit')
DAY40 READING card=pro6000 boot=O2-on32768 burst_status={200: 46, 429: 18} admit_mem_lines_in_burst=82 seed_cap_bound_lines=26 max_cap_cut_bytes=5179748352 peak_pending_prime=(7088517120, '29646729216', 'defer')
DAY40 READING card=pro6000 boot=O2-on8192 burst_status={200: 64} admit_mem_lines_in_burst=64 seed_cap_bound_lines=49 max_cap_cut_bytes=9766656000 peak_pending_prime=(9755701248, '40833921024', 'admit')
```

- **Every DAY34 1.6 term PASS on all eight boots** (V-CRASH, V-ID, V-TRUNC band and conservation, V-ALLOC, V-RETRY,
  V-OOM, G-BOOK, PARK), and `V-DOOR ... -> PASS`. Every cell stays `executed-not-qualified`.
- **SELECT, stated as the rules read, not chosen:** R1 selects none (every arm truncates: 10, 10 and 8 rows at 2,048,
  8,192 and 32,768); R2 selects none (the largest natural completion is 193,178 tokens, above every value); R3's
  registry value is 32,768; R4 is the survey (context).
- **The seed cap bound** on every ON boot: 49 of 64 burst lines at `on2048` and `on8192`, 26 of 82 at `on32768`; the
  largest cut 9.77 GB (5.18 GB at `on32768`).
- **The prime booking:** the burst's peak `pending_prime` 9.76 GB (7.09 GB at `on32768`) against `pending_prime_v1`
  40.8 GB (29.6 GB).
- **Against day 36 (target card):** `off`, `on2048` and `on8192` admit 64 of 64 on both days; `on32768` admits 46 and
  refuses 18 in both orders here, against 44 and 20 on day 36. The final booking admits two more sessions at
  `on32768`.
- **FAULTS.txt on the box is a traceback** (`FileNotFoundError: ... b-day40/boots/server.log.gz`): my chain passed the
  boots root to `day31-faults.py` instead of each boot's directory. The lister, not a verdict, ran locally on the
  mirrored boots (`FAULTS-local.txt`): no panic, respawn, fatal or engine error on any boot; the only non-200 rows are
  the 18 typed 429s at `on32768` in each order.

The 5090 half is queued in `rtx5090-queue-e.sh`.
