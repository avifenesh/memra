# WP-B day 39: O5, the prime workspace the admission door books, measured against what a burst actually allocates

OWED.md O5. Under `MEMRA_ADMIT_BY_MEMORY` every headroom reading of the admission block is reduced by `pending_prime`,
the prime workspace still-priming sessions owe (memra#680, day 33). It is the sum over those sessions of the whole
`prefill_workspace_bytes(rows)`: `call_row_bytes x min(rows, chunk) + prompt_row_bytes x rows`. DAY24 section 2 (ii)
already named the shape this over-books: the `call_row_bytes` term is the per-call slab, which is ONE retained
per-device pool shared by every prime (`prime_slabs_get`, grow-only at the largest call), and on the batched worker
the prime calls of a tick run one after another. On the 27B that term is most of a session's 658 MB at a 1.3k prompt
(DAY33 1.1), so a 64-request burst books it 64 times. The door's ON concurrency rows (DAY36, and O3's rerun) are read
through this term, so it is corrected first, as the door's own booking measured at its best; O3's rerun then runs on
the final booking (the reason O3 follows this day, OWED.md).

Every cell is `executed-not-qualified`; the door stays OFF; no default moves; nothing outside the armed door changes.

## 1. Pre-registration

Committed and pushed before any day-39 code and before any day-39 boot. Nothing in section 1 changes after a number
is seen.

### 1.1 What a burst's primes allocate (from the code on the lane tip)

- **The slab**: `HybridModel::prime_slabs_get(t)` keeps one slab set per device, grown when a call needs more than
  its `t_cap` rows and never shrunk; every prime call (the plain prefill tick's and the MTP walker's trunk chunks)
  uses it in turn. Its owed growth is `slab_row_bytes x max(0, rows_needed - t_cap)`, once per device, where
  `rows_needed` is the largest call any still-priming session will make.
- **The per-call transients** beside the slab (the f16 activation, the pre-norm operand, the `x` and `hn` copies, the
  call's returned `[t, n_embd]` output): allocated per call from the pool and freed after it, so at most one call's
  worth is live at a time.
- **The MTP walker's whole-prompt hiddens stack** (`MtpPrimeState::hiddens`, `tp x n_embd` f32): allocated when the
  walker starts and live until the prime is consumed. Under the cooperative prime (`MEMRA_PRIME_YIELD`, default ON)
  a walker persists across ticks, so every started walker holds its stack at once. A session whose walker has not
  started yet owes it.

### 1.2 The corrected term

Under the door only (unarmed, both terms are 0 as today):

```text
pending_prime' = max(0, (call_row_bytes + prompt_row_bytes) x rows_needed - slab_bytes_resident)   one call, once
               + sum over cooperative spec sessions whose walker has not started of prompt_row_bytes x P
               + max over non-cooperative spec sessions still priming of prompt_row_bytes x P    (a stack for one call)
rows_needed    = max over still-priming sessions of min(rows_remaining, chunk_rows(P))
```

`slab_bytes_resident` is the slab set's allocated bytes on the primary device, read from the engine through a new
accessor (`HybridModel::prime_slab_bytes`). `call_row_bytes` and `prompt_row_bytes` stay the admission shape's own
numbers, so the one-call term keeps the call's extras and its returned rows. A started walker's stack is live and
already in the reading, so it books nothing; a non-cooperative walker lives for one burst call, so only the largest
one is owed. The seed term
(`pending_seed`, capped, day 35 and #705) is unchanged. The admit, defer and refuse lines keep their fields;
`pending_prime=` prints the corrected value, and a new `pending_prime_v1=` field prints day 33's value beside it, so
every line shows both.

### 1.3 Cells, arms, binaries

The day-35 cell unchanged (`day33-client.py` on the day-33 environment, `day33-run.sh` shapes), with the red and green
binaries of this day: `red` = the lane tip at the first boot minus this day's change (the day-33/35 booking), `green`
= the tip. One detached-worktree build each per card.

| shape | door | client args | card |
|---|---|---|---|
| `G2` | ON, open output 8192 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `L64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `R64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | target card |
| `off` | removed | `--chars 5000 --skip-long --burst 0` | both |

5090 boots: `red-G2`, `red-L64`, `green-G2-r1`, `green-L64-r1`, `red-off`, `green-off`, `green-G2-r2`, `green-L64-r2`.
Target card: `red-R64`, `red-off`, `green-R64-r1`, `green-off`, `green-R64-r2`. The admission gate
`tools/admit-mem-burst-gate.sh` runs on green at its defaults once more at the end of the 5090 half.

### 1.4 Acceptance, per card

`day33-compare.py` unchanged (sha256 `4493164c...381f10`), its terms as DAY35.md 1.4 reads them: G-NOOM (0
`CUDA_ERROR_OUT_OF_MEMORY` lines, parks included, and 0 503s) and G-BOOK on every green burst boot, V-ID-FIX (green
ON against red ON, sequential rows), V-ID (green ON against green OFF) and V-OFF (green OFF against red OFF), card
verdict GREEN. The admission gate reads ALL GREEN on green.

Readings, no bound: the burst's 200 and 429 counts red against green per shape, and `pending_prime` against
`pending_prime_v1` at the burst's peak.

### 1.5 What a failure means

A green OOM line means the corrected term misses allocation that day 33's over-count was covering. The line and the
`nvidia-smi` state are quoted, the missing term is diagnosed from the log, and a revision is a new addendum; day 33's
term stays until one passes.

### 1.6 CPU

Unit tests: the slab owed growth (0 when the slab covers the need, the difference otherwise), one call counted once
over N sessions, a started walker's stack not booked, an unstarted one booked, the identity when unarmed, and the
admit line rendering both fields. The existing day-33 and day-35 tests stay green.

### 1.7 Addendum A (2026-09-24, before any day-39 boot): the binaries

The lane tip at the first boot is `c6f9282c2` (DAY37 addendum E's r4, after this day's `6262506fc`). Green is its
`memra-server`, the same file as DAY37's r4 lane binary (`580fe677...`); red is the same commit plus
`day39-red.patch`, the reversed crates diff of `6262506fc` (`0cef991c...`; the red binary carries no
`pending_prime_v1=` string, green does). This cell runs the pooled allocator, which addendum E does not touch. The
first builds from `6262506fc` never ran and are deleted. The target-card chain builds with the same script, from the
same source, reusing the day-37 lane file as green when its source matches.

### 1.8 Addendum B (2026-09-25, after the target-card half read NOT-GREEN, before any code of the revision)

The diagnosis is 2.1. The revised term, under the door only:

```text
pending_prime'' = max(0, call_row_bytes x rows_needed - slab_bytes_resident)      the slab's owed growth, once
                + prompt_row_bytes x rows_needed                                  that call's returned rows, once
                + sum over cooperative spec sessions whose walker has not started of prompt_row_bytes x P
                + max over non-cooperative spec sessions still priming of prompt_row_bytes x P
                + sum over still-priming sessions of snapshot_bytes x captures_owed
```

`captures_owed` counts the checkpoint snapshots the session's prime will still take: a plain session with `ckpt_at`
armed and no `ckpt_snap` yet owes one; a spec session whose turn checkpoint is armed (`ckpt_at` on the session before
the walker starts, the walker state's checkpoint row after) and not yet taken owes one. `snapshot_bytes` is the
session's own cache's conv and ssm bytes (the same arithmetic `prefix_seed_bytes_at` uses for the recurrent part),
exactly what `Cache::snapshot` copies. Prefix-seed captures stay in `pending_seed` (capped, day 35). `pending_prime=`
prints the revised value; `pending_prime_v1=` stays beside it.

Arms and binaries: `v3` = the revision (the tip at the first boot), `v2` = the same tree with the revision reverted
(section 1's term, the red of this addendum), `v1` = the same tree with the whole day-39 change reverted (day 35's
booking). One detached-worktree build each. 5090: `v2-G2`, `v2-L64`, `v3-G2-r1`, `v3-L64-r1`, `v2-off`, `v3-off`,
`v3-G2-r2`, `v3-L64-r2`, `v1-G2`, `v1-L64`; the admission gate on `v3` at its defaults. Target card: `v2-R64`, `v2-off`,
`v3-R64-r1`, `v3-off`, `v3-R64-r2`, `v1-R64`.

Acceptance, unchanged from 1.4, with `v3` as green and `v2` as red for `day33-compare.py` (unchanged): G-NOOM (0 OOM
lines, parks included, 0 503s) and G-BOOK on every `v3` burst boot; V-ID-FIX, V-ID and V-OFF; card verdict GREEN; the
admission gate ALL GREEN on `v3`. Readings, no bound: the burst's 200 and 429 counts for `v1`, `v2` and `v3` per shape,
and the peak `pending_prime` against `pending_prime_v1`. A `v3` OOM line is diagnosed again; day 33's term stays until
a revision passes.

CPU: unit tests of the revised term (the slab growth alone subtracts the resident slab; the returned rows are always
booked; a plain session with an armed and untaken checkpoint owes one snapshot, a taken one owes none; a spec session's
checkpoint before and after its walker starts; the unarmed identity), the existing day-33, day-35 and day-39 tests.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target-card half (one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-24 22:00:26 to 22:10:45Z)

Green `3dc05d17...` (the day-37 lane file, source `c6f9282c2`), red `62ee4d5b...` (green plus `day39-red.patch`), the
27B at the checkpoint's context. `day33-compare.py` (unchanged) and `day39-read.py`, verbatim:

```
== card pro6000
DAY33 V-BOOT boot=red-R64 arm=on32768 door_line=present -> PASS role=red shape=R64
DAY33 V-BOOT boot=red-off arm=off door_line=present -> PASS role=red shape=off
DAY33 V-BOOT boot=green-R64-r1 arm=on32768 door_line=present -> PASS role=green shape=R64
DAY33 V-BOOT boot=green-off arm=off door_line=present -> PASS role=green shape=off
DAY33 V-BOOT boot=green-R64-r2 arm=on32768 door_line=present -> PASS role=green shape=R64
== R-OOM / G-NOOM / G-BOOK (ON boots)
DAY33 R-OOM card=pro6000 boot=red-R64 role=red shape=R64 v=32768 oom_lines=0 burst={429: 20, 200: 44} first_oom=none -> NOT-RED
DAY33 G-NOOM card=pro6000 boot=red-R64 role=red shape=R64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=44 other_non200=0 r429=20 refuse_lines=20 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=pro6000 boot=red-R64 role=red shape=R64 admit_lines=60 admit_lines_in_burst=44 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=green-R64-r1 role=green shape=R64 v=32768 oom_lines=10 burst={429: 13, 200: 51} first_oom=server.log:715 1790287516425 [admit-mem] prefill OOM parked session back to queue (model q38, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") -> RED
DAY33 G-NOOM card=pro6000 boot=green-R64-r1 role=green shape=R64 oom_lines=10 burst_503=0 crash_lines=0 burst_200=51 other_non200=0 r429=13 refuse_lines=13 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=green-R64-r1 role=green shape=R64 admit_lines=77 admit_lines_in_burst=61 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=green-R64-r2 role=green shape=R64 v=32768 oom_lines=12 burst={429: 13, 200: 51} first_oom=server.log:707 1790287751398 [prefix-cache] snapshot failed (DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")); prefix not cached -> RED
DAY33 G-NOOM card=pro6000 boot=green-R64-r2 role=green shape=R64 oom_lines=12 burst_503=0 crash_lines=0 burst_200=51 other_non200=0 r429=13 refuse_lines=13 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=green-R64-r2 role=green shape=R64 admit_lines=78 admit_lines_in_burst=62 est_over_booked_free=0 -> PASS
== V-ID-FIX (green ON against red ON, same shape, sequential rows)
DAY33 V-ID-FIX card=pro6000 shape=R64 green=green-R64-r1 red=red-R64 eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID-FIX card=pro6000 shape=R64 green=green-R64-r2 red=red-R64 eligible=16 equal=16 differ=0 -> PASS
== V-ID (the day-32 term: green ON against green OFF)
DAY33 V-ID card=pro6000 shape=R64 on=green-R64-r1 off=green-off eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID card=pro6000 shape=R64 on=green-R64-r2 off=green-off eligible=16 equal=16 differ=0 -> PASS
== V-OFF (green OFF against red OFF: the default-OFF program unchanged)
DAY33 V-OFF card=pro6000 green=green-off red=red-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS
DAY33 VERDICT card=pro6000 boots=5 v_boot_all=True green_noom_book_all=False v_id_fix_all=True v_id_all=True v_off_all=True -> NOT-GREEN
DAY39 READING card=pro6000 boot=red-R64 burst_status={200: 44, 429: 20} admit_mem_lines_in_burst=92 peak_pending_prime=23053934592(other=None verdict=defer) peak_pending_prime_v1=none
DAY39 READING card=pro6000 boot=green-R64-r1 burst_status={200: 51, 429: 13} admit_mem_lines_in_burst=87 peak_pending_prime=0(other=0 verdict=admit) peak_pending_prime_v1=32592089088(other=0 verdict=defer)
DAY39 READING card=pro6000 boot=green-R64-r2 burst_status={200: 51, 429: 13} admit_mem_lines_in_burst=88 peak_pending_prime=0(other=0 verdict=admit) peak_pending_prime_v1=32926113792(other=0 verdict=defer)
DAY39 READING card=pro6000 boot=red-off burst_status={} admit_mem_lines_in_burst=0 peak_pending_prime=none peak_pending_prime_v1=none
DAY39 READING card=pro6000 boot=green-off burst_status={} admit_mem_lines_in_burst=0 peak_pending_prime=none peak_pending_prime_v1=none
```

**The card reads NOT-GREEN: G-NOOM fails on both green runs** (10 and 12 OOM lines; the first, verbatim,
`[admit-mem] prefill OOM parked session back to queue (model q38, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out
of memory")`, then `[admit-oom] step-OOM teardown complete: 10 session(s) dropped behind a device fence; pool trim
released 14560MB across 1 device(s)`). No 503, no crash line; the parked sessions retried and every burst request
answered 200 or a typed 429. The identity terms pass (V-ID-FIX, V-ID, V-OFF 16 of 16). Red (day 35's booking) has no
OOM line, as on day 35.

The diagnosis, from the log and the code, per 1.5:

- The corrected term read **0 at every admission** of the burst (`peak_pending_prime=0`), against day 33's 32.6 GB.
  The burst's sessions ran the plain path (`[spec-k] ... K=0 source=concurrency`, `path=plain`), so no walker stack is
  owed, and the resident slab already covered one call. Green admitted 51 against red's 44.
- What the primes then allocated that the reading did not hold: at the last admit the measured free was about 14.8 GB
  (`device_free=4821210520` after the booked `pending_seed=9962993664`); the card's used memory rose from 85.7 GB to
  96.5 GB (of 97.9 GB) between the end of admission and the OOM (`samples.csv`, 1 s rows), while 35 seeds of 196.8 MB
  landed (6.9 GB, booked).
- The missing term is the **plain-affinity checkpoint**: every chat-rendered plain session arms `ckpt_at`, and its
  prime takes `Cache::snapshot` at that boundary (`maybe_plain_checkpoint`), a device copy of every recurrent layer's
  conv and ssm state, held by the session until it retires. On the 27B that copy is the seed entry's recurrent part:
  196.8 MB minus 1,344 rows x 29,696 B = 156.9 MB per session, about 8.0 GB for 51 sessions. Day 33's over-count of the
  slab (611 MB per session of call rows) had been covering it. 1.1 read the prime's allocations and missed the capture.
- A second error in 1.2's term, found while diagnosing: it subtracted the resident slab from the call's slab rows AND
  its returned rows (`prompt_row_bytes x rows`), which the slab does not hold.

Addendum B (1.8) is the revision. Day 33's term stays the door's booking until the revision passes.

### 2.2 Addendum B on the target card (one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-25 05:23 to 05:36Z)

`v3` `c65f3076...` (reused from DAY38D's green, the same source), `v2` `6b06ac13...`, `v1` `28573f31...`, all from
`a803d3080`. `day33-compare.py` (unchanged, `v2` as red and `v3` as green) and `day39-read.py`, verbatim:

```
== card pro6000
DAY33 V-BOOT boot=v2-R64 arm=on32768 door_line=present -> PASS role=red shape=R64
DAY33 V-BOOT boot=v2-off arm=off door_line=present -> PASS role=red shape=off
DAY33 V-BOOT boot=v3-R64-r1 arm=on32768 door_line=present -> PASS role=green shape=R64
DAY33 V-BOOT boot=v3-off arm=off door_line=present -> PASS role=green shape=off
DAY33 V-BOOT boot=v3-R64-r2 arm=on32768 door_line=present -> PASS role=green shape=R64
== R-OOM / G-NOOM / G-BOOK (ON boots)
DAY33 R-OOM card=pro6000 boot=v2-R64 role=red shape=R64 v=32768 oom_lines=10 burst={429: 13, 200: 51} first_oom=server.log:708 1790313880872 [admit-mem] prefill OOM parked session back to queue (model q38, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") -> RED
DAY33 G-NOOM card=pro6000 boot=v2-R64 role=red shape=R64 oom_lines=10 burst_503=0 crash_lines=0 burst_200=51 other_non200=0 r429=13 refuse_lines=13 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=v2-R64 role=red shape=R64 admit_lines=77 admit_lines_in_burst=61 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=v3-R64-r1 role=green shape=R64 v=32768 oom_lines=0 burst={429: 18, 200: 46} first_oom=none -> NOT-RED
DAY33 G-NOOM card=pro6000 boot=v3-R64-r1 role=green shape=R64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=46 other_non200=0 r429=18 refuse_lines=18 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=pro6000 boot=v3-R64-r1 role=green shape=R64 admit_lines=62 admit_lines_in_burst=46 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=v3-R64-r2 role=green shape=R64 v=32768 oom_lines=0 burst={429: 18, 200: 46} first_oom=none -> NOT-RED
DAY33 G-NOOM card=pro6000 boot=v3-R64-r2 role=green shape=R64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=46 other_non200=0 r429=18 refuse_lines=18 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=pro6000 boot=v3-R64-r2 role=green shape=R64 admit_lines=62 admit_lines_in_burst=46 est_over_booked_free=0 -> PASS
== V-ID-FIX (green ON against red ON, same shape, sequential rows)
DAY33 V-ID-FIX card=pro6000 shape=R64 green=v3-R64-r1 red=v2-R64 eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID-FIX card=pro6000 shape=R64 green=v3-R64-r2 red=v2-R64 eligible=16 equal=16 differ=0 -> PASS
== V-ID (the day-32 term: green ON against green OFF)
DAY33 V-ID card=pro6000 shape=R64 on=v3-R64-r1 off=v3-off eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID card=pro6000 shape=R64 on=v3-R64-r2 off=v3-off eligible=16 equal=16 differ=0 -> PASS
== V-OFF (green OFF against red OFF: the default-OFF program unchanged)
DAY33 V-OFF card=pro6000 green=v3-off red=v2-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS
DAY33 VERDICT card=pro6000 boots=5 v_boot_all=True green_noom_book_all=True v_id_fix_all=True v_id_all=True v_off_all=True -> GREEN
DAY39 READING card=pro6000 boot=v1-R64 burst_status={200: 44, 429: 20} admit_mem_lines_in_burst=92 peak_pending_prime=22720868352(other=None verdict=defer) peak_pending_prime_v1=none
DAY39 READING card=pro6000 boot=v2-R64 burst_status={200: 51, 429: 13} admit_mem_lines_in_burst=87 peak_pending_prime=0(other=0 verdict=admit) peak_pending_prime_v1=32933781504(other=0 verdict=defer)
DAY39 READING card=pro6000 boot=v3-R64-r1 burst_status={200: 46, 429: 18} admit_mem_lines_in_burst=82 peak_pending_prime=7088517120(other=29647687680 verdict=defer) peak_pending_prime_v1=29647687680(other=7088517120 verdict=defer)
DAY39 READING card=pro6000 boot=v3-R64-r2 burst_status={200: 46, 429: 18} admit_mem_lines_in_burst=82 peak_pending_prime=7088517120(other=29639540736 verdict=defer) peak_pending_prime_v1=29639540736(other=7088517120 verdict=defer)
DAY39 READING card=pro6000 boot=v2-off burst_status={} admit_mem_lines_in_burst=0 peak_pending_prime=none peak_pending_prime_v1=none
DAY39 READING card=pro6000 boot=v3-off burst_status={} admit_mem_lines_in_burst=0 peak_pending_prime=none peak_pending_prime_v1=none
```

**The card reads GREEN.** `v3` has no OOM line on either burst run, 46 x 200 and 18 typed 429s, every admission
within the booked reading; `v2` reproduces section 1's failure (10 parked prefill OOMs, 51 x 200); `v1` (day 35's
booking) reads 44 x 200 and 20 x 429. At the burst's peak the revised term books 7.09 GB where day 33's booked
29.6 GB. The revision admits two more requests of the 64-request burst than day 35's booking, with no OOM. The
admission gate on `v3` and the 5090 half wait for the 5090's reset.
