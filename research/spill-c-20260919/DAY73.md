# WP-C day 73 (2026-09-26): OWED C12, the compaction against the door's process, registered before any cell

Lead, resuming lane C: "BOX15's machine (37297) is still off the market, so DAY71's default-rig core cell stays waiting
on it ... Then continue your owed list (C12 onward)." DAY71's default half stays registered and waits for BOX15's
machine; nothing else stands in for it under `DAY71.md` section 1 (the class line is defined on that machine and a
second one). This day registers the question machine `b`'s receipts raised, as its own cell, runnable on any Ryzen 9
9950X machine. Tree at start: `54c09a0de` plus queue v10.

## 0. What machine `b` showed (`DAY71.md` section 2), and what it cannot say

On BOX24 (a 9950X, 60 GB, kernel 5.15) every slow door run lost cycles at a full clock (12.4 to 14.2 APERF cycles per
chain step at `gate` against 6.0; MPERF over TSC 1.000), and through every door span compaction isolated 32,000 to
94,000 pages and failed to migrate nearly all of them (70,000 to 78,000 failures a second in the slow runs, 42,000 in
the one fast run whose state began at `warm`), while REF's spans read 0 from REF's third run on. `compact_stall` never
moved and kcompactd's own counters moved little, so the compaction is neither a direct-reclaim stall nor mostly the
daemon. A page that fails to migrate has been unmapped and remapped, and each unmap flushes that mapping from the TLB
of every CPU running the process, the owner's included. That fits cycles lost while the clock runs, but `b`'s
container hid `/proc/interrupts`, so the link is a candidate.

What `b` cannot say: which pages are isolated and failing (the door's 27 GB pinned pool, the artifact's mapped pages the
door keeps as bank views, or other memory of the process), what asks for the compaction, and whether the slow state on
BOX15's machine (197 GB) goes with the same counters. Unmovable pinned pages that compaction keeps isolating and
failing to migrate are one candidate; the door's large mapped views are another.

## 1. Pre-registration: the cell `compact` (any Ryzen 9 9950X machine; before its scripts)

A measurement cell; no code, no default, no host setting changes. The binary `p71=6bad38150` (the I15 door program with
DAY71's counters). Runs under DAY65's ONE pin, `--cpu-probe`, `--cpu-probe-phases`, and `--cpu-probe-counters` when
DAY71's counters check passes; the day-18 pressure shape; one collector hold.

**Part A, sampled (20 runs):** REF and the door (DAY71's arms), order 1 (REF, door) x 5, order 2 (door, REF) x 5.
DAY71's sampler, with two more row kinds, both reads of the kernel's own counters: `B` every 250 ms, `/proc/buddyinfo`
(free blocks per order per zone); `P` every 1 s, for each `run-gen` process, `/proc/<pid>/status`'s `VmRSS`, `VmLck`
and `VmPin`, and `/proc/<pid>/smaps_rollup`'s `AnonHugePages`, `Shared_Hugetlb`, `Locked` and `Swap`. At the start and
end: the kernel release and every compaction and huge-page setting readable under `/proc/sys/vm/` and
`/sys/kernel/mm/transparent_hugepage/` (read only).

**Part B, traced (4 runs, after Part A in the same hold):** REF, door, door, REF under `strace -f -c -S calls` (all
threads, counts and time per system call; the traced runs are untimed and their timing decides nothing). A host
without `strace` reads Part B `not_read` and does not void Part A.

**Readings.**
- R1, the state per door run, within the machine: `slow` when the gate probe reads at least 7.5 APERF cycles per chain
  step (6.0 in every fast state read so far; 8.1 to 8.9 in BOX15's slow boots by `compute_ns`, 12.4 to 14.2 on `b`),
  else `fast`. When the counters are unavailable, `slow` when the gate's `compute_ns` is at least 1.25 times REF's
  median gate `compute_ns`.
- R2, compaction over each door run's gate-to-window span, per second: `migrate_fail`, `compact_isolated`,
  `migrate_ok` (`pgmigrate_success`); and REF's.
- R3, the process at `gate` (the `P` sample at or before it): `VmPin`, `VmLck`, `AnonHugePages`, `Locked`, door and REF.
- R4, `/proc/buddyinfo` at each door run's gate: the free blocks of order 9 and above in the Normal zone.
- R5, Part B: per traced run the system calls by count; the ten whose counts differ most between the door's two runs
  and REF's two (medians), deciding nothing.

**The verdict** (`DAY73 COMPACT VERDICT rig=<rig>`, the strict rule of `DAY67.md`, a count beside each field):
- `void` if integrity fails (20 Part A runs, exit 0, argmax `MATCH`, 32 generated and 32 window tokens, one tape, the
  door's fill and `physical_reads=0`, the sampler's `O`, `V` and `P` rows bracketing every door run's span);
- `not_reproduced` if fewer than two `slow` or fewer than two `fast` door runs by R1;
- otherwise `migrate_fail_tracks` when every slow door run's `migrate_fail` is above every fast door run's, and
  `isolated_tracks` likewise for `compact_isolated`; else `none_tracks`.
- Beside it, deciding nothing: `ref_compacts` if any REF span from REF's third run on moves `compact_isolated`, and R3,
  R4, R5 printed.

**What it decides.** Whether the slow state goes with the compaction failures on the machine where it runs, and, from R3
to R5, what the door's process holds that REF's does not and what it asks the kernel for. A remedy (the door's host
memory placed so compaction leaves it alone, or the state's source removed) is its own registration after this reads. On
a machine where nearly every door run is slow (as on `b`, one fast in 20), the cell reads `not_reproduced` by R1 and R2
to R5 are still printed as observations.

## 1a. The sitting, prepared before any cell

`day73-sampler.py` (DAY71's sampler plus the `B` and `P` rows), `day73-cell.sh`, `day73-read.py` (which reuses
`day71-read.py`'s run parsing, sampler rows and gate counters) and `day73-box.sh` were written after section 1. Dry
checks (`day73-cpu/`): the sampler on the local host against a busy stand-in `run-gen` process writes `B` and `P` rows
beside DAY71's (`dry-check-sampler.log`); the cell's control flow with a stub `run-gen-p71` and a stub `strace` (20
sampled runs, then the four traced runs with their summaries, host settings before and after, nothing outliving the
cell; `dry-check-cell.log`); the reader on a synthetic cell built from machine `b`'s DAY71 runs with invented `B`,
`P` and strace rows, and `void` on `b`'s own cell, which has no `P` rows (`dry-check-reader.log`); the driver's
control flow (`dry-check-driver.log`).

Run as `D73_BUILDS="p71=6bad38150" [D73_RIG=<name>] bash /root/wt-c/research/spill-c-20260919/day73-box.sh` on any
Ryzen 9 9950X machine with one RTX PRO 6000 Blackwell Workstation Edition, the approved 35B artifact, `/root/wt-c` at the
lane tip and a detached `/root/wt-c-build`, root in the container, at least 48 GB `MemAvailable`, and `strace` if the
image has it. On BOX15's machine it can follow DAY71's default half in the same sitting (separate receipts roots).
Receipts land in `/root/spill-receipts/c-day73-<rig>/`. Expected: the build is shared with DAY71's when both run (else
about 5 minutes), the cell about 8 minutes (20 sampled runs, 4 traced runs slower under `strace`).

## 2. The registered cell (BOX31, a Ryzen 9 9950X, run by the lead as registered; `pro-single-day73-box31/`)

The lead ran `D73_BUILDS="p71=6bad38150" D73_RIG=box31 bash .../day73-box.sh` on BOX31 (a Ryzen 9 9950X, 123 GB, kernel
7.0.0-31, one RTX PRO 6000 Blackwell Workstation Edition, driver 595.84) on the tree `4130a4916`, right after integ65's
GPU battery on the same card, 01:30Z to `box done 2026-09-26T01:36:50Z`. Receipts: 132 of 132 `OK` against
`box-mirror-manifest.sha256` (re-checked here; `MIRROR-CHECK.txt`), `run-gen-p71` by hash. Regime (`regime.txt`): SM
median 2610 MHz, N=931. Verbatim (`compact/reading.log`):

- `DAY73 PINS rig=box31 home_l3=0-7,16-23 one=0,1,2,3,4,5,6,7,16,17,18,19 sampler_cpu=31`
- `DAY73 COMPACT CHECKS rig=box31 runs=20 counters=on integrity=ok`
- `DAY73 R5 rig=box31 (strace -- version 6.8) calls, door minus REF (medians of two): futex=+57556, pread64=+35062, sched_yield=-11194, rt_sigprocmask=+2904, sigaltstack=+2178, clock_nanosleep=+1329, ioctl=-955, mprotect=+907, sched_getaffinity=+847, clone=+726`
- `DAY73 R1 rig=box31 slow=0 fast=10 of 10 door runs (slow: gate cycles per step >= 7.5, or gate ns >= 1.25 x REF's without counters)`
- `DAY73 REF_COMPACTS rig=box31 none (REF spans from its third run on; deciding nothing)`
- `DAY73 COMPACT VERDICT rig=box31 integrity=ok -> not_reproduced`

**Read as registered: `not_reproduced`.** No door run on this 9950X entered the slow state (6.009 to 6.052 cycles per
step at the gate), and no span, door or REF, moved `compact_isolated` or `pgmigrate_fail` (0 in all 20). The door ran
0.247 to 0.249 s gen-only against REF's 0.243 to 0.244. So the 9950X class does not by itself carry the slow state:
BOX15's machine and machine `b` did, this one did not.

**What the lines show beside the verdict, deciding nothing.**
- R3: `VmPin` reads 0 in every door and REF sample, although the door holds a 27 GB pinned pool: the driver's pinned
  host memory is not accounted as `VmPin` here, so R3 cannot tell the two processes' pinned memory apart. `VmRSS` is
  about 33.4 GB for the door against 22 to 33 GB for REF.
- R4: this container exposes no `/proc/buddyinfo` (no `B` rows); the reader printed `normal_free_order9plus=None`
  where `not read` is meant. R4 decides nothing; the printing is recorded, not changed after the reading.
- R5: the door's two traced runs make 57,556 more `futex` calls and 35,062 more `pread64` calls than REF's (likely the host
  fill's reads and its workers' waits), 11,194 fewer `sched_yield`, and no system-call family that a compaction or a
  remapping would need (`madvise`, `munmap`, `mbind`, `move_pages` are not among the ten).

## 2a. A diagnostic outside the registration (BOX30, a Ryzen 9 9950X3D2; `pro-single-day73-diag-9950x3d2/`)

No plain 9950X was on the market at the time, so the lead ran the same driver on BOX30, a Ryzen 9 9950X3D2 (192 MB
L3 over two domains, 123 GB, kernel 7.0.0-30), as `D73_RIG=diag-9950x3d2`, 23:33Z to `box done
2026-09-25T23:40:21Z`: a diagnostic, not the registered class, deciding nothing. 132 of 132 receipts `OK`. Its lines:
`DAY73 COMPACT CHECKS rig=diag-9950x3d2 runs=20 counters=on integrity=ok`, `DAY73 R1 rig=diag-9950x3d2 slow=1 fast=9
of 10 door runs`, `DAY73 REF_COMPACTS rig=diag-9950x3d2 none`, `DAY73 COMPACT VERDICT rig=diag-9950x3d2 integrity=ok
-> not_reproduced`.

What it shows: its one slow door run is the sitting's first door run (`o1-i15-r1`, 7.723 cycles per step at the gate,
0.306 s gen-only against REF's 0.249 to 0.250), and it is the only span in the sitting with compaction:
`compact_isolated` 1,453,237 a second, `pgmigrate_fail` 531,032, `pgmigrate_success` 461,116 over its span. The
compaction ran from 23:36:33.6 to 23:36:38.7, inside that run (23:36:23 to 23:36:39), and in its last pass kcompactd's
own counter moved (`compact_daemon_migrate_scanned` 8,875,008). The other nine door runs read 6.010 to 6.021 cycles per
step with no compaction in their spans.

**The pattern across the three 9950X-family sittings with the sampler, stated as an observation, not a reading.** On
machine `b` the slow state and compaction in the span came together in 19 of 20 door runs; on BOX30 in 1 of 10; on
BOX31 neither appeared. No door run so far has been slow without compaction in its span, and no REF run from its
third run on has seen compaction; REF's first two runs on `b` saw successful compaction and read 7.4 to 8.4 cycles per
step later in those runs. That points to compaction running while the process runs as the trigger, and to the door's
process being the one it lands on, but no registered cell has read it: `b` and BOX31 read `not_reproduced` by rule
and BOX30 is outside the class. The next registration (`DAY74.md`) tests it directly.
