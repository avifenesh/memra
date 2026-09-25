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
