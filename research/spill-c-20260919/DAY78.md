# WP-C day 78 (2026-09-26): OWED C12, which of the door's pages compaction takes and cannot move, before any code

Lead: "register C12 (which of the door's pages compaction isolates and cannot migrate; the pinned pool is the prime
suspect; candidate fixes a pool compaction does not scan, or the door's other held memory) with a cell that identifies
the pages directly (for example /proc/<pid>/pagemap plus kpageflags, or page_owner if the box allows it; say what a
container permits)". Tree at start: `2983d31d7`.

## 0. What the receipts say, and what a container lets a cell see

`DAY76.md` section 2: the slow runs are the ones whose compaction fails to migrate nearly every page it isolates, in
the door's process, while REF's rare compaction moves its pages. Something in the door holds a large set of pages
that compaction takes off the LRU and cannot move.

What an unprivileged container (root in the container, no `CAP_SYS_ADMIN`, the default capability set these boxes
give) can read, from the kernel's own rules:
- `/proc/<pid>/pagemap` of its own processes: per virtual page, present, swapped, file-backed or shared-anonymous, and
  exclusively mapped; the page frame number reads 0 without `CAP_SYS_ADMIN`.
- `/proc/kpageflags` and `/proc/kpagecount` (per page frame): mode 0400 root, readable by the container's root when
  not masked; useless for a process's pages without the frame numbers, but a host-wide census of flags (LRU,
  unevictable, mlocked, compound) is possible.
- `page_owner` needs `page_owner=on` at boot and debugfs; tracepoints (`compaction`, `migrate`) need tracefs. Neither
  is expected in a container.
- `/proc/<pid>/smaps` and `numa_maps`: per mapping, resident and locked sizes, anonymous or file, huge pages, flags.
So the cell probes each of these and records what it could read; where frame numbers are visible it identifies the
pages directly; everywhere it identifies them by elimination, with an arm that removes the prime suspect.

## 1. Pre-registration: the diagnostic flag and the cell `pages` (the 285K class; before any code)

**The flag (a diagnostic door).** `run-gen --expert-bank-pool-pageable` (with the door): the host tier's pool buffers
come from ordinary heap memory (zeroed, 4 KiB aligned) instead of `cuMemHostAlloc`; everything else is the door's
program (the same buffers, headroom, fill, verification, leases); the copies then leave from pageable memory, which the
driver stages through its own pinned buffers (slower copies; the timing decides nothing here). Without the flag the
pool is today's. Decide-by 2026-10-10 in `MOE-SLOT-CACHE-DOOR.md`; it goes when C12 closes.

**The cell `pages`.** One binary `p78` (the lane tip after the flag), a Core Ultra 9 285K host with one RTX PRO 6000
Blackwell Workstation Edition, every run under `induce-b`'s fragmentation (`DAY74.md` section 4, at least 98 GiB
`MemFree` at the start), DAY73's sampler, the phase probes. Arms: REF+I, D+I (the door), DP+I (the door with the
pageable pool). Order 1 (REF+I, D+I, DP+I) x 4, order 2 reversed x 4: 24 timed runs. Then six census runs, not
counted (one REF+I, one D+I, one DP+I, in each order), each with a page census of the `run-gen` process taken by a
separate reader process (`day78-pages.py`, pinned outside the run's CPUs) at the gate and 0.5 s later: per mapping, its
size, resident pages, file or anonymous, locked, huge pages, and from `pagemap` the present and exclusively mapped
counts; where frame numbers read nonzero, each mapping's frame numbers hashed (so a mapping whose frames changed
between the two passes has had pages migrated) and joined to `/proc/kpageflags` for flag counts per mapping. The
capability probe (`ev/access.txt`) records `CapEff`, whether frame numbers read nonzero, and whether `kpageflags`,
`page_owner` and tracefs are readable.

**Readings per timed run:** the window state (APERF cycles per step where the counters run, else the probe's
nanoseconds per step), and the span's compaction: `compact_isolated`, `pgmigrate_fail`, and `fail_heavy` when
`compact_isolated` > 0 and `pgmigrate_fail` / `compact_isolated` >= 0.5 (the slow runs of DAY76 read 0.41 to 1.00,
the fast compacted ones 0.01 to 0.24).

**The verdict** (`DAY78 PAGES VERDICT rig=<rig>`):
- `void` on integrity (24 timed runs and 6 census runs exit 0 and `MATCH`, one tape, the door arms' fill and
  `physical_reads=0`, the sampler bracketing every span); `not_run` under 98 GiB `MemFree`;
- `not_reproduced` when fewer than 3 of the 8 D+I runs are `fail_heavy`;
- `pool_draws` when at least 3 D+I runs are `fail_heavy` and no DP+I run is;
- `pool_does_not` when at least 3 DP+I runs are `fail_heavy`;
- `undecided` otherwise. Beside it, deciding nothing: slow counts per arm, REF+I's compaction, the census tables.

**What it decides.** `pool_draws`: the pinned pool's pages are the ones compaction takes and cannot move, and the
remedy is a pool compaction leaves alone (its own registration: the candidates are a pinned pool from huge pages, or
from memory outside the movable LRU); `pool_does_not`: the pool is not the cause, and the census names the door's
other held mappings for the next registration. It changes no default.
