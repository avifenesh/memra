# WP-C day 80 (2026-09-26): OWED C12, the fix: a pinned pool compaction does not isolate, registered before any code

Lead, after DAY78: "DAY78 places the slow state on the door's pinned host pool: register the fix next (your candidate:
a pool compaction does not scan, for example a hugetlbfs or 1 GiB-page backed pool registered with cuMemHostRegister,
or pinning in a way the kernel excludes from compaction; weigh what a container allows), with its own qualification on
the 285K and a 9950X, and the door's timing re-read on it." Tree at start: `0a1d65c02`.

## 0. What DAY78's census says the fix must be

The door's pool comes from `cuMemHostAlloc(CU_MEMHOSTALLOC_PORTABLE)`, which on these hosts backs it with a shared
`/dev/zero` mapping: shmem pages on the LRU, pinned by the driver. Compaction skips a pinned page only when it is
private anonymous (its reference count above its map count and no mapping); a shmem page has a mapping, so compaction
isolates the pool's pages, fails to move them, and repeats. Three ways out, weighed against what a rented container
allows (root inside, no `CAP_SYS_ADMIN`, no host settings changed):
- **hugetlbfs or 1 GiB pages** (`MAP_HUGETLB`, then `cuMemHostRegister`): compaction leaves hugetlb pages alone, but
  the pages must be reserved on the host first (`vm.nr_hugepages`, or `hugepagesz=1G` at boot). A container cannot
  reserve them and this lane changes no host settings, so a fix that needs them does not run where the door runs.
  Excluded, stated.
- **Driver-owned pinned memory** (`cuMemHostAlloc(CU_MEMHOSTALLOC_WRITECOMBINED)`, REF's kind: `/dev/nvidiactl`
  mappings compaction never scans): write-combined memory reads on the CPU at about a tenth of a gigabyte a second, and
  the door's fill and demand checksums read the pool's bytes on the CPU (DAY47's reason for a cached pool). It would
  need every CPU read of the pool bounced through cached memory first. Not the first candidate; registered after this
  reads if this one does not clear.
- **Private anonymous memory pinned by registration** (`mmap(MAP_PRIVATE | MAP_ANONYMOUS)`, every page written, then
  `cuMemHostRegister(CU_MEMHOSTREGISTER_PORTABLE)`): cached, CPU-readable, pinned, and exactly the kind compaction's
  check skips without isolating it. Nothing a container forbids. This is the candidate.

## 1. Pre-registration: the pool kind, and the two cells (before any code)

**The flag, a diagnostic door first.** `run-gen --expert-bank-pool-registered` (with the door): the pool's
allocations come from private anonymous mappings, written once page by page, then pinned with
`cuMemHostRegister(CU_MEMHOSTREGISTER_PORTABLE)`, unregistered and unmapped on drop; everything else is the door's
program. Without the flag the pool is today's. The pool line ends ` registered`. It is a diagnostic door (decide-by
2026-10-10 in `MOE-SLOT-CACHE-DOOR.md`) until the cells below read; a default change is the owner's, on these readings.

**Cell `regpool` (the compaction question; the 285K class, then a 9950X).** DAY78's shape with the pageable arm
replaced: REF+I, D+I (today's pool), DR+I (the registered pool), 8 each under `induce-b`'s fragmentation (at least 98
GiB `MemFree` at the start), the phase probes, DAY73's sampler; plus two census runs (one D+I, one DR+I) with DAY78's
page census, not counted. Readings as DAY78's (`fail_heavy`, `slow`). Verdict `DAY80 REGPOOL VERDICT rig=<rig>`:
- `void` on integrity (DAY78's conditions); `not_run` under 98 GiB `MemFree`;
- `not_reproduced` when fewer than 3 of the 8 D+I runs are `fail_heavy`;
- `registered_clears` when at least 3 D+I runs are `fail_heavy` and no DR+I run is `fail_heavy` or `slow`;
- `registered_does_not` when at least 3 DR+I runs are `fail_heavy`;
- `undecided` otherwise. The census rows beside it name the DR+I pool's mapping (expected `rw-p [anon]`, pinned).

**Cell `regtime` (the door's timing on it; the same hosts, no fragmentation).** DAY75's timing shape: REF, D (the
door, today's pool), DR (the door, registered pool), order 1 (REF, D, DR) x 5, order 2 reversed x 5, 30 runs, one
binary. The admissibility clause first; then DR against D (`improves`, `regresses`, `flat` by `DAY61.md` section 2,
gen-only primary, the window beside it) and DR against REF (`beats`, `matches`, `loses`). Integrity: every run exit 0,
`MATCH`, one tape, the door arms' fill and `physical_reads=0`, and DR's host demand sequence equal to D's (the pool's
kind changes no decision). Verdict `DAY80 REGTIME VERDICT rig=<rig>`.

**What they decide.** `registered_clears` on the 285K and on a 9950X (or, where a 9950X host cannot give 98 GiB
`MemFree`, `regpool` `not_run` there and its `regtime` read with any natural slow boots it shows), together with
`regtime` not `regresses` on either: the registered pool removes the slow state without costing the door's timing,
and the default change (the registered pool as the door's pool, today's kept as the rollback seam) goes to the owner
with these receipts. `registered_does_not`: the write-combined pool with bounced reads is registered next. `regtime`
`regresses`: the registered pool is not free, recorded with its size, and the owner reads both.

**CPU gates before any card** (`day80-cpu/`): the flag's parse test; the pool's layout tests (the chunk plan unchanged);
the engine library and tier suites; clippy and fmt; and a local RTX 5090 check that the door with the registered pool
exits 0 with `MATCH` and the door's tape, with its pool line and a page census showing the pool as a private anonymous
mapping.
