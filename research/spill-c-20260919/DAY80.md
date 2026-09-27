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

## 1a. The flag, the sitting, before any cell

The flag landed as `57086efc8` (binary `p80`): `ExpertBankBudget` gains `pool_registered`; `expert_bank_cli` parses
`--expert-bank-pool-registered` only behind the door, without a value, once, and refuses it beside
`--expert-bank-pool-pageable`; `PinnedPool` carries a `PoolKind` (allocated, pageable, registered): a registered
allocation is an `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` written once per page, then `cuMemHostRegister_v2(...,
CU_MEMHOSTREGISTER_PORTABLE)` (unmapped again if the registration fails), unregistered then unmapped in `Drop`; the pool
line ends ` registered`. CPU gates (`day80-cpu/gates.log`): the engine library 574 passed, the tier suites green (the
parse test among them), clippy and fmt clean.

Scripts (`day80-cell.sh` with both cells, `day80-regpool-read.py` from DAY78's reader, `day80-regtime-read.py`,
`day80-box.sh`), written after section 1. Dry checks (`day80-cpu/`): both readers on synthetic cells (`regtime` from
DAY75's target receipts, `regpool` from DAY78's BOX34 receipts; meaningless; `dry-check-readers.log`); both cells'
control flow with stub binaries (30 `regtime` runs; 24 `regpool` runs and 2 census runs; `dry-check-cell.log`); the
driver running `regtime` then `regpool` (`dry-check-driver.log`).

Run as `D80_BUILDS="p80=57086efc8" [D80_RIG=<name>] bash /root/wt-c/research/spill-c-20260919/day80-box.sh` on a Core
Ultra 9 285K host, then on a Ryzen 9 9950X host, each with one RTX PRO 6000 Blackwell Workstation Edition, root in the
container, and at least 98 GiB `MemFree` when `regpool` starts (after the page-cache eviction of unused files; the
driver runs `regtime` first, which needs no eviction). Where a 9950X host cannot give 98 GiB, `regpool` reads `not_run`
there and `regtime` still reads. Expected: the build about 5 minutes, `regtime` about 8, `regpool` about 30.

## 2. The 285K half (BOX37, a Core Ultra 9 285K, run by the lead as registered; `pro-single-day80-box37-285k/`)

BOX37: one RTX PRO 6000 Blackwell Workstation Edition, 249 GB, driver 580.173.02; the page-cache eviction of unused
files before the cells (`MemFree` 206 GiB at `regpool`'s start). `D80_BUILDS="p80=57086efc8" D80_RIG=box37-285k bash
.../day80-box.sh` on the tree `f318def14`, 14:00Z to `box done 2026-09-26T14:42:30Z`. Receipts: 281 of 281 `OK` against
the box manifest (re-checked), `run-gen-p80` by hash. Regimes: `regtime` 36 to 47 C, SM median 2610 MHz, N=1154;
`regpool` 35 to 44 C, N=8283. Verbatim:

`regtime/reading.log`:
- `DAY80 host demand sequence d sha256 4bdc2610c3534e42`, `DAY80 host demand sequence dr sha256 4bdc2610c3534e42`
- `DAY80 REGTIME CHECKS rig=box37-285k runs=30 integrity=ok`
- `DAY80 ADMISSIBILITY rig=box37-285k ceiling=0.005 max_iqr_gen=0.0012 max_iqr_window=0.0013 failing=[] -> admissible`
- `DAY80 gen-only decode medians (N=10 each): ref=0.311 d=0.315 dr=0.315`
- `DAY80 STEP dr_vs_d gen-only decode: pooled=+0.0000 o1=+0.0000 o2=-0.0010 noise=0.0012 -> flat`
- `DAY80 STEP dr_vs_d steady window: pooled=+0.0000 o1=+0.0010 o2=-0.0010 noise=0.0013 -> flat`
- `DAY80 DOOR dr_vs_ref gen-only decode: pooled=+0.0035 o1=+0.0040 o2=+0.0030 noise=0.0010 -> loses`
- `DAY80 DOOR dr_vs_ref steady window: pooled=+0.0005 o1=+0.0010 o2=+0.0000 noise=0.0010 -> matches`
- `DAY80 REGTIME VERDICT rig=box37-285k integrity=ok dr=flat vs_ref=loses (window: dr=flat vs_ref=matches)`

`regpool/reading.log`:
- `DAY80 REGPOOL CHECKS rig=box37-285k runs=24 state=ns_per_step integrity=ok`
- `DAY80 ARM rig=box37-285k refi: fail_heavy=0 of 8 slow=0 of 8 compacted=3 of 8 window_ns_per_step median=1.055 gen median=0.315 window median=0.286`
- `DAY80 ARM rig=box37-285k di: fail_heavy=6 of 8 slow=5 of 8 compacted=7 of 8 window_ns_per_step median=1.456 gen median=0.360 window median=0.318`
- `DAY80 ARM rig=box37-285k dri: fail_heavy=0 of 8 slow=0 of 8 compacted=0 of 8 window_ns_per_step median=1.055 gen median=0.316 window median=0.287`
- `DAY80 REGPOOL VERDICT rig=box37-285k integrity=ok -> registered_clears (fail_heavy: di 6 of 8, dri 0 of 8, refi 0 of 8; slow: dri 0 of 8)`

**Read as registered: `registered_clears` and `dr=flat`.** Under the same fragmentation, the door with today's pool
had failing compaction in 6 of 8 runs and was slow in 5 (1.41 to 1.67 ns per chain step against 1.055); with the
registered pool no run had any compaction at all and every run read 1.054 to 1.063, the same as REF. Its pool is a
15,055,928 kB `rw-p [anon]` mapping in the census, where today's is the `rw-s /dev/zero (deleted)` one. And it costs
the door nothing in the normal regime: DR against D is `flat` on both measures with the same host demand sequence
(0.315 s gen-only, 0.286 window). Under fragmentation it also gives back the time the slow state took: DR+I's gen-only
median 0.316 against D+I's 0.360. The door against REF is this host's usual reading (`loses` gen-only by 3.5 ms over
32 tokens, `matches` the window), unchanged by the pool.

## 2a. The local check on the development host's RTX 5090 (under its lock; `day80-cpu/gpu-check.log`)

The door and the door with `--expert-bank-pool-registered`, 256 generated tokens each: both exit 0 with `MATCH` and the
same tape; the registered run's pool line ends ` registered`; the page census of each run's own process shows the pool
as `rw-s /dev/zero (deleted)` for today's pool and `rw-p [anon]` (15,055,924 kB resident) for the registered one; the
decode times are the same (2.789 s and 2.820 s for 256 tokens, one run each).

## 2b. The RTX 5090's `regtime` (queue v15; `rtx5090-day80/regtime/`)

Verbatim: `DAY80 ADMISSIBILITY rig=rtx5090 ... -> inadmissible` (every arm above the ceiling; the card ran 64 to 85 C
with other lanes' work between holds), `DAY80 REGTIME VERDICT rig=rtx5090 integrity=ok -> void (inadmissible) [as read:
dr=flat vs_ref=matches (window: dr=flat vs_ref=matches)]`. Decides nothing; the same host demand sequence held, and as
read DR is not slower than D (0.382 against 0.383 gen-only, 0.330 against 0.334 window).

## 3. The 9950X half, by section 1, and one reading added before it runs

Lead: no RTX PRO 6000 Workstation offer with a 9950X and more than 123 GB is on the market, and on 123 GB `regpool`
cannot reach 98 GiB `MemFree` with the 35B cached (about 93 GiB at best). By section 1 ("or, where a 9950X host cannot
give 98 GiB `MemFree`, `regpool` `not_run` there and its `regtime` read with any natural slow boots it shows"), a 123
GB 9950X host is a valid 9950X half: the driver runs `regtime`, then `regpool` reads `not_run`, and `regpool` on a 9950X
is not owed by this registration. The 98 GiB floor is `DAY74.md` section 4's and no other way to meet it was
registered; it does not move.

**Registered now, before the 9950X half runs: how "any natural slow boots it shows" is read.** A 9950X host can show the
door's natural slow boots (BOX15's machine did in every door arm, `DAY64.md` section 4), and then the admissibility
clause voids `regtime`'s step and door readings (a bimodal D arm spreads past 0.005 s), as the clause intends. So
`day80-regtime-read.py` gains one reading and one verdict line, the same on every host: per door arm, the runs slower
than REF's median gen-only by more than 0.030 s (DAY64's and DAY70's slow mark); `DAY80 NATURAL rig=<rig> ... ->`
`no_natural_slow` when fewer than 2 of D's 10 runs are slow, `registered_clears_natural` when at least 2 are and none of
DR's are, `registered_does_not_natural` when at least 2 of DR's are, `undecided` otherwise. On BOX37 it reads
`no_natural_slow` (0 of 10 in both arms); on DAY64's BOX15 receipts relabelled (a mechanics check, meaningless as a fix
reading) it counts 6 of 10 (`day80-cpu/dry-check-natural.log`).

**What the 9950X half decides, then.** `regtime` admissible and not `regresses` there, or, where natural slow boots void
it, `registered_clears_natural`: the section 1 condition holds on the 9950X, and section 4's question goes to the owner
complete. `registered_does_not_natural`: the registered pool does not clear the 9950X's natural state, recorded, and the
question goes with that reading.

Run as `D80_BUILDS="p80=57086efc8" D80_RIG=<name> bash /root/wt-c/research/spill-c-20260919/day80-box.sh` with
`/root/wt-c` at the lane tip (the new reading is in the reader only; the binary and the cells are the 285K half's) on a Ryzen 9 9950X host with one RTX PRO 6000 Blackwell Workstation
Edition, 123 GB is enough, root in the container, no page-cache eviction needed. About 15 minutes (the build, `regtime`,
`regpool` reading `not_run` at once).

## 4. The owner's question (prepared now; complete when the 9950X half reads)

**Proposed: the door's host pool made from private anonymous memory pinned with `cuMemHostRegister` (today's
`--expert-bank-pool-registered`) becomes the door's default pool, and today's `cuMemHostAlloc(PORTABLE)` pool stays
behind a rollback flag (`--expert-bank-pool-allocated`, a door with a decide-by, deleted after two weeks unused per the
door rules).** The door itself stays default-off; this changes only the pool the door uses.

What it rests on:
- **The cause, placed** (`DAY78.md` section 2): under fragmented host memory, compaction isolates the pages of today's
  pool (shared `/dev/zero` memory the driver pins) and fails to move them, again and again, and the door's core runs 1.4
  to 1.7 times slower per instruction while it does (the C12 slow state seen on BOX15's 9950X and on machine `b`). With
  the pool made of pageable heap memory the state is gone (`pool_draws`: 8 of 8 against 0 of 8).
- **The fix, measured** (section 2): with the registered pool, the same fragmentation produces no compaction against
  the door at all (`registered_clears`: 6 of 8 failing runs with today's pool, 0 of 8 with the registered one, every
  registered run at REF's core speed), and in the normal regime the door's timing is unchanged (`dr=flat` on both
  measures, the same host demand sequence, admissible). The local RTX 5090 check reads `MATCH`, the same tape and the
  same decode time; the 5090's timing cell is inadmissible and, as read, not slower.
- **What it costs:** the pool's pages are written once at install (15 GB of page faults before the registration), the
  registration pins them; nothing changes after install. No host setting, no privilege.
- **What it does not change:** the door still `loses` to REF gen-only on these hosts (C11's gap, 3.5 to 10 ms over 32
  tokens by host); this fix removes the host-memory-dependent slowdown, not that gap.
Owed before the question is complete: the 9950X half (section 3). The implementation of the default change follows the
owner's answer: the flip, the rollback flag and its decide-by row, the door doc, one qualification sitting of the
flipped default on the 285K class.
