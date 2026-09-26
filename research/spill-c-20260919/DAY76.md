# WP-C day 76 (2026-09-26): OWED C12, the door's one large pinned allocation and the compaction, before any code

`DAY74.md` section 5 closes: across three hosts and both CPU classes the slow state is compaction running against the
door's process while it decodes, and it starts under fragmented or scarce free memory during the door's load, never
during REF's. Tree at start: `985611f34`.

## 0. What the door's process holds that REF's does not

Read from source before any cell:
- **The host tier's pinned pool.** `PinnedPool::new` (`banked_residency/native.rs`, DAY47) makes one
  `cuMemHostAlloc(CU_MEMHOSTALLOC_PORTABLE)` of the whole host plan (the 16 GiB budget these cells request; section 1a
  corrects the size), cached, carved into buffers. REF has no host tier; its pinned host copy of the experts
  is many `cuMemHostAlloc(CU_MEMHOSTALLOC_WRITECOMBINED)` allocations, one per stacked expert tensor (cudarc
  `alloc_pinned`, `model.rs`).
- **Its reads of the artifact.** The installer reads the whole artifact once to hash it and again (record by record)
  to fill the pool, both by `pread` through the page cache. REF reads it once, through its mapping, at load.
The first is one very large pinned allocation made while free memory is fragmented; how the driver backs a 27 GB
pinned request (and what the kernel does to satisfy it) is not visible from user space. The test is to make the same
pool out of smaller allocations and see whether the compaction goes away. If it does not, the artifact reads are next.

## 1. Pre-registration: the diagnostic flag and the cell `chunk` (the 285K class; before any code)

**The flag (log only in effect, a diagnostic door).** `run-gen --expert-bank-pool-chunk-bytes=<N>` (with the door):
the pinned pool is made of allocations of at most `N` bytes each (whole buffers per allocation, each class's buffers
spread over as many allocations as needed; the same flags, the same buffers, the same zeroing, the same headroom);
without it the pool is the one allocation it is today, byte-for-byte the same program. Its decide-by is 2026-10-10 in
`MOE-SLOT-CACHE-DOOR.md`; it goes when C12 closes, or becomes the default by its own registration if it clears the
state and qualifies.

**The cell `chunk`.** One binary `p76` (the lane tip after the flag), on a Core Ultra 9 285K host with one RTX PRO 6000
Blackwell Workstation Edition (the class where `induce-b` reproduced the state in 3 of 6 induced door runs), with
`induce-b`'s fragmentation before every run (all but 2 GiB of free memory in order-0 holes, the inducer's loop from
the gate, the artifact reread before every run; `DAY74.md` section 4), DAY73's sampler and the phase probes. Arms:
REF+I (REF), D+I (the door, one allocation), DC+I (the door with `--expert-bank-pool-chunk-bytes=268435456`, 256 MiB
allocations). Order 1 (REF+I, D+I, DC+I) x 4, order 2 reversed x 4, 24 runs. At the gate of each arm's first run the
cell also saves the process's `/proc/<pid>/smaps` (`ev/<label>.smaps`, a VMA census, deciding nothing).

**Readings per run:** the state at `window` (APERF cycles per step where the counters run, else the probe's
nanoseconds per step), and whether the span had compaction (`compact_isolated` moved). A run is `slow` when its window
state is at least 1.25 times REF+I's median.

**The verdict** (`DAY76 CHUNK VERDICT rig=<rig>`):
- `void` on integrity (24 runs, exit 0, `MATCH`, one tape across arms, the door arms' fill and `physical_reads=0`, the
  sampler bracketing every span); `not_run` when `MemFree` at the start is under 98 GiB;
- `not_reproduced` when fewer than 3 of the 8 D+I runs had compaction in their span;
- `chunk_clears` when 0 of the 8 DC+I runs had compaction and 0 are `slow`;
- `chunk_does_not` when at least 3 of the 8 DC+I runs had compaction;
- `undecided` otherwise. REF+I's compaction count and each arm's gen-only and window medians are printed beside it.

**What it decides.** `chunk_clears`: the one large pinned allocation draws the compaction; the chunked pool becomes a
remedy candidate, whose default change needs its own timing and gate qualification in the unfragmented regime on both
cards. `chunk_does_not`: the allocation's size is not the cause; the artifact reads are the next candidate, registered
after. It changes no default.

## 1a. The flag, the sitting, and one correction, before any cell

The flag landed as `7a162e6b7` (binary `p76`): `ExpertBankBudget` gains `pool_chunk_bytes`; `expert_bank_cli` parses
`--expert-bank-pool-chunk-bytes=<N>` only behind the door, positive, once; `PinnedPool::new` takes it (without it the
one allocation, the same pointer arithmetic as before; with it each class's buffers in allocations of at most `N`
bytes, whole buffers each, freed together); the pool line gains ` chunk_bytes=<N> allocations=<k>` only when the flag is
given. CPU gates (`day76-cpu/gates.log`): the engine library 574 passed, the tier bank tests 82 passed (the budget
struct is compiled there too; a new test pins the flag's parse), a new unit test pins the chunk plan, clippy and fmt
clean. The door row in `MOE-SLOT-CACHE-DOOR.md` carries the flag's decide-by 2026-10-10.

**Correction, stated before any cell.** Section 0's "about 27 GB" for the pool was wrong: with the 16 GiB host budget
these cells request, the pool line reads `bytes=15417016320` (15.4 GB) in the DAY74 receipts. The question is the same.

**Scripts and dry checks** (`day76-cpu/`): `day76-cell.sh` (DAY74 section 4's run shape; every run induced; the three
arms; the smaps saved at each arm's first gate), `day76-read.py`, `day76-box.sh`. The reader on a synthetic cell
relabelled from BOX29's `induce-b` runs (`dry-check-reader.log`, meaningless); the cell's control flow with stub
binaries (`dry-check-cell.log`: 24 runs, each with its inducer; the stub is a script, so its process is named `bash`
and the smaps capture finds nothing there); the smaps capture itself against a real ELF named `run-gen-p76`
(`dry-check-smaps.log`: found, 46 VMAs saved); the driver (`dry-check-driver.log`).

Run as `D76_BUILDS="p76=7a162e6b7" [D76_RIG=<name>] bash /root/wt-c/research/spill-c-20260919/day76-box.sh` on a Core
Ultra 9 285K host with one RTX PRO 6000 Blackwell Workstation Edition, at least 98 GiB `MemFree` at the start (after
the page-cache eviction of unused files), root in the container. Expected: the build about 5 minutes, the cell about 25
(24 runs, each with a fragmentation setup of about 38 s at 120 GiB).

## 2. The cell on the 285K class (BOX32, BOX29's machine, run by the lead as registered; `pro-single-day76-box32-285k/`)

After a page-cache eviction of unused files (`MemFree` 127 GiB at the start), `D76_BUILDS="p76=7a162e6b7"
D76_RIG=box32-285k bash .../day76-box.sh` on the tree `1d4f94cba`, 11:08Z to `box done 2026-09-26T11:31:37Z`, after
DAY77's cell on the same card. Receipts: 133 of 133 `OK` (re-checked), `run-gen-p76` by hash. Verbatim
(`chunk/reading.log`):

- `DAY76 INDUCER rig=box32-285k memfree_gib=127 F_gib=125 induce=1`
- `DAY76 CHUNK CHECKS rig=box32-285k runs=24 state=ns_per_step integrity=ok`
- `DAY76 POOL dci: bytes=[15417016320] allocations=[58]`
- `DAY76 ARM rig=box32-285k refi: compacted=1 of 8 slow=0 of 8 window_ns_per_step median=1.112 gen median=0.258 window median=0.230`
- `DAY76 ARM rig=box32-285k di: compacted=4 of 8 slow=2 of 8 window_ns_per_step median=1.114 gen median=0.298 window median=0.242`
- `DAY76 ARM rig=box32-285k dci: compacted=5 of 8 slow=3 of 8 window_ns_per_step median=1.138 gen median=0.313 window median=0.261`
- `DAY76 CHUNK VERDICT rig=box32-285k integrity=ok -> chunk_does_not (di compacted 4 of 8, dci 5 of 8, refi 1 of 8)`

**Read as registered: `chunk_does_not`.** The pool in 58 allocations of at most 256 MiB draws compaction as often as
the one allocation (5 of 8 against 4 of 8) and its runs are as slow. The allocation's size is not the cause; section 1
names the artifact reads as the next candidate.

**What the per-run lines show beside the verdict, deciding nothing, and it refines what "the state" is.** Every slow
run had compaction, as before, but not every run with compaction was slow, and the difference is how compaction ended:
- the six runs that read 1.34 to 1.70 ns per step (four slow, one partly at 1.336, plus `o2-dci-r3` at 1.447) failed to
  migrate nearly every page they isolated: `pgmigrate_fail` 431,116 of 433,405 isolated a second, 797,018 of 797,018,
  775,270 of 774,957, 768,350 of 769,547, 576,879 of 702,172, and 494,444 of 1,204,963 in the partly slow one;
- the four runs with compaction that stayed at 1.113 to 1.160 migrated most of what they isolated: 217,943 failed of
  1,549,139, 322,062 of 1,349,029, 272,064 of 1,509,450, and REF's one compacted run 20,712 of 1,413,184.
So the slow state goes with compaction that keeps isolating pages it cannot move, not with compaction as such: in the
door's process something holds a large set of pages that compaction takes off the LRU and fails to migrate, again and
again. REF's one compacting run moved its pages and did not slow. What those pages are is the next question; the
pinned pool's allocation granularity is ruled out here.
