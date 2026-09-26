# WP-C day 76 (2026-09-26): OWED C12, the door's one large pinned allocation and the compaction, before any code

`DAY74.md` section 5 closes: across three hosts and both CPU classes the slow state is compaction running against the
door's process while it decodes, and it starts under fragmented or scarce free memory during the door's load, never
during REF's. Tree at start: `985611f34`.

## 0. What the door's process holds that REF's does not

Read from source before any cell:
- **The host tier's pinned pool.** `PinnedPool::new` (`banked_residency/native.rs`, DAY47) makes one
  `cuMemHostAlloc(CU_MEMHOSTALLOC_PORTABLE)` of the whole host plan (about 27 GB for the 35B: 30720 slots of up to
  860,160 bytes plus headroom), cached, carved into buffers. REF has no host tier; its pinned host copy of the experts
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
