# WP-C day 47 (2026-09-24): the MoE slot cache door, improvement I2: bounded pinned host slots

`OWED.md` C1 step (b). After the fill (day 45) and I1 (day 46) a GPU miss is served from a host-resident lease without
a drain, and what the door still pays per miss beyond the legacy program is the H2D from pageable memory: day 40 read
`enqueue` 1.08 ms and `copy_gpu` 2.83 ms per window token (92.3 copies, about 12 us to enqueue and 31 us on the GPU
each; the driver stages every pageable copy through its own pinned bounce buffer), and every host miss allocates and
zero-fills two `Vec`s and copies the read once more (`alloc` 0.75, `step` minus `pread` 1.58 ms per token; the
day-43 dry check showed those allocations grow with the tier). Written before any I2 code; tree at start:
`7455bfa32` (I6, I9, the fill, I1).

## 1. Pre-registration

**The design.**

- (a) **One pinned pool per install.** The gate allocates, once, cached pinned host memory (`cuMemHostAlloc`
  `CU_MEMHOSTALLOC_PORTABLE`, never write-combined: the fill and the verify read these bytes on the CPU, and
  write-combined reads run at about 0.1 GB/s on both hosts, `DAY18.md` hash micro-cell) and carves it into fixed
  buffers per host-plan class: the planned slots of the class plus a headroom of 106 buffers (the 33 open leases I1
  allows, the fill's 64 queued and 8 in its workers, and one), so an evicted lease whose copy is still in flight never
  shares memory with a new record. A buffer is zeroed once, the first time the pool hands it out; after that it is
  always initialized.
- (b) **The tier's buffer hook.** `memra-tier` gains `HostBuffer` (fixed bytes a lease can own; dropping it returns it
  to its pool) and `HostBufferSource`, and `BankService::with_host_buffers(source)`. With a source installed,
  `ReadWork::new` takes each output from it instead of allocating a `Vec`, and `ReadWork::step` reads an extent that
  covers exactly one output's single segment straight into that output (no slot `Vec`, no assembly copy); other
  extents keep today's path. The verify reads the same buffer. A lease's backing is then the pooled buffer, and
  `ExpertBankProxy::with_bytes` lends its bytes exactly as it lends a `Vec`'s. Without a source nothing changes.
- (c) **The fill takes from the same pool**, never below a reserve of 34 buffers per class (so a demand's read always
  finds a buffer); a fill that finds none waits for one or stops with the fill.
- (d) **The H2D** now reads cached pinned memory (`stage_expert` over the buffer): a DMA with no driver staging copy.
- (e) **Release.** A buffer returns to its free list only when its lease's backing is retired, which after I1 is
  after every ticket on it has retired and so after its copy's event: never while a DMA can read it. An exhausted
  class refuses the demand (`Capacity`), never allocates.

**Correctness.** Same bytes, same kernels: a pooled buffer holds exactly the record the read or the fill put there,
verified as before. CPU tests: a pool buffer is returned exactly once and only after the lease retires; a read
straight into a pooled output verifies and leases the same bytes the `Vec` path would (same digest, same content); an
exhausted class refuses and allocates nothing; the fill leaves the demand reserve.

**The cell `pinned` (RTX 5090 first).** Day 46's shape and budget; arms OFF (the I2 binary, no door), I1 (the day-46
binary), I2 (the I2 binary); order 1 (OFF, I1, I2) x 5, order 2 reversed x 5, one collector hold. Integrity as day 46's.

**Clauses.** `noise` as before.
- (i) **I2 beats I1 in the window**: `median(I2 window) < median(I1 window) - noise` in both orders.
- (ii) **The enqueue is a DMA enqueue**: I2's `enqueue` per window token below half of I1's.
- A reading, direction registered: I2's `copy_gpu` per window token below I1's.
I2 stays if (i) and (ii) hold; (ii) holding with (i) failing is recorded as the copies not being what the window paid.

**What each card can decide.** The RTX 5090 decides (i) and (ii) here; the target card reads them in the ladder (its
host is a different CPU and PCIe generation; nothing is compared across cards).
