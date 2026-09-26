# WP-A day 63: OWED items 14 and 19, the host tier's pinned memory on the owner thread (design L, one lease pool)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order. Item 17 is
re-read on top of it (the lead's ruling after DAY52).

## 1. Pre-registration (committed before any code)

**The prices on record, all on the owner thread (the tenant's tick):**

- Frees (item 14): a publication that replaces a long host entry drops its 32 pinned leases in 8.6 ms (DAY52 section
  3, about 270 us per `cuMemFreeHost`, which waits for every stream's work in the context); P's reserve made those frees
  about 1 ms slower (DAY51, DAY52: `placed in insert, kv`).
- Allocations (item 19): a long demote's pre-submit allocates its 32 KV leases fresh in 18 to 26 ms (DAY49 section 3,
  DAY51 section 3, DAY54 section 3), `alloc_host` zero-filling each new backing on the owner thread; the first demote of
  a context allocates the whole staging set in 20 to 78 ms (`spans`, DAY49 and DAY54).
- Read from the code: a new entry's leases are allocated at its pre-submit, before its publication evicts the entry it
  replaces, so in the steady state the memory freed by one publication is exactly what the next demote allocates.

**Design L, in two parts.**

- L1, a pinned backing pool in the transfer engine (`CudaTransfers`): a `CudaPinnedLease`'s backing returns to the pool
  when the lease drops (after its own tracking event completes; no `cuMemFreeHost`), and `alloc_host` takes a pooled
  backing of the request's size class before it allocates one. Size classes: at or below 1 MiB the next power of two,
  above it the next 1 MiB multiple; the lease's length is the request's, and nothing past it is ever exposed (the D2H
  writes the lease's range before any read, as today). A reused backing is not zero-filled (its bytes are initialized:
  a previous lease's). The pool's idle bytes are charged on the pinned ledger under a pool tenant, with the ledger's
  pinned capacity gaining one term (one budget), and capped at one budget; a backing past the cap is freed as today.
  The pool frees at the tier's latch and at shutdown.
- L2, the staging set at boot: the context's span staging set (one buffer per recurrent plane length of each loaded
  model, known at boot) is allocated when the context is built, under the same charge as today, so a context's first
  demote does not allocate it on the tick.
- Not in L (stated): the growth phase, before any entry has left the tier, still allocates fresh on the owner; a
  background allocator for it is the next step if L's readings show the growth phase matters.

**Acceptance, stated before any code.**

- (a) Bytes and ownership: the identity gate door ON default and plain, the fault gate default and plain, the hit gate
  door ON, the pause gate; engine cells: a pooled lease round-trips the same bytes as a fresh one; a reused backing's
  bytes past the lease's length are never in any slice or copy (a cell and a census); the pool's charges and the
  pinned ledger return to their baselines at the latch; a latched tier frees the pool.
- (b) The chain cell (DAY52's shape, base against L): the replaced twin's `kv` drop median at most 1.0 ms per order
  (from about 8.6), and the long pre-submit's `leases` median at most 2.0 ms per order in the chain's steady state (from
  about 25).
- (c) The demote cell: the first demote's `spans` at most 2.0 ms (L2); the boot's time to ready, a reading.
- (d) The tenant's stall and every intruder's e2e at most base's plus 1.0 ms in each cell; the chain's e2e a reading.
- Then item 17: P2 (DAY52) re-applied on L and re-read by DAY52's rule, pre-registered anew with L as its base.

**What each card decides.** The target card (the chain's long entries and the 9950X class's allocation prices); the
5090 half after it.

**Budget.** 1.0 agent-day: L1 with its cells and census 0.5, L2 0.15, the sitting 0.2, item 17's re-read 0.15.

## 2. Design L's program, stated before any code (the lead's order after DAY62: item 14 with 19 now)

Section 1's design and acceptance stand unchanged. This section fixes the details the code must follow, before the
code.

**L1, the lease pool** (`tier_transfer.rs`).

- (L1.1) `LeasePool`, one per `CudaTransfers`, shared by `Rc` with every `PinnedAllocation` that `alloc_host`
  creates. Its idle backings are keyed by (size class, `PinnedKind`).
  - A backing gains a capacity beside its length: the capacity is the class, and every slice, copy and view uses
    the length. Nothing reads past the length.
  - Size classes: at or below 1 MiB the next power of two, above it the next 1 MiB multiple.
- (L1.2) The lease's own charge is its class bytes, the pinned memory it actually holds. Today's charge is the
  requested bytes. Stated now because it moves the ledger: padding is at most one 1 MiB step per lease above 1 MiB,
  so at most 32 MiB per 27B entry.
- (L1.3) `alloc_host_kind`:
  1. Reserve the lease charge (as today).
  2. Take an idle backing of the class and kind. If there is one, release the pool's charge for it and set its
     length to the request; no fill, since its bytes were initialized by the lease before.
  3. Otherwise allocate a fresh backing at the class size and zero-fill the whole class.
  - A refused charge takes nothing from the pool.
- (L1.4) `PinnedAllocation::drop`, after its tracking event is synchronized (as today):
  1. Release the lease charge.
  2. If the pool is open and its idle bytes plus the class fit under its cap, reserve the pool charge under the pool
     tenant (`memra_engine::cache::record::digest("host-tier-lease-pool", ..)` from the worker's ledger adapter) and
     park the backing idle.
  3. Otherwise free it as today (`cuMemFreeHost`).
  - A backing whose tracking event fails is leaked as today.
- (L1.5) The pool's cap is one host budget (`MEMRA_KV_HOST_MB`). The ledger's pinned capacity gains one host budget
  (from twice to thrice) so the transient overlap in (L1.3) never refuses.
- (L1.6) The pool closes at the tier's latch (`HostPrefixCache::disable`) and when `CudaTransfers` drops: every
  idle backing is freed and every pool charge released. After closing, a dropping lease frees as today.
- (L1.7) Log only: the demote pre-submit split's `leases` term gains `pooled P of N`, and the latch prints the
  pool's freed count and bytes.

**L2, the staging set at boot** (`worker.rs`, `host_tier_context`).

- (L2.1) For every loaded model's `ModelPlan` (trunk layers and MTP blocks), take each `StatePlan::Recurrent`
  layer's two span lengths: conv `conv_width x (conv_kernel - 1) x 4` and ssm `state_width x 4`.
  - The set needs one image's buffers, since one demote is in flight per worker. So each length's count is the
    maximum over the models, not the sum.
- (L2.2) Each buffer is allocated through `staging_take` (charged as today) and put back, before the context
  returns. It prints `[prefix-host] contracts door: span staging set allocated at boot: N buffers, X MB in Y ms`.
- (L2.3) A refusal (a charge or an allocation) is a boot line naming it. The set then stays partial, and the first
  demote allocates the rest as today. Never a boot failure.

**The cells, added to section 1's (a):**

- Engine native cells (on a card):
  - A dropped lease's backing is the next same-class allocation's backing (the same host pointer), with the new
    length. Its bytes round-trip through a D2H and an H2D on the transfer engine, covering exactly the lease's
    length (the copy's own byte count).
  - `bytes()`, `read_view()` and the copy's source and destination span the length, not the capacity.
  - The ledger's pinned use equals the idle pool bytes after the drop, and returns to zero after the pool closes.
  - A drop past the cap frees.
- CPU:
  - the class function's table;
  - the L2 length derivation on the 27B's plan shape and on a mixed two-model set (max counts).
- The census:
  - no slice, view or copy of a backing uses its capacity;
  - the lease charge is the class bytes;
  - the pool closes at the latch and at drop;
  - `cuMemFreeHost` is reached only through `PinnedBacking::drop` (the pool's close, a drop past the cap, a closed
    pool).
- The red arm: a pooled backing handed out with its capacity as its length must fail the length cell.

**The sitting**, per section 1:

- The target card: the chain cell (DAY52's shape, base against L), the demote cell, and the gates. The 5090 half
  follows.
- The target's (b) and (c) read the existing split lines: the replaced twin's `kv` drop, the long pre-submit's
  `leases`, and the first demote's `spans`.

**The local builds** keep out of W's 5090 timed window: before every build, the 5090 lock's holder is checked. While
W's hold runs its timed boots, nothing of this lane builds.

## 3. Design L as built (`f6dfe303a`), and its target sitting prepared

- **L1, the lease pool** (`tier_transfer.rs`), as section 2 fixed it:
  - `PinnedBacking` gains a `capacity` beside its `len`; every slice, view and copy uses `len`.
  - `lease_class`; `LeasePool` (`take`, `put` and `close`), shared by `Rc` with each `PinnedAllocation`.
  - `alloc_host_kind` charges the class, takes a pooled backing (its pool charge released), or allocates the class
    zero-filled, then sets the length.
  - `PinnedAllocation::drop` releases the lease charge, then parks the backing under a pool-tenant charge
    (`digest("host-tier-lease-pool", ..)`) when the pool is open and under its cap, and frees it otherwise.
  - The engine's `Drop` closes the pool. `set_lease_pool_cap` (the default of 0 pools nothing, so every existing
    cell and caller keeps today's program until the worker sets a cap), `lease_pool()` and `lease_pool_counts()`.
- **The worker:**
  - the pinned capacity is three host budgets;
  - the pool's cap is set to one budget at the context's build, and the latch closes the pool with a line;
  - the pre-submit split's `leases` term reads `N pinned, pooled P of N`.
- **L2:** `host_staging_lengths` (the plans' recurrent conv and ssm planes, the max count per length over models).
  The context's build allocates them through `staging_take` and puts them back, with the boot line or the typed
  partial line.
- Cells:
  - CPU: `day63_the_lease_class_table`; `day63_nothing_reads_past_a_lease_length_and_frees_stay_in_one_place` (the
    census); `day63_the_staging_set_at_boot_is_one_image_over_the_models`;
    `day63_the_pool_and_the_boot_staging_are_wired_as_stated`.
  - Native (card): `day63_a_dropped_lease_backs_the_next_same_class_lease` (the same host pointer, the new length,
    the H2D spanning it, the ledger holding the idle class then returning to zero at the close, a closed pool
    freeing) and `day63_a_drop_past_the_cap_frees_and_the_engine_closes_the_pool`.
  - `NATIVE_CELLS` goes from 16 to 18.
- The red arm (`day63/red-arm.patch`: a pooled backing keeps its capacity as its length, with a printed marker). On
  the CPU it fails the census (`assertion failed: alloc.contains("backing.set_len(bytes);")`,
  `day63/red-arm-cpu.log`); on the card the length cell must fail too (the sitting's unit cell).
- CPU: engine lib `578 passed; 0 failed; 51 ignored`, tier `301 passed`, server lib `940 passed; 0 failed; 26
  ignored`; clippy `-D warnings` on the engine and the server, all targets; fmt (`day63/`).
- **The sitting** `pro-single-l/`, receipts `/root/spill-receipts/a-l`: `build.sh <tip> <L's parent>` (l, red and base
  from one clone), then `driver.sh`, each step under one collector hold:
  - `unit-cells.sh`: the two native cells green, the length cell red with the marker, the CPU censuses;
  - `gates.sh`: the 11 gates on l;
  - `ab.sh chain` (DAY52's chain cell, base against l, 20 boots);
  - `ab.sh demote` (20 boots);
  - then `l-reading.py`, whose last line is `L VERDICT -> ..`.
  - The reader was dry-run on P2's chain and demote receipts mapped as two arms. It read base's twin `kv` drop at 8.6
    ms, the long `leases` at 24.1 to 24.3 ms and the first `spans` at 28.1 to 28.4 ms, matching DAY52 and DAY54
    (`INCOMPLETE` there only because P2's chain cell had its third arm).
  - About 1.5 hours of card time.
- Item 17 (P2 re-read on top of L) is pre-registered anew once L's verdict is read, with L as its base.
