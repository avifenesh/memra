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
