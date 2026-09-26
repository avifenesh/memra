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

## 4. L's sitting, read as registered: REFUTED (a); the failure placed; L' registered

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh cbe433acc 55684e4bd`
  then `driver.sh`, to 13:03Z. Mirror `pro-single-l/box/`, sha256-checked against the box manifest (0 mismatches);
  the executables are recorded by hash. Start temperatures 48 C to 67 C.
- Verbatim (`box/reading-l.log`):

      L (a) UNIT a1-green=0 a2-green=0 a1-red=101 (marker 1) a2-red=101 (marker 1) censuses=0
      L (a) gates {'contract-fault-plain': '1', 'contract-fault': '1', 'failure-off': '0', 'failure-on': '0', 'hitgate-off': '0', 'hitgate-on': '0', 'identity-default-off': '0', 'identity-default-on': '0', 'identity-plain-off': '0', 'identity-plain-on': '0', 'pause-demote': '0'}
      L READING cell=chain order=o1 twin kv base=9.72 l=0.05 ms (N=95) | long leases base=26.06 l=0.04 ms (N=95, pooled median 32) | stall base=94.72 l=68.05 | e2e base=199.5 l=172.6 | chain base=427.8 l=396.2 ms
      L READING cell=demote order=o1 first spans base=28.61 l=0.17 ms (N=5) | boot staging l=28.5 ms | stall base=63.89 l=64.15 | e2e base=174.9 l=175.3 ms
      L READING cell=chain order=o2 twin kv base=9.62 l=0.05 ms (N=95) | long leases base=25.99 l=0.04 ms (N=95, pooled median 32) | stall base=94.80 l=68.09 | e2e base=199.5 l=172.7 | chain base=431.9 l=396.4 ms
      L READING cell=demote order=o2 first spans base=28.85 l=0.17 ms (N=5) | boot staging l=28.7 ms | stall base=64.02 l=64.14 | e2e base=175.2 l=175.4 ms
      L (b) PASS [True, True]
      L (c) PASS [True, True]
      L (d) PASS [True, True, True, True]
      L VERDICT -> REFUTED ((a) failed): revert in one commit, red receipts banked

  Both fault-gate arms fail the same two checks (`KV-HOST-CONTRACT-FAULT GATE: 2 FAILURE(S)`):
  `FAIL: span-refusal: the staging set filled once and every later demote reused it` and `FAIL: promote-span-refusal:
  the staging set filled once and every later demote and promote reused it`.
- Read:
  - The timing clauses pass by wide margins. In the chain cell the replaced twin's `kv` drop falls from 9.72 to 0.05
    ms, the long `leases` from 26.06 to 0.04 ms (32 of 32 pooled), the tenant's stall from 94.7 to 68.1 ms and the
    chained request from 427.8 to 396.2 ms. The demote cell's first `spans` falls from 28.6 to 0.17 ms.
  - (a) fails, and the rule is the rule: **REFUTED**, reverted in one commit, the receipts kept.
- **The failure, placed from the gate's code** (`tools/kv-host-contract-fault-gate.sh`, `one_staging_fill` and
  `one_staging_fill_promote`):
  - Both checks require exactly one `tier span staging: N fresh pinned buffer(s)` line in the boot, with N equal to
    the refusal's `N f32 spans handed back`. The line is printed when a demote (or promote) allocates fresh staging
    buffers.
  - The receipts: on L the gate's logs carry `span staging set allocated at boot: 96 buffers, 156.9 MB` and no fresh
    line at all (`staging fill line(s) [], refusal N 96`). L2 moved the fill to boot, so the first demote allocates
    nothing, and it prints nothing.
  - What the check was written for (day 31, the span refusal's put-back): a refused span attach must return every
    staging buffer to the set, so the set is filled once and no later demote or promote allocates again. On L that
    property holds: one fill (at boot, 96 buffers, equal to the refusal's 96), and no fresh line after it.
  - The check encodes where the first fill happened (the first demote), not the property. **The gate is wrong for
    L2, not L2 for the gate.**
- **The gate change, registered before any rerun** (its own commit):
  - `one_staging_fill` and `one_staging_fill_promote` count fill events: the boot line `span staging set allocated
    at boot: N buffers`, or the fresh line `tier span staging: N fresh pinned buffer(s)`.
  - They require exactly one fill event, with N equal to the refusal's N. That reads the day-31 program (one fresh
    line, no boot line) and L2's (one boot line, no fresh line) the same way. A second fill event of either kind, a
    boot fill of the wrong size, or a fresh fill after a boot fill fails.
- **Its red arm, registered with it:**
  - A scratch server patch (`pro-single-l2/gate-red-arm.patch`, with a printed marker). The D2H span refusal drops
    each staging buffer instead of `tier.staging_put(destination)`, and the H2D span refusal drops each source
    instead of pushing it to `staged`.
  - On an L' binary the set is then short after the refusal, the next demote or promote allocates fresh, and the
    boot has a boot fill plus a fresh line.
  - The changed gate must FAIL both cells on that binary (rc 1, the same two FAIL lines) and pass on the L' binary.
    A gate change that passes the red arm is refuted with it.
- **L'** = L's code re-applied unchanged, plus the gate change. It reruns whole: section 1's (a) to (d), with the
  unit cells, the 11 gates, the chain and demote cells, and the gate's red arm as a twelfth run. The sitting is
  `pro-single-l2/`, receipts `/root/spill-receipts/a-l2`, with `l-reading.py` plus the red arm's exit and FAIL lines.

## 5. The revert, the gate change and L', as built

- **The revert** (`bb3f4c1d1`) removes L's code, keeps its receipts (`day63/`) and keeps T-H on top. The one conflict
  was the test block where L's two server cells sat beside T-H's; it was resolved by removing only L's cells. Server
  lib `941 passed`, engine lib `576 passed`.
- **The gate change** (`217ace3fd`): `one_staging_fill` and `one_staging_fill_promote` count fill events (the boot
  line or the fresh line) and require exactly one, of the refusal's N. Read offline before any card, on the banked
  logs:

      L's gate logs (the boot fill):       staging fill event(s) [96], refusal N 96 -> rc 0 (both cells)
      P2's gate logs (the first-demote fill): staging fill event(s) [96], refusal N 96 -> rc 0 (both cells)
      L's span-refusal log plus a second fill (a fresh line of 96 appended): staging fill event(s) [96, 96] -> rc 1

  The appended line stands for the regression the check was written against, before its card red arm runs.
- **L'** (`348d2e8f3`): L's code re-applied unchanged (the revert of the revert, no conflict). Server lib `943
  passed`, engine lib `578 passed`, clippy `-D warnings`, fmt.
- **The gate change's red arm** (`pro-single-l2/gate-red-arm.patch`): the D2H span refusal drops each staging buffer
  (instead of `tier.staging_put(destination)`) and the H2D span refusal drops each source (instead of `staged.bufs.push`),
  each with a printed marker. It compiles on L' (`cargo check`).
- **The sitting** `pro-single-l2/`, receipts `/root/spill-receipts/a-l2`: `build.sh <tip> 217ace3fd` (l, redgate, red
  and base from one clone; base is the gate change's tree, L reverted), then `driver.sh`, each step under one
  collector hold:
  - `unit-cells.sh`: L's two native cells green, the length cell red;
  - `gates.sh`: the 11 gates on l, the fault gate with its changed checks;
  - `gates-red.sh`: the fault gate on redgate, which must fail both staging-fill checks;
  - `ab.sh chain` and `ab.sh demote`, 20 boots each;
  - then `l2-reading.py`: L's reader, the red arm's line `L2 GATE RED ARM .. -> caught (as required)`, and the last
    line `L2 VERDICT -> ..`.
  - About 1.75 hours of card time.
- T-H's sitting (DAY65 section 2) has L in its base and waits for L's verdict. Its `build.sh <tip> 1cba80185` stays
  valid on the tip, since L' is L's code.

## 6. L''s sitting, read as registered: ADOPT; the gate change stands

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh 21984b527 217ace3fd`
  then `driver.sh`, to 14:26Z. Mirror `pro-single-l2/box/`: 913 receipts, sha256-checked against the box manifest (0
  mismatches); the seven executables are recorded by hash. `markers.txt`: the red and redgate markers are present in
  their binaries and absent from l's. Start temperatures 47 C to 68 C.
- Verbatim (`box/reading-l2.log`):

      L (a) gates {'contract-fault-plain': '0', 'contract-fault': '0', 'failure-off': '0', 'failure-on': '0', 'hitgate-off': '0', 'hitgate-on': '0', 'identity-default-off': '0', 'identity-default-on': '0', 'identity-plain-off': '0', 'identity-plain-on': '0', 'pause-demote': '0'}
      L READING cell=chain order=o1 twin kv base=9.78 l=0.05 ms (N=95) | long leases base=26.27 l=0.04 ms (N=95, pooled median 32) | stall base=94.91 l=68.26 | e2e base=151.0 l=124.3 | chain base=282.4 l=245.9 ms
      L READING cell=demote order=o1 first spans base=28.83 l=0.18 ms (N=5) | boot staging l=28.6 ms | stall base=63.88 l=64.15 | e2e base=175.4 l=175.9 ms
      L READING cell=chain order=o2 twin kv base=9.87 l=0.05 ms (N=95) | long leases base=26.27 l=0.04 ms (N=95, pooled median 32) | stall base=95.05 l=68.36 | e2e base=151.0 l=124.3 | chain base=282.4 l=246.1 ms
      L READING cell=demote order=o2 first spans base=28.50 l=0.17 ms (N=5) | boot staging l=28.9 ms | stall base=63.94 l=64.18 | e2e base=175.5 l=175.6 ms
      L (b) PASS [True, True]
      L (c) PASS [True, True]
      L (d) PASS [True, True, True, True]
      L VERDICT -> ADOPT (L is the naked program)
      L2 GATE RED ARM rc=1 staging-fill FAILs=2 of 2 marker=True -> caught (as required)
      L2 VERDICT -> ADOPT (L' is the naked program; the gate change stands)

- Read:
  - All 11 gates pass with the changed staging-fill checks. The gate change's red arm is caught: both checks fail
    on the binary whose refusals drop their staging buffers.
  - The chain cell: the replaced twin's `kv` drop falls from 9.8 to 0.05 ms, the long `leases` from 26.3 to 0.04 ms
    (32 of 32 pooled), the tenant's stall from 95.0 to 68.3 ms, the chain's first request from 151.0 to 124.3 ms and
    the chained request from 282.4 to 246.0 ms.
  - The demote cell: the first demote's `spans` falls from 28.7 to 0.17 ms, and the boot pays 28.6 to 28.9 ms for the
    staging set instead.
  - This sitting's chain e2e numbers are lower than the refuted L sitting's (base 151.0 against 199.5). They are the
    same cell on another box session; only the same sitting's base and l are compared.
- **Adopted** as registered: L' is the naked program, and the fault gate's changed checks stand.
  - L' and R1 go to integ69.
  - **Items 14 and 19 close.** No `cuMemFreeHost` and no fresh pinned allocation reach the serving path in the steady
    state, and the staging set is allocated at boot.
  - The growth phase (before any entry leaves the tier) still allocates fresh, as section 1 stated.
- The 5090 half follows under the per-hardware rule. Item 17 is re-read on top of L' (DAY67).

## 7. integ69's worker GPU span cells: 13 failures, placed (before any edit)

- integ69's battery (the lead's BOX35 run, tree `64101769b` = main plus `dba7c0a0c` plus lane C's `9bb17bd8d`) reads
  `worker-span-cells rc=101 test result: FAILED. 5 passed; 13 failed`: the serial `cargo test -p memra-server --lib
  -- --ignored --test-threads=1 option_b_ option_c_`. Every gate is green.
- L's and L''s sittings ran L's own native engine cells but not these worker GPU cells. That is this lane's gap: a
  design that moves the pinned ledger must run the worker's `option_b_` and `option_c_` cells in its sitting.
- **Placing the 12 `gpu_used` failures** (`assert_eq!(gpu_used(host), (3 * 8 * 58, 0, 0))`, `left: (2304, 0, 0)
  right: (1392, 0, 0)`):
  - The fixture's entry (`gpu_entry`) has three KV planes (trunk layers 0 and 2, and the draft), each 8 rows of 34 K
    bytes and 24 V bytes: K 272 bytes, V 192 bytes. The expectation 3 x 8 x 58 = 1392 is the sum of the leases'
    requested lengths.
  - Design L1.2 (section 2, registered before code: "the lease's own charge is its class bytes ... Stated now
    because it moves the ledger") charges each lease its size class: 272 is class 512 and 192 is class 256. Three
    planes of 512 + 256 is 2304, exactly the reading.
  - Nothing else is in the pinned total. The fixture builds its `HostTierContext` directly, with no boot staging and
    the lease pool at its default cap of 0, so the pool holds nothing.
  - **Placed:** the cells encode the pre-L charge. The behaviour is L1.2's, as registered.
- **Placing `option_b_span_staging_charge_refusal_refuses_the_attach_and_keeps_the_tier_on`** (`a staging buffer over
  the ledger was charged`; `tier span staging: 4 fresh pinned buffer(s), 163952 bytes charged`):
  - The cell sizes a co-tenant charge as `2 * (1 << 30) - kv_bytes - first`: the pinned capacity at two budgets, less
    the KV leases at their requested bytes and one staging buffer. That leaves room for exactly one buffer, so the
    second span's staging charge must refuse.
  - Design L1.5 (registered) made the pinned capacity three budgets, so the hog leaves a whole budget free, and the
    staging charges succeed.
  - The refusal path itself is untouched by L: `staging_take` still reserves before it allocates, and a refused
    charge still hands every span back. The fixture has no boot staging, so the first demote allocates, as the cell
    expects.
  - **Placed:** the cell's setup encodes the pre-L capacity and charge, not the behaviour.
- **The fix follows from L1.2 and L1.5 as registered**, and changes no production code. The cells compute the
  expected charge from `lease_class` (`gpu_lease_charge()`: three planes of `lease_class(272) + lease_class(192)`), and
  the refusal cell sizes its hog from the ledger's three-budget capacity and that charge. Each cell asserts the same
  property as before.
- The same cells run in the next target sitting of any design that moves the pinned ledger (recorded for F2 and for
  T-H' if accepted).
- **As built:** `411177fea` on the lane, cherry-picked onto the integ69 branch as `fb639631c`
  (`lane/spill-a-integ69-20260926`). `gpu_lease_charge()` replaces every `3 * 8 * 58` ledger expectation (16 pinned
  totals, and three `+ span_bytes` totals). The refusal cell's hog is `3 * (1 << 30) - gpu_lease_charge() - first`.
  - CPU: the integ69 branch's server lib `942 passed`, the lane's `948 passed`; clippy `-D warnings`; fmt.
  - The GPU verification is the lead's battery rerun on the target card. A local 5090 run of the same 18 cells, from
    the branch's frozen test executable, is queued behind the 5090's lock (`/home/avifenesh/spill-a-cells/i69-gpu/`,
    banked when it reads).
