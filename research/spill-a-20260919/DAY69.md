# WP-A day 69: a purge left a revoked tenant's bytes in the lease pool (revuto on integ69), design P registered

Integ69's review (revuto, comment 4112582951 on `tier_transfer.rs:378`) found a real retention defect in L'. Placed
below, then design P is registered before any code. Every cell is `executed-not-qualified`.

## 1. The defect, placed

- `HostPrefixCache::purge_tenant` drops the revoked tenant's entries, and with them their `CudaPinnedLease`s.
- Since L' (DAY63 section 4), `PinnedAllocation::drop` parks each backing in the lease pool (`LeasePool::put`) without
  a scrub, and a pooled backing is not refilled when a later lease takes it.
- So the tenant's q8_0 / q5_1 KV bytes stay in pinned host memory, up to one host budget, until one of three things:
  - a same-class lease reuses the backing, and even then only its `len` is overwritten, so the tail up to `capacity`
    keeps the old bytes;
  - the tier latches;
  - the engine drops.
- Before L' the drop ran `cuMemFreeHost` at once, and the purge's promise held ("clears the tenant's parked prompt
  bytes .. the pinned-host tier is the retention concern").
- **A second retention of the same class, found while placing the first:** the span staging set. Its idle buffers
  keep the last demote's or promote's recurrent spans (the f32 conv and SSM state) until another span of the same
  length overwrites them.
  - This dates from day 30, before L'. L' only moved the set's allocation to boot.
  - The purge never touched the set.
- **The other purge paths, checked:**
  - The device `PrefixCache::purge_tenant` drops device planes only. They go to the device async mempool, unchanged by
    L', and it holds no host lease.
  - `host_restore_purge_tenant` and `release_promoted_pin_for_tenant` touch device state and pins only.
  - In this door, a host lease reaches the pool only through `PinnedAllocation::drop`. Every such drop on a revocation
    runs inside `HostPrefixCache::purge_tenant`: its entries, and its Promoting entry after the Block settles.
  - A leased-and-leaked lease (a quarantine, a helper past its deadline) is never dropped, before L' as after. Its bytes
    were never freed, and this design does not change that.
- **Outside this door:** the GLM-5 TP startup arena (`MEMRA_GLM5_TP_KV_HOST=1`, which refuses this door at boot)
  returns a released region's extent to the arena without a scrub. That is the same class of retention on that path.
  It is owed as its own item, registered after this fix.

## 2. Is a reused backing ever readable by another tenant (the non-purge reuse)?

No, in production, for three reasons:

- **Nothing reads past `len`.** Every slice, view and copy of a backing spans `len`. DAY63's census
  `day63_nothing_reads_past_a_lease_length_and_frees_stay_in_one_place` pins this, and the `HostSlice` copies and
  `raw_view` span `len`.
- **Within `len`, the lease's own D2H writes every byte before any read.**
  - The engine's `validate` refuses a D2H whose `bytes` differ from the lease's length (`o.bytes !=
    o.host.valid_bytes()` -> `InvalidLayout`, "exactly the whole initialized logical host range, not a short prefix")
    and one whose lease is shared (`strong_count != 1`).
  - `progress` checksums the lease only after `event_done`, and `as_slice` synchronizes the tracking event first.
  - The worker hands every fresh lease straight into its op (`hosts` -> `TransferOp::D2h`) and reads a lease only once
    it has landed: the bind, the H2D source, the export.
- **So a later tenant never reads an earlier tenant's bytes, neither in the tail past `len` nor before its own
  landing.** There is no second defect.
- **A latent hazard, recorded.** The API lets a caller read a fresh pooled lease before its D2H lands. That read showed
  zeros before L' and now shows the previous lease's bytes. No production caller does it. A guard (a pooled lease
  unreadable until written or landed) is owed as its own item after this fix, so the property holds by construction
  rather than by the callers.

## 3. Design P (the purge scrub), registered before its code

- **(P.1)** `LeasePool::drain()` does three things:
  - advances the pool's epoch;
  - frees every idle backing, by the backing's drop and its `cuMemFreeHost`, as before L';
  - releases each idle backing's pool charge.
  - The pool stays open. `close()` is the same free with the pool closed.
- **(P.2)** Every lease records the pool's epoch at allocation. `put` parks a backing only when all three hold: the
  pool is open, the lease's epoch is the current one, and the cap has room. Otherwise the backing frees.
  - So no backing that held bytes before a drain is ever handed to a later lease. The idle ones free at the drain. The
    live ones free when their lease drops, including a lease some holder still has at the purge (the late drop).
- **(P.3)** The pool is generic over its backing, through a `PoolBacking` trait (its class and kind), so its program
  runs in a CPU test with a stand-in backing. Production is `LeasePool<PinnedBacking>`, statement for statement.
- **(P.4)** `HostPrefixCache::purge_tenant` ends with the scrub. It drains the pool and zeroes every idle buffer of the
  span staging set (`HostStaging::scrub`, `PinnedHostBuf::fill(0)`). The scrub runs after the Block settles and after
  the tenant's entries drop, on every purge whichever tenant it names, logged as `[prefix-host] purge scrub: lease pool
  drained (N idle backings, X MB freed; earlier leases free at their drop), staging set zeroed (M buffers, Y MB)`.
  - The set keeps its buffers and charges, so the staging fill counts the fault gate reads are unchanged.
- **Why the drain and not a zero-before-park of the purged leases.** The choice is on correctness.
  - Zeroing only the purge's own leases needs every lease of the tenant enumerated at the purge, and it misses one that
    another holder drops later.
  - The epoch needs no enumeration and covers the late drop. The steady path gains one integer compare per drop.
- **What it costs.**
  - After a purge, other tenants' idle backings free, and their live leases allocated before the purge free at their
    drop (the pre-L' program).
  - So demotes run the fresh-allocation program until leases allocated after the purge refill the pool: per long
    demote, L''s base lease time. DAY63 section 6 read it at 26.27 ms against 0.04 ms, and the twin's kv free at
    9.78 / 9.87 ms against 0.05 ms, both on the target card.
  - The drain's frees run on the owner thread inside the purge, which is already a blocking admin command with Block
    settles.
  - No steady-state change.

## 4. Cells

- **(a) Engine CPU test** `day69_a_drain_frees_every_idle_backing_and_no_earlier_backing_parks_again`, with stand-in
  backings that carry a tenant's bytes and count their frees:
  - after the drain the idle set is empty, every idle stand-in is freed, and the pool's ledger charge is zero;
  - a stand-in leased before the drain and dropped after it frees and does not park;
  - one leased after the drain parks and is taken.
  - Red arm: `put` ignoring the epoch must fail the test.
- **(b) Engine CPU census:**
  - the allocation records the epoch, and `put`'s park condition names it;
  - `drain` advances the epoch before it frees, and `close` runs the same free;
  - `free_host(` sites are unchanged.
- **(c) Worker CPU census:** the purge's scrub (the drain and the staging zero) follows the settles and the entries'
  drop and is the purge's last work. Red arm: the scrub call removed must fail it.
- **(d) GPU cells (native), for the target card's battery:**
  - engine `day69_a_drained_backing_is_never_the_next_lease`: a pooled lease written 0xa5, the drain, and the next
    same-class lease fresh (the counts) and reading zeros;
  - worker `option_b_purge_drains_the_pool_and_zeroes_the_staging_set`: a demote through the real contract route with
    its leases dropped to the pool and its staging back in the set, then a purge. Pool idle is 0, the staging idle
    buffers are all zero, the ledger holds the staging charge alone, and the next lease is fresh.
- **CPU battery:** fmt, clippy `--all-targets -D warnings`, the memra-engine and memra-server lib tests, and
  `tools/check-flags.sh`.
- **Budget:** 0.2 agent-day.

## 5. Design P as built (`4f297e7bd` on `lane/spill-a-integ69-20260926`, `59376ebeb` on this lane)

- (P.1, P.2) Built as registered:
  - `LeasePool::drain` advances the epoch, then frees every idle backing and releases its charge;
  - `close` is `open = false` plus the drain;
  - every `PinnedAllocation` records `lease_pool.epoch()` at allocation, and its drop calls `pool.put(b, self.epoch)`;
  - `put` parks only on `open && epoch == current && room`.
- (P.3) `LeasePool<B: PoolBacking = PinnedBacking>`, with `PoolBacking { class, pool_kind }`. `PinnedBacking`'s `class`
  is its `capacity()`. DAY63's census now counts `.capacity` reads at 2 (set_len's bound and the pool's class read),
  down from 3 (take, put, and the class read), because take and put read `class()`.
- (P.4) `HostPrefixCache::purge_tenant` ends with `self.purge_scrub()`. It drains the pool and runs
  `HostStaging::scrub`, which is `fill(0)` over every idle buffer, keeping the buffers and charges. It logs
  `[prefix-host] purge scrub: ..`.
- **Cells:**
  - (a) The CPU test is green (`day69_a_drain_frees_every_idle_backing_and_no_earlier_backing_parks_again`).
    - Red arm `day69/red-arm-epoch.patch` (`put` ignores the epoch, with a marker) fails it on the late drop,
      verbatim: `assertion 'left == right' failed: the earlier backing freed at its drop / left: 2 / right: 3`. The
      backing leased before the drain parked. The census (b) fails with it: `assertion failed:
      put.contains("|| epoch != self.epoch.get()")` (`day69/red-arm-epoch.log`).
  - (b) The engine census is green (`day69_the_pool_parks_only_in_its_epoch_and_the_drain_advances_it_first`).
  - (c) The worker census is green (`day69_the_purge_ends_with_the_pool_drain_and_the_staging_zero`). Red arm
    `day69/red-arm-purge.patch` (the scrub call removed) fails it on `the purge scrubs` (`day69/red-arm-purge.log`).
  - (d) The GPU cells, engine `day69_a_drained_backing_is_never_the_next_lease` and worker
    `option_b_purge_drains_the_pool_and_zeroes_the_staging_set`, are built. `NATIVE_CELLS` is 18. They are for the
    target card's battery; the local 5090 is held by this lane's DAY68 chain and is not touched.
- **CPU battery on `4f297e7bd`:**
  - `cargo fmt --all -- --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings` clean;
  - engine lib `578 passed; 0 failed; 51 ignored`, server lib `943 passed; 0 failed; 27 ignored` (`day69/engine-lib.log`,
    `day69/server-lib.log`);
  - `tools/check-flags.sh` "no uncovered runtime names";
  - `git diff --check`.
  - On this lane's tip (`59376ebeb`, carrying F): engine `579 passed`, server `949 passed`, and workspace clippy clean.
- No new flag and no steady-path change beyond the epoch compare.

## 6. Two process defects the lead found on BOX31, placed and fixed

- **The unit scripts' documented invocation ran nothing.** The headers of `pro-single-p2l2/unit-rerun.sh` (line 7) and
  `pro-single-f/unit-worker.sh` (line 6) gave `tier-battery.py .. --external-lock --execute ..` without `--out`. The
  collector refuses that (`--execute requires argv, new --out and positive timeout`), so the lead's first units chain
  ran nothing.
  - Both headers now name `--timeout 5400 --out /root/spill-receipts/<dir>-collector`. The lead's rerun used its own
    correct invocation, which is what DAY67 section 5 and DAY64 section 8 read.
- **A driver rc line printed after a `$(..)` expansion reads that expansion's status, not the step's.**
  - For example `echo "$(date -u +%FT%TZ) reading rc=$? .."` always prints `rc=0`. Checked on this rig: `false; echo
    "$(date -u +%T) rc=$?"` prints `rc=0`, and `rc=$?` placed before the expansion prints `rc=1`.
  - **The audit of this lane's scripts:** 31 lines in 23 scripts had the pattern. They are the `reading rc=` line of
    the `pro-single-*` drivers (b1, day54, day59, day60, day62, day64, day64b, f, l, l2, p, p2, p2l, p2l2, r1, r2 x2,
    th, w), the per-boot `rc=` log of `pro-single-day49/cell.sh` and `pro-single-day54/pause.sh` (after
    `$(basename ..)`), `rtx5090-day33/stall-pair.sh` and `rtx5090-day33/nsys/run.sh`, and the mirrored day-38
    `diagN-run.sh` copies under `pro-single-day38/box/`.
  - **Fixed:** each line now captures `step_rc=$?` first and prints `rc=$step_rc`. The mirrors under `*/box/` are left
    as mirrored. Git history keeps every script as it ran.
  - **Did any record use such an rc as evidence? No verdict rests on one.**
    - Every reader takes its codes from its own files, each written from an rc captured at the step: the gates'
      `.exit` files (`run()`: `local rc=$?` right after the gate), the unit cells' `run.log` (`cell()`: `local rc=$?`),
      the collector cells' rc (the drivers' `cell()`: `rc=$?` right after `tier-battery.py`), the replays' `STALL
      REPLAY: PASS` counts, and the receipts' completeness.
    - The records that quote an rc token: DAY38's `diag4-cell rc=0` to `diag8-cell rc=0` came from the mirrored
      `diagN-run.sh` (the date's rc). So they are completion notes, not evidence. Each of those sections reads its
      verdict from its `reading-hump.log`.
    - DAY38's `diagN-build rc=0` tokens print the build log's own last line and are sound. DAY39, DAY49, DAY51,
      DAY52, DAY54 and DAY59's `*-cell rc=0` tokens came from the drivers' `cell()`, which captured the rc, and are
      sound.
    - The per-boot `rc=` of day49 and day54 (the basename's) was never read; those readers count receipts and
      replays.
    - DAY64 section 8's `reading rc=0` is noted there as the date's.
  - This lane's DAY68 scripts (`rtx5090-half*.sh`, `rtx5090-chain-day68.sh`) print `rc=$?` before any expansion, and
    were checked the same way.
