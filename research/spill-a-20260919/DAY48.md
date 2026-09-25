# WP-A day 48: OWED item 4 again, design S4 (S3's revision, DAY46 section 3)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. S3 (`e776b2843`) failed (c) on the target card and is reverted
(`7197c1a9d`; DAY46 section 3). Every cell `executed-not-qualified`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. Pre-registration (committed before any S4 code)

**What S2 and S3 taught, placed** (DAY46 section 3). Both designs' correctness cells are green on the target card, and
both left the landed digests off the landing path (the copy settle moved 0.02 to 0.03 ms). Both failed (c) by the same
+3.1 to +3.3 ms of e2e, and the price is not the digests' grid: at the tick top where a demote's copy settles, the
intruder's seed capture settles right after it, and the capture's settle takes its destination planes back through
`take_plane`, which drains the WHOLE copy stream (`synchronize_copy_stream`) and so waits for the landed digests the
seal has just enqueued (about 3 ms of PCIe-bound reads): the capture settle's owner hold reads 3.09 to 3.26 ms on S2 and
S3 against 0.13 to 0.21 ms on G4, and S2's trace shows the owner thread in `cuStreamSynchronize` for 2.908 ms from 0.066
ms after the landed launch.

**Design S4: S3 whole (DAY42 sections 1 and 1a, DAY46 section 1), with the release paths' drain made precise.**

1. `CudaTransfers::take_plane` and `release_device` keep their owner-stream drain and no longer drain the copy stream.
2. **Why that is safe, stated so its census can hold it.** Every copy-stream operation that reads or writes a
   registered device lease is an item of a ticket that names the lease (`items[i].device`), or a receipt, digest or
   fault over that ticket's items, recorded before one of the ticket's events. `require_unbound` (both release paths'
   first step) refuses with `Busy` while any unretired entry names the lease, and a ticket retires only after its
   landing observed every item event and every receipt event (`producer_done`). So a release that passes
   `require_unbound` cannot have copy-stream work on its lease pending; the drain waited on other memory's work only.
   Copy-stream work that is not a ticket's item (the D2H spans' sources and staging, the receipt lanes and twins, the
   filled staging) never touches a registered lease.
3. `synchronize_copy_stream` keeps its callers that are not release paths, if any; a function left without a caller is
   removed.

**Acceptance, stated before any code** (S2's, whole; each card its own; S's bounds unchanged), plus:

- (a) Semantics: S3's (a), and a native cell, `day48_a_take_back_waits_for_its_own_lease_only`: a retired D2H ticket's
  lease is taken back within 50 ms while a 300 ms spin (`delay_on`) holds the copy stream; a lease an unretired ticket
  names is refused `Busy` and stays registered; the census `one_side_stream_beside_the_owner` and a new census
  `day48_release_paths_drain_the_owner_stream_only` (both release paths call `require_unbound` first, drain the owner
  stream, and never the copy stream; retire requires `producer_done`).
- (b) to (e): S3's (the gates on S4's binary, the demote A/B against G4 with (c)'s wall at most +8.0 ms and e2e at
  most +1.0 ms per order, the promote A/B with PIN and e2e at most +1.0 ms, the hump at most 0.15 ms beside the G''
  control).
- Reading, no clause: the capture settle's owner hold (`the settle held the owner thread`) per arm in the demote A/B,
  and the trace reading.

**Predictions.** The capture settle's owner hold on S4 reads as G4's (about 0.2 ms); (c) e2e within +0.5 ms; the wall
within +2 ms; (d) and (e) as S3 read them.

**What each card decides.** Each card its own. The target card first (the same box, after item 3's reading); the 5090
when it is back.

**Budget.** 0.4 agent-day: the change, the census and the native cell 0.15, the CPU cells 0.05, the target sitting 0.2.

## 2. S4 as built (`ef4b097ad`) and its CPU cells

- Built: S3's code re-applied (the revert `7197c1a9d` reverted, code only); `take_plane` and `release_device` drain the
  owner stream and no longer the copy stream; `synchronize_copy_stream` had no other caller and is removed. Census
  `day48_release_paths_drain_the_owner_stream_only` (`require_unbound` first, the owner stream drained, no copy stream
  touched; `require_unbound` checks every unretired entry's items; `retire` refuses before `producer_done`); the native
  cell `day48_a_take_back_waits_for_its_own_lease_only` (the pool's 15th context); `one_side_stream_beside_the_owner`
  now pins no copy-stream drain.
- CPU cells, green: engine lib `554 passed; 0 failed; 45 ignored`; server lib `912 passed`; the tier crate; clippy `-D
  warnings` on the three crates; fmt; `git diff --check`; `tools/check-flags.sh`.
- The target sitting (`pro-single-day42/run-all-4.sh`, pro-single-s2's scripts with S4's tip as the s2 arm; S3's
  receipts moved to `a-s2-design-s3` first) runs on the same box after item 3's reading.
