# Self-review: integ39 (A day 23: the draft-bearing restore through the door; C day 28: the hit gate's external lock, the isolating stall cell, the 5090 receipt price)

Author's review of the full diff `main..lane/spill-integ39-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier`: `d2d_restore_draft_ready` (the draft plane's item class rides the restore batch under the same
  producer fence and pin; the draft-head read is ordered behind the installed wait) with its red arm (a read before the
  wait is unordered) and CPU bindings; frozen schedules unchanged.
- `crates/memra-engine/src/spec.rs`: `RestoredDraftScratch` (scratch, `pos`, `k_bytes`, `v_bytes`, exact destination
  views, `set_len`, the OFF copier), `alloc_restored_draft_scratch` (every OFF geometry check, unchanged messages), the
  two constructors `spec_session_from_restored_deferred` (OFF: alloc, copy on the owner stream, `set_len`, tail) and
  `spec_session_from_restored_ready` (ON: takes the filled scratch, copies nothing) over one shared tail
  `spec_session_from_restored_scratch`. Census test pins the order and that exactly two constructors reach the tail.
- `crates/memra-server/src/worker.rs`: the probe decides the draft (`host_restore_draft_decision` over the admission
  estimate, the DFlash ownership, the grammar and `spec_restore_refusal` with the load guard and penalty window), allocates
  and sizes the scratch before the submit; `host_restore_submit` appends the draft K and V spans as `D2dRestore` ops
  under the trunk's producer fence and pin; the `Restoring` record owns the scratch (forgotten under `Latched`, dropped
  after `ReceiptMismatch` and at landed drops, handed over at `take_ready` only under `ready`); admission takes
  `spec_session_from_restored_ready` for a ready scratch and the OFF constructor otherwise, with a typed line when the
  probe declined; disagreement arms typed in both directions. Two CPU tests. `docs/FLAGS.md` door sentence; no new
  `MEMRA_*` read.
- `tools/spec-on-cache-hit-gate.sh --external-lock FD` (the identity gate's shape; `stop()` addresses its own child
  only), `--lock-self-test`, `tools/test_spec_on_cache_hit_gate_lock.sh` (7 checks) in `ci.yml`'s gate-teeth step,
  `docs/TESTING.md` bullet.
- Research: A DAY23 and target-card receipts; C DAY28 with the isolating stall cell (target card) and the 5090 price
  receipts; `HOSTPREFIX-DOOR.md` (item 11's day-23 sentence, owed-cell rows, cost rows per card, the E clause); INDEX
  rows; the lead record section; this file; battery receipts.

## What I checked
- Reachability: every new path is behind the door; with the door OFF the restore is the previous synchronous D2D and
  the draft plane is copied on the owner stream by the same OFF constructor as before (its checks moved, its messages
  kept).
- One numeric program: the draft rows become readable only behind the wait installed at the settle
  (`install_consumer_wait` fences every unfenced non-D2H item, the draft items included); the red arm proves the rule;
  the hit gate's identity clause holds byte for byte on the target card with the route engaged on 12 of 16 hits
  (`19 route submission(s)`), zero refused, disabled or declined lines.
- Ownership: the scratch is never freed under a running copy (Latched forgets it with the cache), is freed only after a
  landed copy, and leaves the record only under `ready` beside the cache and the pin; `host_restore_drop` settles a
  pending contract with Block before the record drops; shutdown settles first. The draft spans share the trunk's pin and
  producer fence, so the source cannot be evicted or rewritten under the copy.
- The probe-side decision is a pure function of admission's inputs (`host_restore_draft_decision` table test); a ready
  scratch admission does not read is dropped (landed, safe), a declined draft admission does read takes the OFF copy.
- The isolating stall cell is reported as C read it: both classes under resolution on the target card, the capture
  share unread because the prime-only arm read above the same prime inside a cache-on boot; nothing tuned.
- The 5090 price row sits beside the target-card row, never compared across cards.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, and the hit gate OFF
  and ON (armed) on the 5090 on this tree (the door gates A stated as owed; stated either way).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
