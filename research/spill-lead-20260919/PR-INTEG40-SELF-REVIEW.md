# Self-review: integ40 (A day 24: the spec-boundary capture route through the door; C day 29: decision cell (i) of Move 1 in one window, the hit gate under the collector)

Author's review of the full diff `main..lane/spill-integ40-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier`: `d2d_capture_draft_publish` (the draft plane's items ride the capture batch; landing is every
  item's event, so a publish with the draft unlanded is refused `NotReady`; the receipt term witnessed per class) with
  its red arm and bindings; frozen schedules unchanged. Contracts 87 passed.
- `crates/memra-server/src/worker.rs`: slice 1's submit half factored into `host_capture_submit` over `CaptureSubmit`
  (the seed route's lines byte-unchanged); `prefix_spec_capture_off_tick` as the spec-boundary route (by-name refusals to
  the OFF program, a pending capture settled `Block` before a second, an already-published prefix skipped, `OnTick` hands
  the capture back untouched); the draft scratch rows `[0..pos)` as two more items of the same batch, ticket and
  receipt (`CapturePlaneClass::Draft`, `Done { kv, draft }`); the spec and DFlash demotions settle a pending capture
  `Block` before `into_demoted` drops the scratch. Census tests. `docs/FLAGS.md` door sentence; no new `MEMRA_*` read.
- Research: A DAY24 and target-card receipts; C DAY29 with decision cell (i) (both Move 1 programs in one window on the
  target card) and the collector run of the hit gate; `HOSTPREFIX-DOOR.md` (item 10's day-24 sentence, the cell (i)
  owed-cell row, the Move 1 cost rows' same-window pair, the hit-gate row, section E); INDEX rows; the lead record
  section; this file; battery receipts.

## What I checked
- Reachability: every new path is behind the door; with the door OFF the spec-boundary publisher inserts exactly as
  before (the route returns the capture untouched).
- One numeric program: the published draft rows are bytes copied on the copy stream behind the producer fence recorded
  after the boundary's last draft-head write; they are read again only through the restore route (day 23, behind the
  installed wait) or the OFF copy; the hit gate's identity clause holds byte for byte with all 11 draft-bearing
  publishes and 12 of 16 hits routed on the target card, and on the local 5090 on this tree (stated either way below).
- Source lifetime: the committed prefix `[0..pos)` of a live session's trunk and draft planes is append-only past `pos`
  and never rewritten below it by verification; the one in-tick free of a source (the MTP demotion dropping the
  scratch) settles first, as the session retire already did (integ37). Both-or-neither: the shell's draft slot is
  filled only from a landed batch; a refused registration or retention of a draft plane takes every fresh plane back.
- **Revuto round 1, fixed (the DFlash tail refused by name):** the route never saw the publisher's `dspark_draft`
  tail and would have published trunk and draft without it for a DSPARK or GLM5 cache; the publisher now passes the
  tail's presence and the route refuses by name, census-pinned. The hit gate (qwen) never carries a tail, so its
  receipts are unchanged.
- C's cell (i) is reported as C read it: the copy stream beats the owner stream by 43 ms (demote) and 13 ms (promote)
  in both orders, the pre-registered clause is not met, and the promote arm's 149.6 (against 81.9 on the tree before
  Move 2's restore route) is traced to a second park on the restore route; recorded as owed to A, nothing tuned.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, and the hit gate OFF
  and ON (armed) on the 5090 on this tree (the door gates A stated as owed).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
