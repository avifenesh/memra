# Self-review: integ41 (C day 30: the door decision packet, the capture share read; A day 25: the double park and the retire-settle share priced)

Author's review of the full diff `main..lane/spill-integ41-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs` (A day 25): two `f64` fields on `PendingCapture` (`settle_after_ms`,
  `settle_held_ms`), timed around the settle that landed the copy and printed on the capture publish line as a typed
  clause (`; the settle held the owner thread H ms, entered A ms after submission`). No behavior change, no new
  `MEMRA_*` read, no parser splits inside the parenthesis (the fault gate's regexes read the line's head).
- Research: C's `DOOR-DECISION-PACKET.md` (the owner's packet for 2026-10-05, every number re-read from its receipt),
  C DAY30 with the capture-share cell and receipts, A DAY25 with the double-park cell, the decomposition and the
  retire-settle reading and receipts; `HOSTPREFIX-DOOR.md` (packet pointer, cost rows, the owed-cell table rebuilt by the
  lead after the union: one row per cell); INDEX rows; the lead record section (rulings 35 and 36); this file; battery
  receipts.

## What I checked
- The engine change is measurement only: the two timings wrap the existing settle call and cannot change its result;
  the publish line keeps its head byte-identical so every gate regex over it still matches (day 24's census counts read
  exactly on the target card after the change).
- The owed-cell table: both lanes branched from `ebe3fe17d` and each carried its integ40 half, so the union produced
  three variants of the hit-gate rows and two of the isolating row; rebuilt three-way against that base (C's day-30
  rewrite plus A's day-25 sentence; A's day-24 row kept; every row of main present once; `check-conflict-markers: OK`).
- The packet recommends nothing and traces every number; the three outcomes are stated with what each requires.
- A's double-park reading is reported as measured, with the decomposition that places the tenant's +64 ms on the
  demote's hashes and the request's +106 ms on the second park's re-admission; proposal 1 is approved for A day 26 with
  its acceptance gate, proposal 2 accepted (the seam stays at 0.4 ms).
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, and the hit gate OFF
  and ON (armed) on the 5090 on this tree (stated either way).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
