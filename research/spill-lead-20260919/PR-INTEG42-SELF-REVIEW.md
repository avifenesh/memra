# Self-review: integ42 (A day 26: the unearned second park refused by shape; C day 31: the 5090 pair and whole-budget arm; C day 32: the `MEMRA_ADMIT_BY_MEMORY` packet)

Author's review of the full diff `main..lane/spill-integ42-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs` (A day 26): `host_restore_promoted_this_admission` and its call in
  `host_restore_park_probe` between the lookup and the class check: when the promote's own insertion pin names the hit
  entry (same pool key and `px.id_index(pin)` equal to the hit's index), one typed line and the OFF device-hit copy on
  the tick. No flag, no new state, no numeric change, no new `MEMRA_*` read. A table test over the pin shapes (none,
  this entry, another entry, another namespace, an index shift) and a source census pinning the placement and the line.
- Research: A DAY26 with the acceptance clauses and receipts on both cards; C DAY31 (the 5090 pair cell and whole-budget
  arm, this card's own figures), C DAY32 (`ADMIT-BY-MEMORY-DECISION-PACKET.md`), the door packet updated with A's
  day-25 lines and ruling 36; `HOSTPREFIX-DOOR.md` (three-way merged by the lead against the integ41 base, one row per
  cell); INDEX rows; the lead record section (rulings 37 and 38); this file; battery receipts.

## What I checked
- One numeric program: the refused route takes the OFF copy the identity clause already covers; the hit gate's ON
  census on both cards is identical to day 24 (30 route submissions, 11 spec-boundary captures with the draft plane, 13
  restores, 0 typed refusals), so the refusal fires only on the promote-then-hit shape the gate does not contain.
- Placement: after the lookup (a miss never reaches it), before the class check and before the route's own pin; the
  pin identity is by key and entry id through `id_index`, so an index shift cannot alias (the test's last case).
- The acceptance clauses are reported one by one, with clause 2's FAIL kept as written and re-derived in the record
  (ruling 37): the expected end-to-end move was mis-modelled on day 25 (the token is sampled a tick later), the measured
  14.6 ms equals one tick plus the slack, and the tenant's stall fell 67.6 ms on this shape to 3.4 ms below OFF.
- C's day-32 finding is carried verbatim to the owner: every banked `MEMRA_ADMIT_BY_MEMORY` cell ran with the door OFF,
  so the door's ON arm has no receipt on any card the day before its decide-by; the packet recommends nothing.
- The door doc merge: the union across two lanes branched from the same integ41 base left variant lines; redone as a
  true three-way (`git merge-file`) against that base; one hunk resolved to C's newer row; A's tree prepend on the
  fault row restored; no duplicated row; markers OK.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, and the hit gate OFF
  and ON (armed) on the 5090 on this tree (stated either way).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
