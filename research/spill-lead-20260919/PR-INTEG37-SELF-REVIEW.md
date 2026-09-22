# Self-review: integ37 (A day 21: Move 2 slice 2, the hit restore off the tick under the host-contracts door)

Author's review of the full diff `main..lane/spill-integ37-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier`: the `d2d_restore_ready` schedule (ready only after the landing and the installed reader wait)
  and its red arm (a prime before the wait fails), three bindings including the receipt-less refusal; frozen schedules
  untouched.
- `crates/memra-engine/src/tier_transfer.rs`: `D2dRestore` (borrowed source slice, borrowed destination view, bytes,
  producer fence) and `submit_d2d_restore` (copy stream, layout and owner validated, items unfenced at submit);
  `install_consumer_wait` now fences every unfenced non-D2H item, so a restore's destination is readable on the owner
  stream only after the wait installed at the settle. I read the direction checks: the D2H arm is excluded at both
  sites, the H2D and D2D arms share the wait.
- `crates/memra-server/src/worker.rs`: `Restoring` as a parked-request state with one pending record per worker; the
  probe in the admission loop after the promote probe (the OFF validation split out unchanged as
  `prefix_restore_validate`); the fresh cache, the source entry's pin, the recurrent f32 state and the lengths on the
  owner stream; one batch; park on the requeue; the tick-top poll installs the wait and a three-tick orphan expiry drops
  a restore whose request left; `host_restore_take_ready` at admit's hit site carries the pin over; settle-first at the
  reclaim, the trims, the purge and shutdown; fail-closed arms; the refused-submission shape from #634 mirrored (drain,
  release, latch typed); the ledger's third in-flight term. Three CPU tests including the path census.
- `docs/FLAGS.md`: the door row. No new flag.
- Research: A DAY21 (pre-registration, design finding, what landed, gates, stall), target-card receipts, offload note,
  STATE, INDEX row; the lead record section; this file; battery receipts.

## What I checked
- Reachability: every new path is behind the door; with the door OFF the hit restore is the previous synchronous D2D.
- One numeric program: the restored rows become readable only after the installed wait (the schedule's red arm proves
  the rule); the identity gate OFF versus ON is ALL GREEN in all four arms with the route engaged in the plain ON arm;
  the hit gate is ALL GREEN OFF and ON but its entries are draft-bearing and the route refuses them by name, so the hit
  gate does not yet exercise the restore route (stated by A; the draft-bearing restore is owed).
- The source entry cannot be evicted under a restore (pinned, out of every eviction index); the retiring or parking of
  the restoring request is refused until the restore settles (the census test names every path).
- The stall reading is reported as A read it (admission work split across ticks, D2D share under resolution), not as a
  door win.
- No new `MEMRA_*` read (census clean); `unsafe` unchanged.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, tier tests, engine, server and tier clippy `-D warnings`, marker census, workflow keys, perf
  board, diff-check) and the local 5090 serve-smoke if the lock frees within the window (stated either way).

## Lane C day 25 folded in
- The fixed fault gate live: ALL GREEN on both cards and both arms with the accounting lines; identity ALL GREEN over
  the retire-seam settle; the settle's cost cell `HOLDS` (deltas 0.1 to 0.7 ms against a 3.0 ms bound, the seam
  exercised 10 of 10). C's day-25 changes are receipts, drivers and docs.

## Review round 1 (revuto, addressed in the integ)
- `host_restore_take_ready` decides the record's shape before moving anything out, so its fail-closed arm releases the
  source entry's pin instead of leaking it (the `let`-else scrutinee had moved the pin). CPU test asserts the pin count
  returns to 0.

## Review round 2 (revuto, addressed in the integ)
- The parked-only bounded wait is guarded on a not-ready `Promoting` entry or a not-ready `Restoring` request (a parked
  restore no longer spins the owner thread on an idle box); census test extended.
- Another request's ready restore is an orphan only past `RESTORE_READY_TICKS`; within the grace the probe goes through
  and leaves the state for its owner; the CPU test covers both readings.

## What I did not do
- No GPU cell of my own beyond the smoke; the door stays OFF; the 5090 door gates on this tree are owed (C's next day).
