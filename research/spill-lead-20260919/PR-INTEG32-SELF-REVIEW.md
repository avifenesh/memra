# Self-review: integ32 (A day 18: memra#536 Move 1 promote half under the host-contracts door)

Author's review of the full diff `main..lane/spill-integ32-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/tier_transfer.rs`: `submit_batch` selects the issue stream per item (the copy stream for a
  D2H as on day 17, and now for an H2D too) and keeps the submit-time owner-stream wait for an H2D (its consumer is the
  owner stream); a CPU source census pins the rule.
- `crates/memra-server/src/worker.rs`: the door's promote route is split into submit and settle (the wrapper
  `host_kv_planes_from_contract` keeps the frozen order); `PendingPromote`, `Promoting` state, `host_promote_settle_*`,
  publication at the tick-top poll with the insertion pin held one tick; the promote decision moved from the admission
  body to the admission loop with shared predicates; settle-first at the demote site (both entries), the purge and the
  probe; a one-tick memo on a failed promote so the request serves cold once with one typed line; a pending promote
  missing its shell or ticket fails closed. Six CPU tests; three source-order censuses moved with the code.
- My merge resolution: at the demote site both settle-firsts precede the armed re-check from #622 round 2; in the
  promote hook the armed re-check sits after A's two settles (the hook itself no longer settles on every admission,
  A's run-1 finding); the probe re-decides through `host_promote_probe_decision`, which gates on `armed()`.
- `docs/FLAGS.md`: the door row describes both halves. No new flag.
- Research: DAY18 (pre-registration, what landed, gates, stall), receipts from the target card (two runs), design note,
  STATE, INDEX row; the lead record section; this file; battery receipts.

## What I checked
- Reachability: every new path is behind `MEMRA_KV_HOST_CONTRACTS=1` (default OFF). Door OFF is the previous code; the
  smoke runs door OFF.
- One numeric program: the promoted cache becomes usable only after the copy's event completed (publication at the
  tick-top poll requires the receipt); the identity gate OFF versus ON on the target card is ALL GREEN in every arm on
  A's tree, and the hit and twin gates OFF and ON as well.
- The merged tree compiles and passes the memra-server suite and clippy `-D warnings` in the battery below; the two
  resolution sites were re-read against both parents.
- The 5090 door gates on this exact tree are owed: the lock was held by another session's server for A's whole
  sitting; C day 22 is running the door gates on main plus A's tip on the target card now.
- No new `MEMRA_*` read (census clean); `unsafe` unchanged.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, engine and server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check) and
  the local 5090 serve-smoke if the lock frees within the window (stated either way).

## What I did not do
- No GPU cell of my own beyond the smoke; the door stays OFF; its decide-by review (2026-10-05) reads both halves'
  stall receipts.
