# Self-review: integ34 (A day 19: tier conformance rule 3 and the H2D reader wait at the settle, Move 2 pre-registration; C day 23: promote stall on the fixed tree, the door review table)

Author's review of the full diff `main..lane/spill-integ34-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier/src/conformance/reader_fence.rs` and `tests/contracts/reader_fence_bindings.rs`: rule 3 as a
  frozen schedule (`consumer_fenced` is an installed wait on the reader's stream, never a flag and never the copy's
  landing) with three bindings: the day-18 at-submit program satisfies it, a settle-time flag without a reader wait
  fails it (red arm under `catch_unwind`), and the at-settle wait on the reader's stream satisfies it. `conformance/mod.rs`
  and `tests/contracts/mod.rs` register them. The existing frozen schedules are untouched (their files do not change).
- `crates/memra-engine/src/tier_transfer.rs`: `submit_batch` no longer installs the owner-stream wait for an off-owner
  H2D at submit; `install_consumer_wait(ticket)` installs it on each H2D item's event and records the consumer fence.
  I read the fail-closed arms: an unknown ticket, a quarantined item, a missing event and a stream error each refuse
  (the last marks the ticket `unknown`); the fence sequence is overflow-checked before any wait is installed.
- `crates/memra-server/src/worker.rs`: `host_kv_planes_settle_promote` calls `install_consumer_wait` under `Poll` and
  `Block` before the receipt check and `ready_view`; a refusal aborts through `host_promote_contract_abort` with the
  typed reason; the source-order censuses pin the order. Everything is behind the door (default OFF).
- `docs/FLAGS.md`: the door row. No new flag.
- Research: A DAY19, `OWNER-THREAD-OFFLOAD.md` (Move 2 pre-registration and first slice), day-18 5090 receipts, day-19
  target-card receipts; C DAY23, stall receipts (both arms, both halves), door-gate receipts from both cards, the final
  `HOSTPREFIX-DOOR.md` review table; STATE files; INDEX rows; the lead record section; this file; battery receipts.

## What I checked
- One numeric program: the H2D destination becomes readable on the owner stream only after the installed wait on the
  item's event, now at the settle instead of at submit; the identity gate OFF versus ON on the target card is ALL GREEN
  default and plain on the moved tree, the hit and twin gates OFF and ON as well, and the fault gate ALL GREEN with the
  floor. The red-arm test proves the schedule rejects a flag without a wait.
- No new `MEMRA_*` read (census clean); `unsafe` unchanged (stream and event calls through the existing wrapper).
- C's stall receipts were taken with N=5 per arm per order in one collector hold with a same-window OFF/ON pair; the
  reading was pre-registered and printed by the replay; no cross-box timing.
- The 27B twin `V3=FAIL` on the 5090 is recorded as a local-card reading with its cause open and assigned (B day 29);
  it is not a door delta (both arms identical) and not moved by this integ.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, tier tests, engine, server and tier clippy `-D warnings`, marker census, workflow keys, perf
  board, diff-check) and the local 5090 serve-smoke if the lock frees within the window (stated either way).

## What I did not do
- No GPU cell of my own beyond the smoke; the door stays OFF; the owner decides on 2026-10-05 with the review table.
