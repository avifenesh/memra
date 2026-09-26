# Self-review: integ65 (A days 56 to 58: items 23, 24 and 25 closed; the route book's end ordering fixed)

Author's review of the full diff `main..lane/spill-integ65-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- `route_telemetry.rs`: item 25, each run end leaves `running` before it counts the end (`Release`), and the snapshot
  reads the end counters (`Acquire`) before `running`. The one production behavior change: `/metrics` and the route's
  rate-limit reading no longer count an ended run as running.
- `lib.rs`, `worker.rs`: item 23, the admission reservation path takes its lane counters as a parameter (the global
  array on every production path) so the shed tests run on their own counters.
- `darklane.rs` tests: item 24, the stop-mode cycle waits for acknowledgements under a 30 s hang guard.
- `docs/TESTING.md`. No new `MEMRA_*` name.
- Research: A's DAY56 to DAY58 with their receipts; the integ65 record (ruling 60), both batteries, this file.

## What I checked
- `leave_running` runs once per run (`out`), so a finish followed by drop decrements once; `Drop` still books `failed`
  for an admitted run that never ended; `begin` increments before any end; `decrement` cannot underflow.
- The acquire-release pairing covers the snapshot's claim, and the red arms show each half is needed.
- The refactored admission path passes the same global counters everywhere in production; route-bound guards never
  release a lane slot.
- The verdict lines in the record are copied from A's DAY56 to DAY58.

## Batteries
- CPU battery 15 of 15: server 939, engine lib 573, portable 388 with 0 skipped, tier 301, pytest 87.
- GPU battery on a rented RTX PRO 6000 with the same 9B: every cell green, fault gate 255 ok per arm, the pause gate
  with the 27B `ALL GREEN` (40 ok).

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged).
Every other hook ran. No tag: the release decision stays with the owner. Revuto: if capped or unavailable, this comment
is the review.

## Addendum after revuto
Revuto was right: the isolated shed tests still moved the global `PENDING_ADMITS`. A's F2b passes the pending gauge with
the lane counters as one pair (global on every production path), extends the census to indirect writers, and adds a
deterministic cell that fails on the old path. I read the F2b diff: production passes `AdmitCounters::GLOBAL` at both
reserve sites and the guard releases both gauges where taken. The merge also brings A's log-only fanout copy timers
(two clock reads per device enqueue in snapshot and restore, on every path, printed only on the fanout line). CPU
battery 15 of 15 (server 941) and the GPU battery rerun all green on the merged tree.
