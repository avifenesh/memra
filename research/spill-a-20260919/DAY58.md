# WP-A day 58: OWED item 25, a route book that counts a run's end before the run leaves `running`

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `dfdb1cc85` (item 24 closed). CPU only, this rig. The lead's
note: 25 may be a real route-book ordering defect; register it and find its cause before any fix.

## 1. Pre-registration (committed before any reproduction run or code)

**What is known.** `tests::a_fake_route_memory_door_refuses_defers_and_recovers_through_the_handler` failed 1 of 100 in
arm B's shape (DAY55) and 1 of 400 in arm A's (DAY56), each at the assertion after its cancel: the test loops on
`route.snapshot()` until `cancelled == 1`, then asserts `(waiting, running, inflight) == (0, 0, 0)`, and it read
`(0, 1, 0)`.

**The cause, read from the code** (`route_telemetry.rs`), the hypothesis the cells test:

1. The writer counts the end before the run leaves the gauge. `RouteRun::cancel(mut self)` does
   `cancelled.fetch_add(1, Relaxed)`, and `running` comes down only when `self` drops at the method's end
   (`impl Drop for RouteRun`: `decrement(&self.load.running)`). `finish`, `refuse`, and the failure count in `Drop` have
   the same order.
2. The reader loads the gauge before the end counters. `RouteLoad::snapshot` loads `running` (through
   `self.running()`) before `admitted`, `completed`, `failed`, `cancelled` and `refused`, all `Relaxed`.
3. So a snapshot taken between the two writes, or one whose loads straddle them, reports a run as both ended and still
   running. The book is what `/metrics` and the route's health read; the state is transient (the next snapshot is
   consistent), and the test is the reader that looks right at the boundary.

**The reproduction** (a CPU cell written for it, `day58_a_snapshot_never_counts_an_ended_run_as_running`): 100,000
iterations, each on a fresh book: a writer thread runs one `begin`, `admit` and `cancel` (the test's own sequence),
while the reader loops on `snapshot()` until `cancelled == 1` and records whether that same snapshot read `running ==
1`. Run on the current tree as a probe (the cell is committed with the fix, its first run here is the reproduction).
Reproduced when at least one iteration reads the violation. Then, the item's own test in DAY57's R3 shape (sixteen
burners), 100 runs, as a second reading.

**The fix, if reproduced.**

1. Each terminal path takes the run out of `running` first (`Release`), then counts its end; `Drop` decrements
   `running` only for a run no terminal path already took out, and counts a failure as before.
2. `snapshot` loads the end counters (`Acquire`) before `running`, so a snapshot that sees a run's end also sees it out
   of `running`.

**Acceptance.** The cell 0 violations in 100,000 iterations, run twice; its red arms (the writer's old order; the
reader's old order), each a scratch patch grep-checked in its binary, each reads at least one violation in 100,000; the
item's test 0 of 100 in R3's shape; the route-telemetry and route tests green; a census of both orders; server lib,
clippy and fmt green.

**The rule.** Not reproduced by the cell in 100,000 iterations: the cell is kept as a guard, the item's test is re-read
in R3's shape, and nothing else changes until a reproduction exists. A fix that misses its acceptance is reverted in
one commit.

**Budget.** 0.2 agent-day.

## 2. As run (`day58/`)

- **Reproduced.** The cell on the tree before the fix (`dfdb1cc85` plus the cell), twice: `34011 snapshots counted an
  ended run as running` and `27753 ..` of 100,000 (`repro/`). The item's own test in R3's shape (sixteen burners): 99
  `rc=0`, 1 `rc=101` with `left: (0, 1, 0) right: (0, 0, 0)` (`r3-pre/`). The cause is as section 1 read it.
- **The fix** (`62a29cfe0`): `RouteRun::leave_running` (once per run, the only decrement of `running`) runs before
  `finish`, `cancel` and `refuse` count their ends (`Release`) and before `Drop` counts a failure; `snapshot` loads
  `completed`, `failed`, `cancelled` and `refused` (`Acquire`) before `running`. Census
  `day58_the_book_orders_ends_after_running_and_reads_them_first`.
- **Acceptance:** the cell 0 violations in 100,000, twice (`fix/`); the writer's old order restored (a scratch patch,
  marker in its binary) reads `63926 ..` (`red-writer/`); the reader's old order restored reads `16340 ..`
  (`red-reader/`); a first reader red arm is kept as `red-reader-void/`: it loaded `running` first but still published
  the later load, so it was not the old order and read 0; the item's test in R3's shape 100 of 100 (`r3-fix/`); the
  route-telemetry tests 8 of 8; server lib `932 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt; `git diff
  --check`.

**Verdict, as registered:** a real ordering defect in the route book, reproduced (27 to 34% of snapshots taken at the
boundary counted the ended run as running) and fixed. Item 25 closes. It reaches production readers of the book (the
route's X-RateLimit reading and `/metrics`) as a one-run over-count for the instant between the two writes.
