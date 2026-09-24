# WP-A day 37: OWED item 1, A's finding 5 (the native span cells fail once in parallel in one process)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU (Intel Core Ultra 9 275HX
host); the target card's half rides the next rented sitting. Every cell `executed-not-qualified`. Every engine push in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode. The ledger is `OWED.md` (`0e65c3d0d`).

## 0. Resync

`git fetch`: the lane tip `6d9e0b06e` equal to origin; `origin/main` `17dceb981` (integ58 merged day 36 as
`aadfea44d`, then #712) fast-forwarded into the lane. Records re-read: both CLAUDE.md files, STATE.md, DAY34 to DAY36,
OWNER-THREAD-OFFLOAD.md, the lead's integ45 to integ58 (rulings 41 to 53), lane C's STATE.md and DAY39 section 7.
`OWED.md` committed and pushed before any day-37 code.

## 1. Pre-registration (committed before any day-37 code)

**The finding, as recorded** (`DAY32.md` section 6 finding 5, `rtx5090-day32/unit/engine-span-cells-parallel.log`):
`d2h_span_batch_lands_with_its_ticket_on_the_copy_stream` and `h2d_span_batch_lands_with_its_ticket_on_the_copy_stream`
run in parallel in one test process; the H2D cell panicked at its rule-2 check, `a batch with a running span has not
landed` (`tier_transfer.rs`, the first poll that saw the KV item landed also saw every span landed). Serial runs 3 of 3
green. Every battery since runs these cells with `--test-threads=1`; the lead's ruling: fix the defect, not the harness.

**The class.** Three native cells hold work behind a 300 ms `tier_delay_spin` on the copy stream and then assert an
intermediate state (the KV item landed, the spans not): the two above and
`h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies`. All eleven native cells of `tier_transfer.rs`
retain the device's PRIMARY context (`CudaContext::new(0)`), so cells running on parallel test threads share one
context, each with its own streams and its own `CudaTransfers`.

**The instrument (test code only, no engine change).** Each of the three timed-hold cells records, from the instant the
hold is enqueued: the instant of the first poll that saw the KV item landed, the number of polls to it, the longest
single `poll` call, and the longest gap between two polls. The cell prints one line `HOLD READING cell=<name>
first_item_seen_ms=F polls=P longest_poll_ms=L longest_gap_ms=G batch_landed_at_first_sight=<bool>` before its rule-2
assertion (so a passing run carries the reading too), and the assertion's message carries the same fields. Nothing
else in any cell changes, and no assertion moves.

**The reproduction protocol, 5090, one bounded hold of `/tmp/memra-5090.lock`** (60 x 120 s attempts, never inside
another lane's hold; then no compute app and at least 20000 MiB free, 15 x 60 s). The test binary built once under the
CPU quota and hashed. Two arms, each repeated in the same hold:

- `pair`: the two finding-5 cells alone, in parallel (`--test-threads=2`), 20 runs.
- `all`: the eleven native cells of `tier_transfer.rs` in one process at the default thread count, 20 runs.

Every run's output teed raw; the reader counts, per cell, passes and failures and prints every `HOLD READING` line.

**Hypotheses, each with its sign in the readings, stated before any run.**

- H1, the observation is delayed: a failing run's `first_item_seen_ms` is at least 300 and its `longest_poll_ms` or
  `longest_gap_ms` is at least 100 (the cell's thread was held inside or between CUDA calls while its own hold ran
  out). The KV item landed early; the cell saw it late.
- H2, the spans escaped the hold: a failing run's `first_item_seen_ms` is under 300 with the batch landed (the span
  copies ran before the spin ended): an engine ordering defect.
- H3, the KV item landed late: `first_item_seen_ms` at least 300 with every poll and gap under 100 ms (the poll loop
  ran and the item was not done): the item's copy queued behind the hold or behind other work: an engine or fixture
  ordering defect.

**Decision rule.**

- The reproduction is valid only if at least one timed-hold cell fails in the 40 runs. If none does, the hold is
  repeated with 100 runs of each arm; if none fails there, the finding is recorded `not reproduced on this tree` with
  its receipts and the lane goes on to item 2 without a code change to the cells.
- H2 or H3: the defect is in the engine's ordering (or the cell's own fixture order); the fix goes there, with a census
  pinning the order.
- H1: the defect is the cells' premise that a wall-clock window on a shared context is theirs alone; the fix makes the
  held state independent of the observer's timing (the spans held by a gate the cell itself opens after it has read the
  intermediate state, not by a timed spin), with the gate's own rule stated in its commit. A fix of the harness
  (thread count, serial runs, a lock around the cells) is not a fix.
- Mixed readings: each failing run is classified on its own reading; every class present is fixed.

**Acceptance of the fix** (5090 now; the target card in the next sitting):

- The `all` arm 100 of 100 runs green, then the `pair` arm 100 of 100 green, in one hold, on the fixed test binary.
- Every cell's serial run green (`--test-threads=1`, the batteries' current form).
- The fixed cells still carry their red arms: each timed-hold cell's intermediate-state check still fails if the spans
  are not held (a scratch arm that opens the gate before the check, run once, must fail the rule-2 assertion; banked).
- On the target card: the `all` arm 20 of 20 in the sitting's unit cells.

**What each card can decide.** The 5090 decides the cause and the fix's correctness on this host. The target card's
arm checks that the fix holds on another host and driver; it cannot reproduce the original failure on demand.

**Budget.** 0.25 agent-day.

## 2. The reproduction, as it ran (`rtx5090-day37/finding5/repro/`)

- Test binary `0849f96628e8d5ca..` (the instrument `cdfe75263`), one hold 11:45:09Z to 11:46:01Z, no compute app at
  the start or the end, card telemetry `card-250ms.csv`. The pair arm 20 runs, the all arm 20 runs, each run's output
  in `raw/`.
- **The reproduction is valid**: pair 17 of 20 runs FAILED (21 cell failures: `d2h_span_batch` 8, `h2d_span_batch`
  13); all 20 of 20 runs FAILED (61 failures over the 20 runs: `a batch with a running span has not landed` 40,
  `the owner thread does not wait for the fill at the attach` 19, and once
  `d2d_early_reader_fault_is_refused_by_the_receipt`'s `the early reader saw the fresh plane`).
- **Every reading, verbatim form** (N=61, every one a failing cell: libtest captures a passing cell's stderr, so the
  passing readings were not kept; the next hold runs with `--nocapture`): `batch_landed_at_first_sight=true`,
  `first_item_seen_ms` 302.55 to 1011.23, `polls=1`, `longest_poll_ms` at most 0.02, `longest_gap_ms` 0.00. Example:
  `HOLD READING cell=h2d_span_batch first_item_seen_ms=313.95 polls=1 longest_poll_ms=0.01 longest_gap_ms=0.00
  batch_landed_at_first_sight=true`.
- **Against the signs of section 1, as they read.** Not H2 (the first sight is never under 300 ms). Not H3 as worded
  (the poll loop did not run with the item pending: its first poll saw everything landed). Not H1 as worded either:
  H1's sign put the delay inside or between polls, and both read 0.01 ms or less. What the readings show is H1's
  mechanism (the cell's thread did not observe the batch until its own hold had run out) with the delay located
  BEFORE the first poll: between the hold's enqueue and the loop, where each cell builds its spans (pinned and
  device allocations, and in the D2H cell a pageable host-to-device copy) and submits them. The filled cell's attach
  assertion, which times exactly the span build and the submit, failed in 19 of 20 all runs. The cause is not yet
  placed at a call.

## 2a. Amendment, committed before the next hold: a step clock to place the delay

- The three timed-hold cells time every CUDA-touching step between the hold's enqueue and the first poll (each span
  build's pinned allocation, device allocation and host-to-device copy, each submit call), with the start offset from
  the hold's enqueue and the duration, and print one `HOLD STEPS cell=<name> <label>@<start>+<dur> ...` line beside
  the `HOLD READING`. Test code only; no assertion moves; the evaluation order of every call is unchanged (a span set
  is built before its submit, as the argument evaluation did).
- The same stress protocol, one hold, runs with `--nocapture` so every reading and step line of a passing cell is kept.
  20 runs each arm. No decision in section 1 changes; the decision rule is applied to the step that holds the thread.
