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

## 3. The step-clock hold, as it ran, and the two-thread probe pre-registered (`rtx5090-day37/finding5/steps/`)

- Test binary `4d03dc5ecbe7f3e9..` (`030f0f203`), one hold from 11:50:30Z, `--nocapture`: 40 of 40 runs FAILED (the
  pair arm 20 of 20, the all arm 20 of 20). The step clock places every delay at ONE call per cell, and that call
  returns when a 300 ms spin on the context ends. Verbatim examples (`raw/pair-1.log`, `raw/all-1.log`):
  `HOLD STEPS cell=d2h_span_batch .. delay@0.00+0.03 .. htod-pageable@7.35+294.53 pinned-alloc@301.88+1.74
  submit-busy@303.63+0.01`; `HOLD STEPS cell=h2d_span_batch .. delay@0.00+0.03 .. device-alloc@9.88+290.75
  submit@300.64+2.64 ..`; `HOLD STEPS cell=h2d_span_filled_batch .. delay@0.00+135.44 pinned-alloc@135.45+306.03
  device-alloc@441.48+3.63 ..`. So a pageable host-to-device copy, a stream-ordered device allocation, a pinned host
  allocation and a kernel launch each held their thread for up to about 300 ms in the parallel runs, and none did in
  the serial runs the batteries pass.
- **The probe, pre-registered before it runs** (`day37-hold-probe/`, detached, cudarc 0.19.8 as the engine, nothing of
  the engine). Holder thread H launches a 300 ms `%globaltimer` spin on its own stream and, 20 ms later, performs one
  action X: `none`, `free-host` (cuMemFreeHost of a 4 MiB pinned buffer allocated before the spin), `malloc-host`
  (cuMemHostAlloc 4 MiB), `free-async` (a stream-ordered alloc and free), `stream-sync-own`, `event-sync-own`,
  `ctx-sync`. Victim thread V, 40 ms after the spin's launch, times one call Y on its own stream: `alloc-zeros`,
  `malloc-host`, `htod-pageable` (4 MiB), `launch`, `event-query`. Each (X, Y) pair runs 3 times in a two-thread form
  and a serial form (X then Y on one thread, the spin on a second stream), event tracking on (the cells' context), then
  the whole matrix again with event tracking off (the engine's context). One bounded 5090 hold, raw output teed.
- **What the probe decides, stated now.** An X whose two-thread column holds Y for 200 ms or more in all three runs,
  where the `none` row does not, is a holding action; the serial column says whether the same X holds the calling
  thread's own later calls. If `none` itself holds Y, the spin alone is the cause and no X is needed. The fix is then
  chosen by where the holding action sits: in the cells' own fixtures (a synchronizing call that a concurrent cell's
  timed window cannot survive), or in the engine (a synchronizing call the engine itself makes on the owner thread
  while copy-stream work is queued, which would hold the owner in production too). The fix's acceptance stays section
  1's.

## 4. The probe, as it ran (`rtx5090-day37/finding5/probe/`), and its cross-context extension pre-registered

- Binary `day37-hold-probe` (hash in `binary.sha256`), one hold 11:55:39Z to 11:57:46Z, no compute app. Verbatim
  pattern over both tracking modes (`tracking-on.log`, `tracking-off.log`; N=3 each cell): the `free-host` row holds
  the victim's `alloc-zeros` 260.46 to 260.99 ms, `event-query` 260.49 to 261.01, `htod-pageable` 260.82 to 261.07 and
  `malloc-host` 262.03 to 262.28 in the two-thread form, 3 of 3 each, and holds nothing in the serial form (0.04 to
  1.97 ms); `launch` is not held (0.02 to 0.36). Every other X (`none`, `malloc-host`, `free-async`, `stream-sync-own`,
  `event-sync-own`, `ctx-sync`) holds no Y past 8.54 ms in any run. Event tracking on or off reads the same.
- **The holding action is `cuMemFreeHost`.** While one thread's pinned free waits for the context's outstanding
  device work (here the other thread's 300 ms spin, entered 20 ms in, so about 280 ms), every other thread's
  allocation, event query, pageable copy and pinned allocation on that context waits with it. In the step-clock
  receipts the H2D cell's thread sits 330 ms between its two refused-attach span builds (`pair-1`: the bad=1 set built
  at -345 to -337 ms, the bad=2 set at -6.51 ms), which is the drop of the refused set's three `PinnedHostBuf`s, while
  the D2H cell's `htod-pageable` is held for 294.53 ms.
- **Where the engine frees pinned memory** (read, `pinned_host.rs`, `tier_transfer.rs`, `worker.rs`):
  `PinnedHostBuf::drop` and the contract lease backing's drop (`event.synchronize()` then `free_host`) call
  `cuMemFreeHost`. On the serving path the door keeps a promoted host entry and its leases (the H2D sources are
  retained twins, `host_promote_finish`), so a lease backing is freed only when a host entry leaves the tier (host LRU
  eviction at insert, a VERIFY FAILED drop, the tenant purge, the latch); the staging set frees only at the latch. In
  the serving process the owner thread is the context's only CUDA caller, so the cross-thread hold does not arise on
  one card; the free still waits for whatever the copy stream holds at that moment.
- **The cross-context extension, pre-registered before it runs.** The question the fix's layer turns on: is the
  held lock per context, or does a pinned free in one context hold a thread working in another context (the serving
  shape of a multi-card process, one owner thread and one context per card)? On one card: the holder in a CREATED
  context (`cuCtxCreate`) launches the 300 ms spin there and runs `free-host` (and `none` as the control); the victim
  in the primary context times the same five calls; then the roles reversed. 3 runs each, one hold. Decision: if the
  created-context free holds the primary-context victim for 200 ms or more in 3 of 3 runs, the lock is wider than a
  context, and the engine's pinned frees are a cross-card hazard in a multi-card process; if not, the hold is a
  same-context property only.

## 5. The cross-context probe, as it ran, and the fix pre-registered (revised design)

- Binary `binary-cross.sha256`, one hold 12:05:21Z to 12:05:43Z (`cross.log`, N=3 each). A `free-host` in the CREATED
  context holds nothing in the primary context (`alloc-zeros` 0.05 to 0.20 ms, `event-query` 0.03, `htod-pageable`
  0.33 to 0.34, `malloc-host` 1.66 to 1.70), nor the reverse (0.02 to 2.11 ms); the same-context control in the same
  hold repeats the hold (`alloc-zeros` 260.69 to 260.95, `event-query` 260.48 to 260.93, `htod-pageable` 260.83 to
  261.27, `malloc-host` 262.28 to 262.51). **By section 4's rule the hold is a same-context property only**: a pinned
  free in one context does not hold a thread working in another. A multi-card process (one context and one owner
  thread per card) is not exposed to it across cards.
- **The defect, named.** The engine's contract is one CUDA owner thread per context (`CudaTransfers::check_thread`
  refuses any other thread; the server runs one worker per device). The native cells break it when run in parallel:
  each builds its own `CudaTransfers` on the device's PRIMARY context, so several owner threads share one context, and
  in that shape a pinned free on any one of them (`cuMemFreeHost`, reached through `PinnedHostBuf::drop` or a lease
  backing's drop) holds every other owner's driver calls until the context's device work drains, including another
  cell's 300 ms hold. That is the whole of finding 5: the engine's code under test is not wrong, and the cells were
  not testing the shape the engine runs in.
- **Why section 1's H1 fix is not taken.** Section 1 named a gate the cell opens after it reads the intermediate state.
  With several owners on one context a gate deadlocks instead of flaking: another owner's pinned free waits for the
  gated work while it holds the gate-opener's event query. The rule-2 design is revised here, before any fix code; the
  acceptance of section 1 is unchanged.
- **The fix (engine test code only).** Each of the eleven native cells of `tier_transfer.rs` owns its context: the
  cell builds it with `CudaContext::new_non_primary(0, 0)` (cudarc 0.19.8, `cuCtxCreate_v4`, destroyed on drop) instead
  of retaining the primary, and drives it from its one thread, the shape the engine's contract states. Nothing else in
  any cell changes: every assertion, hold, pattern and step stays; the step clock and the hold reading stay as the
  cells' diagnostics. A CPU census pins it: every native cell in the module builds its context with
  `new_non_primary`, and none calls `CudaContext::new(`.
- **Acceptance (section 1's, restated, unchanged).** On the fixed test binary, one hold: the `all` arm 100 of 100 runs
  green, then the `pair` arm 100 of 100 green; every cell's serial run green; the red arm: a scratch build of the fixed
  cells with the hold removed from `h2d_span_batch` and `d2h_span_batch` (the `delay_on` line deleted), run once, must
  fail both rule-2 assertions (banked, not committed as code); on the target card the `all` arm 20 of 20 in the next
  sitting's unit cells.
- **A reading beside it (not a clause).** The server's native door cells (`option_b_*`, `option_c_*`, built on
  `Engine::new(0)`, the primary context) run in parallel once, 10 runs, to find whether the same class reaches them;
  any failure there is recorded with its reading and gets its own pre-registered fix.

## 6. The per-cell-context fix, as it ran (`rtx5090-day37/finding5/fix/`): acceptance FAILED, the fix reverted

- Test binary on `8c8f1fa1f`-era tree (the fix commit, binary hash in `fix/binary.sha256`), one hold 12:08Z to
  12:16:44Z, no compute app at either end. Verbatim (`fix/run.log`): the pair arm `100 green of 100`; the serial arm
  3 of 3 (`test result: ok. 11 passed`); the all arm **75 of 100** runs green. The 25 failing all runs carry 35 cell
  failures: `h2d_span_batch` 24, `d2h_span_batch` 8, `h2d_span_filled_batch` 3 (`a batch with a running span has not
  landed` 32, `the owner thread does not wait for the fill at the attach` 3).
- The step clock places every one at a PINNED ALLOCATION now, held 164.89 to 306.61 ms across the cells' own
  contexts (`all-12.log`: `HOLD STEPS cell=h2d_span_batch .. pinned-alloc@9.42+295.55 ..`; `all-24.log` the D2H cell
  `pinned-alloc@6.27+295.81`, then `@334.66+192.94`). The cross-context probe of section 5 showed a pinned free in
  another context does not hold a pinned allocation; so something else, reached only when eleven cells in eleven
  contexts overlap, holds `cuMemHostAlloc` across contexts. Candidates by elimination: context creation or
  destruction (every cell now creates and destroys a context; the pair arm's two cells create theirs together and end
  together, and it read 100 of 100).
- **As registered, the fix does not meet section 1's acceptance (all arm 100 of 100); it is refuted and reverted in one
  commit.** Its receipts stay here (`fix/`). Nothing in section 1 moves.
- **The probe's teardown extension, pre-registered before it runs.** The victim's own context carries the 300 ms spin
  (the cells' shape: a cell's calls held until its own hold ends); 20 ms in, a holder thread performs X in ANOTHER
  context: `none`, `free-host` (in its own idle context), `ctx-create` (create a new context), `ctx-destroy` (destroy an
  idle created context); 40 ms in, the victim times Y in its own context (`malloc-host`, `alloc-zeros`,
  `event-query`, `htod-pageable`). Victim primary and victim created, 3 runs each, one hold. Decision: an X that holds
  Y 200 ms or more in 3 of 3 runs, with `none` not holding, is a cross-context holding action. The next fix is
  pre-registered on that reading.
