# WP-A day 53: OWED item 21, the server test that failed once under the full suite (a reproduction first)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `3e2e7f5ed` (P2 reverted; the publication split stays). The
lead's order after DAY52: item 21 first, pre-registered like any item, with a reproduction under the suite's concurrency
first. CPU only: every cell runs on this rig under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=20G`.

## 1. Pre-registration (committed before the probe and before any reproduction run)

**What is known.** One full `cargo test -p memra-server --lib` run (DAY52 section 2) read
`test tests::responses_carry_rate_limit_headers_and_slot_frees ... FAILED`, panicked at
`crates/memra-server/src/lib.rs:22740:9`. That line is `assert_eq!(resp.status(), StatusCode::OK);` for the test's
second request, a streaming `/v1/completions`: the stream was answered with a status other than 200. (The first record
named the next assertion; corrected in OWED item 21 and DAY52 section 2.) The status value was not kept. The rerun of
the suite passed, and the test alone passed 6 of 6. The test holds `drain_lock()`.

**Where a non-200 can come from on that request** (`completions_with_admission`, read from the code), each tied to
process-global state or to time:

- H1, the admission reservations: `reserve_pending_admit` sheds with 429 (`shed_queue` when the lane's backlog is at its
  bound, `max_queue_depth(cap)` = 4 x the lane cap by default; `shed_deadline` when the lane waits and the estimate
  exceeds the deadline). The backlog is `worker::ADMISSION_RESERVATIONS[lane]`, a process-global static every
  concurrently running handler test raises and its fake worker releases at admission; a test that does not hold
  `drain_lock()` can hold reservations while this test runs.
- H2, the drain flag: `draining()` reads the process-global `DRAINING`; a test that raises it without `drain_lock()`
  answers every concurrent handler 503 (`draining`).
- H3, the route registry: `route_telemetry::lookup` is process-global; a route registered for the model name `m` by a
  concurrent test sends this request through the route's own admission (`reserve_route_admit`).
- H4, a deadline under load: `peek_admission` and `peek_first_token` answer 408 or 504 (`deadline_exceeded`) if the fake
  worker thread is starved past the request's default deadline.
- H5, anything else: the probe records it.

**Step 1, the probe (test-only; its own commit; removed with the fix).** In this one test, when the streaming response's
status is not 200, print before the same assertion fails: `ITEM21 PROBE status=S code=C message=M reservations=[i, j,
h] draining=D`, where `C` and `M` are the body's `error.code` and `error.message`, and the reservations and the drain
flag are read from the process-global state at that moment. On a 200 the test is unchanged.

**Step 2, the reproduction.** Arm A, the shape that failed: the full `cargo test -p memra-server --lib` on the probe
tree, default test threads, under the CPU quota above, 40 runs, each run's whole output kept (`day53/arm-a/run-NN.log`;
no pipe before the log). Arm B, only if arm A reads no failure of this test: the same 40 times with `--test-threads 48`
under `CPUQuota=400%` (more contention). Every failure of any test in any run is recorded by name.

**The rule, stated before any run.** Reproduced when at least one run fails this test with a probe line; its code
places the path: `shed_queue` or `shed_deadline` place H1 (the probe's interactive reservations must be above zero);
`draining` places H2; a route code places H3; `deadline_exceeded` places H4; any other code is H5, recorded as read.
Not reproduced when arms A and B read 80 runs with no failure of this test: then a targeted stress (this test looped
beside each suspected peer set) is pre-registered anew. The fix is pre-registered after the placing, with its own
clauses (the placed state's isolation, and this test green in N full-suite runs under both arms' contention).

**What each rig decides.** This rig only (CPU; no card).

**Budget.** 0.2 agent-day: the probe and the runs 0.1, the placing and the fix's pre-registration 0.1.

## 2. The probe (`2f1ea5a2b`) and the reproduction, as run (`day53/`)

- The probe built and passes alone (`1 passed`); clippy `-D warnings` clean. Both arms ran the same test binary
  (`binary.sha256` in each arm's directory), tree `2f1ea5a2b`.
- **Arm A** (40 full suites, default threads, `CPUQuota=1200%`, 24 CPUs): `39 rc=0`, `1 rc=101`. The target test passed
  in all 40 (no probe line). The one red run (`run-14.log`) failed a sibling handler test:
  `tests::same_effort_value_resolves_identically_on_every_surface`, `assertion left == right failed: /v1/responses
  rejected effort "none" .. left: 429 right: 200`: a handler request answered 429 under the full suite.
- **Arm B** (40 full suites, `--test-threads 48`, `CPUQuota=400%`): `33 rc=0`, `7 rc=101`. The target test passed in
  all 40. The red runs failed three other tests, each a timing assertion under starvation:
  `tests::an_extended_stream_commits_prefill_then_injects_the_original_deadline` (4 runs, `the bridge waited for the
  first-token deadline instead of committing`), `worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress`
  (4 runs, `heartbeat declared stalled: .. no forward progress for 80 ms (.. threshold 50 ms)` and `normal decode
  stopped at 9 steps`), `dsv4_serve::c4_host_budget_tests::coalesced_rows_each_get_their_own_token_once_per_step` (1
  run).
- **Verdict, as registered: not reproduced** (80 runs, no failure of the target test). By the rule, a targeted stress
  is pre-registered anew (section 3).
- **What the runs and the code say** (a reading, no verdict): arm A's sibling 429 is the H1 class. The tests that WRITE
  the process-global admission counters serialize on their own lock, `admission_counters_guard()`, and some of them
  set the interactive backlog to the queue bound for their whole body
  (`the_queue_bound_sheds_with_429_retry_after_and_the_ratelimit_trio` swaps `ADMISSION_RESERVATIONS[interactive]` to
  `max_queue_depth(cap)`); the handler tests that READ those counters through a real request hold `drain_lock()`
  instead (the target and `same_effort_value..` both), so nothing orders a reader against a writer. A handler request
  that runs inside a writer's window sheds 429 `shed_queue`. Seven tests in lib.rs write the counters under that lock
  (the swap and store tests), eleven more hold it (the route tests).
- Arm B's three reds are a different class (timing thresholds under a starved CPU); recorded here by name as the rule
  asks, and added to OWED as item 22 (their placing is not item 21's).

## 3. The targeted stress, pre-registered (before it runs)

**Arms** (the same probe binary; this rig; `CPUQuota=1200%`):

- T1: the target test beside every test that holds `admission_counters_guard()`, 200 runs of `cargo test -p
  memra-server --lib -- responses_carry_rate_limit_headers_and_slot_frees <each writer and route test by name>`
  (default threads), each run's output kept.
- T2, the control: the target test alone, 200 runs.
- T3, the drain flag's peers (H2): the target beside every test that stores `DRAINING`, 200 runs.

**The rule.** H1 placed when T1 reads at least one failure of the target whose probe code is `shed_queue` or
`shed_deadline` with the interactive reservations above zero, and T2 reads none. H2 placed when T3 reads a failure with
code `draining`. If T1 and T3 read no failure, the item stays `not reproduced` with 480 more runs recorded, and the next
step is a deterministic cell (the target's handler call inside a held writer window) whose pre-registration comes
first.

**What follows a placing.** The fix is pre-registered with its own clauses before its code: one ordering rule for every
test that reads the process-global admission state through a handler (the same lock as the writers, or a state that no
longer reads the global backlog), the target and its siblings green in arm A's and arm B's full-suite shapes.

## 4. The targeted stress as run, and the next cells pre-registered

- T1 (the target beside the 17 tests that hold `admission_counters_guard()`: the seven counter writers and the ten
  route tests; 17 of 17 names matched by `--list`), T2 (the target alone) and T3 (the target beside
  `draining_rejects_new_requests_with_503_and_retry_after`, the one test that raises `DRAINING`, under `drain_lock()`),
  200 runs each, the probe binary `092bd386..` as arms A and B: **200 of 200 green in each** (`day53/t{1,2,3}/`).
- **Verdict, as registered: not reproduced** (600 more runs; section 3's rule said 480, a counting slip in the text,
  the runs are 600). The deterministic cell comes next, and with it a longer run of the shape that did fail.
- The reading: the writers' windows are a few milliseconds and 18 tests start together; in the full suite the 900-odd
  tests keep the runner's 24 threads busy for 6 s, and a handler request lands in a writer's window by chance (arm A
  caught a sibling once in 40 runs; the target failed once in about ten full runs on day 52).

**Pre-registered now (before either runs):**

- A' (the observation): 200 more full suites in arm A's shape (default threads, `CPUQuota=1200%`, the same probe
  binary), each output kept (`day53/arm-a2/`).
- D (the mechanism, a test-only cell `day53_a_handler_request_inside_a_writer_window_sheds_429`): holding
  `admission_counters_guard()`, the interactive backlog swapped to `max_queue_depth(cap)` (restored on drop, the
  writers' own idiom), the target's streaming `/v1/completions` on a fresh fake state; the cell asserts the answer is
  429 with code `shed_queue`, and that the same request after the restore answers 200 and holds the slot.

**The rule.** H1 is **placed by observation** when A' catches the target with probe code `shed_queue` or
`shed_deadline` and interactive reservations above zero; another code places its own hypothesis. If A' does not catch
the target, H1 is **placed by mechanism** when D passes and at least one handler test in arm A or A' failed with a 429
(arm A already read one): the target's own failure then stays unobserved, and that is said. If D fails, nothing is
placed and the next step is pre-registered anew. The fix follows either placing, pre-registered with its own clauses.
