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
