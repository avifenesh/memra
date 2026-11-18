# The constraint compiler tests' 100 ms waits (memra#1335)

Verdict: fixed, in the test module only. A test of `crates/memra-server/src/constrained.rs` failed on a loaded host although the code was right, and its sibling has the same waits. Both now wait for the event they are about, and the deadline a runaway compile must outlive gives the supervisor a second to pick the job up.

## What failed

`constrained::tests::compiler_refuses_at_cap_then_rearms_after_runaways_finish` failed in two of five full `memra-server` library suite runs in one night, on a host at a 1-minute load of 55 to 98 on 32 threads, with the suite at nice 15:

```
runaway compile did not start: Timeout
```

Alone it passed every time (10 of 10), and the suite rerun right after each failure passed.

The test submits a compile with a 100 ms deadline, then gives the worker thread 100 ms to send its start signal. Two things can be late on a loaded host, and both end in the same message:

- the worker thread starts more than 100 ms after the submit;
- the supervisor thread wakes more than 100 ms after the submit. It drops a job whose deadline has passed before it dequeued it (`if Instant::now() >= job.deadline { continue; }`), so no worker starts and no result comes.

Its sibling `compiler_abandons_runaway_job_and_drains_next_job` has the same two waits.

## The fix

Two constants in the test module:

- `COMPILER_EVENT_WAIT`, 30 s: every wait for something the code does by itself (a worker starting, the supervisor's result, a worker exiting, the rearm), and the deadline of a compile that returns at once. Such a wait ends at the event, so a passing run is not slower for it.
- `RUNAWAY_COMPILE_DEADLINE`, 1 s (was 100 ms): the deadline of a compile that is meant to run away. The supervisor has that long to dequeue the job. It waits each runaway's deadline out before it reports the timeout, so this one does cost time: the two tests take 1 s and 4 s, 4.0 s together, where they took 0.41 s.

Nothing outside `mod tests` changes.

## Red and green

`tools/mutate.py` induces each delay in the test code, on main's test code and on the fixed test code alike:

- `late-start`: the compile closure sleeps 300 ms before it sends its start signal.
- `late-dequeue`: the compile factory sleeps 300 ms on its first call, which the supervisor thread makes before it reads its queue.

`tools/dd-checks.sh` ran both tests in each cell (`logs/summary.txt`; one log per cell, with cargo's per-crate progress lines and the home path left out):

| test code | no delay | late-start | late-dequeue |
|---|---|---|---|
| main 5a6ae2438e | 2 passed | 2 FAILED, `runaway compile did not start: Timeout` | 2 FAILED, the same message |
| the fix | 2 passed | 2 passed | 2 passed |

The fixed tests then passed 20 of 20 runs, at 4.00 to 4.01 s each. The whole `memra-server` library suite on the fixed tree: 1,272 passed, 0 failed, 18 ignored. Clippy on `memra-server` (all targets, `-D warnings`) and `cargo fmt --all -- --check` are clean.

## Review fold (memra#1340)

The independent review approved the change and asked for two folds. Both are test code.

1. **A runaway is reported at its own deadline.** Main's 250 ms wait for the result also checked, by accident, that the supervisor reports a timeout at the job's deadline; the 30 s wait dropped that. Each runaway's deadline is now kept, and its result's `finished_at` (the supervisor's clock at the report) must fall at the deadline or less than `RUNAWAY_COMPILE_DEADLINE` after it (`assert_reported_at_its_deadline`).
2. **The third test with the same race.** `worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress` gave its compiler job the 100 ms deadline on the wall clock, so a supervisor that dequeued it later dropped it and the test failed after its 10 s start wait. The job now gets a 30 s deadline. The pending entry keeps the 100 ms deadline, which that test judges on its step clock.

`tools/mutate.py` has two more arms and `tools/dd-checks-fold.sh` ran them (`logs-fold/summary.txt`):

- `wrong-deadline`, a mutation of the production code: the supervisor waits `CONSTRAINT_COMPILE_TIMEOUT` (5 s) for a worker instead of the job's deadline.
- `worker-late-dequeue`: the third test's compile factory sleeps 300 ms on its first call.

| cell | test code | result |
|---|---|---|
| `wrong-deadline` | before the fold | 2 passed, in 20 s: the lost check (every runaway reported 4 s after its deadline, and nothing saw it) |
| `wrong-deadline` | the fold | 2 FAILED, `runaway compile 0 was reported 4.000227543s after its deadline` |
| `worker-late-dequeue` | main's `worker.rs` | FAILED after the 10 s start wait, `test compiler did not start: Timeout` |
| `worker-late-dequeue` | the fold | passed |
| `late-start` | the fold | 2 passed |
| `late-dequeue` | the fold | 2 passed |
| no mutation, the three tests | the fold | 3 passed |

The three tests then passed 20 of 20 runs, 4.00 s each. The whole `memra-server` library suite on the folded tree: 1,272 passed, 0 failed, 18 ignored. Clippy on `memra-server` (all targets, `-D warnings`) and `cargo fmt --all -- --check` are clean.

## Conditions

CPU only, on 18 pinned threads of a 32-thread host at nice 15, with a 1-minute load of 45 to 64 from other work during the first run's cells and 12 to 33 during the fold's. sm_120a build, release profile. The change is test code only. The impact rules class `constrained.rs` and `worker.rs` by path, so the push gate still asks device cells of it; it lands with a landing whose records cover them.

## Limits

A supervisor that wakes more than a second after a submit would still drop the runaway job, and the test would then fail after its 30 s wait with the same message. The deadline is a wall-clock instant fixed at the submit, so the test cannot remove that bound without a clock seam in the compiler; it is ten times the old one. The review measured the limit with a delay in the production code: a dequeue 900 ms late passes and one 1,100 ms late fails. The same second bounds how late the supervisor may report a runaway. The one-second margin rests on the induced-delay cells and on the plain runs at the loads stated here; no run measured the supervisor's real wake-up lag at a load of 55 to 98.
