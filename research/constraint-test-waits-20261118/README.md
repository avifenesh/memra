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

## Conditions

CPU only, on 18 pinned threads of a 32-thread host at nice 15, with a 1-minute load of 45 to 64 from other work during the cells. sm_120a build, release profile. The fix needs no device evidence: it changes test code only.

## Limits

A supervisor that wakes more than a second after a submit would still drop the runaway job, and the test would then fail after its 30 s wait with the same message. The deadline is a wall-clock instant fixed at the submit, so the test cannot remove that bound without a clock seam in the compiler; it is ten times the old one.
