# cx-sec7 results

## Status

**COMPLETE.** The sec6 constrained-compiler fail-closed latch now rearms after the outstanding
runaway count drops below its cap. The branch is ready for orchestrator review at fix commit
`194f8733`.

## Fix

- Moved abandoned compile-worker handles into a shared per-model `AbandonedWorkers` state.
- Kept the cap at four outstanding runaways. The latch is set exactly while the retained handle
  count is at or above that cap and cleared after finished handles reduce it below the cap.
- When latched, `try_submit` uses `try_lock` and removes only handles already reported finished.
  It neither waits for the tracker lock nor joins a worker on the request-admission path.
- The supervisor uses the same state, so queued jobs preserve the existing
  `AbandonedWorkerLimit` failure path and cannot start a fifth runaway worker.
- Added an operator-visible poison log and rearm log. The shared latch is also published as the
  operator-only `constraint_compiler_fail_closed` per-model 0/1 gauge; tenant metrics cannot see
  it.
- Preserved `AbandonedWorkerLimit` as a retryable overloaded 503 and changed its client message to
  describe temporary saturation with a short retry, without advertising a worker restart.

## Regression coverage

The existing sec6 cap test now proves the complete lifecycle:

1. Four blocked compiles time out and retain four live worker handles.
2. The next submit is refused with `AbandonedWorkerLimit` and no fifth worker starts.
3. The four workers are released and finish while the supervisor has no new job to wake it.
4. A later submit drives the non-blocking front-door reap, observes the latch rearmed, starts, and
   returns the test compiler's ordinary `Invalid` result.

The metrics authorization tests also prove that the gauge is present for an operator principal
and absent for a tenant principal.

## CPU-capped gates

Focused regression:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release \
  constrained::tests::compiler_refuses_at_cap_then_rearms_after_runaways_finish \
  -- --exact --nocapture

running 1 test
[constraint] model "test": compiler fail-closed (4 abandoned workers outstanding; cap 4)
[constraint] model "test": compiler rearmed (0 abandoned workers outstanding; cap 4)
test constrained::tests::compiler_refuses_at_cap_then_rearms_after_runaways_finish ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 181 filtered out
```

Required full gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 182 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

An earlier unqualified `--exact` selector matched zero tests and is not counted as evidence; it was
immediately corrected to the fully qualified focused command above. No GPU workload, merge, tag,
push, formatting command, or perf-board update was run.
