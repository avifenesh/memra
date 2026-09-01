# cx-sec6 results

## Status

**PASS.** All three scoped items have independent green CPU-only crate gates, the flag-drift gate
is green, and the branch is ready for orchestrator review.

## Baseline

At starting revision `126e6642a11727d7f0ee95254ef3dc2c17cf289f`:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 181 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 1 — abandoned constraint-worker cap (finding `e8e860ed`)

- Each model's compiler supervisor retains handles for timed-out disposable workers and reaps
  finished-late workers non-blockingly before admitting more work.
- At four simultaneously outstanding abandoned workers, the supervisor latches a per-model
  fail-closed state until the server worker restarts. This bounds live runaway threads, their
  warmed full-vocabulary `ConstraintFactory` state, and their thread stacks.
- New submits receive `ConstraintSubmitError::AbandonedWorkerLimit`. Jobs already queued across
  the latch receive `ConstraintCompileFailure::AbandonedWorkerLimit`. Both paths map to a clear,
  retryable overloaded-class response and neither starts another compile worker.
- Normal success still returns the warmed factory, and ordinary single timeouts still drain the
  next queued compile through fresh state.

Regression evidence:

- Before the fix, `compiler_refuses_work_after_four_outstanding_runaways` failed with
  `fifth compile was accepted after four outstanding runaways`.
- After the fix, the regression observes four sequential timeouts, proves exactly four workers
  started, and sees the fifth submit refused with the distinct safety-limit variant.
- `compiler_abandons_runaway_job_and_drains_next_job` and
  `slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress` remain green.

Item gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 182 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 2 — constrained-decoding compile isolation (finding `4846ed17`)

`docs/SERVING.md` now records the complete admission boundary next to the constrained-decoding
runtime contract:

- schema compilation and first-use full-vocabulary factory work run off the CUDA scheduler tick on
  the bounded eight-job per-model CPU queue;
- the 512 KiB byte, 64-level depth, and 32 Ki-value node limits are named with their source
  constants and documented as loud pre-admission 400s;
- the five-second deadline and exact retry-with-smaller-schema overloaded timeout are explicit; and
- four outstanding abandoned workers fail-close that model's constrained compiler until the server
  worker restarts, without affecting unconstrained or already-active sessions.

The subsection links the sec5 watchdog evidence and this sec6 cap evidence.

Item gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 182 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 3 — `MEMRA_OPTI_CONTROLLER_Q` catalog (finding `7073b082`)

- `docs/FLAGS.md` now records the memra-server door as fresh-process-only, finite `[0,1]`, and
  absent by default with an explicit **NO-GO** status.
- The row preserves why the current controller stays off: every corrected threshold lost to
  serial, the merged seam, and plain because low-q refusals still pay the online shadow-draft tax.
- It links all opti2 revival conditions: frozen trained/calibrated retained-trace selector; zero-cost
  low-confidence plain path; a retained segment that clears the re-solved economic threshold; and
  an interleaved N=5 admitted win with zero low-q regression followed by the 2x PRO 6000 battery.
- The controller was removed from `research/docsync3-20260811/flags-drift.txt` now that it is
  documented, so deleting the catalog row would become newly uncovered again.

Item gates:

```text
tools/check-flags.sh
check-flags: no new drift beyond research/docsync3-20260811/flags-drift.txt

nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 182 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Final gate evidence

- Starting baseline: 181 passed, 0 failed, 0 ignored.
- Item 1: 182 passed, 0 failed, 0 ignored.
- Item 2: 182 passed, 0 failed, 0 ignored.
- Item 3: 182 passed, 0 failed, 0 ignored; flag drift green.
- Branch shape: four commits — one opening progress commit followed by one commit per item.

## Lane constraints

No GPU runtime or GPU test, benchmark, `cargo fmt`, merge, tag, push, release, serving-default
change, or performance-board change was performed.
