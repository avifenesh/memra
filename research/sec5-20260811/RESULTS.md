# cx-sec5 results

## Status

**PASS.** All three independent fixes are committed, each item-specific full crate gate is green,
and the final base-to-HEAD diff/commit audit is clean.

## Baseline

At starting revision `0e890ccf72c539e040624dd3c3784586c7e419ed`:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 178 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 1 — compile-time watchdog (`bc4d3a70`)

- The bounded per-model queue is now owned by a compiler supervisor. Each accepted job runs on a
  disposable CPU worker with the request deadline as its hard wait budget.
- A normal worker returns its warmed, lazily initialized `ConstraintFactory` to the supervisor, so
  successful compiles retain the existing factory reuse. A timeout or panic discards that state,
  detaches the worker without joining it, and lets the next queued job use fresh compiler state.
- An overrun publishes `ConstraintCompileFailure::TimedOut`, which follows the existing retryable
  constraint-timeout error path. The CUDA scheduler tick remains uninvolved.

Regression evidence:

- Before the fix, `compiler_abandons_runaway_job_and_drains_next_job` failed after 500 ms with
  `fresh compile did not drain while the first worker was stuck: Timeout`.
- After the fix, that regression passes while job 1 remains held, observes its timeout, and receives
  job 2's compiler result from fresh state.
- `slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress` also passes: the
  request receives the existing overloaded timeout classification while normal token events and
  the worker heartbeat keep advancing.

Item gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 179 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 2 — per-tenant ADSD self-baseline fallback (`ad08837f`)

- The cross-tenant comparator remains preferred whenever its post-exclusion population meets the
  existing 16-sample and 512-drafted-token floors.
- If no cross-tenant comparator is eligible, the detector derives a historical baseline from the
  same tenant's older model-history rows. It skips the preceding seven tenant rows before the
  current observation enters the eight-request signal window, keeping the populations disjoint.
- When a historical-baseline incident latches, that comparator is retained until recovery so a
  persistent collapse cannot adapt its own baseline downward and falsely rearm.
- Detection remains observational only. Verification, scheduling, cache, routing, and rate-limit
  behavior are unchanged.

Regression evidence:

- Before the fix, `adsd_detector_fires_on_single_tenant_historical_collapse` failed because the
  synthetic single-tenant collapse emitted 0 incidents instead of 1.
- After the fix, all four ADSD tests pass: cross-tenant collapse emits exactly once, single-tenant
  collapse emits exactly once, ordinary noise emits none, and sustained collapse stays latched.
- `docs/SERVING.md` documents comparator preference/fallback/rearm behavior, and the sec4
  `ad08837f` follow-up entry now records closure.

Item gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 180 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Item 3 — operator-only global prefix/LCP aggregates (`38bca65c`)

- `lcp_histogram`, global `prefix_cache_hits/misses/inserts/evictions/hit_tokens`, and the global
  `cache_hit_token_ratio` now populate only inside the existing operator-scope block.
- Completion credentials retain the safe base counters and their filtered `tenants` row. The
  tenant row's own `prompt_tokens_in`, `cached_tokens_in`, and `cache_hit_token_ratio` are unchanged.
- `docs/SERVING.md` now identifies the global prefix/LCP fields as operator-only at the route,
  visibility-boundary, cache-metering, and keyring descriptions. The sec4 `38bca65c` follow-up
  entry records closure.

Regression evidence:

- Before the fix, `prefix_aggregate_metrics_are_operator_only_but_tenant_ratio_remains` failed on
  a completion credential receiving `lcp_histogram`.
- After the fix, the same test proves all seven global fields absent for the completion key,
  present with seeded values for the operator token, and preserves `t:acme`'s own 0.4 ratio.
- All seven focused metrics/auth tests pass.

Item gate:

```text
nice -n 15 taskset -c 0-7 cargo test -p memra-server --release
test result: ok. 181 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Final gate evidence

- Starting baseline: 178 passed, 0 failed, 0 ignored.
- Item 1: 179 passed, 0 failed, 0 ignored.
- Item 2: 180 passed, 0 failed, 0 ignored.
- Item 3: 181 passed, 0 failed, 0 ignored.
- `git diff --check 0e890ccf72c539e040624dd3c3784586c7e419ed..HEAD`: clean.
- `git rev-list --count 0e890ccf72c539e040624dd3c3784586c7e419ed..HEAD`: 3 commits.

## Lane constraints

No GPU runtime or GPU test, benchmark, merge, tag, push, release, serving-default change, or
performance-board change was performed. `cargo fmt` was not run.
