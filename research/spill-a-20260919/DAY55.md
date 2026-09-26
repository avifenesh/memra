# WP-A day 55: OWED item 22, five server tests that fail under CPU starvation (reproduce, place, fix each)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `dfc9ff6e4` (DAY54 recorded). The lead's order after DAY54: items
22 and 23 next (a flaky or slow suite hurts every lane's CI). CPU only, this rig, every command under
`systemd-run --user --scope`.

## 1. Pre-registration (committed before any reproduction run or code)

**The five tests and what their failures read** (DAY53 sections 2, 5 and 7):

| test | failed | the failing assertion |
|---|---|---|
| T-a `health::tests::no_progress_source_is_the_pre_fix_beat_age_verdict` | 1 of 400 (arm A's shape) | `with no source the published progress age IS the beat age .. left: 41 right: 40` |
| T-b `tests::an_extended_stream_commits_prefill_then_injects_the_original_deadline` | 4 of 40 (arm B) | `the bridge waited for the first-token deadline instead of committing` (elapsed not under 60 ms) |
| T-c `worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress` | 4 of 40 (arm B) | `heartbeat declared stalled .. 80 ms (.. threshold 50 ms)`; `normal decode stopped at 9 steps` |
| T-d `tests::deep_schema_fails_while_normal_decode_keeps_stepping` | 1 of 200 (arm A's shape) | `bad schema stalled or replaced the normal decode` (the normal request had finished) |
| T-e `dsv4_serve::c4_host_budget_tests::coalesced_rows_each_get_their_own_token_once_per_step` | 1 of 40 (arm B), 1 of 400 | the full-batch count (`>= 10` of the widths equal to 3) |

**What each one is, read from the code** (the hypotheses the cells test):

- T-a is a defect in the code under test. `WorkerHealth::snapshot` samples the clock for `beat_age_ms`, then calls
  `forward_progress_age_ms`, which samples it again (`beat_age_ms()` inside it). With no progress source the two are
  documented equal (the rollback seam's contract), but they are two samples; a millisecond boundary between them breaks
  the equality. `live()`'s message also prints a second beat-age sample beside the judged one.
- T-b, T-c and T-d assert logic (the bridge commits at its commit time, not at the deadline; the compile's expiry and
  the loop's progress are independent; a bad schema does not stall or replace a running decode) through wall-clock
  bounds a starved runner cannot meet (60 ms of elapsed time; 2 ms sleeps against a 100 ms deadline and a 50 ms stall
  threshold; a 64 x 5 ms fake decode racing the bad request).
- T-e asserts a scheduling outcome: three lanes keep 500 us batches full (`ROW_BATCH_WAIT`) most of the time. When the
  runner does not deliver the lanes within the window, a partial batch is the mechanism's correct answer.

**The reproduction harness.** Per test, alone (`--exact`), on the current tree's test binary: R0, 100 runs under
`CPUQuota=1200%`; R1, 100 runs under `CPUQuota=25%` (the cgroup is throttled for about 75 ms of every 100 ms period, the
shape of arm B's starvation). Raw output per run kept (`day55/<test>/r{0,1}/`). A test is reproduced when R1 reads at
least one failure of its recorded assertion.

**The fixes, by class, with their acceptance** (each lands as its own commit; the bounds each test states are kept):

- T-a (a defect): `snapshot` samples the clock once and derives both ages from that sample; `live()`'s message prints
  the beat age it judged with. The test is unchanged. A census pins one clock sample in `snapshot` and in the stall
  verdict. Accept: R1 0 of 100 and R0 0 of 100.
- T-b (a logic test on wall time): the test runs on tokio's paused clock (`start_paused`), which every timer on its path
  reads (`tokio::time`), so 10 ms, 60 ms, 80 ms and 150 ms are virtual and exact; every assertion and bound unchanged.
  Accept: R1 0 of 100; the red arm, a bridge that ignores its commit time (a scratch patch, never committed), fails
  the test in 10 of 10 runs.
- T-c (the same class, std time): the loop drives the compile expiry with a step clock (`start + steps x 2 ms`, the
  `now` argument `expire_constraint_compiles` already takes), so the deadline falls at step 50 and `normal_steps >= 10`
  is judged against the same 100 ms; the heartbeat verdict is judged on a clock the test controls (a test-only clock
  for `WorkerHealth`, advanced with the loop), against the same 50 ms threshold. Accept: R1 0 of 100; the red arm, a
  loop that blocks on the compile (a scratch patch), fails in 10 of 10.
- T-d (the same class, a race): the fake worker holds the normal request's stream open until the test releases it, so
  "the normal decode is still running when the bad request returns" is true by construction and is asserted after the
  bad request answered and after the normal stream delivered at least one token; the release then lets it finish and
  its 200 is asserted as before. Accept: R1 0 of 100; the red arm, a bad schema that blocks the worker (a scratch
  patch), fails in 10 of 10.
- T-e (a scheduling outcome): the correctness assertions stay exact (every row once, every result its own, widths
  within 1 to 3, the width sum). The full-batch count is replaced by the mechanism it depends on, which the coalescer
  controls and a starved runner cannot break: from per-deposit timestamps the test records, every batch narrower than
  the members still waiting at its dispatch has waited out `ROW_BATCH_WAIT` from its first deposit. The outcome count
  stays printed as a reading. This is a change of what the test claims, stated here so the lead can overrule it: the
  outcome depends on the OS scheduler, the mechanism does not. Accept: R1 0 of 100; the red arm, `ROW_BATCH_WAIT` set
  to zero (a scratch patch), fails in 10 of 10.

**The suite, after all five.** 100 full suites in arm A's shape and 100 in arm B's shape (DAY53): none of the five
fails; every other failure is recorded by name and becomes its own item.

**The rule.** A test whose R1 reads no failure is recorded `not reproduced` and still gets its fix only if its class is
a defect (T-a); a wall-time test that R1 cannot break is left unchanged and recorded. A fix that does not meet its
acceptance is reverted in one commit and its test re-read under a new pre-registration. No bound moves.

**Budget.** 0.5 agent-day: the reproduction 0.1, five fixes and their red arms 0.3, the suite runs 0.1.

## 2. The reproduction as run (R0 and R1; `day55/<test>/r{0,1}/`)

- The test binary `1e83f63e450ff9c1` (tree `dfc9ff6e4`), each test alone, 100 runs per arm:

| test | R0 (`CPUQuota=1200%`) | R1 (`CPUQuota=25%`) |
|---|---|---|
| T-a | 100 rc=0 | 100 rc=0 |
| T-b | 100 rc=0 | 99 rc=0, 1 rc=101 (`the bridge waited for the first-token deadline instead of committing`, lib.rs:17423) |
| T-c | 100 rc=0 | 100 rc=0 |
| T-d | 100 rc=0 | 100 rc=0 |
| T-e | 100 rc=0 | 100 rc=0 |

- **As registered:** T-b is reproduced; T-a, T-c, T-d and T-e are not by R1. By the rule, T-a gets its fix (a defect),
  T-b gets its fix, and T-c, T-d and T-e are left unchanged by this harness and recorded.
- **What R1 missed** (a reading): a lone test under a 25% quota mostly finishes inside one 25 ms quota slice; the
  failures came from contention, a test's threads waiting behind hundreds of runnable siblings. A harness that
  reproduces contention for one test is registered below; it does not move any bound.

## 3. R2, pre-registered (before it runs and before any fix's code)

- R2: per test, 100 runs of `day55/r2.sh`: the test (`--exact`) inside a scope at `CPUQuota=100%` beside eight CPU
  burners (`burn.py`, a busy loop) in the same scope, so the test's threads wait for CPU as they did in arm B. The
  burners contend only for that scope's one CPU; nothing outside it is loaded.
- The rule for T-c, T-d and T-e: reproduced when R2 reads at least one failure of the recorded assertion; then the fix
  section 1 names, with its acceptance. Not reproduced by R2: the test is left unchanged and the item records it.
- Every fix's acceptance (all five) gains R2 at 0 of 100 beside R1 at 0 of 100 (a stricter clause, added before any fix
  is written).

## 4. R2 as run (`day55/<test>/r2/`), and one addition to T-c's fix, pre-registered

- **The interruption.** The first R2 cell was cut by the rig's reboot during T-a's runs: 51 runs, 50 `rc=0` and run 51
  `rc=143` (the shutdown's signal; its log reads `test result: ok`), on a binary kept in `/tmp` that the reboot removed.
  The cell is void (incomplete, and its binary is gone); its logs are banked as `r2-interrupted/`, and R2 was re-run
  whole on a rebuilt test binary (`ef352d04c48a28b5`, the same tree `f44001fa7`, whose crates are `dfc9ff6e4`'s; the
  R0 and R1 binary read `1e83f63e450ff9c1`).
- **R2, 100 runs per test:**

| test | R2 | the failing lines |
|---|---|---|
| T-a | 100 rc=0 | |
| T-b | 98 rc=0, 2 rc=101 | `the bridge waited for the first-token deadline instead of committing` (2) |
| T-c | 67 rc=0, 33 rc=101 | `normal decode stopped at N steps` (N = 4 to 8; 25), `test compiler did not start: Timeout` (8) |
| T-d | 100 rc=0 | |
| T-e | 86 rc=0, 14 rc=101 | the full-batch count (the widths vectors, mostly width 1 and 2) |

- **As registered:** T-c and T-e are reproduced by R2 and get their fixes; T-b has its fix (R1 and R2); T-a has its fix
  (a defect); **T-d is not reproduced** by R1 or R2 (it failed once in A''s 200 full suites) and is left unchanged; the
  item records it open with its A' observation.
- **T-c, an addition before its code.** R2 found a second wall bound in T-c's harness: the test waits 50 ms
  (`recv_timeout`) for the held compile to START before its loop begins. That wait is plumbing (it orders the loop after
  the compile is running), not one of the test's claims; the fix waits for the start signal with a 10 s safety bound,
  whose expiry still fails the test by the same message. The claims' bounds (the 100 ms deadline, `>= 10` steps, the 50
  ms stall threshold) are unchanged.

## 5. T-b as built, and T-c's fix revised before its code

- **T-b** (`77d03de33`): `#[tokio::test(start_paused = true)]` with tokio's `test-util` as a dev-only feature
  (`Cargo.lock` unchanged); the test and its bounds unchanged. Its red arm (the bridge's commit time dropped, a scratch
  patch never committed) fails 10 of 10: `the extended stream must commit before first token: ()`
  (`an_extended_stream_commits_prefill_then_/red/`). A first red-arm run is void and kept as
  `red-void-stale-binary/`: the dev-dependency moved the test executable to a new name, and those ten runs used the
  stale one (they read 10 `rc=0` on a binary without the patch); the red arm is always grep-checked in its binary now.
- **T-c, the revision (section 1's text did not foresee this).** Driving the expiry with a step clock alone takes the
  test's teeth: a loop that blocks on the held compile would still reach step 50 and pass (only slower). The fix is
  therefore:
  1. the start wait bounded at 10 s (section 4);
  2. the expiry at `start + 2 ms x steps` (the deadline at step 50, `normal_steps >= 10` judged against it);
  3. a test-only virtual clock for `WorkerHealth` on the test's thread (`#[cfg(test)]`, a thread-local read by
     `now_ms()`, reset on drop), advanced 2 ms per step, so the 50 ms stall threshold is judged on the loop's own time;
  4. the teeth: every step's `resolve_constraint_compiles` call is timed on the wall clock and must return under 1 s
     (a non-blocking call; beside eight burners a step's scheduling gaps read tens of milliseconds, R2); a resolve that
     waits on the held compile fails at the first step.
  Red arm: `resolve_constraint_compiles` made to wait on its result channel for 2 s (a scratch patch): fails 10 of 10.

## 6. T-c as built, and T-e's fix revised before its code

- **T-c** (`965f6e9c1`): as section 5 states. Its red arm (`resolve_constraint_compiles` waiting 2 s on its result
  channel, a scratch patch grep-checked in its binary) fails 10 of 10: `a resolve blocked on the held compile
  (2.099645589s)` (`slow_constraint_compile_times_out_while_/red/`).
- **T-e, the revision (section 1's per-deposit timestamps cannot judge it).** Reading `Coalescer::step`: a batch is full
  when the waiting deposits reach `min(target, members - in_flight)`, so a batch of two is full while a row rides a
  batch in flight; the test cannot see `in_flight` at the leader's decision, and inferring it from its own callbacks
  races the leader. A timestamp assertion built that way would be flaky itself. The fix instead:
  1. `Coalescer` takes its window as a field (`Coalescer::new` keeps `ROW_BATCH_WAIT`; a `with_window` constructor for
     tests), the same program in production;
  2. `coalesced_rows_each_get_their_own_token_once_per_step` keeps every correctness assertion; its full-batch count
     is printed, not asserted;
  3. the mechanism gets two cells that a starved runner cannot break: `a_partial_batch_waits_out_its_window` (three
     members, one deposits alone with a 20 ms window: its batch runs one row, after at least 20 ms), and
     `a_full_batch_does_not_wait_for_its_window` (three members all deposit with a 10 s window: the batch runs all
     three rows in under 5 s).
  Red arms: the window ignored (`t0.elapsed() >= window` made `true`) fails the first cell; the fullness ignored (`full`
  made `false`) fails the second. Each 10 of 10, each grep-checked in its binary.
- This touches `dsv4_serve.rs`, which the DSv4 lane also edits; the change is the one field, one constructor and the
  tests, and it is flagged to the lead for the overlap.

## 7. T-e as built, the acceptance, and the suites (`day55/acceptance/`, `day55/suite-{a,b}/`)

- **T-e** (`1ba2af13b`): as section 6 states. Red arms, each grep-checked in its binary: the window ignored fails
  `a_partial_batch_waits_out_its_window` 10 of 10 (`a partial batch ran after 246.844us, inside its 20ms window`); the
  fullness ignored fails `a_full_batch_does_not_wait_for_its_window` 10 of 10 (`a full batch waited 10.00073687s of its
  10 s window`) (`coalesced_rows_each_get_their_own_token_/red-{window,full}/`).
- CPU cells on the fixes tip (`1ba2af13b`): server lib `928 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt.
- **Acceptance, R1 and R2 on the fixes tip** (test binary `2dc7614a5fdd193e`), 100 runs each:

| test | R1 | R2 |
|---|---|---|
| T-a `no_progress_source_is_the_pre_fix_beat_age_verdict` | 100 rc=0 | 100 rc=0 |
| T-a's census `day55_a_snapshot_reads_the_clock_once` | 100 rc=0 | 100 rc=0 |
| T-b `an_extended_stream_commits_..` | 100 rc=0 | 100 rc=0 (was 2 of 100 red) |
| T-c `slow_constraint_compile_..` | 100 rc=0 | 100 rc=0 (was 33 of 100 red) |
| T-e `coalesced_rows_..` | 100 rc=0 | 100 rc=0 (was 14 of 100 red) |
| T-e `a_partial_batch_waits_out_its_window` | 100 rc=0 | 100 rc=0 |
| T-e `a_full_batch_does_not_wait_for_its_window` | 100 rc=0 | 100 rc=0 |

- **The suites** (the fixes tip's test binary run directly from `crates/memra-server`, the cargo test runner's
  directory): arm A's shape (default threads, `CPUQuota=1200%`) **100 of 100 green**; arm B's shape (`--test-threads 48`,
  `CPUQuota=400%`) **98 of 100**. None of the five failed in any of the 200. Arm B's two reds, by name:
  `darklane::tests::stop_mode_full_cycle_launch_yield_resume_shutdown` (run 89, `timed out (3000ms) waiting for: yield
  to T`) and `tests::a_fake_route_memory_door_refuses_defers_and_recovers_through_the_handler` (run 64, lib.rs:19036,
  `(waiting, running, inflight)` read `(0, 1, 0)` against `(0, 0, 0)` after the loop saw `cancelled == 1`). By the rule
  each becomes its own item (OWED 24 and 25).

**Verdict, as registered.** T-a, T-b, T-c and T-e: **fixed and accepted** (R1 and R2 0 of 100 each, every red arm 10 of
10, none red in 200 full suites). T-d: **not reproduced** (R1 and R2 0 of 100; 0 in the 400 F1 and the 200 suites here,
after its one A' failure), left unchanged and recorded. Item 22 closes; items 24 and 25 open.
