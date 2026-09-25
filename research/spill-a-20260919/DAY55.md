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
