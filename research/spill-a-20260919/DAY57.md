# WP-A day 57: OWED item 24, the darklane stop-mode test times out under starvation

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `de4b95215` (item 23 closed). CPU only, this rig.

## 1. Pre-registration (committed before any reproduction run or code)

**What is known.** `darklane::tests::stop_mode_full_cycle_launch_yield_resume_shutdown` failed 1 of 100 in arm B's
shape (DAY55) and 1 of 400 in arm A's (DAY56), each `timed out (3000ms) waiting for: yield to T` (darklane.rs:602).
The test runs a real job (`sh` looping `sleep 0.01`) under the runner thread, flips the busy signal, and waits up to 3 s
for `/proc/<pid>/stat` to read `T`. Its own comment records the same failure at a 500 ms bound in 2026-08 ("the runner
thread didn't get scheduled for >500ms") and names the claim: "the yield itself plus the counters; wall-clock
tightness is poll_ms config, not an OS promise". Its other waits are the same shape: launch 1 s, job running 1 s, resume
1 s, reaped 3 s.

**The hypothesis.** A wall bound on scheduler latency. The runner thread polls every 5 ms (`poll_ms`), and between the
busy flip and `T` sit the runner's wake-up, `kill_group(SIGSTOP)`, the kernel's stop, and the test's own polls; a runner
that the scheduler does not run for 3 s fails the bound with the wiring intact.

**The reproduction.** R2 as DAY55 ran it: the test (`--exact`) inside a scope at `CPUQuota=100%` beside eight burners,
100 runs. If R2 reads no failure, R3: the same with sixteen burners. Reproduced at one or more failures with the
recorded message.

**The fix, if reproduced.** The test's waits become waits for an acknowledgement, each under one hang guard of 30 s:
the runner's own state (`BG_RUNNING`, `BG_YIELDED`) and counters after each signal, then the `/proc` state, as now. Each
wait prints the latency it read. This changes what the bound means, stated so the lead can overrule it: the 1 s and 3 s
bounds become a 30 s guard that catches a wiring break (a runner that never acts never acknowledges) and no longer
asserts scheduler latency, which the test's own comment already disclaims. The other darklane tests are unchanged.

**Acceptance.** R2 (and R3 if run) 0 of 100 on the fixed test; the red arm, the stop arm's `kill_group(SIGSTOP)`
removed (a scratch patch, grep-checked in its binary), fails 10 of 10 at the guard with the `yield` message; server lib,
clippy and fmt green.

**The rule.** Not reproduced by R2 and R3: the test is left unchanged and the item records it. A fix that misses its
acceptance is reverted in one commit.

**Budget.** 0.15 agent-day.

## 2. As run (`day57/`)

- The test binary before the fix: R2 (eight burners) **100 of 100 green**; R3 (sixteen burners, `r3.sh`) **99 rc=0,
  1 rc=101**, `timed out (3000ms) waiting for: yield to T` (darklane.rs:602): reproduced by R3.
- The fix (`a327f486c`): `wait_acknowledged(what, f)`, a 30 s hang guard that prints the latency it read; the test waits
  for the runner's own acknowledgement (`BG_RUNNING`, `BG_YIELDED`, `resumes == 1`) and then the job's `/proc` state at
  each step. The other darklane tests are unchanged.
- The red arm (the stop arm's `kill_group(SIGSTOP)` replaced by a printing marker, grep-checked in its binary, `marker=1`)
  fails 10 of 10: `no acknowledgement within the 30 s guard: yield to T`. A first red-arm run is kept as `red-unmarked/`:
  its marker was an unused `let` the compiler removed, so its binary could not be checked, though it failed 10 of 10
  the same way.
- On the fixed binary: R2 **100 of 100**, R3 **100 of 100**. CPU cells: server lib `930 passed; 0 failed; 25 ignored`;
  clippy `-D warnings`; fmt.
- The latencies the guard prints are captured by the test harness on a pass (no `--nocapture`), so no latency reading
  exists for these runs; a failure prints them.

**Verdict, as registered:** reproduced by R3; the fix meets its acceptance (R2 and R3 0 of 100, the red arm 10 of 10).
Item 24 closes. The bound's change (1 s and 3 s to a 30 s guard) is the lead's to overrule, as section 1 states.
