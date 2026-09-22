# GPU probe recovery (memra#516)

Verdict: one missed `nvidia-smi` probe degrades the process and leaves it live; the fatal GPU fault
latches only when the miss streak reaches `MEMRA_GPU_PROBE_MISSES` (default 3), an answering probe
clears the degradation and never a latched fault, and `/health` publishes the canary's state
(`worker.gpu_probe`). Proven on the real server with a fake `nvidia-smi` on `PATH`
(`tools/gpu-probe-recovery-gate.py`, in local-ci).

## The defect

`mark_gpu_fault` latched on the first steady-state `ProbeErr::Hang` and `live()` rejected the
latch for the rest of the process. On the production B200 box (2026-09-13) NVML stalled past the
10 s deadline once, right after `Engine ready` during graph capture, and the box answered 503 for 28
minutes while `nvidia-smi` answered in 40 ms from a shell. Startup already tolerated six hangs
(#340); steady state tolerated none.

## The change (`crates/memra-server/src/health.rs`)

- `note_probe_hang(deadline, misses)`: the streak grows; below `misses` the process is DEGRADED
  (a `[gpu-watch] DEGRADED:` line, `degraded_reason` set, `live()` unchanged); at `misses` it
  calls `mark_gpu_fault` with the streak, the policy and the last answer's age in the reason.
- `note_probe_ok()`: streak to 0, `last_ok` stamped, degradation cleared with one line. It never
  touches `gpu_faulted`: fatal Xid, ECC, row-remap and the miss bound stay latched.
- `MEMRA_GPU_PROBE_MISSES` (default 3, floor 1; `1` restores the single-hang latch). Three
  misses at the 60 s interval and 10 s deadline is about three and a half minutes with no answer.
- `/health` `worker.gpu_probe`: `degraded`, `miss_streak`, `last_ok_age_ms`, `degraded_reason`,
  `latched_reason`. A guard restarts on `latched_reason`, not on `degraded`.
- The startup answer stamps the first `last_ok`; the six-probe startup window is unchanged.

## Receipts (local RTX 5090, 9B NVFP4 on the plain route, interval 2 s, deadline 2 s, misses 3)

`raw/gate-5090-run1/<arm>/observed.json` is the `/health` sequence at 250 ms; `fake-smi-trace.txt`
the fake tool's call log; `gpu-watch-lines.txt` the server's `[gpu-watch]` lines.

- **A recover** (script ok, hang, hang, ok): t=0.0s status=200 degraded=False streak=0 latched=False; t=2.26s status=200 degraded=True streak=1 latched=False; t=6.27s status=200 degraded=True streak=2 latched=False; t=8.28s status=200 degraded=False streak=0 latched=False
- **B latch** (ok, hang, hang, hang, ok, ok, ok): t=0.0s status=200 degraded=False streak=0 latched=False; t=2.26s status=200 degraded=True streak=1 latched=False; t=6.27s status=200 degraded=True streak=2 latched=False; t=10.29s status=503 degraded=True streak=3 latched=True; t=12.3s status=503 degraded=False streak=0 latched=True. The 503 detail names
  `3 consecutive probe(s)` and `MEMRA_GPU_PROBE_MISSES=3`.
- **C fatal** (ok, ecc, ok, ok): t=0.0s status=200 degraded=False streak=0 latched=False; t=0.25s status=503 degraded=False streak=0 latched=True. The detail names uncorrected ECC; later clean
  answers do not clear it.

CPU teeth: `health::tests::{one_steady_state_hang_degrades_but_stays_live,
an_answer_clears_timeout_only_degradation, the_miss_bound_latches_and_an_answer_does_not_unlatch,
misses_policy_of_one_restores_the_single_hang_latch, fatal_faults_latch_regardless_of_probe_answers}`.
`raw/local-ci/local-ci.log`: the full battery on the lane tree with the new stage inside it.

Review round 1 (revuto) found two gaps: after a latch, an answer reset the streak and a later miss
republished `degraded` beside `latched_reason`; and one pre-#516 sentence survived in SERVING.md.
Both fixed: no streak or degradation bookkeeping runs beside a latched fault (the answer time is
still stamped), arm B of the gate now hangs again after the latch and asserts the two states are
never published together, and the sentence names the miss streak.

## What stays open in #516

The default deadline stays 10 s; the issue's ask to scale it with card or model size is not
measured here (no B200 on hand), and the miss policy makes a single long stall survivable
regardless. Startup hangs remain the six-probe window.
