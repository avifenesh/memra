# WP-C day 72 (2026-09-26): OWED C11, the door's remaining gap to REF re-attributed at I15, registered before any cell

Lead, resuming lane C: "DAY64 section 5, cell i15b ran on BOX14 ... Read it as registered", then "continue your owed
list". `DAY64.md` section 5b reads `i15b` as registered: admissible, `i14=flat i15=flat door=i15 vs_ref=loses`. DAY64
section 5 says what follows: "if admissible and the door still loses to REF, the next improvement from its split,
registered before code". Tree at start: `e07b98ee0`.

## 0. What the split says, and why the next step is an attribution, not an improvement

At I15 on the 285K class the door runs 0.264 s gen-only and 0.232 s window against REF's 0.255 and 0.226 (0.28 and
0.19 ms per token), exactly where I13 stood on BOX13. Between I13 and I15 the door's CPU-side work halved on the card
(the owner's demand 0.216 to 0.119 ms per window token, the cache's lease retire 0.075 to 0.039, the bank's `stage`
0.149 on 92.3 stages to 0.074 on 33.3), and neither step moved the wall (`flat`, `flat`, noise 0.001).

So the split no longer names an improvement. DAY60's attribution (`cpu_side`, `top=prefetch_ns`) was read on the I10
program, where the door's CPU brackets exceeded REF's by 0.699 ms per window token against a 0.437 wall gap. The
premise of that reading (the CPU side on the critical path) held through I11 and I13 (`improves` twice) and stopped
holding at I14 and I15. What is left is either CPU time still on the path in a part the stage clock does not bracket,
or the GPU side: the compute stream waiting on the door's copies (a later issue point, a different stream ordering, a
per-lease event wait), or kernels that run longer under the door. An improvement registered against the split would
guess. This day registers the measurement that separates those.

## 1. Pre-registration: the cell `gap15` (the 285K class first; before its scripts)

A measurement cell; no code, no default change. One binary for every arm: `i15=2243b1fe2` (the door's tuned program;
its REF path is the legacy slot cache with `MEMRA_MOE_PREFETCH=1`, the same program as `c60`'s, and its
`--moe-dispatch-clock` brackets are DAY60's). The day-18 pressure shape and DAY60's run shape, one collector hold.

**Part A, timed (40 runs):** DAY60's four arms REF, REFC (REF with `--moe-dispatch-clock`), ON (`--experts-via-tier
--expert-bank-host-bytes=17179869184`) and ONC (ON with the clock); order 1 (REF, REFC, ON, ONC) x 5, order 2 reversed
x 5. Read by `day60-gap.py` unchanged (integrity, R1 the gaps, R2 the instrument's cost, R3 the CPU brackets, R4
`cpu_gap` against `window_gap`, the verdict line), after DAY64 section 5's admissibility clause (every arm's gen-only
and window IQR at most 0.005 s; an inadmissible Part A decides nothing and is recorded as it reads).

**Part B, profiled (4 runs, after Part A in the same hold):** one REF run and one ON run under Nsight Systems, each
twice (order REF, ON, ON, REF): `nsys profile --trace=cuda --sample=none --cpuctxsw=none`, the same argv as Part A's
unclocked arms, exported to SQLite. Its numbers are under the profiler and read only REF against the door in the same
condition. The window is located from the run's own log: its stamped `STEADY-STATE window` line is the window's end
and the printed window seconds its length, mapped onto the trace through the trace's recorded session start (UTC). Over
that window, per window token:
- B1 `gpu_busy`: the union of all kernel intervals, and `gpu_idle` = the window minus it;
- B2 `h2d`: host-to-device copies' count, bytes and the union of their intervals; `h2d_exposed`: copy time during
  which no kernel runs;
- B3 `kernel_sum`: the sum of kernel durations, and the five kernel names whose summed duration differs most between
  the arms.

**What each reading decides, stated before the cell.** Part A's `cpu_gap` against `window_gap`: `cpu_side` (at least
75 percent) says the CPU side still carries the gap, in terms the clock names; `not_cpu_side` says it does not. Part B,
the door against REF (medians of each arm's two runs): if the door's `gpu_idle` exceeds REF's by at least half of Part
A's `window_gap`, the gap is the GPU waiting (a stall: the door's copies, their ordering or their event waits, and B2
says whether copies are exposed); if the door's `kernel_sum` exceeds REF's by at least half of it, the door's kernels
run longer; if neither, the profile does not place it and that is recorded. The line is `DAY72 GAP15 VERDICT rig=<rig>
integrity=<ok|FAIL> admissible=<yes|no> partA=<cpu_side|not_cpu_side|void> partB=<gpu_stall|kernels_longer|
unplaced|not_read>`. It changes no code; the improvement it points to is its own registration.

**Integrity.** Part A as DAY60 section 1a. Part B: 4 runs, each exit 0 and `MATCH`, the same tape as Part A, the door's
fill complete and `physical_reads=0`, an SQLite export with kernel rows inside each window; a Part B that cannot be read
(no `nsys` on the host, an export failure) is `not_read` and does not void Part A.

**Where.** The target card on the 285K class first (the admissible class of `i15b`); the RTX 5090's half joins its
queue after the cells already queued there. Box needs as `DAY64.md` section 3a plus `nsys` from the CUDA 13 toolkit
(checked and recorded by the cell).
