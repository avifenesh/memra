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

## 1a. The sitting, prepared before any cell

`day72-cell.sh`, `day72-read.py` and `day72-box.sh` were written after section 1; two details are stated here.
- **What is mirrored.** The profiler's `.nsys-rep` reports and SQLite exports are too large for the repository; after
  the reader, the driver moves them into `profiles-hash-only/` with their SHA-256 in `profiles.sha256` (like the ELF
  binaries, by hash). The raw rows the reading uses stay in the cell: `ev/<label>.window.tsv`, every kernel and copy
  that overlaps each profiled window, written by the reader.
- **Part B's window.** Located as section 1 says, through `TARGET_INFO_SESSION_START_TIME.utcEpochNs` (checked present
  in a CPU-only `nsys` 2025.5.2 export on the local host); kernels from `CUPTI_ACTIVITY_KIND_KERNEL` (names through
  `StringIds`), copies from `CUPTI_ACTIVITY_KIND_MEMCPY` with `copyKind` 1 as host to device. A schema that lacks them
  reads `not_read`.

Dry checks (`day72-cpu/`): the cell's control flow with a stub `run-gen-i15` and a stub `nsys` (40 timed runs, then
the four profiled runs in the order REF, ON, ON, REF, each with its report and export; `dry-check-cell.log`); the
reader on a synthetic cell whose Part A is DAY60's own `gap` receipts (it reprints DAY60's lines and its verdict) and
whose Part B is invented (`make-synthetic.py`), and with one export removed or no `nsys`, where Part B reads
`not_read` and Part A still reads (`dry-check-reader.log`); the driver's control flow, including the move of the
profile files (`dry-check-driver.log`). The local 5090 was not used: lane B's queues hold it.

Run as `D72_BUILDS="i15=2243b1fe2" bash /root/wt-c/research/spill-c-20260919/day72-box.sh` on a Core Ultra 9 285K
host with one RTX PRO 6000 Blackwell Workstation Edition (the class of BOX12, BOX13 and BOX14; box needs as `DAY64.md`
section 3a, plus `nsys` in the CUDA 13 toolkit, which the cell records or reads `nsys=none`). Receipts land in
`/root/spill-receipts/c-day72/`. Expected: one build about 5 minutes, the cell about 20 (40 timed runs, 4 profiled
runs and their exports).

## 2. The RTX 5090 (queue v10, 2026-09-26 01:43Z to 02:08Z; `rtx5090-day72/gap15/`)

Queue v10 ran the cell after queue v9, on `run-gen-i15` `de00c256...` (tree `2243b1fe2`, rebuilt after the rig's
reboot by `c-local-build.sh`, CUDA 13.1), behind `/tmp/memra-5090.lock` with the card idle before the hold; Nsight
Systems 2025.5.2 from the CUDA 13.1 toolkit. The four reports and exports are kept by hash (`profiles.sha256`); the
reader's window rows are in `ev/p-*.window.tsv`. Regime: 56 to 68 C, SM median 1590 MHz, N=2043. Verbatim
(`reading.log`):

- `DAY72 ADMISSIBILITY rig=rtx5090 ceiling=0.005 max_iqr_gen=0.0090 max_iqr_window=0.0062 failing=['on:gen_s=0.0090', 'on:window_s=0.0062', 'onc:gen_s=0.0065'] -> inadmissible`
- `DAY60 GAP CHECKS rig=rtx5090 runs=40 integrity=ok`
- `DAY60 R1 window_gap_ms_per_token on_minus_ref pooled=+0.109 o1=+0.219 o2=+0.094 | medians ref=0.402 refc=0.400 on=0.406 onc=0.404 (N=10 each)`
- `DAY60 R4 window wall_gap=+0.109 cpu_gap=+0.356 residual=-0.246 top=prefetch_ns (+0.378) -> cpu_side`
- `DAY72 B door_minus_ref per window token (ms, medians of two): gpu_busy=-0.1117 gpu_idle=+0.1586 kernel_sum=-0.1113 h2d_busy=-0.0091 h2d_exposed=-0.0049 h2d_count=+0.0000 h2d_mb=+0.0000`
- `DAY72 B3 kernels differing most (door minus ref, ms per window token): qmatvec_expert_q8=-0.0274, moe_gate_up_silu8_q8=-0.0157, qmatvec_q8_0_mmvq_fused2=-0.0128, quantize_q8_1=-0.0080, fa_decode_f32=-0.0072`
- `DAY72 B rule: half of Part A's window_gap=+0.0545 ms per token`
- `DAY72 GAP15 VERDICT rig=rtx5090 integrity=ok admissible=no partA=cpu_side partB=gpu_stall`

**Read as registered: `admissible=no`, so Part A decides nothing on this card,** and Part B's threshold is half of that
inadmissible gap, so its `gpu_stall` is recorded as it reads and decides nothing either. The ceiling does not move; the
cell runs again on this card as a new hold (queue v11, `rtx5090-day72-rerun1/`).

**What the profiled runs show beside it, deciding nothing.** Under the profiler (411 to 413 ms windows against about
400 unprofiled) the door and REF launch the same kernels (1866.6 to 1867.2 per window token) and the same copies (94.3
host-to-device copies and 45.57 MB per window token in every run). The door's GPU is busy 0.11 ms per window token
less than REF's (its expert kernels run slightly shorter) and idle 0.16 ms more: the GPU waits on something between
kernels in the door's decode that it does not wait on in REF's, with no extra copy time exposed. That is the kind of
reading the target card's Part B is registered to decide.
