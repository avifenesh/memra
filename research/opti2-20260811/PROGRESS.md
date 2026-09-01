# OPTIPIPE increment 2 — progress

Date: 2026-08-11
Lane: `lane/cx-opti2`
Starting head: `60672886`
Rig: box1 (`ubuntu@<rented-box-ip>`), 2x RTX PRO 6000 Server Edition; every GPU block will hold `flock /tmp/memra-gpu.lock`.

## Scope and frozen constraints

- Build only the diagnostic, default-off depth-1 controller on top of the merged increment-1 fork/reconcile primitives.
- While round N resolves, allow at most one optimistic round N+1 stage-0 issue. Carry its real boundary ticket and fork generation until hit retirement or miss reconciliation. Never issue the optimistic successor's stage 1 before resolution.
- Admit only the increment-1 fail-closed shape: session-only, greedy K=1, device acceptance, PP-2 with primary stage 0. Ring, round-stream, replay, sampled/constrained, host-bounce, and other shapes remain refused.
- Measure spill-independent controller economics honestly. The frozen floor triple is seam 63.082 / serial 55.365 / plain 121.051 tok/s. The q-gate stub sweep is q* in {0.0, 0.5, 0.7, 0.9}; per arm retain hit/miss rates, reconcile count, wasted-draft tokens, and tok/s.
- Exactness precedes performance: serial fresh-process golden hash 10/10 (`21b8293f...`), `run-spec` K=1..8, `kernel-check`, `run-gen` argmax, and controller-ON byte identity against serial on identical greedy prompts.
- Raw combined stdout/stderr is retained before parsing. Every reported median states N and thermal regime. Failure causes are quoted, never inferred.
- No `cargo fmt`, push, merge, tag, release, or perf-board edit in this lane.

## Required context read

- `~/.lanectl/inbox/cx-opti2.md`
- `/home/avifenesh/projects/bw24/CLAUDE.md`
- `research/opti1-20260811/RESULTS.md`
- `research/optipipe-20260810/DESIGN.md`, including increment 2 controller/gates
- `research/opti0-20260810/RESULTS.md`

## Plan

1. Audit the merged seam and fork APIs plus existing optipipe harness, then write the smallest controller state-machine tests.
2. Implement actual K=1 validity wiring, one-successor stage-0 issue/finish, hit-retire, miss-reconcile plus serial rerun, q-gate stub, breaker/telemetry, and phase timings without changing round math.
3. Run local compile/tests without formatting; checkpoint code and harness work frequently.
4. Build and run the full locked exactness battery on box1 under the GPU flock, retaining raw receipts.
5. Run the c=2 N=5 interleaved floor remeasurement and q* sweep under one lock hold, then write the evidence-bounded GO/NO-GO verdict in `RESULTS.md`.

## Checkpoints

- Initial lane checkpoint: required reads complete; worktree clean on the required branch; no engine changes yet.
- 03:59 +03 audit checkpoint: increment 1 splits stage-0 issue from stage-1 finish, but the
  finish immediately publishes a stage-1 event wait onto the primary/stage-0 stream. Same-session
  runahead therefore requires a deferred-publication form: enqueue N stage 1, extend the draft,
  enqueue N+1 stage 0, then publish N stage 1 before the existing accept readback.
- The carried successor must own the actual boundary ticket, its `VerifyCkpt`, verify tokens,
  generation tag, pre-successor snapshot slot, and optimistic scratch length. A hit promotes that
  snapshot/ticket into the next loop iteration; a miss drains stage 0, restores stage-local lengths,
  retires the generation, and leaves the next round on the unchanged serial path.
- The q-gate is explicitly a diagnostic proxy, not a calibrated confidence head. Following
  DSpark's current prefix-survival definition, the stub score will be the product of the first two
  draft-token top probabilities; the sweep must report its empirical validity rather than call the
  score calibrated.
- The pinned Step target exposes no recurrent cache. The real controller will fail closed when
  recurrent state exists; increment 1's synthetic conditional-restore proof is not sufficient to
  generalize an overlapped full-accept/bonus-miss path to GDN-bearing models.
- 04:10 +03 implementation checkpoint: controller-only policy/ticket/counter scaffolding is in
  place, stage-1 completion can defer publication to the caller stream, and the miss reconcile
  primitive now accepts the actual accepted-token count plus base length. The decode loop wiring
  is intentionally still incomplete at this checkpoint and has not yet been compiled.
- 04:19 +03 controller checkpoint: the K=1 loop now defers round N stage-1 publication, computes
  the explicit uncalibrated two-token probability-product probe, optionally enqueues exactly one
  N+1 stage 0, and resolves its real boundary against `(n_acc == 1 && bonus == optimistic_pending)`.
  Hits carry the boundary/checkpoint/scratch tail; misses conditionally reconcile stage-0 lengths
  and retire the generation; a three-consecutive-miss breaker stops new probes. `cargo check
  -p memra-engine` passes locally (including CUDA compilation); no GPU execution yet.
- 04:24 +03 harness checkpoint: `optipipe-gate ... controller` now runs byte/state/continuation
  identity at `MEMRA_OPTI_CONTROLLER_Q`, and the server has a fresh-process-only research door
  with the same name (absent by default, bounded to [0,1], engine admission still fail-closed).
  Focused controller-policy and server-door unit tests pass, as does the gate binary check.
- 04:34 +03 box1 correction checkpoint: the first real-artifact smoke refused before mutation
  with `controller-requires-greedy-draft-graph`. The pinned Step MoE drafter deliberately takes
  the eager draft fallback, so graph-only admission could never exercise this lane. The ticket
  now carries the eager continuation hidden alongside its token/probability; the same two extra
  draft steps can continue through either graph replay or the existing eager head forward. The
  widened carrier compiles locally; the failed box1 refusal is retained for the final raw set.
- 04:39 +03 box1 runtime checkpoint: the eager-capable q=0 gate passed exact output, full cache /
  scratch / hidden state identity, and disabled-controller continuation after three real misses;
  counters were attempts 3, misses/reconciles 3, wasted/shadow drafts 6, breaker trips 1. The
  first server q=0 request also produced the pinned golden hash, but its wrapper exited afterward
  because a misquoted `grep` became `grep: match: No such file or directory`; that parser failure
  is retained and the clean wrapper receipt will be rerun before broader exactness.
- 04:44 +03 serving-order checkpoint: the clean server rerun produced the golden hash but exposed
  the increment-1-only `requires-primary-stage0-cross-device` refusal. Production intentionally
  keeps the worker primary on stage 1/head; DESIGN's rollback race analysis assumes that shape.
  Controller admission now accepts either stage-0 primary (forced-harness compatibility) or
  stage-1 primary (controller only), and stage 0 explicitly waits the primary accept-decision
  event before peer-reading `acc_out` and conditionally reconciling. q*=0 is now a true
  unconditional measurement arm (the breaker remains active for thresholded policy). Five focused
  state-machine tests and the full server type-check pass locally.
- 04:48 +03 production-path exactness checkpoint: a fresh q*=0 server request on box1 matched the
  pinned 64-token golden SHA-256 exactly (`21b8293f...`) with the controller actually issuing 38
  successors: 8 hits, 30 misses, and 30 device reconciliations. Both GPUs returned to 0 MiB after
  shutdown. This is the first clean production-head proof of the accept-decision ordering fix;
  its complete build, server, driver, QoS, and before/after GPU logs are retained under
  `raw/box1/server-controller-q0-golden-3/` before broader exactness or performance work.
- 04:57 +03 serial-oracle checkpoint: ten fresh server processes on box1 independently matched
  the pinned golden hash 10/10. The run held `/tmp/memra-gpu.lock` throughout, captured every
  server/QoS/before-after log, rejected known failure signatures, and ended with both GPUs at
  0 MiB. The receipt is retained at `raw/box1/serial-boots-1/`.
- The fixed c=2 protocol is now encoded in `box1-perf.sh`: N=5 for plain, serial K=1, merged seam,
  and q*={0.0,0.5,0.7,0.9} under one interleaved lock hold, plus excluded q0/q0.7 anatomy traces.
  Every controller run retains admitted and shadow `v_N` labels, hit/miss/reconcile/breaker counts,
  wasted/shadow draft tokens, and resolution timing. Tail cancellation now correctly charges its
  two unused draft tokens to the diagnostic wasted-work counter.
- 05:01 +03 combined-gate wrapper correction: `exact-gates-1` completed `kernel-check` ALL GREEN
  and a matching decode argmax, then the wrapper stopped because the one-token default `run-gen`
  prompt emits one `MATCH` line while the assertion expected the established two-comparison text
  gate. The captured failure is a wrapper-input mistake, not a model mismatch. The rerun pins the
  prior CPU-pipeline text prompt so both prefill/decode and batched-prime/tokenwise comparisons
  must match; the failed driver and full kernel/run-gen logs remain under `raw/box1/exact-gates-1/`.
- 05:08 +03 exactness-complete checkpoint: `exact-gates-2` passed all 376 kernel `OK` cells and
  ended ALL GREEN; run-gen reported both required argmax `MATCH` comparisons; run-spec passed
  self-consistency at every K=1..8; and the controller gate preserved output, full session state,
  and a subsequent disabled-controller continuation at q*=0.0, 0.5, 0.7, and 0.9. The q0 arm
  exercised 30 real misses/reconciliations; the thresholded synthetic prompt retained 30 labeled
  shadow rejects per threshold. Both GPUs returned to 0 MiB. The c=2 interleave began only after
  this receipt printed `EXACT_PASS`.
- 05:09 +03 final local checks pass on the evidence head: engine/controller and server type-checks,
  all five focused controller-policy/generation tests, and the explicit server-door unit test.
  Combined and per-command logs are retained under `raw/local/final-checks-1/`.
- 05:15 +03 performance checkpoint (first interleaved repetition only; not the N=5 verdict):
  plain 121.603, serial 55.613, seam 63.183, q0 40.712, q0.5 51.622, q0.7 52.339, and
  q0.9 53.676 aggregate tok/s. q0 observed 150/594 hits (25.25%) and paid 444 reconciliations /
  888 wasted draft tokens. The thresholded arms selected much stronger admitted subsets
  (76.5--80.6% hit in this repetition), but their mandatory shadow probes still left every
  controller arm below serial, seam, and plain. Four more predeclared rotations remain; no verdict
  is drawn from this single repetition.
- 05:24 +03 interleave checkpoint: repetition 2 reproduced the first ordering-independent result
  within a narrow band: plain 121.594, serial 55.599, seam 63.271, q0 40.667, q0.5 51.591,
  q0.7 52.349, and q0.9 53.639 aggregate tok/s. Repetition 3 is in flight under the same single
  GPU-lock hold. This remains an interim observation; the predeclared N=5 summary and excluded
  anatomy traces will determine the final verdict.
- 05:30 +03 interleave checkpoint: repetition 3 is complete and remains consistent despite its
  different cyclic position: plain 121.626, serial 55.548, seam 63.159, q0 40.686, q0.5 51.654,
  q0.7 52.301, and q0.9 53.431 aggregate tok/s. Controller labels are deterministic across the
  first three repeats: q0 resolves 150/594 opportunities as hits; q0.5, q0.7, and q0.9 admit
  124/562, 74/554, and 34/538 with admitted hit rates 80.65%, 78.38%, and 76.47%. No scored arm
  has tripped the three-miss breaker. Repetition 4 is in flight; the N=5 verdict is still pending.
- 05:36 +03 final-head exactness hardening: the controller comparator can now optionally tokenize
  a real prompt (and apply the model chat template), so the post-performance receipt can require
  both model-selected hit and miss terminals while comparing all live cache/scratch/hidden bytes
  and a disabled-controller continuation. The added diagnostic path and final invalid-confidence
  guard pass engine/server type checks, all five focused state-machine tests, and the server-door
  test; generated logs are retained under `raw/local/final-checks-2/`. No scored binary changed
  during the still-active interleave.
- 05:47 +03 performance-complete checkpoint: the uninterrupted one-lock block printed
  `PERF_PASS` after 35 scored points / 280 clean measured requests and two excluded anatomy points.
  N=5 medians are plain 121.594, serial 55.548, seam 63.183, q0 40.675, q0.5 51.613,
  q0.7 52.339, and q0.9 53.639 aggregate tok/s. The best controller arm is therefore 3.44%
  below serial, 15.11% below the seam, and 55.89% below plain: provisional promotion verdict
  NO-GO pending the final-head hit-bearing state receipt. The copied 193-file remote payload
  matches byte-for-byte by aggregate SHA-256 (`46ab1072...`); local `summary.json` is the 194th
  file. Both GPUs ended at 0 MiB.
- 05:58 +03 schedule-audit correction: the first anatomy receipt invalidates that provisional
  promotion verdict. Successor tickets resolved only ~0.08 ms after issue because N stage 1 and
  the q/d2/d3 continuation had already run on the primary stream before N+1 stage 0 was enqueued;
  the scored controller therefore did not implement the requested overlap. The raw block remains
  retained as a negative schedule control, not as the lane verdict. The controller now prepares
  d2/q/d3 first, enqueues N stage 0, captures only stage 0's post-N checkpoint, enqueues N+1 stage
  0, then enqueues N stage 1 and captures its matching checkpoint half. This gives both GPU queues
  the same N boundary while preventing either checkpoint half from observing the wrong generation.
  Engine/server checks and all six focused tests pass locally; box1 exactness and N=5 performance
  must be repeated before any verdict.
- 06:09 +03 overlap-exactness correction: the reordered queueing made ticket resolution real
  (~12.2 ms instead of ~0.08 ms), but the retained real-prompt gates then caught an output
  divergence and, in the reduced N=8 repro, a layer-0 K-cache mismatch immediately after one miss
  and one hit. Step35's row-wise verify helper was ignoring the ticket's explicit position and
  deriving RoPE rows from mutable `cache.pos`; that happened to be valid only when N stage 1 ran
  before N+1 stage 0. All four verify entry points now thread immutable `pos0`, and Step35 uses
  `pos0 + row`. Local engine type-check and all five controller-policy/generation tests pass. The
  failed reduced receipt is retained at `raw/box1/debug-hit-n8-1/`; no performance rerun will begin
  until the corrected N=8 and full exactness gates pass on box1.
- 06:10 +03 reduced box1 proof: corrected N=8 real-prompt controller execution passed output,
  complete state, and disabled-controller continuation identity after exactly one miss/reconcile
  and one hit. Both tickets resolved in ~12.2 ms, preserving the intended overlap rather than the
  invalid earlier ~0.08 ms schedule. Both GPUs returned idle; the receipt is retained under
  `raw/box1/debug-hit-n8-2/`. The full N=128 real-prompt gate is next.
- 06:11 +03 full real-prompt proof: corrected N=128 execution passed full state and continuation
  identity across 80 actual controller opportunities: 20 hits, 60 misses/reconciliations, zero
  refusals, zero abort drains, and 12.08--12.34 ms ticket resolution. The q*=0 diagnostic arm
  charged all 120 miss-side draft tokens as wasted and ended with both GPUs idle. Receipt:
  `raw/box1/final-hit-state-4/`. The complete rebuilt kernel/run-gen/run-spec/controller battery
  remains required before performance.
- 06:18 +03 rebuilt exactness checkpoint: all four gate binaries were rebuilt from corrected
  source, then `exact-gates-3` passed 376 kernel `OK` cells / ALL GREEN, both run-gen argmax
  comparisons, run-spec K=1..8 self-consistency, and controller state plus continuation identity
  at q*=0.0, 0.5, 0.7, and 0.9. q*=0 resolved 30 real misses with 30 reconciliations at the
  now-overlapped ~12.3 ms schedule; thresholded arms retained their shadow labels. Both GPUs
  returned idle. Fresh server rebuild and serial 10/10 golden boots remain before performance.
- 06:28 +03 serving-gate checkpoint: the corrected `memra-server` rebuilt under the shared GPU
  lock (SHA-256 `33b49fa2...`). The reproducible fresh-boot harness is now tracked as
  `box1-serial-boots.sh`; its first two independent processes each emitted the pinned
  `21b8293f...` golden hash with exactness `match`. The remaining eight boots are running under
  the same uninterrupted lock hold; performance remains blocked on 10/10.
- 06:34 +03 exactness-complete checkpoint: all ten fresh corrected-server processes independently
  matched the pinned 326-byte completion (`21b8293f...`) 10/10. Every boot retained its request,
  server, and before/after GPU receipt; the lock block ended with no compute process. Together
  with `exact-gates-3` and `final-hit-state-4`, every pre-performance exactness requirement is now
  green. The corrected c=2, N=5 interleave may begin.
- 06:38 +03 queue-only correction: another lane acquired the shared lock between exactness and
  performance. `perf-c2-2` never acquired the lock or touched a GPU; its queued waiter was
  deliberately terminated after the competing job proved longer than the fixed 900-second
  horizon, yielding the captured lines `Terminated` and `FAIL: GPU lock timeout`. The harness now
  accepts `OPTI2_LOCK_WAIT` (default unchanged at 900 seconds); the authoritative rerun will use a
  fresh `perf-c2-3` directory and a 3600-second wait. No scored point exists in `perf-c2-2`.
- 06:40 +03 final local checks: engine/server checks pass; the engine library suite is 59 passed,
  0 failed, 1 CUDA-only test ignored; and the server suite is 162 passed, 0 failed. The first
  wrapper invocation incorrectly requested `memra-server --lib` even though that package has no
  library target, and lacked `set -e`, so its misleading trailing `PASS` is explicitly invalid
  and retained in `final-checks-pos0-1`. The corrected fail-fast rerun is retained in
  `final-checks-pos0-2`. `perf-c2-3` remains queued without GPU use behind the competing lock.
- 06:55 +03 corrected performance checkpoint (repetition 1 only): after acquiring the lock at
  03:48:02Z, the first rotation produced plain 120.085, serial 55.258, seam 62.551, q0 40.506,
  q0.5 51.350, q0.7 52.043, and q0.9 53.360 aggregate tok/s. Controller opportunity labels are
  unchanged, but the real schedule is now visible: median issue-to-decision time is
  12.269--12.296 ms across q arms, versus ~0.084 ms in invalid `perf-c2-1`. q0 resolved 150 hits /
  444 misses; q0.5 100/24; q0.7 58/16; q0.9 26/8. Four cyclic rotations remain; no verdict is
  drawn from this hot-start repetition.
- 07:09 +03 interleave checkpoint: repetition 2 stayed narrow at plain 120.878, serial 55.309,
  seam 63.098, q0 40.513, q0.5 51.360, q0.7 52.018, and q0.9 53.376 tok/s. Repetition 3
  controls were plain 121.109, serial 55.427, and seam 63.140; q0 was 40.648 and q0.9 53.481,
  while q0.5/q0.7 had lower 50.569/51.268 points. All loads remained 8/8 clean with no sheds or
  errors and deterministic controller label counts. Repetition 4 is in flight; the N=5 medians
  remain pending.
- 07:23 +03 scored block complete: all 35 points / 280 measured requests are clean. N=5 medians
  are plain 121.109, serial 55.427, seam 63.098, q0 40.513, q0.5 51.350, q0.7 52.018, and
  q0.9 53.376 aggregate tok/s. The best controller arm is therefore still about 3.70% below
  serial, 15.41% below the seam, and 55.93% below plain despite genuine stage overlap. The two
  excluded instrumented traces are now running; final summary/accounting and GPU teardown remain
  before the verdict.
- 07:26 +03 authoritative receipt complete: `perf-c2-3` printed `PERF_PASS` after 37 total load
  points (35 scored plus two excluded traces), 296/296 successful measured requests, no sheds,
  and no captured failure signatures. The 193-file remote payload matches the local tree digest
  exactly (`214352a6...`); generated `summary.json` is file 194. Thermal snapshots span 30--37 C,
  180--2415 MHz, and 31.44--98.84 W; both GPUs ended at 0 MiB with no compute process. Final
  verdict remains NO-GO for promotion: every controller threshold loses to serial, seam, and
  plain, so the door stays default-off and diagnostic-only.
- 07:33 +03 final-report checkpoint: `RESULTS.md` records the corrected N=5 medians, controller
  accounting, fixed-trajectory ranking diagnostic, excluded phase anatomy, invalid schedule
  control, corrected exactness gates, receipt hashes, and the promotion NO-GO. Its local evidence
  links resolve, all manually reported table values match the deterministic summary, and rerunning
  `summarize-perf.py` reproduces the committed `summary.json` byte-for-byte.
