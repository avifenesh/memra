# cx-sigrouter2 progress — 2026-08-11

Lane: `lane/cx-sigrouter2`

## Objective

Keep the Step-3.7 sigmoid router's selected ids and weights on the device through expert
dispatch, removing the remaining per-layer pinned readback and stream synchronization without
weakening host-oracle exactness or mixed/pruned-expert safety.

## Required gates

- Fail closed, identically on host and device, when `active_count < n_used`; add a manifest-backed
  kernel-check rejection cell at `active_count = n_used - 1`.
- Establish host `expf` trust with either a boot-time byte probe or a vendored scalar oracle, and
  record the choice in `RESULTS.md`.
- Capture served router-logit rows during `run-gen` and replay them through host and device routing
  in a manifest-backed kernel-check cell.
- Local RTX 5090: kernel-check ALL GREEN under `/tmp/memra-gpu.lock`.
- Box1: required-manifest kernel-check ALL GREEN, `run-gen` MATCH, `run-spec` K=1..8 PASS, and
  10/10 golden boots at `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
- Box1: one-lock interleaved N=5 at c=1 and c=8 against increment 1; update the SOL model with the
  measured medians. Retain raw logs and state N plus thermal regime.

## Checkpoints

- Started from `30418923`, the merged increment-1 sigmoid-router and rowwalk baseline.
- Read the lane brief, increment-1 receipt, SOL-gap report, and repository law before inspection.
- Worktree was clean and already isolated on `lane/cx-sigrouter2`.
- Added one shared `active_count >= n_used` validator to the host oracle and device launcher;
  the kernel-check rejection cell pins identical text with `active_count=7, n_used=8`.
- Added a once-per-process 24-case scalar host-`expf` byte probe. Sigmoid-router model loading
  refuses the enabled device route if the pinned runtime results move; the explicit host rollback
  `MEMRA_SIG_ROUTER=0` remains available.
- `cargo check -p memra-engine --lib`, `cargo check -p memra-engine --bin kernel-check`, and the
  two focused contract unit tests pass locally. No formatter was run.
- Added opt-in `MEMRA_SIG_ROUTER_LOGIT_TRACE`: a run-gen decode captures one bit-preserving real
  router row per layer, including correction bias, original-id active mask, scaling, and route
  normalization. `MEMRA_SIG_ROUTER_REPLAY` drives the required host-vs-device replay cell; missing,
  duplicate, selection-mismatched, or weight-bit-mismatched records fail the gate.
- Added the Step-only resident device arm before grouped/sequential dispatch. It is fail-closed to
  local resident slabs, uniform q8 layouts, original-id masks, and macro-free experts. Unclamped
  layers consume device `sel/w` through the established rows twins; clamped layers retain the
  separate gate/up, clamp, down, and slot-ordered scatter chain without a router readback.
- The device arm compiles for the library and all engine binaries. Spill, mixed-layout, remote-slab,
  macro-scaled, non-Step sigmoid, and route-observation calls still use their established paths.
- Local 5090 synthetic kernel-check: `ALL GREEN` (77 cells, 22 model-backed skips); both new
  contract cells passed. Box1 run-gen engaged all 42 resident Step MoE layers, including both
  clamped layers, and both argmax comparisons reported `MATCH`.
- Box1 required-manifest kernel-check: 42 served records across 42 unique layers, zero id or
  weight-bit mismatches; full battery `ALL GREEN` (83 cells, 20 unavailable-model skips).
- Box1 `run-spec` passed self-consistency at every K=1..8.
- Local manifest-backed kernel-check completed `ALL GREEN (101 cells, 0 skipped)`, including
  replay of the 42 box1-served rows with zero selected-id or weight-bit mismatch.
- Added the fixed one-lock interleaved N=5 harness. Its increment-1 arm keeps device sigmoid
  routing but sets `MEMRA_MOE_DEV=0`, isolating removal of the selected-id/weight readback from
  the earlier full-logit host rollback.
- Box1's fresh-process golden battery completed 10/10 exact matches at
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
- The first performance-harness invocation stopped before its first measured point: Bash
  `set -u` rejected a same-declaration `label` reference. The failed driver log is retained as
  `raw/box1-perf-attempt1/`; the declaration was split before restarting.
- Box1's fixed interleaved N=5 matrix completed with zero request errors and 5/5 paired wins at
  both loads: c1 medians 101.281657 vs 84.520490 tok/s (+19.8309%); c8 medians 169.004876 vs
  165.505592 tok/s (+2.1143%). The CUDA transfer/API census is still running under the same lock.
- The first `tools/local-ci.sh --perf` invocation passed kernel-check (`101 cells`, the lane replay
  intentionally absent from this separate standing manifest), then its wrapper found that the
  release `prime-gate` binary had not been built. The missing gate binaries were built and the
  full, unmodified battery was restarted; the failed wrapper log is retained.
- Nsight completed successfully. Relative to increment 1, the default removes every 32-byte
  decode routing DtoH (18,144 to zero), 98.09% of all DtoH API calls, and 96.87% of all stream
  synchronizations in the whole run-gen trace. The exact 9,213-sync delta matches the eliminated
  router-call count.
- Re-running the frozen SOL model with the measured medians moves c1 from 28.812% to 34.526% SOL
  (+5.714pp) and c8 from 27.168% to 27.742% (+0.574pp).
- The unmodified local-CI rerun has completed its correctness stage GREEN: model-backed
  kernel-check, prime gate, Qwen K=1..8, Gemma run-gen/verify/spec, both decode-batch dtype gates,
  graph stress plus canary, serve smoke, c64 serve stress, and the served-acceptance gate all pass.
  Its full `--perf` cell stage is now running under the same local lock.
- The full perf stage exposed a real non-Step acceptance regression in `26b-spec-d1736`
  (`0.880 -> 0.646`). The shared-expert exact-form adjustment had been unnecessarily broad, so it
  was gated by `cfg.step35` before attribution; an on-window 26B rerun and full battery remain
  required before push.
- A post-scope-fix 26B smoke still reads acceptance 0.646. That disproves the initial causal
  attribution above: the cross-day tripwire is real, but it may predate this lane. An exact
  `30418923` baseline binary is built and the required same-window interleaved N=5 comparison is
  queued behind the local GPU lock.
- The one-lock same-window settle completed N=5 per arm. The exact `30418923` lane base and the
  candidate were identical in all ten correctness observations: acceptance `0.646`, 47 rounds,
  127 drafted, and 82 accepted. Median throughput was 242.03 vs 241.91 tok/s (`-0.0496%`), which
  is flat. The standing rolling-baseline alert therefore predates this lane and is retained as an
  explicit receipt rather than attributed to the sigmoid work.
- The pre-push freshness hook correctly required a receipt newer than the final engine-scope
  commit. After rebuilding every release binary from HEAD, the supported `--perf-quick` battery
  completed under the local GPU lock. Its correctness stage is green, including kernel-check
  `ALL GREEN (101 cells, 1 standing-manifest skip)`, prime-gate, Qwen run-spec K=1..8, Gemma
  generation/verification/specification, both decode-batch dtypes, graph stress and canary,
  serving, c=64 stress, and served acceptance. The four 31B perf cells finish `0 fail, 0 warn`.
- The normal repository pre-push hook accepted the lane with the generated perf board current;
  `lane/cx-sigrouter2` is published to `origin` without an override.

## Complete

Implementation, mandatory evidence, settle receipts, final local battery, and branch publication
are complete. Merge, tag, and release remain owner actions outside this isolated lane.
