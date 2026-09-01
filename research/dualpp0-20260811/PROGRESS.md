# Dual-active PP-2 decode increment 0 progress

Started: 2026-08-11T07:54:21Z  
Lane: `lane/cx-dualpp0`  
Base: `3d485a227222` (post-sigrouter2 and rowwalk merge)  
Target rig: box1, `<rented-box-ip>`, 2x RTX PRO 6000 Server Edition

## Contract

- Prototype dual-active plain decode only at fixed concurrency 8 and 16; keep the default OFF.
- Split requests into two waves, A first and B second, with A on boundary slot 0 and B on slot 1.
- Overlap stage 0 of B with stage 1 of A without changing any request's arithmetic or KV ownership.
- Use only the always-alternating `tx_pipelined` / `prepare_overlap_slots` boundary primitives.
- Refuse dual mode if the PP boundary is single-slot; the negative gate must quote the reason and emit no tokens.
- Preserve the honest serial path at c=1 and wrap both waves in the existing EXACT-16 scope.
- Gate exactness, manifest coverage, liveness, run-gen, and strict batched-decode identity before performance.
- Measure fresh N=5 interleaved dual/serial points at c=8 and c=16 under one box1 lock hold.
- Kill and delete the arm below +15% at c=16; otherwise HOLD it for increment 1 with the default still OFF.

## Progress

- [x] Read the lane brief first, then `research/dualpp-20260811/DESIGN.md` including its binding amendment, then repo law.
- [x] Confirm the dedicated branch/worktree is clean and based on the fresh sigrouter2/rowwalk tip.
- [x] Confirm current CUDA stream/event semantics against primary NVIDIA documentation.
- [x] Map the current PP-2 batch walk, boundary-slot setup, gate manifests, and box1 harnesses.
- [x] Implement the dual-active fork, liveness counter, and fail-closed boot check.
- [x] Add positive exactness/liveness cells and the negative single-slot refusal cell.
- [x] Add CUDA-event A/B stage-span diagnostics and document the experimental flag contract.
- [x] Pass focused local compile/tests without running `cargo fmt`.
- [x] Pass manifest kernel-check, c1..8 exactness/liveness, run-gen, strict decode-batch,
      and run-spec K=1..8 gates on box1.
- [x] Pass the fresh-source c1..8 replay after CUDA-event instrumentation on box1.
- [x] Pass the fresh-boot one-hash golden matrix on box1.
- [x] Capture N=5 interleaved c=8/c=16 serial-vs-dual raw evidence on box1.
- [x] Apply the frozen c=16 kill rule: HOLD, default OFF, for increment 1.
- [x] Land and re-gate the post-review timing-drop and host-bounce-refusal amendment.
- [x] Record the final verdict in `RESULTS.md`.

## Notes

- No performance denominator from before `3d485a227222` is authoritative for this lane.
- Raw logs and manifests belong under `research/dualpp0-20260811/raw/`; failures are quoted from captured output.
- No perf-board update, merge, tag, push, or default promotion is in increment-0 scope.
- CUDA 13.3.1 documents that a stream wait observes the event state captured at the time of
  the wait and is unaffected by a later re-record; this preserves the existing per-slot
  TX-record / RX-wait and RX-record / next-TX-wait sequence.
- First implementation check: `cargo check -p memra-engine --lib` passed in 2m23s, including
  the local CUDA 13.1 sm_120a build (`raw/local/cargo-check-implementation.log`).
- Gate compile passed for `decode-batch-gate` and `kernel-check`. Engine unit tests passed
  63/63 runnable with one CUDA-only spill test ignored (`raw/local/cargo-check-gates.log`,
  `raw/local/cargo-test-lib.log`).
- `kernel-check-step35.cells` now requires `dual-pp-wave-split` and
  `dual-pp-single-slot-refusal`. The model gate's negative cell serial-primes with slot 1
  unprepared, then requires the exact refusal and unchanged cache positions; positive dual
  replays require the overlap counter to advance.
- First box1 correctness hold passed at source `824452f81852`: kernel-check reported 85
  green cells (21 model-inapplicable cells skipped), dual c1..8 was bit-identical with
  liveness at every c>=2, strict batch was all green, run-gen reported both argmax MATCH
  checks, and run-spec reported self-consistency PASS for K=1..8
  (`raw/box1/correctness/`). A supplemental direct B=16 probe was correctly refused by
  the pre-existing Step serial oracle (`B=16 > cap 8 with no exact tier`) before it reached
  this lane's arm; live c=16 serving is scheduler-chunked into exact B<=8 ticks and is gated
  through the end-to-end hash/perf harness instead.
- The 08:21Z inbox amendment is implemented: `MEMRA_DUAL_PP_TIMING=1` brackets all four
  A/B stage layer ranges with CUDA events and exports cumulative spans through `/metrics`.
  The box1 script collects a separate unscored c8/c16 diagnostic process, keeping the frozen
  N=5 throughput points instrumentation-free. `docs/FLAGS.md` now records the double-slot,
  `tx_pipelined`-only, and fail-closed relationships.
- Instrumented source `2cca9e63ef1a` rebuilt on box1 and repeated the full battery green:
  manifest 85/85 applicable cells, negative refusal, c1..8 bit identity and liveness,
  strict batch, both run-gen argmax MATCH checks, and run-spec K=1..8
  (`raw/box1/correctness-instrumented/`). Local compile plus the two focused dual tests
  and the server metrics route test also pass. The first remote build attempt is retained
  with its literal `cargo: command not found`; the explicit `/home/ubuntu/.cargo/bin/cargo`
  retry passed in 29.61s (`raw/box1/build/`).
- First one-hash attempt produced matching `21b8293f...` bytes for every executed boot and
  c1..8 point, then its receipt failed `assert len(summaries) == 26` with 12: a compound
  Bash `local` declaration expanded the caller's matrix label before assigning the function
  label, so c1..8 artifacts overwrote one directory per arm. The failed raw run is retained;
  the declaration is split and the complete 26-summary matrix will be rerun fresh.
- Corrected one-hash rerun passed: 10 alternating fresh boots plus serial and dual c1..8,
  26 distinct summaries / 82 requests, every request exactly
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
  All five dual-armed c1-only boot logs have no `[dual-pp]` engagement marker, proving the
  honest serial fallback on the two-card pair (`raw/box1/hash-matrix/`).
- First perf invocation failed before server launch with the captured literal
  `box1-perf.sh: line 113: label: unbound variable`: two perf helpers had the same dependent
  compound-`local` expansion class found by the hash receipt. No point was scored. The raw
  failure is retained in `raw/box1/perf-harness-failed/`; all dependent declarations are split
  before the fresh restart.
- Fresh frozen perf run started on box1 at 09:12:06Z from `ed4ff393969b` under the single GPU
  lock. Round 1 completed without errors: serial c8/c16 168.842/169.913 tok/s and dual c8/c16
  203.408/202.339 tok/s. These are preliminary single-round observations; the binding verdict
  remains the completed N=5 median at c16.
- Rounds 2-3 completed without request errors. Their serial c16 points are 169.596/170.166
  tok/s and dual c16 points are 203.497/202.301 tok/s; all completed arms also produced the
  required 512 tokens per request. Rounds 4-5 and the unscored stage diagnostic remain running.
- Frozen box1 perf completed PASS with 20 summaries / 240 requests / zero errors. At c16 the
  N=5 median moved from 169.879 to 202.339 tok/s (+19.108%), clearing the binding 15% floor;
  c8 moved from 169.009 to 203.340 tok/s (+20.313%). The unscored companion recorded 1,024
  c16 overlaps and a 1.057 stage-balance ratio (stage0 11.593 ms, stage1 12.257 ms mean).
  Verdict is HOLD, default OFF, for increment 1. Raw logs are in `raw/box1/perf/`.
- The 09:2xZ post-review amendment is implemented before verdict: CUDA event creation and
  `elapsed_ms` failures are diagnostic-only, warn once, increment an exported dropped-sample
  counter, and cannot propagate into decode. Dual mode now refuses active host-bounce transport
  with a quoted reason; model-free manifest and model-level no-token/no-cache-advance negative
  cells cover it. Local all-target compile and four focused dual tests pass without `cargo fmt`
  (`raw/local/post-review/`). Box1 rebuild and affected-cell replay remain.
- Final source `4f32f3b25a16` rebuilt on box1 in 44.04s. The amended manifest is 86/86
  applicable cells green (21 model-optional skips), including `dual-pp-hostbounce-refusal`.
  The live Step replay then passed both quoted no-token/no-cache-advance refusal cells plus
  c1..8 bit identity and +8 liveness at every c>=2. The same-lock full battery also passed
  strict batch, both run-gen MATCH checks, and run-spec K=1..8. Every gate exit is zero;
  final-source receipts are in `raw/box1/correctness-post-review/` and `raw/box1/post-review/`.
