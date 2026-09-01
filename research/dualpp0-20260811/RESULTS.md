# Dual-active PP-2 decode increment 0 results

Date: 2026-08-11

Lane: `lane/cx-dualpp0`

Fresh base: `3d485a227222` (post-sigrouter2 and rowwalk)

Final implementation commit: `4f32f3b25a160c4d589a16c159c6fe6d440b1638`

## Verdict

**HOLD for increment 1. Keep the feature default OFF.**

On box1, the frozen c=16 N=5 median improved from 169.879 to 202.339 aggregate decode-window
tok/s, a **+19.108%** gain. This clears the binding +15% increment-0 floor with every required
exactness gate green. It does not authorize a default flip, merge, tag, or release: the arm
remains available only through `MEMRA_DUAL_PP=1` together with `MEMRA_PP_OVERLAP=1`.

The companion timing run rules out dead mechanics and material stage imbalance. At c=16 the
host-overlap counter advanced 1,024 times, stage 0 averaged 11.593 ms, stage 1 averaged
12.257 ms, and the longer/shorter stage ratio was 1.057. The measured gain is therefore
consistent with live overlap across a balanced PP-2 cut.

## What landed

- `decode_step_batch_dual` splits the batch into wave A=`ceil(c/2)` and wave B=`floor(c/2)`;
  c=1 takes the existing serial PP-N path.
- Wave A uses boundary slot 0 and wave B slot 1. The only boundary primitives in the dual body
  are `prepare_overlap_slots` and `tx_pipelined`; it never calls the env-conditional `tx()`.
- A scoped second host walker issues stage0(B) while the caller drives stage1(A). Results are
  concatenated in original request order, and one final publication orders both waves behind
  the caller stream.
- The existing EXACT-16 scope, when the checkpoint admits that tier, covers both waves and both
  stage-owned engines.
- Host-thread liveness and optional per-wave CUDA-event spans are exported through `/metrics`.
  Timing event creation or elapsed-time errors now warn once, increment
  `dropped_timing_samples`, record no invalid span, and cannot fail decode.
- Dual mode refuses before cache advance or token output when either required transport contract
  is absent:

  - single slot: `decode_step_batch_dual: refused: PP boundary is single-slot; set
    MEMRA_PP_OVERLAP=1 so both alternating boundary slots are prepared before dual-active decode`
  - unvalidated host bounce: `decode_step_batch_dual: refused: MEMRA_PP_HOST_BOUNCE=1 is
    unvalidated for dual-active decode; disable MEMRA_DUAL_PP or use peer transport`

`docs/FLAGS.md` records the default-off status, overlap dependency, tx-pipelined-only rule,
host-bounce refusal, and diagnostic timing behavior.

## Correctness gates

### Final-source battery

The final implementation was rebuilt on box1 (2x RTX PRO 6000 Blackwell Server Edition) in
44.04 seconds and tested under one `/tmp/memra-gpu.lock` hold.

| Gate | Result |
|---|---:|
| `kernel-check --require-manifest tools/kernel-check-step35.cells` | 86 applicable cells green; 21 model-optional cells skipped |
| `dual-pp-single-slot-refusal` | PASS; exact reason, zero output, unchanged cache positions |
| `dual-pp-hostbounce-refusal` | PASS; exact reason, zero output, unchanged cache positions |
| Step dual replay, B=1..8, 8 steps each | 37,122,048 split-arm logits, zero differing bits |
| `DUAL_PP_OVERLAPS`, B=2..8 | +8 at every width |
| strict decode-batch battery | ALL GREEN |
| `run-gen` | prefill/decode MATCH and batched-prime/tokenwise MATCH |
| `run-spec` | self-consistency PASS at K=1..8 |

All five gate exit receipts are zero. Raw final-source evidence is in
`raw/box1/correctness-post-review/`; the release build is in `raw/box1/post-review/`.
Local CUDA 13.1 sm_120a all-target compile, four focused dual policy/timing tests, and the server
metrics route test are in `raw/local/post-review/`. `cargo fmt` was not run.

### One-hash serving matrix

The corrected serving matrix passed 10 alternating fresh boots plus serial and dual c=1..8:
26 summaries, 82 requests, and every request exactly
`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
All five dual-armed c=1-only boot logs lack the dual engagement marker, proving the honest serial
fallback. Receipt: `raw/box1/hash-matrix/summary.json`.

The first matrix execution produced the correct hash for every request it retained but failed its
receipt count (12 instead of 26) because a Bash compound `local` declaration caused per-width
artifact overwrites. That failed evidence is preserved in `raw/box1/hash-matrix-failed/`; it is
not included in the passing matrix.

### Direct B=16 qualification

The Step IQ4_XS checkpoint's existing serial oracle refuses a direct B=16 tick with the captured
reason `B=16 > cap 8 with no exact tier` before the dual fork. The failed probe is retained in
`raw/box1/correctness-c16/`. Live c=16 serving is scheduler-chunked into exact B<=8 ticks, so the
scored c=16 result below is end-to-end 16-request traffic whose decode chunks each exercise the
gated dual path. Direct B=16 admission for this checkpoint was not claimed.

## Frozen performance block

Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition.

Protocol: N=5 interleaved serial/dual arms, rotating c=8/c=16 order, one lock hold, 512 generated
tokens per request, no artificial cooldown.

Metric: aggregate completion tokens after the first visible token divided by decode-window time.

Receipt: 20 summaries, 240 requests, 240 successful full-length completions, zero errors.

| Live concurrency | Serial samples (tok/s) | Dual samples (tok/s) | Serial median | Dual median | Delta |
|---:|---|---|---:|---:|---:|
| 8 | 168.842, 170.506, 168.840, 170.212, 169.009 | 203.408, 200.703, 203.603, 200.507, 203.340 | 169.009 | 203.340 | **+20.313%** |
| 16 | 169.913, 169.596, 170.166, 169.406, 169.879 | 202.339, 203.497, 202.301, 203.034, 201.983 | 169.879 | 202.339 | **+19.108%** |

The sampled thermal regime was 31-47 C and 2,287-2,407 MHz over 5,752 250-ms samples. Full
request rows, server logs, GPU traces, hashes, and the reducer output are in
`raw/box1/perf/`; `raw/box1/perf/summary.json` is the machine-readable verdict.

The scored processes had `MEMRA_DUAL_PP_TIMING` unset. A separate unscored companion process
enabled CUDA events so instrumentation could not perturb the kill-rule denominator:

| Concurrency | Overlap counter delta | Stage0 mean | Stage1 mean | Balance ratio |
|---:|---:|---:|---:|---:|
| 8 | 512 | 11.692 ms | 12.481 ms | 1.067 |
| 16 | 1,024 | 11.593 ms | 12.257 ms | 1.057 |

The scored source was `ed4ff393969bb94bf167ce4ea530e4a6bb5e77f1`. The subsequent binding
post-review patch changes only diagnostic error handling while timing is enabled and refuses the
host-bounce transport; the scored arm had timing unset and peer transport active. The scored
control flow and denominator are therefore unchanged, while the final source was separately
rebuilt and re-gated above.

The first perf invocation failed before server/model launch with the captured literal
`box1-perf.sh: line 113: label: unbound variable`; its points file is empty. That zero-score
failure is preserved in `raw/box1/perf-harness-failed/` and excluded from N=5.

## Decision boundary and next increment

The c=16 gain is 4.108 percentage points above the frozen floor, exactness is green, host and GPU
timing both show live overlap, and the stage split is balanced. Increment 0 therefore ends at
**HOLD**, not KILL.

Increment 1 must retain the default-off posture while it adds the worker scheduler integration
and the required cross-device slot-collision soak. Dual+bounce remains explicitly closed until a
dedicated host-bounce c=1..8 one-hash matrix passes. No perf board was changed, and this lane did
not merge, tag, push, or publish a release.
