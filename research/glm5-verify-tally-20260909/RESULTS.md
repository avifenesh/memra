# GLM verify six-projection fusion, 2026-09-09

## Current disposition, 2026-09-09

Component KEEP OFF, unserved, code removed 2026-09-09. The owner removed
unserved default-OFF candidates from runtime. The door, dispatch, dedicated
CUDA kernels, counter and oracle/bench executable are gone. Measurements
below describe the archived candidate. Re-derive from
`7c3ddf0db45295dcd2ec67c963f3295ec8d8c14b` (merged in #388 as
`cec4f5a05f6a1ab01e3d8e91247df9b0636e20e5`); there is no retained runtime
switch or serving qualification. rev: 2026-09-23.


Verdict: **KEEP, 1.789758071 ms/round weighted saving**. rev: 2026-09-23

The real-input byte-exact oracle passed before timing. Warmed ABBA x5 with
all 34 KDA layers' weights rotating beats the fixed >=0.5 ms/round bar.
The measured `MEMRA_GLM5_VERIFY_E4M3_FUSED6` door was default OFF; it is now removed.
This is a component result. No sampled candidate serving qualification or
default deployment is claimed.

## Candidate and source

Measured source `3def5881dff5795f4f2a274bceafc8868f9c7076`, based on the
banked tally commit `e0e1ed4eff862ffc2a948f4818ff11d3d4007641` and engine
base `dcfeab7c738912a150ebbfea277112724bb99de4`. One B200, CUDA 13.1.115,
sm_100a. Every capture/oracle/bench GPU phase held the common lock.

The six KDA q/k/v/f_a/g_a/b projections share one Q8 input quantization and
one block-offset batched MMVQ grid. Each output keeps the current batched
E4M3 dot order; the per-tensor scale moves into an explicitly rounded final
multiply. At t2..8, six quantizes + six matvecs + six scales become one
quantize + one matvec: 544 launches removed across 34 layers. t1 retains the
existing six-group implementation; wider or incompatible numerical classes
retain their current path. No F16 activation or weight-layout substitution.

Measured oracle/bench executable SHA256:
`17b66eede0a44006b1eb665fdce85c19cda9b507672c05c52c2c70df31ed3fdb`.
Capture server SHA256:
`6a031a2e3ff62176f21af835577c7e9a9a066484a4c10bbb84c357759e96bca2`.

## Oracle first

Inputs are first-observed real KDA activations at each width from a complete
p32k sampled PP1 DFlash2 HTTP request. Sampling parameters were omitted;
PMIN=0.7, K cap=6. Capture ran the control, 29781 prompt/64 output tokens,
21 rounds, 51 drafted, 43 accepted. All 34 KDA layers were captured at every
requested width, with context metadata. Both oracle arms replay the same
input bytes and the same loaded native E4M3 weights and scales.

| t | KDA layers | Six-projection row checks | Differing f32 bits | Row argmax |
|---|---:|---:|---:|---|
| 2 | 34 | 408 | 0 | 408/408 PASS |
| 4 | 34 | 816 | 0 | 816/816 PASS |
| 7 | 34 | 1428 | 0 | 1428/1428 PASS |

All compared outputs are finite. Normalized mean/max delta is 0/0 for every
row; exact equality is stricter than a nonzero error band. Every layer-width
call increments the candidate-specific engagement counter exactly once.
`fusion/oracle.tsv` contains all 2652 row records. No benchmark invocation
occurred before the all-width oracle marker was written.

## Warmed ABBA x5

All layer weights remain resident. For each width, 40 complete current/fused
round pairs warm the cell, then five ABBA blocks time 100 complete 34-layer
six-projection rounds per position. Weights rotate across all 34 layers,
3.467 GB of E4M3 planes, rather than repeatedly timing one layer's hot data.
The timed chain includes quantization, projection and scale handling.
Position boundaries synchronize the CUDA stream; no weight upload is timed.

| t | Current ms/round | Fused ms/round | Paired saving ms/round |
|---|---:|---:|---:|
| 2 | 2.968260055 | 1.542080860 | 1.426119340 |
| 4 | 4.471642270 | 2.728016835 | 1.743582405 |
| 7 | 8.380124855 | 4.571146255 | 3.808917405 |

Each ABBA block averages its two current and two fused positions. Columns
are medians of five block values; paired saving is the median of the five
paired differences, so it need not equal the difference of the medians.
Fixed conditional weights 48/162, 103/162 and 11/162 yield
**1.789758071 ms/round**. Widths 3/5/6 are outside that inherited weighting,
so this is not a whole-traffic throughput estimate. Other verify phases,
drafting, rollback and HTTP overhead are outside this component.

An additional warmed per-layer ABBA x5 pass reports all 102 layer-width
rows in `fusion/per-layer.tsv`. Those hot-layer measurements are diagnostic;
the rotating whole-component row decides KEEP. `fusion/bench.tsv` retains
all 2100 position records. `summarize-fusion.py` reconstructs the result.
250 ms telemetry covers capture through bench; full-cell GPU temperature
spans 36-51 C, including load/idle transitions. Power and clocks are metadata.

## Final scope and validation

The capture/NVTX diagnostics are removed from runtime code after the cell.
`instrumentation.patch` preserves the tally instrument and
`capture-instrumentation.patch` preserves the real-input capture instrument.
The measured source commit and full private source bundle retain the exact
executable harness and bootstrap. The final runtime change is the default-OFF
fusion; FLAGS.md and KERNELS.md document its arms, rollback and decision date.
The retained oracle/bench executable replays the banked input files.

Remote final formatting and release Clippy -D warnings passed. The final
engine unit suite passed 457 tests, failed 0, ignored 19; the full output
is retained in final-validation.log. No rig cargo, GPU gate, bench or smoke server ran.
Push uses MEMRA_SKIP_PERF_CI=1. Hosted checks remain the PR gate.

The three settled component verdicts remain unchanged. TALLY.md contains the
original ranked tables and archive hash. Full fusion inputs, request/SSE,
logs, telemetry, measured source bundle and validation logs are in the private
fusion archive; its hash and member manifest accompany this result.

publicity: skipped - maintenance research record.

Full fusion archive SHA256: `b1710dde9b4ab697c67a72a470bf7da1e26b44c7ec29c6d41dc025a366fa1164`.

Hosted all-targets Clippy found the engagement counter after a test module.
The declaration was moved before tests; kernel arithmetic and measured
source/binary receipts are unchanged. Hosted checks run on the final PR head.
