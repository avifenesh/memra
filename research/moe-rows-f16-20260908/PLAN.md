Concluded on 2026-09-09: NEGATIVE numeric no-go; F16 door removed.
See [final results](../moe-rows-f16-20260909/RESULTS.md). This document retains the prior interrupted attempt.

# F16 rows real-input component cell, 2026-09-08

PR #294 stays draft. Rebased without conflicts onto main 72aa777c3.
No serving dispatch, new default, or serving qualification is part of this cell.

Capture native mint prompt inputs with the existing input and binary selection
trace diagnostics. Replay the same f32 activation, exact expert IDs and f32 route
weights in both arms at t=2/4/7, 4096 -> 2048 -> 4096, top8, preclamp10.
Require exactly 42 routed layers and complete token coverage. The capture is a
native short prompt pass, not an archived DFlash2 verification activation tape.
The timing geometry matches DFlash2 verification; this distinction stays explicit.

Gate the resident V1-to-V2 weight repack against the established host converter.
Then compare gate/up and each arm's own complete conversion + gate/up + activation
+ conversion + down output against the f64 chain of the same weight bytes. Require
finite outputs, candidate mean <= 1.05 * current mean, max <= 2 * max(current max,
1e-3), normalized by the reference tensor max absolute value. Require each complete
output row's argmax to agree. Stop on the first failing layer/tensor. The existing
synthetic gate also exercises the wrong-layout red arm and accumulation twins.
Do not benchmark after an oracle failure.

Only after all shapes pass: ten warmups per weight copy, four rotating copies,
five ABBA blocks per layer and shape, 100 complete chains per interval, final
stream drain included. Setup/upload/resident repack is recorded separately. Include
both per-call activation conversions. The current arm is the interleaved NVFP4
ILP rows chain; F16 uses the V2 slot-major rows chain. Router and shared expert
costs are unchanged and excluded from this routed-component delta.

All 42 routed layers have weight 1 per verification round. In the profile's first
64 rounds for each of three requests, widths 2/4/7 occur 48/103/11 times. Report
per-width sums and their 48/162, 103/162, 11/162 weighted mixture over measured
widths. Also report the observed-window contribution with denominator 192;
30 rounds at widths 3/5/6 remain unmeasured. Do not interpolate them. Candidate
retention requires >=0.5 ms per round in the measured-width mixture. A failing
numeric oracle is a no-go independently of timing.

Every GPU phase uses device 0 and the common GPU lock. Persist each shape's raw
receipt and copy it off the interruptible host before starting the next shape.
No local rig build, gate or test runs.
