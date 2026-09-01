# glm5-hyper-ppn-gate — OVERLAY TWIN arm (lane/glm5-vision-default-on, 2026-08-30)

Blocker-2 gate for the vision default-ON flip: the mixed-embedding splice under the
ppN door. Binary: `glm5-hyper-ppn-gate` (arm 5 added this lane; arms 1-4 unchanged).
Rig 5090, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, exactness only.

TRUTH BY SUBSTITUTION (not arm-equality): overlay rows are `m.embed()` rows of known
substitute tokens, so `prime_cache_overlaid` over the placeholder prompt must be
BIT-IDENTICAL to `prime_cache` over the substituted prompt. Three overlay arms per
invocation — door-OFF serial, door-ON monolithic, door-ON two-call windowed
(`EmbedOverlay::window`, the prefill-tick shape) — each with a decode continuation
off the primed cache.

| log | knobs | verdict |
|---|---|---|
| 10-n2-even.log | stages=2, even split | 6/6 arms PASS, all bit-identical |
| 11-n2-split1.log | stages=2, MEMRA_PP_SPLITS=1 | 6/6 PASS |
| 12-n2-split3.log | stages=2, MEMRA_PP_SPLITS=3 | 6/6 PASS |
| 13-n2-streams0.log | stages=2, MEMRA_PP_STREAMS=0 (same-stream seam) | 6/6 PASS |
| 16-n3-asym.log | stages=3, MEMRA_PP_SPLITS=1,3 (the serving stage COUNT — middle-stage rx→tx path live) | 6/6 PASS |
| 17-n4-even.log | stages=4 (one layer per stage) | 6/6 PASS |
| 19-n2-longer.log | stages=2, P=16 N=24 | 6/6 PASS |
| 20-n2-chunked-prime.log | stages=2, MEMRA_PRIME_CHUNK=8, P=16 — the CHUNKED ppN prime (93927b1fa, merged mid-lane): two `hyper_prime_ranges` per call, overlay windowed per range in the chunk loop | 6/6 PASS |
| 90-RED-span-shift-n2.log | MEMRA_GLM5V_GATE_RED=span-shift, stages=2 | RED BITES: overlay-serial / overlay-ppn / overlay-ppn-windowed each 9/9 mismatched, exit 1 (text arms stay green — the red is the SPLICE PLACEMENT, nothing else) |
| 91-RED-span-shift-n3.log | red + stages=3, splits=1,3 | RED BITES: 3 overlay arms 9/9 mismatched, exit 1 |

Splice seam truth (verified in code, hybrid_forward.rs): `prime_cache_hyper_ppn`
embeds tokens on STAGE 0 ONLY; stages s>0 receive the already-expanded
`[t, streams, hidden]` boundary payload — so the overlay splices at stage-0
embedding intake (`EmbedOverlay::splice_into`, shared with the serial hyper walk)
and no other stage carries overlay arithmetic. Overlay rows are primary-resident; a
placement with stage 0 off the primary device refuses loudly. Under the CHUNKED
ppN prime (93927b1fa, merged into the bringup head mid-lane) each range takes the
overlay windowed to its own call-relative slice (`EmbedOverlay::window` inside the
chunk loop), so splice placement is chunk-schedule-invariant — arm
20-n2-chunked-prime holds that down.

Rig scope: same-device stages (one card). Cross-device placements share the same
stage-0 intake code path; the cross-device overlay refusal (stage 0 not primary) is
compile-visible, not exercised here.

Composition note (same as the text arms): this gate anchors split-vs-unsplit and
overlay-vs-substitution; `tests/hyper_connections_gpu.rs` anchors the unsplit hc
walk to `memra_reference`. Cite both.
