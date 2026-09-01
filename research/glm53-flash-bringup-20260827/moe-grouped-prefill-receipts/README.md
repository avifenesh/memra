# glm5_next GROUPED MoE PREFILL: acceptance receipts (lane/glm5-grouped-prefill, 2026-08-29)

Prefill-gap plan L1 (`../prefill-gap-20260829/PREFILL-GAP.md` §3). `MEMRA_MOE_GROUPED_PREFILL`:
landed default OFF (FLAGS.md row in the same commit as the flag), then flipped **DEFAULT ON on
2026-08-29** after the box A/B (`box-ab-20260829/`) with the B5550 near-tie argmax movement
accepted verbatim by the owner (the 8-draw census in `box-ab-20260829/tie-off.out` /
`tie-on.out` is the acceptance receipt: the OFF arm itself draws the ON arm's token at the
flipped position under vendor defaults). `=0` is the rollback seam. The engine-side logit-delta
cell remains OWED as follow-up (no logprobs surface; needs an offline logits dump).
Gate: `crates/memra-engine/tests/glm5_moe_grouped_prefill_gpu.rs`, on the fused-epilogue lane's
fixture family (`../moe-epilogue-receipts/`).

All runs: RTX 5090 laptop (rig), `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`,
release build, `--test-threads 1`. **Rig law: exactness only. No timing number was taken here
and none is claimed.**

## Files

| file | what it is |
|---|---|
| `gpu-gates-GREEN.txt` | all 8 GPU gates passing against the honest engine, both arms' announce lines included (full --nocapture output). |
| `gpu-gate-4096-GREEN.txt` | the chunk-cap gate (t=4096, `PRIME_CHUNK_MAX_TOKENS`) passing, both arms. |
| `gpu-gates-RED-tokensort-permutation.txt` | RED: two pairs' expert assignments swapped after the sort. |
| `gpu-gates-RED-scatter-off-by-one.txt` | RED: every token accumulated the next token's pairs. |
| `nsys-launch-counts.md` | the MEASURED launch-count delta, grouped vs sequential, and its exact decomposition. |
| `nsys-{sequential,grouped}-kernsum.csv` | the raw `nsys stats --report cuda_gpu_kern_sum` rows behind it. |
| `AB-PLAN.md` | the box A/B protocol that the default flip requires. |
| `box-ab-20260829/` | the A/B, RUN (owner window, 4x RTX PRO 6000, PP3 full residency): **7.2-7.4x TTFD, 85 -> 616-639 tok/s prefill**, engagement 42/42, sampled twin green; default flip still blocked by a first-token argmax flip on 1 of 3 prompts at a measured-soft position. `BOX-AB.md` is the receipt. |

## GREEN: per width, with the class each t rides

```
                 vs reference        grouped dispatches   class
prefill T=1     1.340e-3             0                    per-token routed loop + matvec-class shexp
prefill T=2     3.087e-3             0                    per-token routed loop + matvec-class shexp
prefill T=15    1.732e-3             0                    per-token routed loop + matvec-class shexp
prefill T=16    3.627e-3             0                    per-token loop + GEMM-class shexp (the knee row)
prefill T=17    7.976e-5             1                    grouped f16 GEMM (routed) + GEMM-class shexp
prefill T=64    1.424e-4             1                    grouped f16 GEMM (routed) + GEMM-class shexp
prefill T=4096  1.972e-4             1                    grouped f16 GEMM (routed) + GEMM-class shexp
prime  T=20     (in band)            1                    prime chunk (the real serving path)
decode x4       (in band)            0                    decode t=1 (never grouped)
sequential control: WORST 4.568e-3 (T=64) / 4.930e-3 (T=4096)   (tol 1.0e-2)
```

TWO PROPERTIES IN THAT TABLE ARE FINDINGS, not decoration:

**The grouped rows sit an ORDER BELOW the sequential control's own floor** (7.976e-5 vs
4.568e-3). The control's floor is the q8_1 ACTIVATION quantization its dp4a expert matvecs pay
against the f32 reference; the grouped arm's f16-mirror class (row-normalized f16 activations
into the NVFP4-direct sk GEMM) does not pay it. The band bar (1e-2, the epilogue gate's
calibrated class) is kept; the arm is NOT claimed bit-stable (the grouped GEMM class is
measured nondeterministic run-to-run in the step37 lane), only in-band.

**The t=16 knee is BITWISE.** With the flag ON, every row the arm does not dispatch
(t in {1,2,15,16}, all decode steps) is 0-bits-different from the flag-OFF run: the flag
provably does not leak into programs it does not dispatch. t=17/64 rows differ in 543/544 and
2048/2048 bit positions (the numeric-class change, in-band), and the seam sits exactly at
`MOE_DEV_MAX_T` = `GEMM_M_THRESHOLD` = 16.

Routing exactness is MEASURED, not only argued: `MEMRA_MOE_TRACE` + `MEMRA_MOE_WEIGHT_TRACE`
files are byte-identical between arms (718 route bytes, 4,858 weight bytes), on top of the
by-construction argument (one shared `moe_router_logits` + `moe_route_sigmoid_cfg` invocation).

Fail-closed is EXERCISED: on the SLRU placement (no resident slab) the ON arm records 0
dispatches and the sequential fallback stays green.

Reference-side mutations, measured against the SAME grouped GPU output the passing assertion
uses (closest grouped row):

```
mutation[post-for-pre-clamp]:      1.909e-1
mutation[plain-swiglu (no clamp)]: 1.601e0
mutation[shared-expert-dropped]:   3.857e-1
mutation[softmax-for-sigmoid]:     2.597e-1
mutation[macro-plane-flattened]:   7.160e-2   (engine-loaded bytes, worst grouped row)
                                              (tol 1.0e-2, floor 3.0e-2)
```

## RED: the gate proved on deliberately wrong engine arms

The reference mutations never touch the engine, so the arm itself was broken twice (the two
failure shapes a token-sort implementation actually produces), the gate re-run, and the output
banked. Both were reverted; the committed tree is the GREEN one, re-verified after revert
(8/8 GPU tests).

| deliberately wrong arm | grouped vs reference | sequential control |
|---|---|---|
| token-sort permutation (`ex_pairs.swap(0, n_pairs-1)` after the sort) | **FAIL 2.068e-1** on `prefill T=17` | still 4.568e-3, PASS |
| scatter off-by-one (`tids[p] = (p + n_used) % n_pairs`) | **FAIL 3.934e-1** on `prefill T=64` | still 4.568e-3, PASS |

Both are 20x-39x above tolerance, and in both the sequential control stayed at its floor: the
failures are attributable to the grouped arm, not the harness. `the_two_arms_agree...` and
`wrong_programs_fail_the_gate` failed alongside in both runs (three independent detectors per
breakage).

## What this lane did NOT do

- **No throughput claim.** The launch-count reduction (measured -1889 on the fixture workload,
  ~8.43M -> ~2k per 4096-chunk extrapolated at the real geometry, `nsys-launch-counts.md`)
  converts to prefill tok/s only through the box A/B in `AB-PLAN.md`, which needs an owner
  window on the 4-card box (the 1M cell holds it).
- **The 42 per-layer router host readbacks remain.** The arm calls the same
  `moe_route_sigmoid_cfg` host oracle as the sequential loop, deliberately, since that is what
  makes routing exactness hold by construction. Killing the D2H is PREFILL-GAP L4, downstream,
  and `MEMRA_PRIME_PROF=1` (`[moe-grouped-prefill-prof]`) is wired to size it on the box.
- **No claim for any other sigmoid-router arch.** The arm is keyed to `cfg.glm5` (no
  generic-model support claims); step37 keeps its own TP grouped prime, and a POST-clamp layer
  reaching the arm errs rather than picking a clamp form by default.
