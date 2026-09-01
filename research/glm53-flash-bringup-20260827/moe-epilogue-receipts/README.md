# glm5_next fused MoE epilogue — acceptance receipts (lane/glm53-epilogue, 2026-08-28)

Roadmap step 4, the fused-epilogue half. `MEMRA_MOE_FUSED_EPI`, default OFF.
Gate: `crates/memra-engine/tests/glm5_moe_epilogue_gpu.rs`.

All runs: RTX 5090 laptop (rig), `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`,
release build, `--test-threads 1`. **Rig law: exactness only. No timing number was taken here
and none is claimed.**

## Files

| file | what it is |
|---|---|
| `host-gates.txt` | the four host-only gates (no CUDA). Non-vacuity + the four plan mutations measured on the reference alone. |
| `gpu-gates-GREEN.txt` | all five GPU gates passing against the honest engine. |
| `gpu-gates-RED-post-clamp.txt` | RED: the fused kernel's epilogue swapped to step35's POST form. |
| `gpu-gates-RED-no-macro-fold.txt` | RED: the fused arm's per-expert macro scales replaced with 1.0. |
| `nsys-launch-counts.md` | the MEASURED launch-count delta, fused vs unfused, and how it decomposes. |
| `nsys-{fused,unfused}-kernsum.csv` | the raw `nsys stats --report cuda_gpu_kern_sum` rows behind it. |

## GREEN

```
fused:            WORST 4.822e-3 on `decode step 2`   (tol 1.0e-2)
unfused control:  WORST 4.822e-3 on `decode step 2`   (tol 1.0e-2)
fused-epilogue dispatches: 51 of 89 token-layer opportunities   (unfused arm: 0)
SLRU  fused   hits=498 misses=645 staged=2,972,160 B  slots=12
      unfused hits=414 misses=387 staged=1,783,296 B  slots=12
two arms:  0.000e0 and 0/N elements differ IN BITS on every one of the 11 rows
           (prefill T=1/3/8/65, prime last row, decode steps 0..5)
```

The fused arm and the shipped sequential loop are **bit-identical** on this fixture, and both sit
on the same distance from `memra-reference`. That distance is therefore the fixture's floor (q8_1
activation quantization against an f32 reference), not the fusion's.

The miss counts are ASSERTED, not assumed (`check_cache_pressure`): both arms must report
`hits > 0` and `misses > 0`, so the arm is gated under recurring eviction rather than on a warm
layer. That is the whole difference from gdec, and the sibling residency gate's header records
the same by-construction argument going silently vacuous on its first run.

TWO NUMBERS IN THAT BLOCK ARE FINDINGS, not decoration.

**51 of 89, not 89 of 89.** The other 38 token-layers hit the arm's pass-2 re-verification: an
admission evicted one of the token's own already-admitted blocks and the arm fell closed to the
sequential loop. At `SLOTS = 12` against a 9-block working set that is expected — the margin is
three slots and `dispatch_source` has no keep-list, so SLRU can take a block admitted moments
earlier in the same token. It costs nothing numerically (the fallback is the bit-identical
sequential loop, which is why every row still matches) and it means the fail-closed seam is
EXERCISED by this gate rather than merely written.

**The fused arm staged 1.19 MB more than the unfused one (645 vs 387 misses).** Same cause: it
admits all 9 blocks BEFORE running anything, where the sequential loop admits one projection at a
time and consumes it immediately, so at a three-slot margin the up-front admission evicts more.
This is a real cost of the arm at tight slot counts and it is stated rather than hidden. It does
not describe the serving regime — 12000 slots against a 24-block working set — but it does say
plainly that the arm's profitability is conditional on slots comfortably exceeding `3 * n_used`,
not merely reaching it. Sizing that margin is box work, not rig work.

Mutations, measured against the SAME GPU output the passing assertion uses:

```
mutation[post-for-pre-clamp]:       closest row 7.563e-2   worst 3.034e-1
mutation[plain-swiglu (no clamp)]:  closest row 8.615e-1   worst 3.984e0
mutation[shared-expert-dropped]:    closest row 1.297e-1   worst 4.226e-1
mutation[softmax-for-sigmoid]:      closest row 5.301e-2   worst 2.585e-1
mutation[macro-scale-dropped]:      worst 6.244e-2
                                                  (tol 1.0e-2, floor 3.0e-2)
```

## RED — the gate proved on deliberately wrong fusions

The four plan mutations above are the reference answering "what if the engine had fused THIS".
They are necessary but not sufficient: they never change the engine. So the fusion itself was
broken, twice, and the gate re-run.

| deliberately wrong fusion | `fused` vs reference | `two arms` | `unfused` control |
|---|---|---|---|
| kernel epilogue -> POST clamp (`min(silu(g*gs), l) * u`) | **FAIL 1.747e-1** on `prefill T=65` | **FAIL 1.251e-1**, 32/32 bits differ | still 4.822e-3, PASS |
| dispatch -> `gs = us = 1.0`, `wv = w` (no macro fold) | **FAIL 1.624e-1** on `prefill T=8` | **FAIL 7.698e-2**, 32/32 bits differ | still 4.822e-3, PASS |

Both breakages are 16x-35x above the tolerance, and in both the unfused control stayed green — so
the failure is attributable to the fused arm and not to the harness.

Reverted after each run; `gpu-gates-GREEN.txt` is the tree as committed.

## Launch count — MEASURED (nsys, counts only) and confirmed against source

Per token-layer, `n_used = 8`, the SLRU dp4a path glm5_next actually takes today
(`hybrid_forward.rs` `moe_ffn_inner`, the `cache_dispatch && moe_q8` branch):

| | launches |
|---|---|
| `quantize_q8_1_view(z)`, once per token | 1 |
| per routed expert x8: gate `qmatvec_expert_q8`, up `qmatvec_expert_q8`, `swiglu_preclamped_mul_scaled`, `quantize_q8_1(act)`, down `qmatvec_expert_q8`, `axpy_into` | 8 x 6 = 48 |
| **sequential total** | **49** |
| `quantize_q8_1_view(z)` | 1 |
| `moe_gate_up_preclamp8_q8` | 1 |
| `quantize_q8_1(act, n_used, n_ff)` | 1 |
| `moe_down8_fma_q8` | 1 |
| **fused total** | **4** |

Over 42 MoE layers: 2058 -> 168 launches per token, a reduction of 1890.

**That structure is measured, not only derived.** `nsys-launch-counts.md` profiles the gate's own
two arms (`TOP_K = 3`, so n_used=3): total kernel launches 2698 -> 1933, delta -765, and it
decomposes EXACTLY — `qmatvec_expert_q8` -459 (= -9 x 51), `swiglu_preclamped_mul_scaled_f32`
-153 (-3 x 51), `axpy_f32` -153 (-3 x 51), `quantize_q8_1` -102 (-2 x 51), plus 51 each of the two
fused kernels. -15 per engaged token-layer x 51 dispatches = -765, the entire total. No other
kernel's count moved. The n_used=8 figures above are that same structure extrapolated.

The per-layer `moe_out` allocation is unchanged in kind; the fused arm takes it uninit because
`moe_down8_fma_q8` fully overwrites its row. H2D admissions are the same blocks through the same
`dispatch_source`, but at this fixture's slot pressure their COUNT is not unchanged — see the
staging finding above.

`ATTRIBUTION.txt` sized the MoE loop at ~1750 of ~3200 launches/token; the count above is the same
quantity derived per branch and is consistent with it.

**No throughput claim.** The rig is exactness-only by law and the 190.7 GB artifact has never been
on it. Whether 1890 fewer launches per token converts into the ~5.3 us x N the attribution implies
is an interleaved A/B on the 2x RTX PRO 6000 bench box, with the vendor-default sampled twin, and
it has not been run.

## What this lane did NOT do: the 42 router DtoH copies

The brief scoped "kill the 42 per-layer router device-to-host copies and their stream syncs" into
this step. Source says it is gated on residency (roadmap step 3), not on the epilogue:

- The sigmoid router is ALREADY a device kernel (`moe_router_sigmoid_topk_f32`,
  `hybrid_forward.rs:471`, default ON). What crosses the bus is only the tiny `[t, n_used]`
  `sel`/`w` pair, not the full logits — but that readback is still a sync per MoE layer.
- The host needs `sel` because it drives ADMISSION: a miss is a host-issued `memcpy_htod` into an
  SLRU slot. The zero-DtoH arms (`moe_ffn_dev`) avoid it only by reading a DEVICE pointer table of
  fixed slot addresses, which exists exclusively for a layer whose blocks are already all resident
  (`moe_cache::layer_dev_row`, `prewarm_layer`) or for a load-time resident slab (`dev_exps`).
- glm5_next has neither today. One MoE layer is `288 experts x 3 = 864` blocks; the serving recipe
  runs 12000 SLRU slots across 42 layers, i.e. ~285 blocks per layer. No layer can be fully
  resident, so no layer can carry a device pointer table, so the readback cannot be removed on
  this placement.

So the DtoH kill is downstream of step 3 (full expert residency across both cards), exactly as the
roadmap's own "residency is the keystone, not the epilogue" note says for gdec. The fused epilogue
does not need residency and lands independently; the sync removal does.
