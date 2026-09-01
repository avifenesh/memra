# nsys kernel-LAUNCH-COUNT delta, fused vs unfused (rig 5090, 2026-08-28)

COUNTS ONLY. The rig is exactness-only by law: no duration column from this profile is
quoted anywhere, and none was used to justify anything.

Command (per arm, under `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`):

    nsys profile --trace=cuda -o <arm> \
      target/release/deps/glm5_moe_epilogue_gpu-<hash> --ignored --exact <test> --test-threads 1
    nsys stats --report cuda_gpu_kern_sum --format csv <arm>.nsys-rep

Arms are the gate's own `unfused_control_matches_the_reference` (MEMRA_MOE_FUSED_EPI=0)
and `fused_epilogue_matches_the_reference` (=1) — identical fixture, identical workload
(prefill T=1/3/8/65 + prime 6 + 6 decode steps = 89 token-layer opportunities), identical
model load. `TOP_K = 3` in this fixture, so read the per-token-layer figures at n_used=3.

    TOTAL kernel launches   unfused 2698   fused 1933   delta -765

Every kernel whose count changed, and nothing else did:

| kernel | unfused | fused | delta | per engaged token-layer (51 dispatches) |
|---|---:|---:|---:|---:|
| `axpy_f32` | 267 | 114 | -153 | -3 (one per expert) |
| `moe_down8_fma_q8` | 0 | 51 | 51 | +1 |
| `moe_gate_up_preclamp8_q8` | 0 | 51 | 51 | +1 |
| `qmatvec_expert_q8` | 801 | 342 | -459 | -9 (3 experts x gate/up/down) |
| `quantize_q8_1` | 356 | 254 | -102 | -2 (3 per-expert act quantizes -> 1 batched) |
| `swiglu_preclamped_mul_scaled_f32` | 289 | 136 | -153 | -3 (one per expert) |

**-15 launches per engaged token-layer, x51 dispatches = -765, which is the whole total
delta to the launch.** Nothing else in the program moved: the profile is the arithmetic.

At the real model's `n_used = 8` the same structure gives **49 -> 4** per token-layer
(1 z-quantize + 8 x {gate, up, preclamp, act-quantize, down, axpy} versus z-quantize +
gate/up + act-quantize + down/FMA), i.e. **2058 -> 168 launches per token** across 42 MoE
layers. That extrapolation is arithmetic; what is MEASURED here is the n_used=3 structure
it extrapolates from.

H2D admissions are not in this table and are unchanged in kind — the same blocks through
the same `dispatch_source`. Their COUNT is not unchanged at this slot pressure: see the
staging note in README.md.
