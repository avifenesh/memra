# Grouped MoE prefill: MEASURED launch-count delta (nsys, counts only)

Rig 5090, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, release. **Rig law:
exactness/counts only. No timing number was taken here and none is claimed** (the kern_sum CSVs
carry nsys's time columns because the report format does; they are not quoted anywhere).

Two profiles of the gate's own arms over the IDENTICAL workload (widths {1,2,15,16,17,64} +
prime T=20 + 4 decode steps on the 2-layer NVFP4 macro fixture, `TOP_K = 3`):

- `nsys-sequential-kernsum.csv`: `sequential_control_matches_the_reference`
  (`MEMRA_MOE_GROUPED_PREFILL=0`)
- `nsys-grouped-kernsum.csv`: `grouped_prefill_matches_the_reference` (`=1`, exact filter, so
  the chunk-cap test's mixed-arm process is NOT in this profile)

## The measured table

Total kernel instances 3712 -> 1823, delta **-1889**, and it decomposes EXACTLY into the arm's
structure. The grouped rows in this workload are T=17, T=64 and prime T=20: 3 (layer, chunk)
dispatches on the one MoE layer, 101 tokens total, n_used = 3.

| kernel | sequential | grouped | delta | = |
|---|---|---|---|---|
| `qmatvec_expert_q8` | 1251 | 342 | -909 | -9 x 101 tokens (3 experts x 3 projections) |
| `quantize_q8_1` | 556 | 152 | -404 | -4 x 101 (1 z-quantize + 3 act-quantizes per token) |
| `axpy_f32` | 417 | 114 | -303 | -3 x 101 |
| `swiglu_preclamped_mul_scaled_f32` | 439 | 139 | -300 | -3 x 101 + 3 (ONE preclamp per chunk) |
| `moe_kq_sktail_kernel<7>` | 0 | 9 | +9 | **3 chunks x 3 projections: ONE grouped GEMM kernel per projection, measured** |
| `gather_act_f16_kernel` | 0 | 6 | +6 | 3 chunks x 2 (z gather-to-f16 + act gather-to-f16) |
| `scale_rows_f32` | 0 | 6 | +6 | 3 chunks x 2 (gate/up macro fold) |
| `rows_permute_f32` | 0 | 3 | +3 | 3 chunks x 1 (CSR -> pair order) |
| `moe_pairs_scatter` | 0 | 3 | +3 | 3 chunks x 1 (weighted per-token accumulation) |
| **TOTAL** | **3712** | **1823** | **-1889** | -19 x 101 + 9 x 3 - 3 = -1889, the entire delta |

No other kernel's count moved: router (`router_gemv` + sigmoid top-k), shared expert, trunk and
the t<=16 / decode rows are IDENTICAL in both profiles, which is the same-program property the
gate's bitwise knee assertion proves from the output side.

So per (layer, chunk) the arm replaces the sequential loop's `(1 + 6*n_used) * t` MoE launches
with **9 chunk-wide arm launches + 1 preclamp = 10** (router and shexp unchanged in both arms).

## Extrapolation to the real model (n_used = 8, 42 MoE layers, t = 4096), ARITHMETIC

The same structure at the real geometry, anchored on the epilogue lane's banked count
(`../moe-epilogue-receipts/nsys-launch-counts.md`: 49 launches per token-layer at n_used = 8):

| | per (layer, 4096-chunk) | x42 MoE layers per chunk |
|---|---|---|
| sequential (shipped default) | 49 x 4096 = **200,704** | **8,429,568** |
| grouped arm | 3 GEMM + 2 gather + 2 scale_rows + 1 preclamp + 1 permute + 1 scatter = **10** | **420** |

A ~20,000x reduction of the MoE dispatch term per chunk (the whole-chunk total goes ~8.43M ->
~2k once the unchanged trunk/router/shexp launches are counted back in). The weight-traffic side
of the same change: each routed expert's NVFP4 rows stream through the grouped GEMM ONCE per
layer per chunk instead of once per (token, expert), the PREFILL-GAP.md §1.1 arithmetic
(4.76 GB of expert VRAM re-reads per token -> 0.023 ms/token equivalent at slab bandwidth).

**No throughput claim.** Whether fewer launches converts to prefill tok/s is the interleaved x5
A/B on the serving card class with the sampled vendor-default twin (`AB-PLAN.md` in this
directory), and it has not been run.
