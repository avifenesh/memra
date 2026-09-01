# Kernel inventory

Derived from code (build.rs, cu/, FFI shims) 2026-08-21 **at commit 53710511dc** — line
references resolve against that commit (`git show 53710511dc:<path>`), not necessarily HEAD.
Every row comes from a grep or a read; UNKNOWN means not determinable from the code without
deeper tracing — never guessed. Known drift at merge time (v0.100.0 train, 1e292853de):
kernels.cu 71→73 entry symbols, qmatvec.cu 251→252; other per-file counts unchanged.
Companion docs: FLAGS.md (dispatch seams), MODELS.md (model-level support), ARCHITECTURE.md
§3 (component map). Regenerate discipline: when a `.cu` file or FFI shim changes, update the
affected table rows in the same change — and refresh the pin commit in this header.

## Build/binding model (build.rs)

Two artifact kinds (build.rs:251-257, 337-458, 583-642):

- **Fatbin modules** — loaded at runtime via `ctx.load_module`, kernels bound **by name
  string** (src/lib.rs:1130-1148): `kernels.cu`, `hybrid.cu`, `qmatvec.cu`,
  `flash_attn.cu`, `qmatvec_gemm.cu`, `moe_router.cu`, `spec_sample.cu`. flash_attn gets
  KV-format fatbin variants selected by `MEMRA_KV_K`/`MEMRA_KV_V` (build.rs:291-332,
  src/lib.rs:301-302).
- **Static lib `libmemra_mmq.a`** — host launchers bound via Rust `extern "C"`:
  `mmq_fp4.cu`, `mmq_q45k.cu`, `mmq_nvfp4_w4a8.cu`, `mmq_iq_experts.cu`, `mmq_q8_0.cu`,
  `mmq_q4_0.cu`, `fp8_prefill.cu`, `f16_prefill.cu`, `mmq_nvfp4_f8f4.cu`,
  `fa3_prefill.cu`, `moe_f16_grouped.cu`, `fp8_blk_dequant.cu`, `mmq_fp8_blk.cu`,
  `mmq_q8_0_f32acc.cu` (build.rs:423-442).
- **Separate static lib**: `cutlass_fp4_sm120.cu`, sm_120a-only, opt-in `MEMRA_CUTLASS`
  build env (build.rs:236-237, 583-642).
- **Stub swaps** (fail-closed twins, build.rs:445-457): non-120a → `mmq_fp4_stub.cu`;
  portable (89/90a) → `mmq_nvfp4_w4a8_stub.cu`, `mmq_fp8_blk_stub.cu`; arch != 90a →
  fa3_prefill compiled with `-DMEMRA_FA3_STUB` (build.rs:525-526; stubs return rc 3 /
  nonzero).
- Fatbins are single-arch SASS; arch chosen by `MEMRA_CUDA_ARCH` or nvidia-smi detect,
  fallback 120a (build.rs:136-162, 221-239). `MEMRA_PORTABLE_CUDA=1` defined for 89/90a
  builds (build.rs:233, 265).

Facts worth knowing before "cleaning up":

- **`fattn_vendor.cu` is NOT BUILT** — appears nowhere in build.rs or src/; header says
  "WIP SKELETON" (fattn_vendor.cu:1). Zero exported symbols.
- **`mma_tile.cuh` is dead** — no `#include` of it anywhere in cu/ or src/.
- **`wgmma_common.cuh`** is included only by `hybrid.cu:2202`; guard
  `__CUDA_ARCH__ == 900` at wgmma_common.cuh:9.
- `MEMRA_MMQ_STREAMK` arm is documented as removed (mmq_ffi.rs:714).

## Per-file inventory

Legend: FFI binding "fatbin/by-name" = loaded from module by kernel name string;
"extern in <file>" = Rust `unsafe extern "C"` declaration.

### cu/kernels.cu — 84 symbols (recounted 2026-08-31 by `grep -c 'extern "C" __global__'` — the original inventory's 71 predated several lanes; lane/glm5-matvec touched this file) (fatbin `MEMRA_ENGINE_FATBIN`)

Header: "Stage-1 kernels: correctness-first, all f32, no tensor cores" (kernels.cu:1).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `argmax_*`, `prob_of_token_*` | device argmax / token-prob reduction | f32 logits | PDL macro sm≥900 non-portable (kernels.cu:15) | UNKNOWN | fatbin/by-name |
| `topk_rows_shard_f32` + `topk_rows_shard_merge_f32` (lane/glm5-matvec 2026-08-31) | EXACT sharded twin pair of `topk_rows_f32` (per-(row, column-shard) partial top-k + per-row shard merge, the standing kernel's insertion/tie rules verbatim on global column indices — discrete selection under (value desc, index asc), output-identical by construction; the standing kernel puts n_rows blocks on the card, 15 blocks/188 SMs at 7 GB/s on the DFlash2 selector) | f32 logits | — | `MEMRA_TOPK_SHARDS` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam, engages at n_cols >= 16384) — `glm5_matvec_doors_gpu` planted-tie bit-gate + rotated-row red | fatbin/by-name |
| `rms_norm*` family (~15 incl. fused add/scale/QKV/rope, q8_1 out) | RMSNorm variants | f32 in; f32/f16/q8_1 out | same PDL guard | per-model wiring; MEMRA_QKVNORM_W in lib.rs | fatbin/by-name |
| `rope_neox*` (4 variants) | RoPE NeoX | f32 (+bf16 echo) | — | UNKNOWN | fatbin/by-name |
| activation family (gelu_tanh_mul, silu_mul incl. scaled/q8_1, swigluoai_mul_scaled, swiglu_clamped_mul_scaled (step35, clamp AFTER silu), swiglu_preclamped_mul_scaled (glm5_next, clamp BEFORE silu, one-sided gate), gelu_tanh) | gated-FFN activations | f32/q8_1 | — | MEMRA_Q8_FFN_FUSE2 (q8 fused arm) | fatbin/by-name |
| elementwise/util (~25: add/scale/mul/softcap/mask/convert/pack/gather/permute/l2_norm/layer_norm_bias/row_softmax/prefetch_l2) | glue ops | f32/int | — | UNKNOWN | fatbin/by-name |
| `sdpa_naive_f32`, `_w_`, `_island_` | naive SDPA fallback for head dims FA doesn't cover (flash_attn.cu:71) | f32 | — | fallback when FA doesn't match head_dim | fatbin/by-name |
| `router_gemv_f32[_w8][_batch]`, `sigmoid_dot_rows_f32` | MoE router GEMV | f32/bf16-w8 | — | MEMRA_ROUTER_KERNEL, MEMRA_ROUTER_BATCH (lib.rs:211, 236) | fatbin/by-name |

### cu/hybrid.cu — 67 symbols (fatbin `MEMRA_HYBRID_FATBIN`)

Header: "Qwen3.5/3.6 hybrid linear-attention: depthwise causal conv1d + SiLU, Gated
DeltaNet scan; all f32" (hybrid.cu:1-3).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `ssm_conv1d_*`, `conv_left_pad_f32`, `conv_assemble_and_roll_f32`, `ssm_conv_ring_*` | causal conv1d + ring KV state | f32 | — | UNKNOWN | fatbin/by-name |
| `gdn_*` scan family (prep_decode, scan_s128/_dc/_b, chunk_* incl. `_vl`) | Gated DeltaNet recurrent/chunked scan | f32 | mma-body region `#if !defined(MEMRA_PORTABLE_CUDA) \|\| defined(MEMRA_HOPPER_MMA)` (hybrid.cu:489, 1531, 2192) | MEMRA_GDN_MMA | fatbin/by-name |
| `gdn_chunk_state_mma[_vl]`, `gdn_chunk_output_mma[_vl]`, `gdn_p_bf16_masked` | fused wgmma K4+K5 chunk kernels | bf16 mirrors, f32 acc | sm_90a-only wgmma via wgmma_common.cuh:9 (include at hybrid.cu:2202) | MEMRA_GDN_WGMMA (hybrid.cu:2195 "env+cfg gated") | fatbin/by-name |
| glue (~17: sigmoid, gated_rmsnorm incl. f16out/q8_1, transpose, repeat_heads, axpy, scatter/gather/reduce slots, f32_to_bf16_bulk, q_gate_split, qkv_to_gdn_repack) | hybrid-path glue | f32 | — | UNKNOWN | fatbin/by-name |

### cu/qmatvec.cu — 323 symbols (fatbin `MEMRA_QMATVEC_FATBIN`)

Header: "Resident-quantized matmul: weights stay in GGUF block format in VRAM,
dequantized in-register" (qmatvec.cu:1-2). Guards: PDL sm≥900 (qmatvec.cu:17); dp4a
`#if __CUDA_ARCH__ >= 610` (qmatvec.cu:464).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `qmatvec_{q8_0,q4_K,q6_K,q5_K,q3_K,nvfp4,iq4_XS}_dp4a`, `qmatvec_f32` | decode matvec, int8 dp4a | per-name | dp4a ≥610 | MEMRA_MMVQ / MEMRA_NO_BATCHED | fatbin/by-name |
| `qmatvec_*_mmvq_b{2,4,8,16}[_r2/_rp/…]` batched families (q4_0, q8_0, q4_K, q5_K, q6_K, nvfp4 fused2/3/4) | MMVQ batched decode matvec; `_rp` = split-plane repacked weights | per-name | — | MEMRA_MMVQ, MEMRA_MMVQ_ROWS, MEMRA_MMVQ_BV, MEMRA_RP, MEMRA_Q4RP, MEMRA_B8, MEMRA_Q40_MR | fatbin/by-name |
| `q{8_0,4_0,4_K,6_K}_split_rp_build` | build split-plane repack of weights | per-name | — | MEMRA_RP / MEMRA_Q4RP | fatbin/by-name |
| `moe_pairs_*`, `moe_gate_up_*`, `moe_down8_*`, `qmatvec_expert_q8`, `moe_w_scale_by_expert`, `moe_w_exscale` (~60 variants) | MoE expert matvec/FFN decode kernels | Q8_0 experts (+IQ4 csr variant), f32 | — | UNKNOWN per-variant (decode dispatch in lib.rs; MEMRA_MOE_CACHE etc.) | fatbin/by-name |
| `moe_gate_up_preclamp8_q8` (glm5_next, 2026-08-28) | the ONLY clamped fused MoE epilogue in the family: `moe_gate_up_silu8_q8`'s dots/grid/warp-reduction verbatim with `swiglu_preclamped_mul_scaled_f32`'s PRE-clamp expression and per-slot `weight_scale_2` macro scales (`gs`/`us`). Paired with the unmodified `moe_down8_fma_q8`, whose macro folds into the routing weight. Every other `moe_gate_up_*` hardcodes plain `silu(gate)*up`, which is a different program above the limit — do not substitute. | NVFP4/IQ/k-quant experts via `expert_dot_g`, q8_1 activations | needs `limit > 1e-6` (at 0 every gate collapses to `silu(0)`); debug_assert in `Engine::moe_gate_up_preclamp8_q8` | MEMRA_MOE_FUSED_EPI (default OFF; hybrid_forward.rs `moe_fused_epi_enabled`) | fatbin/by-name |
| `moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows` (glm5_next, lane/glm5-vrest 2026-08-31) | verify-rows twins of the fused preclamp epilogue pair: ONE launch pair covers ALL t x n_used routed pairs of a spec-verify batch (pair p = tok*n_used + j, dense slot-major). Per pair the bodies are `moe_gate_up_preclamp8_q8` / `moe_down8_fma_q8` VERBATIM (same `expert_dot_g` g-strided order per (pair,row) == `qmatvec_expert_q8`'s chain, same warp tree, same PRE-clamp expression, same slot-ordered `__fmaf_rn` down chain); inputs are plane-major (gate\|up\|down) `[3*n_pairs]` u64 pointer + f32 scale tables (gs\|us\|w*macro_down) host-built from the resident slab base + ex*stride. | NVFP4/IQ/k-quant experts via `expert_dot_g`, q8_1 activations | needs `limit > 1e-6`; pairs dense slot-major (`n_pairs % n_used == 0`) | rides `MEMRA_GLM5_VERIFY_BATCH` (verify walk only; no flag of its own) — bit-gated per row vs the sequential chain by `glm5_verify_batch_gpu` gate 4 (swapped-pair + dropped-macro reds) | fatbin/by-name |
| `moe_vrows_tables_from_sel` (glm5_next, lane/glm5-moe-loc 2026-08-31) | builds the verify-rows pair's `[3*n_pairs]` plane-major pointer + scale tables ON DEVICE from the sigmoid router's own `sel_idx`/`sel_w`, so the layer no longer reads the selection back to evaluate `slab_base + ex*expert_stride` and three `macro_scale(ex)` lookups on the host. One thread per pair (`n_pairs = t*n_used <= 128` on every serving shape), 128-thread blocks. Removes a full `cuStreamSynchronize` + 2 DtoH + 2 pageable HtoD per MoE layer-call = 42 drains + 84 DtoH + 84 HtoD per ship round. Bit-identical to the host loop term by term: exact integer pointer arithmetic, the same f32 macro plane at the same index (resident mirror, uploaded once per `(layer, plane)`), and ONE IEEE-754 single multiply `selw[p] * macro_down[ex]` matching the host's `w * macro_scale(ex)` operand order (no FMA contraction is possible in a bare product). Door `MEMRA_MOE_VROWS_DEV_TABLES` (default OFF). Gate: `glm5_moe_loc_doors_gpu` (table-level bitwise vs the host build across t=2..=8 with and without macro planes; pair-output bitwise; wrong-down-stride and dropped-macro reds bite). |
| `moe_gate_up_preclamp8_q8_rows_w4` + `moe_down8_fma_q8_rows_w4` (glm5_next, lane/glm5-matvec 2026-08-31) | WARP-PACKED twins of the verify-rows pair: `MEMRA_MMVQ_ROWS`=4 warps/block on threadIdx.y (`o = blockIdx.x*4 + threadIdx.y`), per-(row,pair) warp body VERBATIM — the one-warp-block form caps residency at the blocks/SM limit (<=67% of warp slots) and schedules ~65k one-warp blocks/launch; packing moves no bits (no `__syncthreads` in either body, ragged tail returns early) | same as the unpacked pair | same | `MEMRA_MOE_VROWS_PACK` (default OFF; box prices the flip) — bit-gated packed-vs-unpacked + re-bitten gate-4 reds in `glm5_matvec_doors_gpu` | fatbin/by-name |
| `matvec_bf16_f32acc_x4_tcols16` (lane/glm5-matvec 2026-08-31) | wide-t twin of `matvec_bf16_f32acc_x4_tcols` for t=9..=16 (the DFlash2 drafter block head: nd=15 rows over the target's 1.269 GB lm head, which the t<=8 tcols launcher refuses — the ship census caught the `_rows` fallback re-reading the head 15x/round, 11.7% of capture GPU). SEPARATE kernel so the priced t<=8 class keeps its acc[8] footprint/SASS (the `_tw32` acc-sizing lesson); body otherwise verbatim, bit-identical per (row,token) to the t=1 program | bf16 weights, f32 acc | — | `MEMRA_BF16_TCOLS_WIDE` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam) — `glm5_matvec_doors_gpu` bit-gate + shifted-row red | fatbin/by-name |
| `matvec_bf16_f32acc_x1_tcols` (lane/glm5-matvec 2026-08-31) | one-row-per-block grid twin of the tcols kernel (grid.x = out_f, p-loop dropped): the trunk kda grids (512..2048 blocks) are ~one resident wave and phase-lock their bit-pinned tree reduces (census 1.05 TB/s = 59% of peak vs the SAME kernel at 1.43 TB/s on the head's 38720-block grid); per-row body + red[256] tree verbatim, bit-identical | bf16 weights, f32 acc | — | `MEMRA_BF16_TCOLS_X1` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam) — `glm5_matvec_doors_gpu` x1-vs-x4 bit-gate + swapped-weight-row red | fatbin/by-name |

### cu/flash_attn.cu — 91 symbols (fatbin `MEMRA_FLASH_FATBIN` + KV-variant fatbins)

Header: "hand-written FlashAttention for RTX 5090 (sm_120a), m16n8k16 bf16 mma"
(flash_attn.cu:1-6). PDL sm≥900 non-portable (flash_attn.cu:59). Head dims: `template<int
HD>` stamped at 256 (base) and 128 (`_hd128`) (flash_attn.cu:66-72); `_hd512`/`_512_tb`
variants; other dims fall back to `sdpa_naive` (flash_attn.cu:72).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `append_quantize_kv_q8_0_q5_1*` (rows/dc/seqs/inc) | KV append+quantize | K=q8_0, V=q5_1 defaults; fp8/q4_0 via KV fatbin variants | — | MEMRA_KV_K / MEMRA_KV_V select fatbin | fatbin/by-name |
| `fa_prefill_f32*`, `fa_prefill_w_f32*`, `_pp`, `_w2`, `_hd128` | f32 FA prefill | f32 | — | MEMRA_FA_FLOOR etc. | fatbin/by-name |
| `fa_prefill_*bf16*` (p1, p1h2, pp, g4, g4o2, bf16kv_pp, bf16kv_vl, hd512, hd512_sp*) | bf16 FA prefill incl. hd512 | bf16 | — | MEMRA_FA_SPW, MEMRA_FA_SP512, MEMRA_FA512_MIN, MEMRA_FA_F16PV | fatbin/by-name |
| `fa_prefill_q*`, `fa_prefill_qw*` (_hd128, _db*) | FA prefill over quantized KV | q8_0/q5_1 KV | — | MEMRA_PRIME_DEQW_DB | fatbin/by-name |
| `fa_decode_f32`, `fa_decode_vec_q*` (~30 variants) + `fa_decode_combine*` | split-K FA decode + combine | q8_0/q5_1 KV, f32/q8_1 out | — | MEMRA_FA_V2/V3/V4, MEMRA_FA_V4_MAX, MEMRA_NO_FA_VEC, MEMRA_FA_SMEM_TKV, MEMRA_FA_SPLIT | fatbin/by-name |
| VL family (`fa_mirror_vl`, `q_gate_split_vl`, `attn_rms_vl`, `attn_rope_vl`, `append_kv_vl`, `fa_prefill_bf16kv_vl*`) | varlen batched-attention pre/post | f32/bf16 | — | UNKNOWN | fatbin/by-name |
| conversions (`f32_to_f16_flat`, `bf16_to_f16_flat`, `f32_to_bf16_flat`, `fa_dequant_kv_ws_*`) | KV workspace dequant / dtype flat converts | — | — | UNKNOWN | fatbin/by-name |

### cu/qmatvec_gemm.cu — 10 symbols (fatbin `MEMRA_GEMM_FATBIN`)

Header: "batched tensor-core int8 quant GEMM for the memra PREFILL path (sm_120a)"
(qmatvec_gemm.cu:1).

| symbol | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `qmatvec_gemm_q8_0` | int8-MMA prefill GEMM | Q8_0 | — | default prefill; MEMRA_NO_GEMM = dp4a fallback (lib.rs:14622); MEMRA_GEMM_K1_LAUNCH tile pin (lib.rs:331-339) | fatbin/by-name |
| `qmatvec_gemm_{q4_K,q4_0,q4_0_rp,q5_K,q6_K}` | int8-MMA prefill GEMM | per-name | — | same seam; MEMRA_Q5K_ISSUE | fatbin/by-name |
| `qmatvec_gemm_nvfp4[_rp]` | NVFP4 W4A8-style GEMM | NVFP4 | — | same seam | fatbin/by-name |
| `qmatvec_gemm_nvfp4_fp4` | native FP4 block-scale GEMM (mxf4nvf4) | NVFP4 W4A4 | `#if !defined(MEMRA_PORTABLE_CUDA) && !defined(MEMRA_DISABLE_NATIVE_FP4)` (qmatvec_gemm.cu:1234); omitted from 100a fatbin via `-DMEMRA_DISABLE_NATIVE_FP4` (build.rs:273) | MEMRA_FP4 | fatbin/by-name |
| `qmatvec_gemm_q8_0_wgmma` | Hopper wgmma Q8_0 prefill GEMM | Q8_0 | `#if defined(MEMRA_HOPPER_MMA) && __CUDA_ARCH__ >= 900` (qmatvec_gemm.cu:1565) | MEMRA_WGMMA=1 opt-in (mmq_ffi.rs:627) | fatbin/by-name |

### cu/moe_router.cu — 3 symbols (fatbin `MEMRA_ROUTER_FATBIN`)

Header: "fused MoE router… bit-identical to host path" (moe_router.cu:1-4).

| symbol | purpose | dispatch flag |
|---|---|---|
| `moe_router_topk_f32` | softmax + stable top-k + renorm | MEMRA_ROUTER_KERNEL=0 rollback (lib.rs:58, 211) |
| `moe_router_sigmoid_topk_f32` | sigmoid-scored top-k | MEMRA_ROUTER_V2 (lib.rs:465, 2100) |
| `moe_router_topk_scaled_f32` | scaled top-k | MEMRA_ROUTER_KERNEL / _PREFILL_EXACT (lib.rs:202-205) |

### cu/spec_sample.cu — 30 symbols (fatbin `MEMRA_SAMPLE_FATBIN`)

Header: "Sampled speculative decoding — device sampling primitives; Philox4x32-10
counter-based" (spec_sample.cu:1-5).

| symbol (family) | purpose | dispatch flag |
|---|---|---|
| `gumbel_perturb_*`, `softmax_gather_*`, `residual_sample_*`, `filter_stats_f32`, `scatter_trim_logits_*`, `penalize_logits_*` (including heterogeneous sparse serving rows), `mask_logits_f32`, `memra_sctr_inc` | Gumbel-max sampling, top-k/p filtering, penalties, residual (rejection) sampler | sampling chain via MEMRA_TEMP/TOP_K/TOP_P/MIN_P/PENALTY_* plus batched serving `MEMRA_SERVE_DEVPENALTY` |
| `spec_accept_greedy[_dc]`, `spec_seed_gather`, `spec_rollback_kv/stream`, `spec_fork_*`, `spec_assemble_verify`, `spec_ring_commit`, `spec_adapt_k`, `plain_tok_ring`, misc int copies | spec-decode accept/rollback/fork machinery | MEMRA_SPEC_* family (MEMRA_SPEC_DUAL_T lib.rs; MEMRA_SPEC_DFLASH FLAGS.md:25) |

### MMQ static-lib TUs (prefill GEMM per weight format)

| file | host entry symbols | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| mmq_q8_0.cu | `memra_mmq_q8_0`, `_act_bytes` | Q8_0 (W8A8 int8) | portable sm_75+ (header) | MEMRA_PP_Q8MMQ, default ON since 2026-07-09 (mmq_ffi.rs:479-490) | mmq_ffi.rs:183-187 |
| mmq_q4_0.cu | `memra_mmq_q4_0`, `_act_bytes`, `_quant_act`, `_gemm`, `_gemm_sk`, `_fixup_bytes`, `_set_clc` | Q4_0 (W4A8) | CLC arm `__CUDA_ARCH_LIST__ >= 1000` (mmq_q4_0.cu:45) | MEMRA_PP_Q4MMQ (mmq_ffi.rs:516-520), MEMRA_MMQ_SK / _SK_FORM (mmq_ffi.rs:929-949), MEMRA_MMQ_CLC (mmq_ffi.rs:267) | mmq_ffi.rs:229-284 |
| mmq_q45k.cu | `memra_mmq_q4_K`, `memra_mmq_q5_K`, `_act_bytes` (DS4 layout: Q4_K/Q5_K carry min-offset) | Q4_K, Q5_K (W4A8) | sm_89 L40S branch noted (mmq_q45k.cu:23); hard guard UNKNOWN | MEMRA_MMQ_W4A8 seam, default-on (mmq_ffi.rs:450-457) | mmq_ffi.rs:157-181 |
| mmq_fp4.cu | `memra_mmq_nvfp4`, `_ex`, `_ex2`, `_act_bytes` (+residual-correct) | NVFP4 (W4A4 mxf4nvf4) | sm_120a-only; stub for `cuda_arch != "120a"` (build.rs:445-450) | MEMRA_MMQ=1 explicit opt-in (mmq_ffi.rs:452, 534); MEMRA_MMQ_RESIDUAL_K (mmq_ffi.rs:464-472) | mmq_ffi.rs:30-82 |
| mmq_nvfp4_w4a8.cu | `memra_mmq_nvfp4_w4a8`, `_act_bytes`; second TU section (line 1476+): `memra_mmq_nvfp4_f8f4`, `_quantize_act` | NVFP4 weights + q8_1 acts; f8f4 route = e4m3 acts | stubbed on 89/90a (build.rs:451-454); f8f6f4 arm needs sm_100a+ | MEMRA_MMQ_W4A8 (default-on, =0 escape); MEMRA_MMQ_F8F4; MEMRA_MMQ_F8F4_PLAIN (build.rs:411) | mmq_ffi.rs:84-123 |
| mmq_nvfp4_f8f4.cu | `memra_mmq_nvfp4_f8f4_act_bytes` (:96), `_quantize_act` (:105) | e4m3 activation quantizer (compiles on sm_89 too) | — | MEMRA_MMQ_F8F4 | mmq_ffi.rs:105-111 |
| mmq_fp8_blk.cu | `memra_mmq_fp8_blk`, `_act_bytes`, `_scale_rows`, `_scale_cols`, `memra_fp8_blk_count_nan` | FP8 E4M3 block-128 | .kind::f8f6f4 sm_100a+ (stub header); stub on 89/90a (build.rs:455-457) | MEMRA_ST_E4M3_BLK (fp8_ffi.rs:87); native path fp8_blk_mmq_native_enabled (fp8_ffi.rs:273) | mmq_ffi.rs:125-155 |
| mmq_iq_experts.cu | `memra_mmq_iq_experts`, `memra_mmq_iq4xs_dense` (needs `in_f % 256 == 0`, :870), `_quantize_act`, `_fused_act_quant`, `_act_bytes` | IQ3_S, IQ4_XS (W4A8), 35B MoE prefill | smem guard vs sm_120a ~99KB (:849, :870) | MEMRA_MOE_MMA (mmq_ffi.rs:286), MEMRA_PP_IQMMQ (mmq_ffi.rs:502-506), MEMRA_IQ_FAST=0 kill (mmq_ffi.rs:574) | mmq_ffi.rs:288-345 |
| mmq_q8_0_f32acc.cu | `memra_accprobe_act_bytes`, `_gemm_s32`, `_gemm_f32` — "THE Q1 INSTRUMENT for the FP8-ST v3 gate (research-only)… never linked into a serving path" (:1; build.rs:439) | Q8_0, f32-acc probe | `#if __CUDA_ARCH__ >= 1000` (:193) | NONE (research) | mmq_ffi.rs:206-227 |

### Other static-lib TUs

| file | host entry symbols | purpose | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| f16_prefill.cu | `memra_f16_pp_gemm[_pre]`, `memra_f16_cvt`, `memra_{q8_0,q4_0,q6_K,q4_K,q5_K}_dequant_f16` | cuBLASLt FP16 TN prefill on resident f16 dequant mirror of quantized weights | host cuBLASLt | MEMRA_PP_F16, MEMRA_PP_F16_BUDGET_MB, MEMRA_W8A8_SIM | f16_ffi.rs:22-62 (+build_*_raw wrappers f16_ffi.rs:725-816) |
| fp8_prefill.cu | `memra_fp8_pp_gemm` (:90) + `__global__` amax/scale/quant kernels | cuBLASLt FP8-E4M3 TN prefill + per-batch activation quantize | — | MEMRA_PP_FP8, MEMRA_PP_FP8_BUDGET_MB, MEMRA_FP8_MMQ, MEMRA_ST_E4M3 (fp8_ffi.rs:27, 45-87, 239) | fp8_ffi.rs:27 |
| fp8_blk_dequant.cu | `memra_fp8_blk_q8_0_bytes` (:220), `memra_fp8_blk_dequant_q8_0` (:228) | device-side dequant of block-128 FP8 weights into GGUF Q8_0 blocks at model load | — | MEMRA_FP8_BLK_GPU (fp8_ffi.rs:476) | fp8_ffi.rs:458-474 |
| fa3_prefill.cu | `memra_fa3_prefill`, `memra_fa3_vl` (+stub twins rc=3, :19-23) | FA3 v10 engine shim, head_dim 256 only | wgmma/TMA sm_90a-only; `-DMEMRA_FA3_STUB` otherwise (build.rs:525-526) | promoted default-ON on hopper 2026-07-27, `MEMRA_FA3=0` reverts; engages only head_dim==256 causal t==t_kv (lib.rs:15399-15409; file header ":1-7 opt-in" is stale) | lib.rs:889-904 |
| moe_f16_grouped.cu | `memra_moe_f16g_{dequant,gemm,gemm_sk,gather_act,h2f,h2f_scaled,w_bytes,act_bytes}`, `memra_moe_kq_gemm_sk` | per-layer expert dequant to f16 + ONE grouped f16 GEMM per projection over CSR groups; per-qtype dequant kernels for Q4_0/IQ4_XS/IQ3_S/Q6_K/Q4_K/Q3_K (:109-246); "SASS portable across 89/90a/100a/120a" (:336) | smem opt-in >48KB, 1 CTA/SM on sm_120a (:483-486) | MEMRA_MOE_F16G (=2 single-kernel, mmq_ffi.rs:394), MEMRA_F16G_SK/_TAIL/_DIRECT/_DEBUG | mmq_ffi.rs:348-424 |
| cutlass_fp4_sm120.cu | `memra_cutlass_fp4_{workspace,sfa_size,sfb_size,fp4_gemm,repack_sfa,repack_sfb}`, `memra_nvfp4_{quant,dequant}_ref`, `memra_gguf_nvfp4_deinterleave` | CUTLASS 4.x NVFP4 W4A4 GEMM wrapper | hard sm_120a gencode (build.rs:604); build asserts 120a-only (build.rs:236-237) | MEMRA_CUTLASS (build env) | cutlass_ffi.rs:47-113 |

### Stub files (fail-closed twins)

- `mmq_fp4_stub.cu` — `memra_mmq_nvfp4{,_ex,_ex2,_act_bytes}` for non-120a.
- `mmq_nvfp4_w4a8_stub.cu` — `memra_mmq_nvfp4_w4a8{,_act_bytes}`, `memra_mmq_nvfp4_f8f4` for sm_89/90a.
- `mmq_fp8_blk_stub.cu` — `memra_mmq_fp8_blk{,_act_bytes,_scale_rows,_scale_cols}`, `memra_fp8_blk_count_nan` for sm_89.

## Shared MMQ headers vs remaining private copies

Deduplicated 2026-08-21 (lane/kernel-dedup-20260821, three SASS-identical increments —
receipts: `research/kernel-dedup-20260821/RECEIPTS.md`; every modified TU × arch gated
`cuobjdump -sass` byte-identical before/after):

- **`cu/mmq_common.cuh`** (adopters: mmq_q8_0.cu, mmq_q4_0.cu, mmq_q45k.cu,
  mmq_nvfp4_w4a8.cu): WARP_SIZE, GGML_PAD, QK8_1, QI8_1, MATRIX_ROW_PADDING,
  MMQ_TILE_NE_K, MMQ_TILE_Y_K, MMQ_WARP_SIZE, the `#ifndef MMQ_X` guard (the
  `-DMMQ_X` tune seams still work), CUDA_QUANTIZE_BLOCK_SIZE_MMQ,
  `mmq_get_granularity_device`, `get_int_b2`, and the D4 `block_q8_1_mmq` +
  static_assert behind `#ifndef MMQ_BLOCK_Q8_1_MMQ_LOCAL` (q45k defines that and keeps
  its DS4-commented local struct).
- **`cu/mmq_mma_i8.cuh`** (adopters: mmq_q8_0.cu, mmq_q4_0.cu, mmq_q45k.cu): the int8
  `ggml_cuda_mma` tile machinery (struct tile, load_generic, load_ldmatrix, m16n8k32 s8
  mma wrapper). w4a8's variant is a different ISA form and stays local.
- Still private by design (different values or different programs): `QI8_0`,
  `MMQ_ITER_K` (256/128/512 per TU), `MMQ_MMA_TILE_X_K_*` per qtype, `MMQ_NWARPS`/`MMQ_Y`
  (w4a8 derives Y and guards it), everything in mmq_fp4.cu, mmq_fp8_blk.cu
  (FP8_MMQ_* naming), mmq_iq_experts.cu (compressed idiom), and mmq_q8_0_f32acc.cu
  (research instrument, deliberately isolated).
- **wgmma helpers — deduplicated 2026-08-21** (lane/wgmma-dedup-20260821, 12/12
  SASS-identical across {120a,100a,90a,89}; receipts in
  `research/kernel-dedup-20260821/RECEIPTS.md`): fa3_prefill.cu and qmatvec_gemm.cu now
  include `wgmma_common.cuh` for the smem descriptor builder, fence/commit, and the
  m64n64k16.bf16 wrapper. Still local by design: fa3's `_tb` (transpose-B imm) and
  templated wait, qmatvec's m64n64k32.s8 form and raw asm statements.

## Known UNKNOWNs

Per-variant dispatch flags inside the four giant fatbin TUs (kernels/qmatvec/flash_attn/
hybrid) are family-level here; per-variant selection lives across ~411 `MEMRA_` read
sites in src/lib.rs and was not traced symbol-by-symbol. Rows say UNKNOWN where the
specific gate was not found. FLAGS.md is the authoritative flag catalog.
