# Kernel inventory

## Qwen FA2 attention experiment, 2026-09-09

Both entries carry the same numerical body: BF16 MMA, FP32 direct PV accumulation,
a BF16-rounded MMA denominator and log2-domain online softmax over 32-key tiles.
The door is qualified only for the Qwen 24 Q / 4 KV / d256 causal prefill at
t=16..1039 on the 170-SM sm_120a target with `MEMRA_PRIME_CHUNK=1024`.

| Symbol | Purpose | Types | Architecture | Door | Binding |
| --- | --- | --- | --- | --- | --- |
| `fa_prefill_qw_fa2` | Six query heads share three rotating BF16 KV staging planes; FP32 direct PV and online softmax | BF16 KV, f32 Q/O | sm_120a, 170 SM | `MEMRA_PRIME_ATTN_FA2`, default OFF, decide-by 2026-09-23 | `Engine::fa_prefill_view_ws` |
| `fa_prefill_qw_fa2_prime_table` | Same numerical body with true causal depth from replay table slot 7 | BF16 KV, f32 Q/O | sm_120a, 170 SM | Same door; the carried graph reuse key includes the attention class | `Engine::fa_prefill_view_ws`, `qwen_prime_graph::run` |

## DSV4 dense wide-prefill tiling, 2026-09-10 (memra #463, #468, #470)

No new kernel and no changed kernel body. Above `DSV4_TMAX` the dense entry
points decompose a transaction into tiles of `DSV4_TMAX` rows and relaunch,
which selects the widest `M` instantiation the dispatch switches already carry.
The width was a hard-coded 8 until #470; it is now the constant itself, and
there is no door.

| Symbol | Purpose | Types | Architecture | Door | Binding |
| --- | --- | --- | --- | --- | --- |
| `dsv4_gemv_fp8_m_kernel<M, false>` | Per-row FP8 dense GEMV; `M` independent register accumulators over one shared weight row | FP8 e4m3 weights with f32 block scales, BF16 activations, f32 out | sm_120a | None; `M = DSV4_TMAX` above the constant | `memra_dsv4_gemv_fp8_m` |
| `dsv4_gemv_bf16_m_kernel<M>` | Same shape for BF16 dense weights | BF16 weights and activations, f32 out | sm_120a | None | `memra_dsv4_gemv_bf16_m` |
| `dsv4_dots_f32acc_mrow_kernel<M>` | f32-accumulated dense dots, `M` rows per launch | BF16 or f32 weights, f32 activations and out | sm_120a | None | `memra_dsv4_dots_f32acc_mrow` |
| `dsv4_dots_f32_mrow_kernel<M>` | f64-accumulated dense dots, `M` rows per launch | BF16 or f32 weights, f32 activations and out | sm_120a | None | `memra_dsv4_dots_f32_mrow` |

Numeric class SAME across every `M`: the per-row accumulation order and the
128-leaf reduction tree are properties of the kernel body, not of `M`. Gated by
`dsv4_dense_tile_gate <model-dir> <real-source.txt>`, which asserts every width
above `DSV4_TMAX` byte-identical to the untiled width-32 walk and carries two red
arms. The measurement that made this the default is in the removed-doors ledger
in `docs/FLAGS.md` and in darklanes `research/dsv4f-dense-tile-20260910/`.

## Whisper CPU reference operators, 2026-09-09

These are native reference operations, not CUDA support or serving qualification.
The measured numeric programs and bounds are in
[the encoder receipt](../research/asr-modality-20260909/ENCODER-NUMERICS.md).

| Operator | Program | Numeric class | Source |
| --- | --- | --- | --- |
| `WhisperFrontend::compute` | Periodic Hann, direct real DFT, Slaney filters and Whisper log normalization | F32 PCM/output; reference DFT accumulation | `crates/memra-reference/src/speech/frontend.rs` |
| `WhisperEncoder::encode` | Biased strided convolution, positions, LayerNorm, full attention and residual FFNs | F32 or binary16 values with FP32 accumulation; no external executor | `crates/memra-reference/src/speech/encoder.rs` |
| `gelu_erf` | Owned evaluation of A&S 7.1.26 matching the pinned HF CPU vector program | FP32 fused polynomial, final destination rounding | `crates/memra-reference/src/speech/encoder.rs` |
| Speech reference matrix product | Four fixed FMA partial sums; AVX2 vectorizes independent time rows; rows are packed into cache-resident groups and output columns split across `MEMRA_SPEECH_THREADS` workers | Same scalar/AVX2 FP32 reduction order at every group size and thread count, bit-identical by construction | `crates/memra-reference/src/speech/matrix.rs` |
| `WhisperDecoder::step` and `speech::decode` | Cached self and cross attention, tied logits, then the pinned beam-1 suppression, timestamp grammar and forced-timestamp rules | Same F32 or binary16 class as the encoder | `crates/memra-reference/src/speech/decoder.rs`, `decode.rs` |

## Carried Qwen prime replay, 2026-09-09

All entries are Memra-owned twins. The qualified Qwen geometry on the 170-SM
sm_120a target uses carried-prime replay as its default, without a runtime door.
Session addresses and absolute depth come from the refreshed replay table.
Numerical bodies preserve the corresponding eager entry's operation order.

| Symbol | Purpose | Binding |
| --- | --- | --- |
| `append_quantize_kv_q8_0_q5_1_rows_prime_table` | Quantized KV append and live length publication | `Engine::append_kv_quantized_rows` |
| `fa_dequant_kv_ws_bf16_prime_table` | True-depth dequantization into stable BF16 workspace | `Engine::fa_prefill_view_ws` |
| `fa_prefill_qw_db_prime_table` | Existing four-plane attention with live causal depth | `Engine::fa_prefill_view_ws` |
| `fa_prefill_qw_t3_prime_table` | Existing three-plane attention with live causal depth | `Engine::fa_prefill_view_ws` |
| `ssm_conv1d_gdn_state_f32_prime_table` | Carried convolution reads the live ring | `Engine::ssm_conv1d_gdn_state_pad` |
| `ssm_conv_ring_update_f32_prime_table` | Publishes the live convolution ring | `Engine::ssm_conv1d_gdn_state_pad` |
| `gdn_chunk_state_mma_prime_table` | Chunked GDN reads/writes live ping-pong state | `Engine::gdn_scan_chunked` |
| `prime_tap_table` | Bulk copy of exact residual bits to live strided tap destination | `Engine::prime_tap_table` |


## Qwen attention prime staging, 2026-09-09

| Symbol | Purpose | Types | Architecture | Door | Binding |
| --- | --- | --- | --- | --- | --- |
| `fa_prefill_qw_t3` | Two K staging planes plus one V plane, register P operands; unchanged head-dim 256 prime arithmetic | BF16 KV, f32 Q/O | existing warp-MMA support | `MEMRA_PRIME_KV_T3`, default OFF | `Engine::fa_prefill_view_ws` |

## GLM TP pool-split indexer, 2026-09-08

Measured f32 TP-2 receipt (combined door): prime improves 10.3% at 128k and 34.6% at 1M. Decode is NEGATIVE at 128k and FLAT at 1M; merge plus exchange consumes score/select savings. Decode split dispatch was deleted. These kernels serve grouped prime (`t>1`) only; decode (`t=1`) stays replicated, and the rows-exact verify walk never enters the door. `MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME` is **ON by default since 2026-09-10**, on the prime-only pair cell the OFF default was waiting for: 1M prime -34.25%, 128k prime -10.36%, decode flat at both (-0.21% and +0.29%), tapes byte-identical at both contexts and both CHECK arms byte-identical to the replicated selection. `=0` restores the replicated prime, and that seam retires after two served weeks per door hygiene, review 2026-09-24. See `research/glm5-tp-indexer-split-20260908/RESULTS.md`; the flip's receipt is the dev-pair lane in [darklanes #585](https://github.com/avifenesh/darklanes/pull/585), section "Cell 3".

| CUDA entry / kernel | Contract | Dispatch and FFI |
|---|---|---|
| `memra_mla_kpool_candidates_f32` / `memra_mla_kpool_candidates_kernel` | Packs existing selector output as original f32 score bits plus global pool id; invalid slots use id -1. No arithmetic on scores. | `cu/mla_attn.cu`, `mla_ffi.rs::mla_kpool_candidates`; `MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME`, default ON since 2026-09-10. |
| `memra_mla_kpool_merge_f32` / `memra_mla_kpool_merge_kernel` | Exact score-desc/id-asc key ordering over both ranks' candidates, then ascending selected pool ids, raw-token expansion, causal tail and -1 padding. Canonical signed zero and nonfinite exclusion use the existing selector helper. One CTA/query, up to 2,048 candidates/rank. | `cu/mla_attn.cu`, `mla_ffi.rs::mla_kpool_merge`; same door. |
| `memra_tp_ar_gather_i32` / `memra_tp_ar_gather_i32_kernel` | Opaque candidate words gathered in global rank order through `MemraArSignal` start/end barriers. Two distinct peer-access devices; timeout traps, no host synchronization. | `cu/tp_ar.cu`, `tp_ar.rs::ArLink::gather_i32`; same door. |

Scoring reuses `memra_mla_kpool_score_f32` and `memra_mla_kpool_score_dsa_f32`, including
the RP arm, through `mla_ffi.rs::mla_kpool_score_range`. Key-pointer offset and relative
causal position restrict the pool domain without introducing a numeric twin. Existing
`__fmaf_rn`, `__fmul_rn` and `__fadd_rn` define the arithmetic. The TC scorer range passed the same bit-identity gate at levels 1 and 2. No live-position/captured-middle twin is added.
Evidence: `research/glm5-tp-indexer-split-20260908/DESIGN.md`, and the 2026-09-10 dev-pair cell on vast 50431646 (2x B200 SXM, TP-2) that flipped the door ON.

## DSV4 small-kernel diet, 2026-09-07

Both kernels live in `cu/dsv4_gpu.cu`, compiled with `-fmad=false`, and use
`MEMRA_DSV4_SMALL_KERNEL_DIET` (default OFF). Gate status and receipt routing:
`research/dsv4f-small-kernel-diet-20260907/README.md`.

| Kernel | Replaced launches and numeric contract | Geometry |
| --- | --- | --- |
| `dsv4_small_hc_f32_fixed_order_kernel` | rowsq f32x + Sinkhorn + collapse, 3 to 1. `dsv4_hc_f32_fixed_order`: same 128-thread rowsq tree, register Sinkhorn with ascending sums, same iteration count, ascending collapse. Bitwise gate required. | One block of 128, t=1, HC4, hidden4096. |
| `dsv4_small_norm_pack_f32_fixed_order_kernel` | Q-LoRA RMSNorm f32x + bf16 conversion, 2 to 1. `dsv4_norm_pack_f32_fixed_order`: same 128-thread reduction tree and f32 intermediate, bf16 RNE. Retains normalized f32 Q as well as packed Q. Bitwise gate required. | One block of 128, t=1. |

The gate counts successful enqueues in the replaced families and requires 8 to
3 launches per layer per rank. This counter excludes all other kernels; total
launch counts require the profile. A PASS with the old targeted count fails.

Derived from code (build.rs, cu/, FFI shims) 2026-09-02 **at commit 6a131edb** — line
references resolve against that commit (`git show 6a131edb:<path>`), not necessarily HEAD.
Every row comes from a grep or a read; UNKNOWN means not determinable from the code without
deeper tracing — never guessed. Known drift at merge time (v0.100.0 train, 1e292853de):
kernels.cu 71→73 entry symbols, qmatvec.cu 251→252; other per-file counts unchanged.
qwen4exp-bringup-20260829: kernels.cu +11 (sdpa_naive_mask_f32, sdpa_blocklist_f32,
qsa_index_score_f32, qsa_index_topk_u32, rope_neox_ffm_f32, gdn_scan_naive_f32,
dwconv_causal_f32 — eager-arm correctness oracles, plus qsa_index_topk_u32 which is a
PERF kernel: the 262k lane's host indexer top-k moved device-side; qmatvec_nvfp4_modelopt_sel_f32 and the
three hc_* gate kernels — the decode perf lane; rows in the kernels.cu table); mtp-spec
lane +4 (qmatvec_bf16w_mt_f32, hc_diet_stage{0,1,3}_mt_f32 — weight-shared verify
kernels) + token grid dims on the diet stages + gufuse tok_map (sel warp packing tried, reverted — spec/mtp6).
mtp9 added NO kernel: the FR-Spec draft-head trim reuses `qmatvec_bf16w_f32` over a D2D
row-gather of the shared lm head (only `out_f` changes, so a trimmed row is bit-identical to
its full-vocab twin), and the verify scan-chain segment graphs replay existing launches.
Both measured NEGATIVE and both stay default-OFF (trim −16.6%, graphs 0.999× — perf/PROFILE-6.md);
the launch-issue class is closed for this model at every t, so do not re-propose graph work
here without a new mechanism.
frspec-dflash2 lane (2026-09-02) added NO kernel: the glm5 DFlash2 draft-head rank trim is a
host byte row-gather of the trunk lm head at load (`frspec_gather_rows` -> one `[n_ranks x d]`
slab, same dtype program as the head: BF16 rows stay `FloatBf16`, NVFP4 rows take the same A6
repack) and the round's existing `matmul` (bf16 `matvec_bf16_rows_into` at m=7) + `topk_rows_f32`
over `n_ranks` columns instead of `n_vocab`, only the column count changes; the verify walk is
untouched.
devtwin lane (device host-twins, 2026-08-31) +3: `qwen4exp_route_topk_f32`,
`qmatvec_bf16w_sel_f32`, `copy_rows_col_f32` (rows in the kernels.cu table) — the router
and indexer HOST TWINS moved device-side, default ON on receipts (spec 1.12-1.19x every
shape, byte-identical chains — perf/PROFILE-9.md). This did NOT reopen the launch-issue
class above: the win is the removal of 60 blocking dtoh per forward, and the seam wins
with decode graphs ON and OFF alike.
Companion docs: FLAGS.md (dispatch seams), MODELS.md (model-level support), ARCHITECTURE.md
§3 (component map). Regenerate discipline: when a `.cu` file or FFI shim changes, update the
affected table rows in the same change — and refresh the pin commit in this header.

## Build/binding model (build.rs)

Two artifact kinds (build.rs:251-257, 337-458, 583-642):

- **Fatbin modules** — loaded at runtime via `ctx.load_module`, kernels bound **by name
  string** (src/lib.rs:1130-1148): `kernels.cu`, `hybrid.cu`, `qmatvec.cu`,
  `flash_attn.cu`, `qmatvec_gemm.cu`, `moe_router.cu`, `spec_sample.cu`. flash_attn also
  gets the kf8vf8 e4m3 KV fatbin gemma's GKV/WKV layers load alongside the default
  (build.rs, `func_g` in src/lib.rs).
- **Static lib `libmemra_mmq.a`** — host launchers bound via Rust `extern "C"`:
  `mmq_fp4.cu`, `mmq_q45k.cu`, `mmq_nvfp4_w4a8.cu`, `mmq_iq_experts.cu`, `mmq_q8_0.cu`,
  `mmq_q4_0.cu`, `fp8_prefill.cu`, `f16_prefill.cu`, `mmq_nvfp4_f8f4.cu`,
  `fa3_prefill.cu`, `moe_f16_grouped.cu`, `fp8_blk_dequant.cu`, `mmq_fp8_blk.cu`,
  `mmq_q8_0_f32acc.cu` (build.rs:423-442).
- **Separate static lib**: `cutlass_fp4_sm120.cu`, sm_120a-only, opt-in `MEMRA_CUTLASS`
  build env (build.rs:236-237, 583-642).
- **Stub swaps** (fail-closed twins, build.rs): arches other than 120a/100a →
  `mmq_fp4_stub.cu`, `mmq_nvfp4_w4a8_stub.cu`, and `mmq_fp8_blk_stub.cu`;
  arch != 90a →
  fa3_prefill compiled with `-DMEMRA_FA3_STUB` (build.rs:525-526; stubs return rc 3 /
  nonzero).
- Fatbins are single-arch SASS; arch chosen by `MEMRA_CUDA_ARCH` or nvidia-smi detect,
  fallback 120a (build.rs:136-162, 221-239). `MEMRA_PORTABLE_CUDA=1` defined for 89/90a
  builds (build.rs:233, 265).
- **SM100 block-scale layouts** — `cu/sm100_blockscale_layout.cuh` owns the exact row-outer,
  K-core-outer, 4X-scale, and 1X-scale address helpers used by the NVFP4 and FP8 twins. The
  host-only `research/b200-kernel-twins-dry-20260901/check-layouts.sh` gate proves each value
  layout is a bounded bijection and the scale atoms have the intended coverage.

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

### cu/kernels.cu — 135 symbols (lane/f32-gemv-rows-20260905 +1: `gemv_f32_rows`; lane/hc-mixes-gemv-20260905 +1: `hc_mixes_gemv_f32`; lane/b200-q8-fuse-20260902 +1: `rms_norm_zq8_f32`; prior recount 2026-09-02, 132, includes the TP verified-prefix batched-copy lane) (fatbin `MEMRA_ENGINE_FATBIN`)

Header: "Stage-1 kernels: correctness-first, all f32, no tensor cores" (kernels.cu:1).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `argmax_*`, `prob_of_token_*` | device argmax / token-prob reduction | f32 logits | PDL macro sm≥900 non-portable (kernels.cu:15) | UNKNOWN | fatbin/by-name |
| `topk_rows_shard_f32` + `topk_rows_shard_merge_f32` (lane/glm5-matvec 2026-08-31) | EXACT sharded twin pair of `topk_rows_f32` (per-(row, column-shard) partial top-k + per-row shard merge, the standing kernel's insertion/tie rules verbatim on global column indices — discrete selection under (value desc, index asc), output-identical by construction; the standing kernel puts n_rows blocks on the card, 15 blocks/188 SMs at 7 GB/s on the DFlash2 selector) | f32 logits | — | `MEMRA_TOPK_SHARDS` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam, engages at n_cols >= 16384) — `glm5_matvec_doors_gpu` planted-tie bit-gate + rotated-row red | fatbin/by-name |
| `topk_rows_wshard_k8_f32` / `_k16_` / `_k32_` + `topk_rows_wshard_merge_f32` (lane/draft-topk-warp-20260905) | EXACT warp-merged twin pair of `topk_rows_f32`: stage 1 (grid n_rows x n_shards, 256 threads) keeps each lane's list in registers at a compile-time k (`tkw_insert<K>`: rank-count + predicated shift, the standing while-loop's landing slot), merges the 32 lane lists by shuffle argmax per output slot inside each warp (`tkw_argmax`, a butterfly all-reduce under the total order value desc / column asc; exhausted head = (-inf, SENTINEL)) and the warp lists the same way, writing the shard's top-k; stage 2 (one warp per row) merges the shard lists. Output-identical by construction (discrete selection; top-k of a union = top-k of the parts' top-k lists; NaN never enters a list). Door `MEMRA_TOPK_WARP` (default OFF). Gate: `glm5_matvec_doors_gpu::gpu_topk_warp_matches_standing_kernel_exactly`. |
| `rms_norm*` family (~16 incl. fused add/scale/QKV/rope, q8_1 out) | RMSNorm variants | f32 in; f32/f16/q8_1 out | same PDL guard | per-model wiring; MEMRA_QKVNORM_W in lib.rs | fatbin/by-name |
| `rms_norm_zq8_f32_v2` (lane/glm5-norm-zq8-ilp-20260904) | ILP twin of `rms_norm_zq8_f32`: four loads in flight per round in BOTH passes (pass 1 same per-thread element order into one accumulator, reduce verbatim; pass 2 hoists four 32-blocks' `(x, w)` loads ahead of the v1 epilogue, which is independent per block), bit-identical by construction; gate `tests/norm_zq8_ilp_gpu.rs`. Why: the 2026-09-04 door-ON census on the 2x B200 pair read v1 at 46/token x 13.8 us on one 256-thread block | f32 in; f32 + q8_1 out | same PDL guard | `MEMRA_NORM_ILP` (default ON) at the `rms_norm_zq8_f32` launcher | fatbin/by-name |
| `hc_mixes_gemv_f32` (lane/hc-mixes-gemv-20260905) | native t=1 GEMV `y[r] = sum_k w[r*in_f+k] x[k]` for the hyper-connection MIXES projection (24 x 16384 on GLM-5.3-Flash, every hc pre site): one block per output row, 1024 threads x 16 contiguous elements as float4, pinned `__fmaf_rn` order, fixed reduction tree (shuffle-down 16..1, warp partials in smem, warp 0 tree); DETERMINISTIC, numeric class against the cuBLASLt `dot_kernel`+`reduce_1Block_kernel` pair it replaces (3.3+1.8 us per site, ~9 us host latency each in the eager MLA layers, trace 2026-09-05); gate `tests/hc_mixes_gemv_gpu.rs` (tolerance + determinism + red arm + shape refusal); rig row cuBLAS 4.92 vs native 3.06 us | f32 in; f32 out | none | `MEMRA_HC_MIXES_KERNEL=1` (default OFF) at `hyper::hc_mixes_into` | fatbin/by-name |
| `gemv_f32_rows` (lane/f32-gemv-rows-20260905) | native f32 row GEMV `y[j][r] = dot(w[r,:], x[j,:])` for the decode/verify tier (grid (out_f, m), block 256, float4 chunks at stride 1024 into one `__fmaf_rn` accumulator, fixed shuffle + smem tree): replaces cuBLASLt's `dot_kernel` + `reduce_1Block_kernel` pair on f32-resident `linear` (the DSA indexer's `wk` / `kpool_gate` / `weights_proj`, 33 pairs per token); deterministic and per-row identical for every m; numeric class vs cuBLAS | f32 | none | `MEMRA_F32_GEMV_KERNEL` (`Engine::gemv_f32_rows_into`, dispatched at the top of `linear_device_into`, m <= 16, `in_f % 1024 == 0`) | fatbin/by-name; gate `tests/f32_gemv_rows_gpu.rs` |
| `rms_norm_zq8_f32` (lane/b200-q8-fuse-20260902) | z = rms_norm(x,w) emitted BOTH as f32 (kept for callers that also read the un-quantized row — MoE router-logits GEMV, an ungated shexp arm) AND its q8_1 quantization, one launch. Pass 1 = rms_norm_f32's reduction verbatim; pass 2 epilogue = `add_rms_norm_zq8`'s warp-per-block q8_1 form minus its `a+b` add — BIT-IDENTICAL to `rms_norm_f32` then `quantize_q8_1` (qmatvec.cu:585) | f32 in; f32 + q8_1 out | same PDL guard | `MEMRA_GLM5_Q8_FUSE` (default OFF) — wired at the glm5_next mHC T=1 decode FFN-input norm (`hyper_range_decode`/`hyper_range_decode_ws_body` in hybrid_forward.rs) | fatbin/by-name |
| `rope_neox*` (4 variants) | RoPE NeoX | f32 (+bf16 echo) | — | UNKNOWN | fatbin/by-name |
| activation family (gelu_tanh_mul, silu_mul incl. scaled/q8_1, swigluoai_mul_scaled, swiglu_clamped_mul_scaled (step35, clamp AFTER silu), swiglu_preclamped_mul_scaled (glm5_next, clamp BEFORE silu, one-sided gate), gelu_tanh) | gated-FFN activations | f32/q8_1 | — | MEMRA_Q8_FFN_FUSE2 (q8 fused arm) | fatbin/by-name |
| elementwise/util (~25: add/scale/mul/softcap/mask/convert/pack/gather/permute/l2_norm/layer_norm_bias/row_softmax/prefetch_l2) | glue ops | f32/int | — | UNKNOWN | fatbin/by-name |
| `sdpa_naive_f32`, `_w_`, `_island_` | naive SDPA fallback for head dims FA doesn't cover (flash_attn.cu:71) | f32 | — | fallback when FA doesn't match head_dim | fatbin/by-name |
| `router_gemv_f32[_w8][_batch]`, `sigmoid_dot_rows_f32` | MoE router GEMV | f32/bf16-w8 | — | MEMRA_ROUTER_KERNEL, MEMRA_ROUTER_BATCH (lib.rs:211, 236) | fatbin/by-name |
| `sdpa_naive_mask_f32` | masked-visibility sdpa_naive twin: causal ∧ u8 mask row per query (qwen4_exp QSA overlay, eager arm) | f32 | — | qwen4exp_gpu eager path only | fatbin/by-name |
| `sdpa_blocklist_f32` | qwen4_exp QSA long-context attention: per query row, attend the row's own ASCENDING selected-position list (<= 2052 entries on real geometry) instead of a dense [t, t_kv] mask — smem and BYTES bounded at any depth; BIT-IDENTICAL to `sdpa_naive_mask_f32` on the same selection (masked entries contribute exact-0.0 terms in the same order) | f32 | — | qwen4exp_gpu (`set_longatt` AUTO) | fatbin/by-name |
| `qsa_index_score_f32` | qwen4_exp QSA indexer block scoring: thread-per-(row, complete block), relu-sum over the 4 index heads / sqrt(head_dim) in fp32 over the device pooled-key mirror; explicit `__fmul_rn`/`__fadd_rn`/`__fdiv_rn` (NO FMA contraction) so scores are BIT-IDENTICAL to the host twin and the top-k selects the same set | f32 | — | qwen4exp_gpu (`set_idx_dev` ON) | fatbin/by-name |
| `rope_neox_ffm_f32` | `rope_neox_ff_f32` + the YaRN attention factor on cos/sin (qwen4_exp long-context lane); identity inputs (ones divisor table, mscale 1.0) reproduce `rope_neox_f32` bit-for-bit | f32 | — | qwen4exp_gpu yarn arm | fatbin/by-name |
| `qsa_index_topk_u32` | qwen4_exp DEVICE QSA indexer top-k SELECTION (262k perf lane): per query row, the pinned top-`budget` micro-block selection over the score slab, emitted ASCENDING by block index — `top_blocks_ascending` on device. 8-pass radix select (one byte/pass, 256 smem bins) fixes the k-th smallest u64 key `(~f32_total_asc_u32(score) << 32) \| block_index` EXACTLY, then one ordered warp-ballot compaction emits every block at or below it. Ascending key order IS the host `sel_cmp` (score desc under `total_cmp`, index asc) over the WHOLE f32 domain — `f32_total_asc_u32` is `f32::total_cmp` verbatim, so unlike `qwen4exp_route_topk_f32`'s `~bits(w)` shortcut it carries NO non-negativity claim. Deliberately NOT the route kernel's k-rounds-of-block-min: that is O(k*n) and the geometries differ by three orders (k=10 over 512 experts there, k=512 over up to 65,536 blocks here), so the same pattern would be SLOWER than the host it replaces. Replaces up to a 128 MB blocking `dtoh` per sub-batch plus the host top-512 that measured 83% of a deep prefill chunk | u64 keys over f32 scores | — | qwen4exp_gpu (`set_idx_sel`, default OFF at introduction) | fatbin/by-name |
| `qwen4exp_route_topk_f32` | qwen4_exp DEVICE MoE router (devtwin lane): the full `host_route_softmax_topk` program per token row — softmax over `experts` (order-sensitive reductions SEQUENTIAL on thread 0, host op order verbatim; exp through double, the one non-bit-pinned op), top-`selected` under the pinned tie rule (weight desc, index asc — unsigned bit compare on the non-negative weight domain == total_cmp), renorm with the 6.1035156e-5 floor; writes sel/w rows + an optional slot->token map. Selection ids+order gated EXACT vs the host twin (tiny arm 0f + the `MEMRA_Q4E_ROUTER_AUDIT` live cross-check); weights ULP-documented | f32 (exp via f64) | — | qwen4exp_gpu (`set_router_dev`, **default ON** 2026-08-31 on receipts — pair with `set_idx_cache`) | fatbin/by-name |
| `qmatvec_bf16w_sel_f32` | device-SELECTED expert twin of `qmatvec_bf16w_f32` over a DeviceBf16 bank (the card-1 draft): grid (out_f, 1, n_sel), slot s reads its expert id from the DEVICE sel array and its rows at sel[s]*out_f*in_f; per-row fmaf chain + reduce tree VERBATIM => BIT-IDENTICAL to the per-slot off_into chain (bf16 oracle sel mode, dup slots, shared-x + slot-x strides) | bf16 w / f32 x | — | qwen4exp_gpu (`set_router_dev` on the DeviceBf16 grouped arm) | fatbin/by-name |
| `copy_rows_col_f32` | row-window column-slice copy (devtwin indexer cache): dst[dst_row+r] = src[r*stride+col..+width] — exact byte moves, appends idx_proj k-part rows to the DEVICE raw-key cache without a host round trip | f32 | — | qwen4exp_gpu (`set_idx_cache`, **default ON** 2026-08-31 on receipts) | fatbin/by-name |
| `copy_batch_uniform_kv_u8_set_len` | TP speculative-verify accepted-prefix repair: one block per uniform attention layer gathers strided full-width canonical K/V rows into the launching rank's contiguous quantized cache slice, then publishes that layer's device length; one launch per rank replaces per-layer K/V copies and len writes | q8_0 K / q5_1 V bytes + i32 len | — | native-P2P, uniform-geometry TP verify restore; per-layer repair fallback otherwise | fatbin/by-name |
| `gdn_scan_naive_f32` | geometry-generic sequential GDN scan, in-kernel q/k l2 + sigmoid(beta); memra-reference twin (any hk/hv incl. tiny 4/4; z-gate composed by caller) | f32 | — | qwen4exp_gpu eager path only | fatbin/by-name |
| `ssm_conv1d_fused_decode_tloop_f32` (lane/dspark-gdn-packed) | The per-row decode conv (`ssm_conv1d_fused_decode_b_f32`) iterated over T rows of ONE sequence inside one launch: state resolved from the replay-refreshed `[conv, canonical ssm, alternate ssm]` pointer table, window and taps in registers, the ring shifted per row exactly as the per-row kernel wrote it, optional post-row ring snapshots (`[T-1, conv_dim, pad]`, the verify-checkpoint slab layout). Bit-identical per row by construction; gate `gdn_packed_gpu`. Door `MEMRA_SPEC_GDN_PACKED` (default ON since 2026-09-08, `0` = per-row). | f32 | none | hybrid.cu |
| `gdn_scan_s128_tsnap` (lane/dspark-gdn-packed) | `gdn_scan_kernel<128,32>`'s body (the same t-loop `gdn_scan_s128` runs) with post-step state snapshots for steps < T-1 (`[T-1, H, S_v, S_v]`, the checkpoint slab layout) and non-restrict state pointers so even-T verifies run in place; resolves canonical/alternate state from the replay-refreshed table, choosing alternate only for odd T; the batched-class verify runs it once per layer over the draft rows and still hands the rollback the per-row states. Bit-identical to T chained `gdn_scan_s128_b` rows; gate `gdn_packed_gpu`. Door `MEMRA_SPEC_GDN_PACKED` (default ON since 2026-09-08, `0` = per-row). | f32 | none | hybrid.cu |
| `dwconv_causal_f32` | token-major depthwise causal conv, arbitrary dilation + history rows (GDN conv dil=1, PLE conv dil=max_ngram); modes conv / silu / silu-add | f32 | — | qwen4exp_gpu eager path only | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_f32` | selected-experts NVFP4 matvec over the AS-STORED modelopt layout (codes [E,out,in/2] + UE4M3 scales [E,out,in/16] + per-expert macro epilogue), W4A16 f32 activations, one launch per projection over all routed experts; x_stride 0 = shared row (gate/up), in_f = per-slot rows (down) | u8-codes/f32 | — | qwen4exp_gpu grouped decode path (t==1, NVFP4 banks; set_moe_sel_path A/B seam) | fatbin/by-name |
| `hc_lowrank_reduce_f32` | qwen4_exp gated-residual read gate: low_act = silu(mean over streams of the rank-320 down parts) in one launch over the stream-major slab; bit-identical to the axpy chain + scale + silu_mul it replaces | f32 | — | qwen4exp_gpu read gate (set_hc_fused_gate seam) | fatbin/by-name |
| `hc_mix_epilogue_f32` | same gate's mix epilogue: mixed = mean_s sigmoid(up_s) * normed_s in one launch over all streams; bit-identical to the per-stream sigmoid/mul/axpy/scale chain | f32 | — | qwen4exp_gpu read gate (set_hc_fused_gate seam) | fatbin/by-name |
| `hc_inject_gates_f32` | same gate's block-inject scalars: out[s,t] = 2*sigmoid(mean_s2 <w[s][s2], normed[s2,t]>), one launch for all streams x tokens (was streams^2 GEMVs + reduces). ACCUMULATION CLASS — different reduction tree than the per-(s,s2) cuBLAS GEMVs | f32 | — | qwen4exp_gpu read gate (set_hc_fused_gate seam) | fatbin/by-name |
| `qmatvec_bf16w_f32` | batched/strided bf16-WEIGHT matvec (W [batch,out,in] bf16, x/y f32, f32 accumulate, uint4=8-bf16 vector loads; x_bstride 0 shares one activation across the batch). qwen4_exp bf16 trunk residency: gdn/qsa projections, lm_head, and the read gate's down/up as ONE batched launch per projection. Guards: in_f%8==0 + exact bf16 representability (loader falls back f32 per tensor). ACCUMULATION CLASS vs cuBLASLt gemvx | bf16-w/f32 | — | qwen4exp_gpu trunk path (set_trunk_bf16 seam); oracle `gate_qmatvec_bf16` | fatbin/by-name |
| `hc_inject_gates_bf16w_f32` | bf16-weight twin of `hc_inject_gates_f32` (same grid/loop/reduction; exact widening => BIT-IDENTICAL to the f32 arm on representable weights) | bf16-w/f32 | — | qwen4exp_gpu read gate (set_trunk_bf16 seam) | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_f32_v2` | v2 of the grouped sel matvec: uint4 code loads (32 codes/load) + float4 activations + 2 output rows per warp sharing the activation registers (ornith sel-kernel craft on the modelopt dialect); same per-element products, accumulation class. Geometry in_f%32==0 && out_f%2==0, else v1 | u8-codes/f32 | — | qwen4exp_gpu grouped decode path (set_sel_v2 seam, v1 fallback) | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_f32_v3` | v3 of the grouped sel matvec: 4 output rows per warp sharing the activation registers + u16 scale loads — quadruples outstanding code loads per warp (v2 ran ≤1 iteration/thread at the artifact's down geometry). Same per-row `acc += scale*group_dot` chaining and p-partition as v2 — accumulation class. Geometry in_f%32==0 && out_f%4==0, else v2/v1 | u8-codes/f32 | — | qwen4exp_gpu grouped decode path (set_sel_v3 seam, v2/v1 fallback); oracle `gate_nvfp4_sel_matvec` v3 modes | fatbin/by-name |
| `gdn_scan_step_f32` | decode-step (t==1) twin of `gdn_scan_naive_f32`: grid (nv, hv) with one state ELEMENT per thread (the naive kernel runs nv blocks with a state row per thread — latency-bound at 48 blocks/188 SMs); same per-element math, block reduction trees for the three row sums — accumulation class. Geometry hk%32==0 && hk<=1024, else naive | f32 | — | qwen4exp_gpu GDN decode (set_gdn_step seam, naive fallback + prefill); oracle `gate_gdn_step_kernels` | fatbin/by-name |
| `rms_sigmul_f32` | dst = rms_norm(x,w) * sigmoid(z) in one launch — replaces the GDN mixer's rms_norm + sigmoid + mul chain (3 serialized small kernels/layer); rms_norm_f32-VERBATIM reduction + sigmoid_f32 gate, no contraction seam => BIT-IDENTICAL to the chain | f32 | — | qwen4exp_gpu GDN norm+gate (set_gdn_fuse seam, Sigmoid arm only); oracle `gate_gdn_step_kernels` | fatbin/by-name |
| `q4e_push_f32` | UVA store of a small f32 vector into a PEER device buffer (raw address; P2P + pool access) — shared by the qwen4_exp TP2 direct join and the generic `Tp2ReplicatedRowJoin`: each card stores its [hidden] partial into the other's persistent ping-pong staging, values verbatim | f32 | — | qwen4exp_gpu TP2 join; generic replicated-row TP2 collective | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_f32_v3c` | count-gated v3 twin for the TP2 graphs: fixed grid over max_sel slots, live slot count + expert ids from a device pack blob (a captured graph cannot re-bake the variable expert split); per-slot arithmetic identical to v3 | u8-codes/f32 | — | qwen4exp_gpu TP2 MoE tail (seg C graphs) | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_g_f32` | SUB-WARP pair-group generalization of v3 (`downsel` lane, mtp14): `g` lanes cooperate on one output row, the warp carries `32/g` groups of `rows` rows each, and the reduce is log2(g) shfl_down steps INSIDE the group. Fixes the lane starvation v3 leaves at this artifact's DOWN geometry — `in_f = expert ff = 640` gives `pairs = 20` against a 32-lane loop, so lanes 20-31 hold no pair for the whole kernel (62.5% occupancy, ONE iteration per active lane). `(g=32, rows=4)` reproduces v3 EXACTLY — same per-lane pair set, same 5-step tree, same write lane — and is gated bit-identical; every other `g` changes the pair summation order (accumulation class, like v3 vs v2). Rows per warp is `(32/g)*rows` so the grid SHRINKS as lanes fill; the launcher refuses any `out_f` it cannot tile exactly and falls back to v3. Dead groups run the reduce with zeroed accumulators rather than returning early: groups in one warp have different `o0`, so an early return would leave `__shfl_down_sync(0xffffffff, ..)` with an incomplete mask. Geometry in_f%32==0 && out_f%((32/g)*rows)==0 | u8-codes/f32 | — | qwen4exp_gpu grouped decode + merged verify (`set_sel_group` seam, **default OFF** — no timing hardware existed to flip it; v3 is the OFF arm); oracle `gate_nvfp4_sel_group` at REAL geometry | fatbin/by-name |
| `axpy_rows_seq_pack_f32` | count-gated slot combine twin of axpy_rows_seq_f32 (weights + count from the pack blob; sequential p-order chain; count 0 writes zeros — the empty-card case) | f32 | — | qwen4exp_gpu TP2 MoE tail (seg C graphs) | fatbin/by-name |
| `hc_norm_planes_f32` | per-(stream, token) RMSNorm over a device plane-pointer table into the stream-major slab — one launch replaces `streams` one-block rms_norm launches per read gate; same per-row formula as rms_norm_f32 | f32 | — | qwen4exp_gpu read gate (set_hc_micro seam) | fatbin/by-name |
| `hc_inject_partials_f32` / `_bf16w_f32` | inject stage 1: chunked partial dots of block_inject rows vs the wide normed token (grid streams×t×C — the single-stage kernel ran `streams` blocks on a 188-SM card); deterministic, no atomics | f32 / bf16-w | — | qwen4exp_gpu read gate (set_hc_micro seam) | fatbin/by-name |
| `hc_inject_reduce_f32` | inject stage 2: sequential chunk sum + 2*sigmoid(mean) into the [streams, t] slab | f32 | — | qwen4exp_gpu read gate (set_hc_micro seam) | fatbin/by-name |
| `hc_write_planes_f32` | slab write gate: plane_s += block_out ⊗ inj[s] for all streams in one launch over the pointer table (replaces `streams` add_scaled_rows + `streams` inject-row d2d copies per gate) | f32 | — | qwen4exp_gpu write gate (set_hc_micro seam) | fatbin/by-name |
| `qmatvec_bf16w_multi4_f32` | row-stacked trunk projection launch (qwen4_exp proj-stack seam): up to 4 same-activation projections (GDN qkv/z/beta/alpha, QSA wq/wk/wv, shared gate/up) over ONE load-time stacked bf16 mat, each output row routed to its original buffer by row range (raw device ptrs, no copies); per-row math is qmatvec_bf16w_f32 VERBATIM => BIT-IDENTICAL, t==1 only | bf16-w/f32 | — | qwen4exp_gpu decode (set_proj_stack seam; per-mat launches are the OFF arm) | fatbin/by-name |
| `hc_diet_stage1_f32` | qwen4_exp hyper-gate diet stage 1 (token grid dim added by the mtp-spec lane — per-token program identical to t==1, verify chunks): per (row-chunk, stream) block recomputes the stream's RMS scale from the raw plane (redundant per block, deterministic), materializes the normed row in smem, then runs its DOWN rows + INJECT partial rows against it — replaces the norm launch + batched down GEMV + inject partials. ACCUMULATION CLASS (new reduce widths) | bf16-w/f32 | — | qwen4exp_gpu read gate (set_hc_diet seam); oracle `gate_hc_diet_kernels` | fatbin/by-name |
| `hc_diet_stage2_f32` | diet stage 2: low_act = silu(mean_s parts) (hc_lowrank_reduce association VERBATIM) + inj = 2*sigmoid(mean_s2 inj_parts) in one tiny launch | f32 | — | qwen4exp_gpu read gate (set_hc_diet seam); oracle `gate_hc_diet_kernels` | fatbin/by-name |
| `hc_diet_stage3_f32` | diet stage 3: per dim-chunk block runs the UP dots for all streams from a smem low_act copy, then the mix epilogue (s-ascending sigmoid*normed, post-sum scale — the hc_mix_epilogue association) with normed recomputed from the stage-1 inv scalars (identical values to stage 1's smem row). Replaces the batched up GEMV + mix epilogue + inject reduce boundary | bf16-w/f32 | — | qwen4exp_gpu read gate (set_hc_diet seam); oracle `gate_hc_diet_kernels` | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_gu_silu_f32` | fused gate+up+silu sel matvec (activations stay f32 — the W4A4 activation-quant lever is owner-retired 2026-08-30): each warp runs 4 GATE + 4 UP rows off shared f32 activation registers, per-row arithmetic v3-VERBATIM, epilogue silu_mul_f32-VERBATIM => BIT-IDENTICAL to the gate+up+silu chain; `pack != 0` = TP2 count-gated twin; `tok_map != 0` = the mtp-spec verify MERGE (slot->token map picks the activation row so ONE launch covers every verify column's routed experts, per-slot program unchanged); warp-packed blocks were tried and REVERTED (decode +0.75 ms, verify flat — spec/mtp6 receipts; kernels keep lane-based indexing, launch stays one warp/block). Geometry in_f%32==0 && ff%4==0 | u8-codes/f32 | — | qwen4exp_gpu MoE tail (set_sel_gufuse seam; the v3 chain is the OFF arm) + spec verify chunks (set_verify_mt); oracle `gate_nvfp4_sel_matvec` gufuse/tok_map modes | fatbin/by-name |
| `qmatvec_nvfp4_modelopt_sel_gu_silu_g_f32` | sub-warp pair-group twin of the fused gate+up+silu sel matvec (`downsel` lane, mtp14), same `(g, rows)` partition as `..._sel_g_f32`, `pack` and `tok_map` modes carried unchanged. Fixes the gate+up tail: `in_f = hidden = 2560` gives `pairs = 80`, i.e. 3 warp iterations for 2.5 iterations of work (83.3% occupancy). `(g=32, rows=4)` is `..._sel_gu_silu_f32` EXACTLY (gated bit-identical, pack + tok_map included), and at every shape the fused arm is gated BIT-IDENTICAL to the same-shape `..._sel_g_f32` gate + up + silu_mul chain, so the fusion property survives the reshape. Interleaving gate/up inside the reduce tree is free for bit-identity: each accumulator's own addition sequence is what fixes its bits. Geometry in_f%32==0 && ff%((32/g)*rows)==0 | u8-codes/f32 | — | qwen4exp_gpu MoE tail + spec verify chunks (`set_sel_group` seam, **default OFF**); oracle `gate_nvfp4_sel_group` gu modes | fatbin/by-name |
| `qmatvec_bf16w_mt_f32` | multi-token WEIGHT-SHARED bf16 matvec (mtp-spec verify): one block per output row, each weight uint4 loaded ONCE and FMA'd into every token's accumulator — per (row,token) chain qmatvec_bf16w_f32 VERBATIM => BIT-IDENTICAL to per-token launches; only weight-read counts drop (the qwen38 t-parallel pattern). 2<=t<=12 | bf16-w/f32 | — | qwen4exp_gpu trunk linears at exact verify chunks + small prefills (set_verify_mt seam; the per-token grid is the OFF arm); oracle `gate_qmatvec_bf16` mt mode | fatbin/by-name |
| `hc_diet_stage0_mt_f32` | hc-diet MT stage 0: the stage-1 RMS sumsq reduce EXACTLY (same block-256 stride, same tree, same rsqrtf) per (token, stream) -> bit-equal inv scalars for the weight-shared stages | f32 | — | qwen4exp_gpu read gate at verify chunks (set_verify_mt) | fatbin/by-name |
| `hc_diet_stage1_mt_f32` | hc-diet MT stage 1: down/inject weight rows read ONCE, tokens iterated inside with INLINE (x*inv)*nw normalization — per-(row,token) fma chain the stage-1 program VERBATIM => BIT-IDENTICAL to the token-grid stage 1; t<=12 | bf16-w/f32 | — | qwen4exp_gpu read gate at verify chunks (set_verify_mt); oracle `gate_hc_diet_kernels` mt mode | fatbin/by-name |
| `hc_diet_stage3_mt_f32` | hc-diet MT stage 3: up rows read once, ALL T low_act rows resident in smem, per-token lane chains + the stage-3 epilogue with inv[t,s] — BIT-IDENTICAL to the token-grid stage 3 | bf16-w/f32 | — | qwen4exp_gpu read gate at verify chunks (set_verify_mt); oracle `gate_hc_diet_kernels` mt mode | fatbin/by-name |

### cu/hybrid.cu — 67 symbols (fatbin `MEMRA_HYBRID_FATBIN`)

Header: "Qwen3.5/3.6 hybrid linear-attention: depthwise causal conv1d + SiLU, Gated
DeltaNet scan; all f32" (hybrid.cu:1-3).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `ssm_conv1d_*`, `conv_left_pad_f32`, `conv_assemble_and_roll_f32`, `ssm_conv_ring_*` | causal conv1d + ring KV state | f32 | — | UNKNOWN | fatbin/by-name |
| `gdn_*` scan family (prep_decode, scan_s128/_dc/_b, chunk_* incl. `_vl`) | Gated DeltaNet recurrent/chunked scan | f32 | mma-body region `#if !defined(MEMRA_PORTABLE_CUDA) \|\| defined(MEMRA_HOPPER_MMA)` (hybrid.cu:489, 1531, 2192) | MEMRA_GDN_MMA | fatbin/by-name |
| `gdn_chunk_state_mma[_vl]`, `gdn_chunk_output_mma[_vl]`, `gdn_p_bf16_masked` | fused wgmma K4+K5 chunk kernels | bf16 mirrors, f32 acc | sm_90a-only wgmma via wgmma_common.cuh:9 (include at hybrid.cu:2202) | MEMRA_GDN_WGMMA (hybrid.cu:2195 "env+cfg gated") | fatbin/by-name |
| glue (~17: sigmoid, gated_rmsnorm incl. f16out/q8_1, transpose, repeat_heads, axpy, scatter/gather/reduce slots, f32_to_bf16_bulk, q_gate_split, qkv_to_gdn_repack) | hybrid-path glue | f32 | — | UNKNOWN | fatbin/by-name |

### cu/qmatvec.cu — 347 symbols (fatbin `MEMRA_QMATVEC_FATBIN`; lane/kda-narrow-q8-20260905 +1: `qmatvec_q8_0_mmvq_f32in_narrow`; lane/moe-gateup-ilp2-20260905 +2: `moe_gate_up_preclamp8_q8_rows_ilp2`, `_w4_ilp2`; lane/moe-down-ilp2-20260905's `moe_down8_fma_q8_rows_ilp2` / `_w4_ilp2` removed 2026-09-06 on the gemvab receipt)

Count basis, stated because BOTH merge parents moved it and neither number survives the union:
344 is a MEASURED `grep -c 'extern "C" __global__'` on this file at commit bbbef06b17 —
343 at its parent e208899d plus the one `matvec_bf16_f32acc_x4_range` entry below. The prior
321 count was itself measured after the
glm5 door-r/dedup merge, but later merged kernels moved the file again. This file is now measured;
every other file's count in this document still is not (the vrest lane's named
inventory-recount follow-up).

Header: "Resident-quantized matmul: weights stay in GGUF block format in VRAM,
dequantized in-register" (qmatvec.cu:1-2). Guards: PDL sm≥900 (qmatvec.cu:17); dp4a
`#if __CUDA_ARCH__ >= 610` (qmatvec.cu:464).

| symbol (family) | purpose | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| `qmatvec_{q8_0,q4_K,q6_K,q5_K,q3_K,nvfp4,iq4_XS}_dp4a`, `qmatvec_f32` | decode matvec, int8 dp4a | per-name | dp4a ≥610 | MEMRA_MMVQ / MEMRA_NO_BATCHED | fatbin/by-name |
| `qmatvec_*_mmvq_b{2,4,8,16}[_r2/_rp/…]` batched families (q4_0, q8_0, q4_K, q5_K, q6_K, nvfp4 fused2/3/4) | MMVQ batched decode matvec; `_rp` = split-plane repacked weights | per-name | — | MEMRA_MMVQ, MEMRA_MMVQ_ROWS, MEMRA_MMVQ_BV, MEMRA_RP, MEMRA_Q4RP, MEMRA_B8, MEMRA_Q40_MR | fatbin/by-name |
| `q{8_0,4_0,4_K,6_K}_split_rp_build` | build split-plane repack of weights | per-name | — | MEMRA_RP / MEMRA_Q4RP | fatbin/by-name |
| `moe_pairs_*`, `moe_gate_up_*`, `moe_down8_*`, `qmatvec_expert_q8`, `moe_w_scale_by_expert`, `moe_w_exscale` (~60 variants) | MoE expert matvec/FFN decode kernels | Q8_0 experts (+IQ4 csr variant), f32 | — | UNKNOWN per-variant (decode dispatch in lib.rs; MEMRA_MOE_CACHE etc.) | fatbin/by-name |
| `qmatvec_nvfp4_{bf16,q8}_ep_*` (`dual_slots`, `paired_slots`, `dual_pairs`, `quad_pairs`, `down_{fma,pairs,slots}`) | automatic whole-expert EP gate/up/down over global fixed route slots; exact W4A16 and optional internal W4A8/mixed-Q8 scopes write canonical slot rows at every qualified batch width. `q8_ep_paired_slots` computes gate+up together with independent accumulator/reduction chains while sharing activation reads; the separate-CTA `q8_ep_dual_slots` remains the explicit rollback after its current-main composition produced degenerate HY3 output | NVFP4 W4A16 exact; optional internal W4A8 or mixed W4A8/W4A16 | — | MEMRA_PARALLEL_EP_DEVICE_ROUTER; MEMRA_PARALLEL_EP_GRAPH; MEMRA_PARALLEL_EP_PAIR_DOWN; MEMRA_PARALLEL_EP_Q8_ACT; MEMRA_PARALLEL_EP_Q8_SCOPE; MEMRA_PARALLEL_EP_Q8_GU_PAIRED | lib.rs by-name wrappers |
| `moe_gate_up_preclamp8_q8` (glm5_next, 2026-08-28) | the ONLY clamped fused MoE epilogue in the family: `moe_gate_up_silu8_q8`'s dots/grid/warp-reduction verbatim with `swiglu_preclamped_mul_scaled_f32`'s PRE-clamp expression and per-slot `weight_scale_2` macro scales (`gs`/`us`). Paired with the unmodified `moe_down8_fma_q8`, whose macro folds into the routing weight. Every other `moe_gate_up_*` hardcodes plain `silu(gate)*up`, which is a different program above the limit — do not substitute. | NVFP4/IQ/k-quant experts via `expert_dot_g`, q8_1 activations | needs `limit > 1e-6` (at 0 every gate collapses to `silu(0)`); debug_assert in `Engine::moe_gate_up_preclamp8_q8` | MEMRA_MOE_FUSED_EPI (default OFF; hybrid_forward.rs `moe_fused_epi_enabled`) | fatbin/by-name |
| `swiglu_preclamped_mul_scaled_q8_1_f32` (cu/hybrid.cu, lane/launch-collapse-20260906) | `swiglu_preclamped_mul_scaled_f32` and `quantize_q8_1` in one launch: one warp per 32-block computes the pre-clamped SwiGLU element per lane with the character-identical expression, then quantize_q8_1's warp amax / scale / `__float2int_rn` chain verbatim, writing the q8_1 pair the down MMVQ reads; the f32 activation row is never materialized. Host refuses `n % 32 != 0`. Door `MEMRA_SHEXP_SWIGLU_ZQ8` (default OFF); gate `swiglu_zq8_gpu` (bitwise vs the two-launch chain, red arms for limit, up scale and width). |
| `l2_norm2_f32` (cu/kernels.cu, lane/launch-collapse-20260906) | `l2_norm_f32` on two tensors in one launch (grid `rows x 2`, blockIdx.y picks the pair): the KDA core's q and k L2 norms, 34 launch pairs per token on GLM-5.3-Flash folded to 34 launches. Per-row body and block shape verbatim, so every row is byte-identical. Always on (a launch fold, no arithmetic change); gate `kda_small_folds_gpu` (four shapes, red arm: the second output follows its own input). |
| `memra_kda_gate_beta_f32` (cu/kda.cu, lane/launch-collapse-20260906) | `memra_kda_gate_f32` and `sigmoid_f32(beta_raw)` in one launch on the gate's grid: threads below `heads` also write `beta = 1/(1+exp(-beta_raw))` verbatim. 34 launch pairs per token folded. Always on; gate `kda_small_folds_gpu` (three shapes, red arms: lower_bound reaches g, beta_raw reaches beta, a non-dividing head_dim is refused). |
| `moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows` (glm5_next, lane/glm5-vrest 2026-08-31) | verify-rows twins of the fused preclamp epilogue pair: ONE launch pair covers ALL t x n_used routed pairs of a spec-verify batch (pair p = tok*n_used + j, dense slot-major). Per pair the bodies are `moe_gate_up_preclamp8_q8` / `moe_down8_fma_q8` VERBATIM (same `expert_dot_g` g-strided order per (pair,row) == `qmatvec_expert_q8`'s chain, same warp tree, same PRE-clamp expression, same slot-ordered `__fmaf_rn` down chain); inputs are plane-major (gate\|up\|down) `[3*n_pairs]` u64 pointer + f32 scale tables (gs\|us\|w*macro_down) host-built from the resident slab base + ex*stride. | NVFP4/IQ/k-quant experts via `expert_dot_g`, q8_1 activations | needs `limit > 1e-6`; pairs dense slot-major (`n_pairs % n_used == 0`) | rides `MEMRA_GLM5_VERIFY_BATCH` (verify walk only; no flag of its own) — bit-gated per row vs the sequential chain by `glm5_verify_batch_gpu` gate 4 (swapped-pair + dropped-macro reds) | fatbin/by-name |
| `moe_vrows_tables_from_sel` (glm5_next, lane/glm5-moe-loc 2026-08-31) | builds the verify-rows pair's `[3*n_pairs]` plane-major pointer + scale tables ON DEVICE from the sigmoid router's own `sel_idx`/`sel_w`, so the layer no longer reads the selection back to evaluate `slab_base + ex*expert_stride` and three `macro_scale(ex)` lookups on the host. One thread per pair (`n_pairs = t*n_used <= 128` on every serving shape), 128-thread blocks. Removes a full `cuStreamSynchronize` + 2 DtoH + 2 pageable HtoD per MoE layer-call = 42 drains + 84 DtoH + 84 HtoD per ship round. Bit-identical to the host loop term by term: exact integer pointer arithmetic, the same f32 macro plane at the same index (resident mirror, uploaded once per `(layer, plane)`), and ONE IEEE-754 single multiply `selw[p] * macro_down[ex]` matching the host's `w * macro_scale(ex)` operand order (no FMA contraction is possible in a bare product). Door `MEMRA_MOE_VROWS_DEV_TABLES` (default OFF). Gate: `glm5_moe_loc_doors_gpu` (table-level bitwise vs the host build across t=2..=8 with and without macro planes; pair-output bitwise; wrong-down-stride and dropped-macro reds bite). |
| `moe_gate_up_preclamp8_q8_rows_ord` + `moe_down8_fma_q8_rows_tmaj` (glm5_next, lane/glm5-dedup 2026-08-31) | DEDUP-SCHEDULE twins of the verify-rows pair: the grids are TRANSPOSED so the deduplicating index is the FASTEST one. `_ord` runs `(n_pairs, n_ff)` and takes its pair from an EXPERT-MAJOR order plane appended to the pointer table (`pr = ptrs[3*n_pairs + blockIdx.x]`), so two verify rows sharing an expert read the identical gate row and up row in ADJACENT blocks; `_tmaj` runs `(t, out_f)` (token fastest) so the t rows at one output row are adjacent and a repeated expert's down row is read once for all of them. Motivation is a measurement, not a guess: **21.96% repeat fraction** over 2.55M expert visits on the ship shape (6.9x the independent-routing bound) against a pair already at 90.2% of theoretical DRAM peak. Bodies are their twins' VERBATIM (same `expert_dot_g` g-strided chain, same warp tree, same PRE-clamp epilogue, same slot-ordered `__fmaf_rn` down chain in its ORIGINAL slot order); every output is a pure function of its `(o, pr)` / `(o, tok)` coordinate and no block communicates, so re-indexing which block computes which output moves no bits. | same as the shipped pair | `_ord` needs the 4th (order) plane present and `n_ff <= 65535`; `_tmaj` needs `out_f <= 65535`; both refused when `MEMRA_MOE_VROWS_PACK` is armed | `MEMRA_MOE_VROWS_DEDUP_ORDER` / `MEMRA_MOE_VROWS_DOWN_TMAJ` (both default OFF; box prices the flips) — bit-gated vs the shipped schedule in `glm5_dedup_sched_gpu` (28 + 42 arms, 3 valid-shuffle INERT arms, two non-permutation reds that bite) | fatbin/by-name |
| `moe_vrows_order_from_sel` (glm5_next, lane/glm5-dedup 2026-08-31) | builds the verify-rows pair's EXPERT-MAJOR order plane on device from the router's own `sel`, into the pointer table's fourth plane `ptrs[3*n_pairs ..)`. STABLE COUNTING RANK — thread p counts how many pairs sort strictly before it on `(expert id, pair index)` and stores itself at that rank — chosen because it needs no scratch, no scan and no order-of-execution dependence, and is therefore bit-identical to the host arm's stable sort (`vrows_expert_major_order`). O(n_pairs^2) with `n_pairs = t*n_used <= 64` on every serving shape (4096 comparisons), one 128-thread block. Exists ONLY in the door-D arm: with host tables the plane rides the `htod_u64_into` that was already uploading the pointers, so the door costs zero extra transfers there. Door `MEMRA_MOE_VROWS_DEDUP_ORDER` (default OFF). Gate: `glm5_dedup_sched_gpu` (device plane == host stable sort at t=2..=8; permutation, stable-expert-major and run-count==distinct all asserted; the changed-selection red bites). | | | | fatbin/by-name |
| `moe_gate_up_preclamp8_q8_rows_w4` + `moe_down8_fma_q8_rows_w4` (glm5_next, lane/glm5-matvec 2026-08-31) | WARP-PACKED twins of the verify-rows pair: `MEMRA_MMVQ_ROWS`=4 warps/block on threadIdx.y (`o = blockIdx.x*4 + threadIdx.y`), per-(row,pair) warp body VERBATIM — the one-warp-block form caps residency at the blocks/SM limit (<=67% of warp slots) and schedules ~65k one-warp blocks/launch; packing moves no bits (no `__syncthreads` in either body, ragged tail returns early) | same as the unpacked pair | same | `MEMRA_MOE_VROWS_PACK` (default OFF; box prices the flip) — bit-gated packed-vs-unpacked + re-bitten gate-4 reds in `glm5_matvec_doors_gpu` | fatbin/by-name |
| `moe_gate_up_preclamp8_q8_rows_ilp` / `_w4_ilp`, `moe_down8_fma_q8_rows_ilp` / `_w4_ilp` (lane/glm5-moe-rows-ilp-20260904) | ILP twins of the verify-rows MoE pair: the same per-warp program with the loads of four (then two) groups per lane hoisted ahead of their math (`nvfp4_v1_load_g` -> `expert_dot_nvfp4_core_regs`, the pinned core on registers); same per-plane accumulation order, warp tree, epilogues verbatim; interleaved NVFP4 only (NaN poison otherwise). Why: the pair is 22% of a plain glm5_next t=1 token on 2x B200 with long-scoreboard stalls at 55-60% (rig ncu) | NVFP4 experts, q8_1 activations | none | `MEMRA_MOE_VROWS_ILP` (default OFF), composes with `MEMRA_MOE_VROWS_PACK` | fatbin/by-name |
| `moe_gate_up_preclamp8_q8_rows_loadonly`, `moe_gate_up_preclamp8_q8_rows_mathonly` (cu/qmatvec.cu, lane/moe-rows-ceiling-20260906) | BENCH-ONLY ceiling probes for the verify-rows MoE gate/up kernel, reached only through `Engine::moe_gate_up_rows_ceiling_probe` and `moe-rows-ceiling-bench`; no door, no serving dispatch, removed when the lane closes. `_loadonly` is the served `_ilp` loop verbatim (same addresses, same four groups of each plane in flight) with the table lookup and dp4a chain replaced by an integer add, so its rate is what the ACCESS PATTERN can reach; `_mathonly` loads one group per lane and repeats the served arithmetic, so its rate is what the INSTRUCTION STREAM can reach with memory free. Their outputs are meaningless by construction: this measures rate, never values. Rig receipt 2026-09-06 (ordering only): served 156-176 us, loadonly 1.19x served, mathonly 4.2-5.0x served, i.e. the access pattern is the wall. |
| `moe_gate_up_preclamp8_q8_rows_loadonly_v2`, `moe_gate_up_preclamp8_q8_rows_mathonly_v2` (cu/qmatvec.cu, lane/rows-v2-ceiling-20260906) | slot-major (`QT_NVFP4_V2`) twins of the two ceiling probes above: identical bodies and ILP, only the loader template argument differs. They exist to separate the layout's two effects, because the interleaved block hides a group's two UE4M3 scale bytes inside a 36-byte block (a warp's 32 lanes fetch 32 sectors for 32 scale bytes) while slot-major keeps them contiguous at `nsb*16 + g*2` (2 sectors). Rig receipt 2026-09-06: V1 served 108.2 us, V1 loadonly 87.7, V2 served 84.8 (**-21.6% vs V1 served, and below V1's load-only floor**), V2 loadonly 82.6 (-5.9% vs V1 loadonly) -- so most of the layout's win is on the SERVED path, not the pure-load path. Bench-only, no door; removed when the lane closes. |
| `moe_gate_up_preclamp8_q8_w4` + `moe_down8_fma_q8_w4` (glm5_next, lane/b200-matvec-occupancy-20260902) | WARP-PACKED twins of the PLAIN-DECODE preclamp epilogue pair (not the verify-rows pair above) — same `MEMRA_MMVQ_ROWS`=4 warps/block idiom applied to `moe_gate_up_preclamp8_q8`/`moe_down8_fma_q8`'s `block=(32,1,1)` grid. Per-(o,j)/o warp body VERBATIM (same `expert_dot_g` chain, same warp tree, same PRE-clamp epilogue, same slot-ordered `__fmaf_rn` down chain); packing only changes which block/warp computes an output. Motivated by the B200 decode census (2026-09-02): the unpacked pair is 20.0%+10.5% of GPU time on 2x B200, ~9x its NVFP4 roofline estimate — an occupancy signature, not bandwidth. | NVFP4/IQ/k-quant experts via `expert_dot_g`, q8_1 activations | needs `limit > 1e-6` | `MEMRA_B200_MATVEC_ARM` (default OFF, sm_100a builds only; box A/B pending) — dispatched in `hybrid_forward.rs`'s `moe_fused_epi_launch` | fatbin/by-name |
| `qmatvec_nvfp4_mmvq_fused2_rp_g2` (lane/b200-matvec-occupancy-20260902) | SUB-WAVE GRID-FILL twin of `qmatvec_nvfp4_mmvq_fused2_rp`: instantiates the existing `nvfp4_mmvq_fused_seg_rp<1>` template (shipped instantiates `<2>`) — RPW halved, grid doubled for the same out0+out1. Per (tensor,row) the seg body is the template VERBATIM regardless of RPW, so a given row's output is bit-identical between the two. The sibling grid-fill for `qmatvec_nvfp4_mmvq_mr2_rp` needs no new kernel: it reuses the ALREADY-SHIPPED `qmatvec_nvfp4_mmvq_rp` (RPW=1) via a pure dispatch-policy change in `qmatvec_mmvq_into` (lib.rs), same sub-wave-grid check (`Engine::sm_count`, `cudaDevAttrMultiProcessorCount`). | NVFP4 split-plane rp, q8_1 activations | m==1 only (fused2 contract) | `MEMRA_B200_MATVEC_ARM` (default OFF, sm_100a builds only) — dispatched in `matmul_nvfp4_fused2`/`matmul_nvfp4_fused2_into` and `qmatvec_mmvq_into` (lib.rs) | fatbin/by-name |
| `matvec_bf16_f32acc_x4_rows_pf` (lane/b200-matvec-occupancy-20260902) | PREFETCH (software-pipelined) twin of `matvec_bf16_f32acc_x4_rows`: the K-loop double-buffers its weight/activation loads — the NEXT iteration's `uint4`/`float4` reads issue before the CURRENT iteration's 8-fma chain runs, hiding DRAM latency. Same grid/block mapping, same 4-row sequential loop, same `red[]` tree reduction, same per-thread fma order for the same `i` -> bit-identical per (row,token); only load-issue TIMING changes. Motivated by the B200 census: the shipped kernel is 17.0% of GPU time on 2x B200, ~3x its bf16 roofline estimate for the KDA `[8192,4096]`/`[4096,8192]` projections. | bf16 weights, f32 acc | — | `MEMRA_B200_MATVEC_ARM` (default OFF, sm_100a builds only; box A/B pending) — dispatched in `matvec_bf16_rows_into` (lib.rs) | fatbin/by-name |
| `memra_bf16_pp_gemm` (cu/f16_prefill.cu, pre-existing) | NEW DISPATCH SITE ONLY (lane/b200-gemv-hbm-20260902, no kernel code added): the `MEMRA_B200_BF16_GEMV_LT` REFERENCE door calls the already-shipped cuBLASLt TN bf16 GEMM at **m=t** so a B200 box can measure what a tuned vendor GEMV reaches on the same bytes the decode row matvec moves (census: 2.7 TB/s = 34% of the 8 TB/s HBM3e wall for `matvec_bf16_f32acc_x4_rows`, 2.1 TB/s = 26% for `qmatvec_kda6_bf16f32`). Per-device `cublasLtHandle_t` + (dev,m,n,k) plan cache unchanged. NAMED NUMERIC CLASS `bf16_gemv_lt`, NOT bit-identical: activation cast f32 -> bf16, library K summation order. Reference instrument, never a serving default. | bf16 weights + bf16 activation, f32 acc/out | — | `MEMRA_B200_BF16_GEMV_LT` (default OFF, sm_100a builds only) — dispatched in `matvec_bf16_rows_into` (lib.rs) via `Engine::bf16_gemv_lt_into` (f16_ffi.rs); `MEMRA_KDA_FUSED_PROJ`'s bf16 arm declines while it is on | C-ABI (f16_ffi.rs) |
| `matvec_bf16_v2`, `matvec_bf16_v2_sk`, `matvec_bf16_v2_sk_combine` (lane/b200-gemv-hbm-20260902) | HBM-speed rewrite of `matvec_bf16_f32acc_x4_rows` for sm_100a. 8 rows/block accumulated CONCURRENTLY with the f32 activation loaded once per K step and reused across all eight (weight:activation bytes 4:1, where the shipped kernel's four sequential rows are 1:2); two-stage software pipeline whose steady-state loop issues ten back-to-back `LDG.E.128.CONSTANT` (160 B/thread) before the first `FFMA`; the eight reductions run in lockstep so a block pays ONE barrier chain instead of four, with a `__shfl_down_sync` tail that replays the shipped smem tree's last five steps exactly; `__launch_bounds__(256)`. BIT-IDENTICAL per (row, token) — same per-thread K subset, same order, same fma expressions. `_sk`/`_sk_combine` are the split-K pair for shapes whose row grid cannot cover two CTA waves: NAMED numeric class `bf16_gemv_v2_splitk`, fixed ascending-chunk combine, never selected by the GLM-5.3 decode shapes. | bf16 weights, f32 activation, f32 acc | sm_100a builds (door-gated) | `MEMRA_B200_GEMV_V2` (default OFF; box A/B pending) — dispatched in `matvec_bf16_rows_into` via `Engine::matvec_bf16_v2_raw` (lib.rs); `Engine::gemv_v2_ksplit` picks the class | fatbin/by-name |
| `qmatvec_kda6_bf16f32_v2` (lane/b200-gemv-hbm-20260902) | v2 twin of the fused KDA six-projection BF16 kernel — the census's single hottest launch (93.8 us for ~200 MB = 2.1 TB/s, 26% of the HBM3e wall). Its three BF16 ranges take the `matvec_bf16_v2` eight-rows-per-block walk instead of `kda6_bf16_rows4`'s four sequential rows; the three f32 ranges keep `f32_mmvq_row1` verbatim. Same six ranges in the same block order, `R=8` block partition, dynamic smem = 8 * blockDim.x * 4 B. BIT-IDENTICAL per row to `qmatvec_kda6_bf16f32`, which is itself bit-identical per row to `matvec_bf16_f32acc_x4_rows`. | bf16 + f32 weights, f32 activation, f32 acc | sm_100a builds (door-gated) | `MEMRA_B200_GEMV_V2` (default OFF) — `Engine::kda_proj_fused6_bf16_arm_raw` (kda.rs), under the `MEMRA_KDA_FUSED_PROJ` bf16 arm | fatbin/by-name |
| `matvec_bf16_v3`, `qmatvec_kda6_bf16f32_v3` (lane/b200-gemv-hbm-20260902, door level 2) | The v2 walk with its weight tiles staged global->shared by `cp.async` (`__pipeline_memcpy_async`, 16 B/thread/row) instead of held in registers, so the in-flight budget stops being register-bound: v2 kda6 is 96 registers -> ~102 KB outstanding per SM, v3 is 2 stages x 8 rows x (blockDim*8) x 2 B = 32 KB per CTA plus a 4 KB reduction window = 36 KB of smem -> 6 CTAs/SM -> **~192 KB per SM, 1.9x**, at 58/64 registers. Adds NO barriers: each thread issues the copy for its own 16 B lane in every row and later reads THAT SAME address, so the pipeline commit/wait are per-thread and no `__syncthreads` is needed between stages. The K chunk is PINNED to `blockDim.x * 8` (the shipped per-thread stride), so chunk `c` hands thread `tid` exactly one index `i = c*kch + tid*8` and walking chunks ascending reproduces the shipped `i` sequence exactly -> BIT-IDENTICAL to v2 and to the shipped kernels. SASS: 24 `LDGSTS` + 2 `LDG` in `matvec_bf16_v3`. Declines to v2 per call when a shape wants split-K or when 36 KB would exceed the 48 KB default dynamic-smem cap (it does at `mmv_block()=256`). Chosen over TMA/`cp.async.bulk`, which has the same in-flight arithmetic and an mbarrier protocol this lane cannot test on-part; and over persistent CTAs (in-flight bytes unchanged) and 2-CTA activation sharing (capped near 12% of load instructions). | bf16 weights, f32 activation, f32 acc | sm_100a builds (door-gated) | `MEMRA_B200_GEMV_V2=2` (default OFF; receipt pending) — `Engine::matvec_bf16_v3_raw` (lib.rs) and `kda_proj_fused6_bf16_arm_raw(arm=2)` (kda.rs) | fatbin/by-name |
| `qmatvec_q8_0_mmvq_rp_v2`, `qmatvec_q8_0_rows_tw_v2` (lane/b200-gemv-hbm-20260902 round 3) | v2 twins of the kernels that actually serve the `MEMRA_GLM5_W8` posture: `qmatvec_q8_0_mmvq_rp` at t=1 (W8-census 9.6 us for 35.65 MB = 3.71 TB/s, 46% of the wall, ~211 launches/token, 8.1% of GPU) and `qmatvec_q8_0_rows_tw` at the t<=8 verify width. The rp4 mirror is a 32 B quant plane + a 2 B scale plane, so every weight fetch is ALREADY an aligned 16 B `__ldcs` — no load-width lever. The lever taken: the shipped t=1 kernel reads 36 B of ACTIVATION per 34 B of weight, per lane, per block-iteration, and that activation is identical for all rows in the launch, so the v2 twin stages it into shared memory ONCE per CTA (`in_f + nblk*4` = 4.6 KB at in_f=4096, one barrier), packs 8 warps per block instead of 4, and unrolls the block walk by two. The `_tw_v2` twin keeps the weight-once t-column structure and does not stage (t*in_f would be 32 KB at t=8); its levers are the packing and the unroll. BIT-IDENTICAL per output: same warp-per-row mapping, `blk` walk, dp4a order, `acc += dw * ad[blk] * (float)sumi` association and `warp_reduce_sum`. | q8_0 rp4 mirror weights, q8_1 activation | sm_100a builds (door-gated) | `MEMRA_B200_GEMV_V2=1` (default OFF; receipt pending) — dispatched inside `matvec_bf16_via_q8_mirror` and `matvec_bf16_via_q8_mirror_t` (lib.rs) | fatbin/by-name |
| `qmatvec_kda6_q8f32_rp_v2` (lane/b200-gemv-hbm-20260902 round 3) | NEW FUSION, not a twin: the W8 posture had none. `qmatvec_kda6_q8f32_mmvq` addresses interleaved 34 B blocks (resident plain-layout Q8_0) while the `MEMRA_GLM5_W8` mirror is the split-plane rp4 form, and `MEMRA_KDA_FUSED_PROJ`'s bf16 arm declines whenever W8 is on — so the KDA stage-1 group runs as 3 q8 launches + 3 f32 projections that each cost a cuBLAS `dot_kernel` + `reduce_1Block_kernel` pair, plus six redundant `quantize_q8_1` calls on the same `x` = 9 launches per layer. This makes it 1: over 34 KDA layers, 306 -> 34 launches per token. Same six-unequal-range block split as `qmatvec_kda6_bf16f32`, the three mirrored ranges on the rp v2 body, the three f32 low-rank/beta ranges on `f32_mmvq_row1`, 8 warps/block, one staged activation for the whole CTA. NUMERIC CLASSES (both pre-existing): q8 ranges bit-identical to `qmatvec_q8_0_mmvq_rp` per row; f32 ranges take the same deterministic warp tree the q8 arm of `MEMRA_KDA_FUSED_PROJ` already ships and has pinned. | q8_0 rp4 mirrors + f32 weights, q8_1 activation | sm_100a builds (door-gated) | `MEMRA_KDA_FUSED_PROJ=1` **and** `MEMRA_B200_GEMV_V2=1` (both default OFF) — `Engine::kda_proj_fused6_q8rp_raw` (lib.rs), armed in `kda_proj_fused6` (kda.rs); counter `KDA_FUSED6_Q8RP_DISPATCHES` | fatbin/by-name |
| `memra_kda_gated_rmsnorm_zq8_f32` (cu/kda.cu, lane/kda-onorm-zq8-20260905) | the KDA sigmoid-gated o_norm emitting its q8_1 pair beside the f32 row: same reduction and per-element expression as `memra_kda_gated_rmsnorm_f32`, then `quantize_q8_1`'s per-32-block arithmetic in the same layout (block (t*H+h, blk) is token block (t, h*ncols/32+blk)), so the `wo` MMVQ input is bit-identical to norm-then-quantize; drops the standalone quantize launch per KDA layer (34 per token) | f32 in; f32 + q8_1 out | none | `MEMRA_KDA_ONORM_ZQ8` (`Engine::kda_gated_rmsnorm_zq8`, `kda_core` hands the pair to `matmul_q8_fast(wo)`) | fatbin/by-name; gate `tests/kda_onorm_zq8_gpu.rs` + the decode-graph fixture arm |
| `qmatvec_e4m3_mmvq_fused6_ilp` REMOVED 2026-09-05 (door sweep; `e4m3_row_dot_ilp<1>` remains the serial walk) | `MEMRA_E4M3_ROW_ILP` twin of the e4m3 row walk and of the six-group above: four blocks' weight and activation loads per lane issued ahead of the dependent fma chains, then the four block sums folded into `acc` in the shipped ascending order. Same range split, same warp tree, same bytes read; bit-identical per row by construction, and `e4m3_row_dot` is `e4m3_row_dot_ilp<1>` so the serial program is an instance of the same template rather than a copy. The per-block body (`e4m3_blk_dot`) is shared by both walks so they cannot drift. WHY: rig ncu reads every big matvec long-scoreboard-bound at 71-76%; the same lever paid on `matvec_bf16_v2`/`v3` and `q8_0_mmvq_row1_rp_v2_ilp` (+2.16% on the pair), and the B200 hybrid mint puts 34 of 45 layers' stage-1 time in this loop. | per-tensor e4m3 weights, shared q8_1 activation, f32 acc | — | `MEMRA_E4M3_ROW_ILP` (default OFF pending its model-scale row; kernel receipt **1.003-1.015x** on 2x B200, 3 reps, bit-identical) — `Engine::e4m3_fused6_into_arm` (lib.rs); gate `tests/kda_fused6_e4m3_gpu.rs` drives both arms at `in_f=4096`. Wider blocks and a CTA-staged activation were measured here and REMOVED (r8 0.99x, r16 0.97x, staged ~0.92-0.93x, r32+staged 0.74x): VERDICT:e4m3-fused6-wider-blocks-and-staged-activation-KILLED | fatbin/by-name |
| `qmatvec_e4m3_mmvq_fused6` (lane/glm5-b200-mint-consume-20260904) | Six-range extension of `qmatvec_e4m3_mmvq_fused2`/`fused3` for the KDA six-projection group on a UNIFORMLY e4m3 checkpoint. The GLM-5.3-Flash B200 hybrid mint quantizes all six KDA projections (q, k, v, f_a, g_a, b) to per-tensor e4m3, so under `MEMRA_ST_E4M3` every one is QT_F8_E4M3-resident at **1.0 B/weight** -- cheaper than the bf16 serving recipe's 2.0 and the Q8_0 re-encode's 1.0625, with no lossy re-quant hop. Neither existing arm of `MEMRA_KDA_FUSED_PROJ` binds on that shape (both need a FloatBf16 trio plus an f32 trio), so without this kernel the mint's cheapest operand runs as SIX launches per KDA layer on 34 of 45 layers, with six redundant `quantize_q8_1` calls on the same `x`; this makes it one launch and one quantize. Same block-offset split as the pair/triple, extended to six unequal ranges; all six share `in_f` and one `row_bytes` (an e4m3 row is `in_f` bytes) and each range keeps its OWN per-tensor weight scale, because in this class the scale is a per-tensor property. Per (range,row) the body is `e4m3_mmvq_row1` => **BIT-IDENTICAL** to six separate m=1 launches. m=1 only; t>1 keeps the caller's arm. | per-tensor e4m3 weights, shared q8_1 activation, f32 acc/out | — | `MEMRA_KDA_FUSED_PROJ=1` (existing door, no new flag) plus `MEMRA_ST_E4M3` residency; declines under `MEMRA_FAST=0`, no MMVQ support for the qtype, or the `MEMRA_E4M3_DUAL=0` rollback. `Engine::e4m3_fused6_into` / `qmatvec_e4m3_fused6_raw` (lib.rs), armed in `kda_proj_fused6_pre` (kda.rs); counter `KDA_FUSED6_E4M3_DISPATCHES`; gate `tests/kda_fused6_e4m3_gpu.rs` (bit identity + two red arms) | fatbin/by-name |
| `qmatvec_e4m3_mmvq_b{2,4,8}_r2` (lane/e4m3-batch-r2-20260905) | The batched F8-E4M3 tier at TWO output rows per warp (`e4m3_mmvq_batched_r2<MCOLS>`): per k32 block the warp decodes both rows' weights, loads the activation block once per column and converts each int8 once, then runs both rows' fmaf chains on the shared values; per (row, column) the chain is `e4m3_mmvq_batched_row`'s verbatim (same decode, same j order, same `ad` scale, same `warp_reduce_sum`). Halves activation L1 bytes and I2F per output row; rows_per_block doubles (r2-class convention). Bit-identical to the shipped batched kernels by construction. Door `MEMRA_E4M3_BATCH_R2` (default OFF). Gate: `e4m3_batch_r2_gpu::gpu_e4m3_batch_r2_matches_one_row_per_warp_bitwise`. |
| `qmatvec_q8_0_mmvq_b{2,4,8}_r2` + `_r2_rp` (lane/q80-batch-r2-20260905) | The batched Q8_0 tier at TWO output rows per warp (`q8_0_mmvq_batched_r2<MCOLS>`, and `_r2_rp` over the split planes): per k32 block the warp decodes both rows' weight ints, loads the activation block once per column and runs both rows' dp4a chains on it; per (row, column) the chain is `q8_0_mmvq_batched_row` / `_row_rp`'s verbatim (same ints, same k order, same `dw * ad * sumi` accumulate, same `warp_reduce_sum`). Halves activation L1 bytes per output row; rows_per_block doubles (r2-class convention). Bit-identical to the shipped batched kernels by construction. Door `MEMRA_Q80_BATCH_R2` (default OFF). Gate: `q80_batch_r2_gpu::gpu_q80_batch_r2_matches_one_row_per_warp_bitwise`. |
| `qmatvec_kda6_q8f32_rp_v2_ilp`, `qmatvec_q8_0_mmvq_rp_v2_ilp` (lane/glm5-q8-row-ilp-20260904) | `MEMRA_Q8_ROW_ILP` twins of the W8 q8_0 row walk: four blocks' loads per lane issued ahead of the dp4a chains (`q8_0_mmvq_row1_rp_v2_ilp`), then the shipped two-deep round and tail; same per-lane accumulation order, warp tree verbatim, bit-identical by construction; the kda6 kernel and its twin are one templated body. Why: 1.8 ms of a plain glm5_next token on 2x B200 with long-scoreboard stalls at 71-76% (rig ncu) | q8_0 rp4 mirrors, staged q8_1 activation | same PDL guard | `MEMRA_Q8_ROW_ILP` (default OFF) | fatbin/by-name |
| `moe_gate_up_preclamp8_q8_v2`, `moe_down8_fma_q8_v2` (lane/b200-gemv-hbm-20260902) | v2 twins of the plain-decode glm5_next NVFP4 W4A16 expert pair (12% and 11% of the HBM3e wall in the census with `_w4` on). gate/up: 8 warps/block on `threadIdx.y` plus a `g` walk unrolled by two so BOTH groups' weight, scale and activation loads issue before either dp4a chain runs (603 LDG vs the `_w4` twin's 201) — the 36 B NVFP4 block layout leaves quant bytes only 4 B aligned, so depth is the only lever, not wider loads. down: ONE BLOCK per output row with warp `j` owning expert slot `j`, taking the launch from `out_f` warps wide (0.43 of a full-occupancy B200 wave) to `out_f * n_used`, with the slot-ordered `__fmaf_rn` chain still run by one thread in ascending `j`. Both BIT-IDENTICAL per output. | NVFP4 (and every `expert_dot_g` qtype) weights, q8_1 activation | sm_100a builds (door-gated) **NOT DISPATCHED** (box receipt 2026-09-02): gate/up 54.4 us vs `_w4`'s 53.3 (1.011x, inside noise, exactly as its 18 -> 19 LDG-burst receipt predicted) and down 43.2 vs 36.0 (**0.870x, a regression** — eight warps per block contend for the same L2 sectors and the shipped single warp was already covering the latency it was accused of exposing). Both bit-identical, so this is a speed verdict. `moe_fused_epi_launch` keeps the shipped/`_w4` pair unconditionally; these kernels stay in the fatbin and in `b200_matvec_bench` as measured arms | fatbin/by-name |
| `encode_q8_0_rows_from_bf16`, `qmatvec_q8_0_rows_t`/`_tw`/`_tw32`, `qmatvec_mmvq` (family, all pre-existing) | NEW DISPATCH SITE ONLY (lane/b200-glm5-w8-20260902, no kernel code added): `MEMRA_GLM5_W8` reuses the SAME pointer-keyed q8_0 mirror building block `MEMRA_STEP_TP_W8`'s hybrid half already calls (`matvec_bf16_via_q8_mirror`/`_t`, lib.rs) — encode a resident bf16 slab to its q8_0 twin on first decode use, then route the glm5_next KDA/MLA projection matvec through the existing mmvq/rows kernels instead of `matvec_bf16_f32acc_x4_rows`. Independent door from `MEMRA_STEP_TP_W8`; same `w8_mirrors`/`w8_act` caches, shared and idempotent. | bf16 weights -> q8_0 mirror, q8_1 activation | — | `MEMRA_GLM5_W8` (default OFF) — dispatched in `matvec_bf16_rows_into` and `matmul_rows_exact` (lib.rs); composes with `MEMRA_KDA_FUSED_PROJ`'s bf16 arm, which declines when this door is on | fatbin/by-name |
| `matvec_bf16_f32acc_x4_range` (lane/hy3-device-token-20260901) | t=1 BF16 row-parallel partial: every rank computes all output rows over one aligned global K range from a compact local activation shard. BF16 expansion, per-thread multiply/add order, and `red[256]` tree match the standing x4 row kernel inside the range; the cross-range add is an explicit TP numeric class. Produces one full-width f32 partial for `Tp2ReplicatedRowJoin`, with no allocation or host fence in either primitive. | bf16 weights, f32 acc | — | gate-only until the composed shared/dense FFN schedule has real-artifact correctness and sampled-serving profile receipts | fatbin/by-name |
| `matvec_bf16_f32acc_x4_tcols16` (lane/glm5-matvec 2026-08-31) | wide-t twin of `matvec_bf16_f32acc_x4_tcols` for t=9..=16 (the DFlash2 drafter block head: nd=15 rows over the target's 1.269 GB lm head, which the t<=8 tcols launcher refuses — the ship census caught the `_rows` fallback re-reading the head 15x/round, 11.7% of capture GPU). SEPARATE kernel so the priced t<=8 class keeps its acc[8] footprint/SASS (the `_tw32` acc-sizing lesson); body otherwise verbatim, bit-identical per (row,token) to the t=1 program | bf16 weights, f32 acc | — | `MEMRA_BF16_TCOLS_WIDE` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam) — `glm5_matvec_doors_gpu` bit-gate + shifted-row red | fatbin/by-name |
| `matvec_bf16_f32acc_x1_tcols` (lane/glm5-matvec 2026-08-31) | one-row-per-block grid twin of the tcols kernel (grid.x = out_f, p-loop dropped): the trunk kda grids (512..2048 blocks) are ~one resident wave and phase-lock their bit-pinned tree reduces (census 1.05 TB/s = 59% of peak vs the SAME kernel at 1.43 TB/s on the head's 38720-block grid); per-row body + red[256] tree verbatim, bit-identical | bf16 weights, f32 acc | — | `MEMRA_BF16_TCOLS_X1` (default ON since the 2026-08-31 mv-battery flip, `=0` = rollback seam) — `glm5_matvec_doors_gpu` x1-vs-x4 bit-gate + swapped-weight-row red | fatbin/by-name |
| `matvec_bf16_f32acc_x1_tcols_rf`, `..._x4_tcols_rf`, `..._x4_tcols16_rf` (lane/glm5-door-r 2026-08-31) | door R fused-reduce-tail twins of the three tcols kernels (designed+sized moe-loc LANE.md §2.2): the standing tail runs t separate strided trees = 9 block-wide barriers per token column (~30 at t=3.34, 135 at the drafter head's t=15) against a 4-trip main loop — barrier/tail-bound at 67.0% of peak on the kda trunk after door X. The `_rf` tail shares ONE barrier sequence across the t columns (`red[t*blockDim]` dynamic shared) and runs levels s<=16 as an intra-warp `__shfl_down_sync` chain at the IDENTICAL pairing and operand order — 9t -> 3 barriers/block (x1 form, 128 threads); main loop verbatim, bit-identical per (row, token). Requires power-of-two blockDim (launcher-enforced) | bf16 weights, f32 acc | — | `MEMRA_BF16_TCOLS_RED_FUSED` (default OFF — no box receipt yet, predicted -1.0 to -2.0 ms/round, box prices the flip) — `glm5_matvec_doors_gpu` bit-gate t=1..=16 both grid forms + shifted-pairing red | fatbin/by-name |
| `matvec_bf16_f32acc_x1_tcols_rf_redshift` (lane/glm5-door-r 2026-08-31) | GATE-ONLY shifted-pairing RED twin of `..._x1_tcols_rf`: the warp phase runs ASCENDING shuffle offsets (1,2,4,8,16) — the same 32 partials under a different association, so f32 rounding must move and the door-R bit bar must see it. Never dispatched by any route; launched only through `Engine::matvec_bf16_tcols_gate_kernel_into` (an allowlist, not a name proxy) by `glm5_matvec_doors_gpu` | bf16 weights, f32 acc | — | none (gate harness only) | fatbin/by-name |

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
| `memra_router_fused_f32` / `_dexp` | router GEMV + last-block sigmoid top-k, one launch | MEMRA_ROUTER_FUSED (default OFF; gate `tests/router_fused_gpu.rs`) |
| `moe_router_topk_scaled_f32` | scaled top-k | MEMRA_ROUTER_KERNEL / _PREFILL_EXACT (lib.rs:202-205) |
| `nvfp4_ep_stage_inputs`, `moe_sel_w_mirror` | one-launch peer-read input BF16 staging and grid-capable route-id/weight mirroring for automatic EP | MEMRA_PARALLEL_EP_DEVICE_ROUTER; MEMRA_PARALLEL_EP_Q8_ACT |
| `silu_mul_scaled_host_expf_{bf16,q8}_ep_slots` | owner-range SwiGLU using the frozen host-expf transcription; exact BF16 or optional q8_1 emission | MEMRA_PARALLEL_EP_DEVICE_ROUTER; MEMRA_PARALLEL_EP_Q8_ACT |

### cu/spec_sample.cu — 30 symbols (fatbin `MEMRA_SAMPLE_FATBIN`)

Header: "Sampled speculative decoding — device sampling primitives; Philox4x32-10
counter-based" (spec_sample.cu:1-5).

| symbol (family) | purpose | dispatch flag |
|---|---|---|
| `gumbel_perturb_*`, `softmax_gather_*`, `residual_sample_*`, `filter_stats_f32`, `scatter_trim_logits_*`, `penalize_logits_*` (including heterogeneous sparse serving rows), `mask_logits_f32`, `memra_sctr_inc` | Gumbel-max sampling, top-k/p filtering, penalties, residual (rejection) sampler | sampling chain via MEMRA_TEMP/TOP_K/TOP_P/MIN_P/PENALTY_* plus batched serving `MEMRA_SERVE_DEVPENALTY` |
| `spec_accept_greedy_dc`, `spec_seed_gather`, `spec_rollback_stream`, `spec_assemble_verify`, `spec_ring_commit`, `spec_adapt_k`, `plain_tok_ring`, misc int copies | spec-decode accept/rollback/fork machinery | MEMRA_SPEC_* family (MEMRA_SPEC_DUAL_T lib.rs; MEMRA_SPEC_DFLASH FLAGS.md:25) |

ROOFLINE PROBES (lane/b200-roofline-recalibrate, 2026-09-06, BENCH-ONLY, no serving path
reaches them): `memra_bw_read` (`memra_bw_read_scalar_kernel<ILP>` / `memra_bw_read_v4_kernel<ILP>`)
and `memra_bw_read_slabs` (`memra_bw_read_slabs_kernel`). Read-only streamers whose accumulator
escapes through a comparison that cannot hold, so the loads stay live with no write in the timed
loop. They exist because the pair's quoted 4062 GB/s "measured wall" was taken with
`dsv4_hc_collapse_kernel` standing in as a streamer (`MEMRA_HC_BW_PROBE`): one thread per output
element, no grid-stride reuse, four SCALAR 32-bit loads per thread, so four times the load
instructions a 128-bit access issues for the same bytes. The `mode=0` arm reproduces that access
shape so the recalibration is anchored to the number it replaces; `mode=1` is the same bytes
through `float4`; the slab launcher reads N disjoint slabs at caller-given offsets, defaulting to
a whole token's expert read (45 x 9 x 14 MiB = 5.5 GiB, sized past B200's 126 MB L2 on purpose).
Driver: `bw-roofline` (`src/bin/bw_roofline.rs`), no flag, no default.

`memra_dsv4_hc_pre_v4_norm_zq8` (`dsv4_hc_pre_v4_norm_zq8_body` + `dsv4_hc_pre_v4_e16_norm_zq8_kernel`,
cu/dsv4_gpu.cu, lane/hc-pre-norm-fuse-20260906, `MEMRA_HC_PRE_NORM_FUSE`): `memra_dsv4_hc_pre_v4`
with `rms_norm_zq8_f32`'s epilogue folded in, deleting that launch (~3.9 us x ~79 per token). The
first instance of the fusion doctrine the decode census points at: 39% of the token sits in kernels
that move almost no bytes, ncu says hc-pre cannot be fixed from the inside (4 active warps, 0.15
eligible, 88.4% of cycles with nothing to issue, grid 1 on a 148-SM part), so the lever is deleting
the neighbouring launch rather than speeding it up. BIT-IDENTICAL on an index coincidence: at BLOCK
1024 and d 4096 both passes of `rms_norm_zq8_f32` own exactly the four elements the hc-pre combine
tail already holds in registers, in the same order, so the reduction tree and every expression
replay unchanged; the launcher returns 40026 at any other width rather than reordering quietly.
Gate: `tests/hc_pre_norm_fuse_gpu.rs` (bitwise on z, q8 codes, q8 scales and y against the two-launch
pair, plus a red arm on the norm weight).

### cu/tp_ar.cu — TP decode all-reduce (static-lib TU, default flags)

`memra_tp_ar_push` (`memra_tp_ar_push_kernel`) and `memra_tp_ar_fold` (`memra_tp_ar_fold_kernel`),
`float4` grid-stride throughout. The two-rank decode join: each rank pushes its partial straight
into the peer's staging buffer over NVLink (peer access from `tp::grant_peer_access`, so the peer
pointer is dereferenceable from a kernel in this context) and each rank folds the buffer its peer
wrote. Ordering is one cross-stream event per direction, the contract
`tp_transport::PeerPullLink::publish` already uses; no host boundary, no `synchronize`, and
nothing occupying the device while it waits. EXISTS BECAUSE the TP-2 join costs about 500 us
today: `tp_transport`'s default `host-canonical` bounces every hop through host and
`Engine::dtoh` drains the stream, so each of the ~90 joins per token waits for that layer's
compute twice, which is the whole reason TP-2 measured 3.1x slower than the pipeline split it
would replace. NEW `memra_tp_ar_1stage` (`memra_tp_ar_1stage_kernel` + `memra_ar_barrier`, lane/tp-ar-1stage-20260906): the ONE-SHOT arm, and the shape the push-then-fold pipeline above should have been. Taken from vLLM's `cross_device_reduce_1stage` and its `barrier_at_start`/`barrier_at_end` pair: ONE kernel per rank and NO CUDA events, synchronising through flags written into the peer's memory, reading BOTH ranks' inputs directly (peer access makes the peer pointer dereferenceable) and computing the whole sum locally. No staging buffer, no copy, nothing for the host to do per reduce beyond two launches, against the pipeline's 4 launches and 8 cross-context event operations that measured 20-26 us for 16 KB on a 956 GB/s fabric. Operands are indexed by GLOBAL RANK so every rank evaluates the same expression and the result is bitwise identical across ranks by construction. Two alternating counter sets because a peer block can reach the second barrier while this block still spins on the first. The spin is BOUNDED where vLLM's is not, writing a refusal word (40043 entry, 40044 exit) rather than hanging the card. Consumers: `tp_ar::ArLink`, whose `all_reduce`, `broadcast` and `all_gather` are the same push under different offsets, and `all_reduce_1stage` (driver `tp-ar-bench`). The movement pair matters more than the reduce for the walk as it stands: the glm5 TP MLA layer moves its bytes in three PURE-MOVEMENT hops (fan out `h` and the positions, all-gather the head parts, concat the column-parallel `wo` parts onto root), which is why the current arm is byte-identical to the unsharded walk by construction, so swapping only the transport keeps that property exactly. Gate:
`tests/tp_ar_gpu.rs` (bitwise against the host sum at n 1 / 4 / 1024 / 4096 / 16384 / 65536, a
repeat-call case, and a red arm where a fold with no peer push must not reproduce the sum).
REQUIRES TWO DEVICES and skips otherwise: unlike every other TP arm, whose bytes go through host
or `cudaMemcpyPeer`, this one dereferences the peer's pointer inside a kernel, and two contexts on
one card share no address space and cannot grant each other peer access
(`cudaDeviceCanAccessPeer(d, d)` is false). On the rig that read correctly at some sizes and
returned the local partial at others, so `ArLink::new` refuses two engines on one ordinal rather
than let a gate pass on an accident. The sweep reaches 256 KiB deliberately: the first cut spun on
a peer-armed flag, which starves the peer wherever they share a device, and was bitwise-correct
below that size and wrong at it.

### cu/dsv4_gpu.cu — mHC (hyper-connections) glue kernels, partial inventory (static-lib TU, not a fatbin; compiled `-fmad=false` for bit-parity with the CPU oracle — build.rs:505-510)

This ~3,700-line file (`memra_dsv4_*` FFI namespace) carries the dsv4/glm5_next dense +
MoE decode/verify machinery end to end (embed, RoPE, indexer, sink attention, MoE
route/combine, per-qtype GEMV arms, act-quant, and more) and is shared by BOTH the native
dsv4 model class (`dsv4_gpu.rs`) and the glm5_next hyper-connections trunk (`hyper.rs`,
via `dsv4_ffi`). Only the mHC (manifold-constrained hyper-connections) pre/post glue
family is inventoried below (research/b200-sinkhorn-fusion-20260902); the remaining
~130 symbols in this TU are not yet in this table.

| symbol | purpose | dispatch flag | FFI binding |
|---|---|---|---|
| `dsv4_rowsq_scale_kernel` (:975) | mHC site pre-chain stage 1: rescale the site's raw mixing coefficients by `rsqrt(mean(x^2) + eps)` over the `[streams, hidden]` slab (block per token, f64 8-wide accumulate at pinned blockDim=128) | always, unless subsumed by `MEMRA_HC_FUSED_PRE` | dsv4_ffi.rs |
| `dsv4_hc_sinkhorn_kernel` (:1395) / `dsv4_hc_sinkhorn_m_kernel` (:2963) | mHC site pre-chain stage 2: Sinkhorn-normalize the rescaled mixes into `pre`/`post`/`comb` gates (single-position / one-block-per-token batched twins; iters serial in one launch) | always, unless subsumed by `MEMRA_HC_FUSED_PRE` | dsv4_ffi.rs |
| `dsv4_hc_collapse_kernel` (:1025) | mHC site pre-chain stage 3: collapse the `streams` state into the one branch input `y[t,:] = sum_c pre[t,c]*x[t,c,:]` | always, unless subsumed by `MEMRA_HC_FUSED_PRE` | dsv4_ffi.rs |
| `dsv4_hc_pre_fused_kernel` (:3080) | `rowsq_scale` + `hc_sinkhorn_m` + `hc_collapse` as ONE launch per (site, token); BIT-IDENTICAL to the three-kernel chain by construction (verbatim per-stage bodies, shared-memory operand staging, bit-preserving Sinkhorn stationarity exit) | `MEMRA_HC_FUSED_PRE` (default OFF; lane/glm5-decode-diet 2026-08-31) | dsv4_ffi.rs |
| `dsv4_hc_pre_fused_v2_kernel` (:3262) | same three stages as `dsv4_hc_pre_fused_kernel`, stage 1 and stage 3 VERBATIM; stage 2 (Sinkhorn) runs warp-0-only with `__syncwarp()` in place of `__syncthreads()` when hc<=4 (a synchronization-primitive substitution only — BIT-IDENTICAL to `=1` and the unfused chain by construction), falling back to `dsv4_hc_pre_fused_kernel` internally for hc>4 | `MEMRA_HC_FUSED_PRE=2` (default OFF; lane/b200-sinkhorn-fusion-20260902) | dsv4_ffi.rs |
| `dsv4_hc_post_kernel` (:1053) | mHC post: `out[t,k,:] = post[t,k]*f[t,:] + sum_j comb[t,j,k]*residual[t,j,:]`. `f` is the SITE'S OWN attention or FFN branch output — a separate multi-kernel program (QKV/RoPE/attention core, or MoE/FFN) runs between the pre-chain's collapse write and this kernel's read, which is why no launch fuses this kernel with the pre-chain (research/b200-sinkhorn-fusion-20260902/LANE.md) | always | dsv4_ffi.rs |
| `dsv4_hc_head_pre_kernel` (:1453) / `dsv4_hc_head_pre_m_kernel` (:3026) | dsv4's `HcCollapse::GatedHead` trunk-exit gate (sigmoid-gated pre-only collapse), single-position / batched twins — distinct from glm5_next's `Mean` exit (`dsv4_hc_mean_kernel`) | `HcCollapse::GatedHead` plans only | dsv4_ffi.rs |

Selected-expert FP4 projection (same TU, `-fmad=false`):

| symbol | purpose | dispatch flag | FFI binding |
|---|---|---|---|
| `dsv4_fp4_gemm_sel_kernel` | NVFP4 group-16/E4M3+F32 and MXFP4 group-32/E8M0 weights with FP8 activations; the existing 128-leaf block reduction remains selected by DSV4. | The per-model FP4_REDUCE door and dedicated gate were removed after flat full-model measurements. | `memra_dsv4_fp4_gemm_sel_g_arm`; historical result: `research/dsv4f-2card-1m-20260904/fp4-warp-reduce.md`. |

DSV4 indexer candidate (same TU, separate multiply/add rounding):

| symbol | purpose | dispatch flag | FFI binding |
|---|---|---|---|
| `dsv4_indexer_score_tiled_kernel` | 64-head, width-128 scorer; one thread owns a candidate and all heads, 128 candidates reuse 16-element query/key slabs. Ascending dimension and head sums use explicit separate round-to-nearest multiply/add, unlike the FMA-based MLA scorer. Absolute-position and fixed-limit masks share the same launch. | `MEMRA_DSV4_INDEXER_SCORE=scalar/tiled`, default scalar. Tiled is unqualified and opt-in; device f32x only. | `memra_dsv4_indexer_score_tiled`; `tools/dsv4-indexer-tiled-gate.cu` anchors to scalar CUDA and CPU witnesses, masks, write guards and a corruption control. Target-card component and one-load full-model parity pass; long-context serving remains unqualified. |

Active-C4 residency experiment: `dsv4_c4_gather_kernel` /
`memra_dsv4_c4_gather` copies selected logical rows from pinned host C4 or device
SWA/transient storage into one bounded per-stage attention buffer. It preserves
index order, pads and all f32 bits, with no quantization or arithmetic rewrite.
The explicit offload/direct-allocation APIs and default-OFF
`MEMRA_DSV4_C4_HOST_MB` serving door select it. Target model/snapshot/rollback
and short HTTP cold/warm/refusal gates pass; long HTTP and concurrency remain
pending. The host-store owner drains its stream before CPU reads or
deallocation. C128 and indexer keys remain device-resident.

Recent-C4 sidecar experiment: `dsv4_c4_recent_write_kernel` /
`memra_dsv4_c4_recent_write` mirrors the bounded suffix of GPU emissions or
resets/seeds the cache from canonical pinned history. `dsv4_c4_gather_recent_kernel`
/ `memra_dsv4_c4_gather_recent` uses an exact absolute-row tag match before a
GPU hit and otherwise preserves the original host read. Values, masks and
selection order are unchanged; future-write/rollback collisions cannot alias
an older live row. Existing APIs request zero rows. The new explicit recent
allocation/restore APIs charge values plus tags to GPU cache bytes; no serving
default or environment flag enables them. Bindings: `dsv4_ffi.rs`, `dsv4_c4.rs`.

Gate-only C4 host-copy elision (`set_c4_host_copy_elision_for_gate`) skips the
per-emission D2H publication when a complete compressed-history recent sidecar
covers the bounded state capacity. Prime/snapshot/restore stay on canonical host
publication; the profile arm enables elision only after restore, and snapshots
refuse while it is enabled. The 8192-token composition capture removed 168 C4
D2H calls and preserved the frozen sampled stream, but the remaining sampled
logit D2H calls still dominate API time. No serving or graph default is implied.
Compiled component/model/performance gates and remaining target qualification:
`research/dsv4f-2card-1m-20260904/c4-recent.md`.

Narrow verification selector: `dsv4_topk_idx_numeric_kernel` /
`memra_dsv4_topk_idx_numeric` uses the shared bitonic body with numerical-zero
normalization, matching the CPU descending-score/ascending-index comparator
for non-NaN inputs. The old `dsv4_topk_idx_kernel` specialization retains its
raw-bit zero ordering and ABI. `MEMRA_DSV4_VERIFY_TOPK=device` is default-OFF
and removes the short verification score D2H/sort/H2D boundary. The 45-cell
CUDA gate covers ties, both zero signs, infinities, power-of-two boundaries,
input preservation and output guards. The composed C4/full-model gate passes;
performance remains pending. This selector is a graph-capture prerequisite.

The gate-only `dsv4_topk_idx_radix_m1_kernel` /
`memra_dsv4_topk_idx_radix_m1` is a separate exact candidate for plain t=1,
`K=512`, `2048<=N<=4096`: three MSD bytes cut the exact order-key prefix, then the
retained prefix uses the same integer bitonic order. It normalizes signed zeros
and preserves ascending original-index ties, with an all-equal primary-key
index-order fast path. Scratch is persistent in `VerifyWs`; no host score/index
copy is introduced. `set_index_topk_radix_for_gate` is process-local and
default-OFF, and `index_topk_radix_dispatches` counts successful accepted
launches. N>4096 and all non-t=1/K!=512/batched paths retain their existing
selectors. This is an attribution candidate, not a production qualification.

Whole-expert EP: `dsv4_fp4_gemm_sel_kernel<WarpReduce,true>` /
`memra_dsv4_fp4_gemm_sel_ep` applies an ownership mask before the existing GEMV
arithmetic, addresses code/scale slabs with local ids and keeps global macro-scale
ids. Existing non-partitioned callers keep the false specialization.
`dsv4_ep_merge_slots_kernel` / `memra_dsv4_ep_merge_slots` copies peer-owned
contribution bits into original slots, without a reassociated partial sum.
`MEMRA_DSV4_EP=pair` defaults OFF. The local gate compares all three projections,
both reductions, 1/6/32 rows and real 4096/2048 dimensions against the original
full-bank launcher. Full-model and serving/performance gates are pending.

Gate-only dense wo_a grouping: `dsv4_gemv_fp8_m_kernel<1,true>` shares the
original FP8 GEMV dot/reduction body, using global grouped weight/scale rows
and separate activation/output strides. `memra_dsv4_gemv_fp8_grouped_m1`
replaces eight t=1 FP8 wo_a launches with one; BF16, prefill and wider verify
rows keep the old loop. The process-local grouped gate defaults OFF and
counts successful submissions. The ignored
`cuda_gemv_fp8_grouped_m1_matches_eight_slices_and_counts_one_enqueue` test
compares the real 8x1024x4096 shape and padded two-group case bitwise and
checks invalid-stride refusals. Full-model gate: `dsv4_plain_perf_gate wo-a`,
holding half2 ON in both arms. Both-device memcheck and all 28 full-model
token/logit/KV rows pass. Measured +1.897%/+1.491% at 256/8192; serving
qualification remains separate. Record:
`research/dsv4f-2card-1m-20260904/wo-a-20260907.md`.

`dsv4_graph::capture_layer` retains an explicitly armed graph and executes the
recorded operations once, with event tracking disabled before allocation.
Window-only layers, the head and the embedding prefix have separate probe
APIs. Normal decode initializes all those probes disabled. Whole-layer EP
capture remains refused. The stateless HC/Q/KV prefix performance candidate was
removed after its 2026-09-07 exact 84-row comparison measured a regression
(`research/dsv4f-2card-1m-20260904/plain-fronts-20260907.md`). The later rank-local
grouped expert-island candidate was also removed: all 86 entries / 860 kernels
engaged, tokens/final logits/KV matched, and there were zero stale fallbacks,
but plain rate changed -0.46%/+0.15% at 256/8192. Its device route/input mirror/
GU/down boundary did not remove the dominant cost. Record:
`research/dsv4f-2card-1m-20260904/expert-half2-20260907.md`.
No graph serving default is enabled.

Corrected DSV4 grouped-prefill experiment:

`MEMRA_DSV4_MOE_PROGRAM=matrix` is the separate default-OFF whole-request
experiment: the existing grouped CUDA kernels run through one native batched
executor for prefill, one-row decode and speculative verification, including
the initial token. No new CUDA arithmetic is introduced by this dispatch.
Program-tagged state refuses matrix/reference crossings. Full-layer capture
still refuses the host-validation path; component graph evidence is not a
whole-model graph receipt. `dsv4_matrix_program_gate` passes on the PRO pair for
phase/state/storage and sampled plain/spec identity. Quality and performance
remain separate: `dsv4_matrix_distribution_gate` characterizes fixed
teacher-forced source windows and does not grant admission. Receipts:
`research/dsv4f-2card-1m-20260904/matrix-request-program.md` and
`research/dsv4f-2card-1m-20260904/matrix-distribution.md`.

The matrix workspace now retains both checked FP8/half mirrors, their scale and
status buffers, and the CSR contribution plane. Seven explicit GPU scratch
allocations per routed MoE call are removed; the existing quantization and MMA
kernels are unchanged. `GroupedWork::bytes` charges the full resident workspace,
including `slots * (6 * hidden + 2 * inter + 16)` bytes beyond route metadata.
The fresh-storage gate control compares zero-initialized and reused buffers;
local GPU tests cover changing live row counts, stable pointers, invalid/stale
status and actual allocation-byte accounting. This is not a speedup receipt.

The opt-in `MEMRA_DSV4_GROUPED_ROUTE=device` arm adds
`dsv4_grouped_{count,prefix,scatter}_kernel`, bound by
`memra_dsv4_grouped_routes`. It preserves stable expert/slot order and copies
route weights and all three outer scales without arithmetic. Empty expert
groups remain explicit. Per-workspace metadata bytes are charged at allocation.
The ModelOpt-only null-host-offset ABI of `memra_moe_kq_gemm_sk` derives real
tile counts from each visitor's device prefix; existing host-count callers are
unchanged. The normal correctness arm keeps scalar invalid-id/live-count and
FP8/half checks host-visible. A gate-only route-validation-off arm keeps the
device prefix authoritative, clears compacted tail rows and skips those D2H
checks/synchronizes; it is not a production or graph-admission claim. Component gate:
`tools/dsv4-grouped-route-gate.cu`; full-model gate:
`dsv4_wide_prefill_gate` host/device/host routes at widths 32/128/512.

The native matrix/EP preparation component adds
`memra_dsv4_grouped_routes_partition` (`dsv4_ffi.rs`), using the partitioned
count/scatter specializations plus `dsv4_grouped_partition_prefix_kernel`.
Router ids and all three macro scales remain global; output group ids index a
partition-local expert table. `offsets[local_expert_count]` gives the live row
count, including zero. Slot arrays beyond that prefix are untouched and must
not be consumed. Invalid global ids reject even when they are outside the
owned range. Rust `GroupedRoutes::new_partition` retains a synchronized
status/live-count check; this is not a full-MoE graph or overlap implementation.
The full-bank caller continues to use the original route entry and requires
all selected slots to remain present. Matrix plus EP now has an experimental
staged executor: prepare both partitions, queue both gate/up chains, then
validate intermediate mirrors and queue down/return. It preserves source FP8
codes/scales on the wire and overwrites original slots without reassociated
reductions. Complete local/peer pointer tables and partition ownership are
checked; graph capture remains refused. Both existing feature defaults remain
OFF after the bounded full-model composition gate passed; quality, serving and
performance admission remain separate. Component:
`tools/dsv4-grouped-ep-route-gate.cu`;
receipt: `research/dsv4f-2card-1m-20260904/grouped-ep-routing.md`.
Full execution contract and candidate pins:
`research/dsv4f-2card-1m-20260904/matrix-ep-chain.md`.

| symbol | purpose | dispatch flag | FFI binding |
|---|---|---|---|
| `dsv4_fp8_gather_half_kernel` | Reorders the existing FP8-QAT codes and per-128 scales into a half matrix with a power-of-two row scale. Every value is round-tripped exactly; a row-status vector rejects nonrepresentable/NaN values before grouped GEMM. | `MEMRA_DSV4_PREFILL_MOE=grouped`, default OFF | `memra_dsv4_fp8_gather_half`; gate `tools/dsv4-fp8-half-mirror-gate.cu` covers duplicate row mapping, finite values, tails and underflow/NaN refusals. |
| `moe_kq_sk{32,128,tail}v_kernel<QT_NVFP4_MODELOPT>` | Reads consecutive E2M1 codes and separate signed-E4M3/16 scales from six pointer planes, with FP32 macro weight scale applied after projection. No GGUF or duplicate weight bank. Grouped MMA changes reduction order; routed slot restoration, combine and the entire shared expert remain explicit common work. | `MEMRA_DSV4_PREFILL_MOE=grouped`, default OFF, mode-2 visitor/direct loader required | `memra_moe_kq_gemm_sk`; actual-model `dsv4_grouped_prefill_gate` verifies total=routed+shared and characterizes forced-path logits. No production qualification yet. |
| `moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT>` | DSV4 matrix plain-decode `m_e=1` gate/up pair: one shared FP8-QAT-mirrored f16 A tile, two unchanged ModelOpt NVFP4 f32-MMA accumulators, then exact macro/clamp/SiLU/route-weight epilogue into the intermediate H row. Down, FP8 intermediate quantization, macro2 and original-slot scatter remain common. | `MEMRA_F16G_GU_FUSE=1`, default OFF; ModelOpt qtype 108, one-row transaction, deep tail only | `memra_moe_kq_gemm_sk_gu`; component gate must compare H/FP8 codes/full routed output bitwise against the shipped two-projection path. No target timing receipt yet. |
| `moe_kq_sktail_kernel<QT_NVFP4_MODELOPT,true>` | DSV4 gate-only m_e=1 tensor-core down candidate: same B tile and valid-row m16n8k16 chain as the shipped deep tail, with invalid-row warps and duplicate A-stage loads elided. Existing FP8 mirror, macro2 and scatter remain the comparison path; this is not the removed scalar visitor. | `MEMRA_F16G_M1_TC=1` or gate setter, default OFF; one-row/deep-tail candidate only | `memra_moe_kq_gemm_sk_m1`; full-model identity/sanitizer/rate gates required before dispatch. |
| `moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT,true>` | Gate-only GU m_e=1 specialization: skips the duplicate A row tile and invalid-row MMA warps while retaining valid-row gate/up accumulation and the fused epilogue. Both EP workspaces use the shared dispatch. | `set_moe_f16g_gu_m1_tc_for_gate`, default OFF; no environment or serving flag | `memra_moe_kq_gemm_sk_gu_m1`; actual launch counter plus `cuda_gu_m1_matches_gu_reference` and plain ABBA gate. |
| `moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT,false,true>`, `moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT,true,true>` and `moe_kq_sktail_kernel<QT_NVFP4_MODELOPT,true,true>` | Packed ModelOpt GU/down stores retain MMA order, using a 256-entry half2 LUT and 1024 extra shared bytes. Existing half2 conversion, full-chain and sampled model identity gates pass; the half2-only model gain is +0.646%/+1.014% at 256/8192. The additional GU M1+half2 conjunction selects `<108,true,true>` for the existing one-token GU visitor; batched groups stay unchanged. Component R7 is 4–7% faster with H bit identity; sampled model composition is +1.2365%/+1.1242% at 256/8192 with full token/logit/KV identity. Receipt: `research/dsv4f-2card-1m-20260904/gu-m1-half2-model-20260907.md`; no serving default. | Existing process-local GU-M1/GU-half2/down-half2 setters, default OFF; no new environment flag. Clearing overrides restores the corresponding prior path. | `memra_moe_kq_gemm_sk_gu_half2`, `memra_moe_kq_gemm_sk_gu_m1_half2`, `memra_moe_kq_gemm_sk_m1_half2`; one combined GU enqueue advances the Rust GU-M1 and CUDA GU-half2 counters once each, not a third receipt. `cuda_half2_chain_identity` and the sampled plain gate retain arithmetic/dispatch coverage. |

DSV4 prefill work-elision dispatch (no new CUDA arithmetic):

The experimental sink-score tile (`dsv4_sink_scores_tiled_f32acc_kernel`) is
bound by `memra_dsv4_sink_scores_tiled_f32acc` and composed with the unchanged
softmax/output kernels in `memra_dsv4_sink_attn_dec_mq_f32acc_tiled`.
`memra_dsv4_sink_scores_tiled_init` checks/configures 84096 bytes of dynamic
shared memory before capture. It tiles 8 heads x 32 selected keys, retains the
ordered 512-dimensional FP32 dot and negative-index mask, and writes the same
score layout. Rust FFI: `dsv4_ffi.rs`; dispatch: `MEMRA_DSV4_SINK_SCORE`, default
scalar. Component gate: `tools/dsv4-sink-score-tiled-gate.cu`; full model:
`dsv4_sink_score_gate`. Receipt: `research/dsv4f-2card-1m-20260904/sink-score-tiled.md`.

| dispatch | purpose | flag | gate |
|---|---|---|---|
| `verify_batch_dev_output` -> existing `head_logits_dev` | Preserve every trunk layer and cache transaction, discard unused intermediate head work, and compute the final row with existing single-row head kernels. Public verification still returns all requested rows/argmaxes. | `MEMRA_DSV4_PREFILL_HEAD=all/last`, default all | `dsv4_prefill_work_gate`: full live cache, logits, sampled DSpark output, public API result shape and head-call counters. |
| `dspark_continue_prefix_chunked` -> existing `dspark_commit_prefill_taps` | Prime only the suffix's final window using the same m=1 projections and absolute positions. No batching/numeric fork; preserve restored slots for short suffixes and always update the newest tap. | `MEMRA_DSV4_PREFILL_DRAFT=all/tail`, default all | Same gate, independently exercising both flags and their combination at 128-position boundaries and restored suffixes. |

DSV4 sampled proposal coupling reuses the existing drafter kernels, the CPU
position-keyed categorical sampler and the unchanged target sample-match walk.
`MEMRA_DSV4_DSPARK_PROPOSAL=coupled` (default OFF) replaces the per-slot greedy
selection only in sampled serving. It forces the host chain, counts every draft
draw and profiles its readback/sampling wall separately. The public greedy
proposal and verifier remain unchanged. `dsv4_coupled_proposal_gate` checks
plain/spec tokens and persistent state, with and without target penalties;
target-card performance qualification is pending.

### MMQ static-lib TUs (prefill GEMM per weight format)

| file | host entry symbols | qtype | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| mmq_q8_0.cu | `memra_mmq_q8_0`, `_act_bytes` | Q8_0 (W8A8 int8) | portable sm_75+ (header) | MEMRA_PP_Q8MMQ, default ON since 2026-07-09 (mmq_ffi.rs:479-490) | mmq_ffi.rs:183-187 |
| mmq_q4_0.cu | `memra_mmq_q4_0`, `_act_bytes`, `_quant_act`, `_gemm`, `_gemm_sk`, `_fixup_bytes` | Q4_0 (W4A8) | CLC arm `__CUDA_ARCH_LIST__ >= 1000` (mmq_q4_0.cu:45) | MEMRA_PP_Q4MMQ (mmq_ffi.rs:516-520), MEMRA_MMQ_SK / _SK_FORM (mmq_ffi.rs:929-949), MEMRA_MMQ_CLC (mmq_ffi.rs:267) | mmq_ffi.rs:229-284 |
| mmq_q45k.cu | `memra_mmq_q4_K`, `memra_mmq_q5_K`, `_act_bytes` (DS4 layout: Q4_K/Q5_K carry min-offset) | Q4_K, Q5_K (W4A8) | sm_89 L40S branch noted (mmq_q45k.cu:23); hard guard UNKNOWN | MEMRA_MMQ_W4A8 seam, default-on (mmq_ffi.rs:450-457) | mmq_ffi.rs:157-181 |
| mmq_fp4.cu | `memra_mmq_nvfp4`, `_ex`, `_ex2`, `_act_bytes` (+residual-correct) | NVFP4 W4A4, same GGUF bytes on both Blackwell families | sm_120a warp MMA; sm_100a `tcgen05.mma` + TMEM twin (`MEMRA_SM100_TCGEN05`); stub elsewhere | `MEMRA_RP=0 MEMRA_MMQ=1` explicit opt-in; B200 exact/model gates pass, but pp1483 is 0.521x raw W4A8 | mmq_ffi.rs |
| mmq_nvfp4_w4a8.cu | `memra_mmq_nvfp4_w4a8`, `_act_bytes`; optional `memra_mmq_nvfp4_f8f4` | NVFP4 weights + q8_1 acts; optional f8f4 route = e4m3 acts | 120a uses int8 W4A8 + fast block-scale F8F4; 100a uses the same W4A8 plus bit-identical plain-E4M3 F8F4; full TU stub on 89/90a | MEMRA_MMQ_W4A8 default-on; MEMRA_MMQ_F8F4 remains opt-in | mmq_ffi.rs |
| mmq_nvfp4_f8f4.cu | `memra_mmq_nvfp4_f8f4_act_bytes` (:96), `_quantize_act` (:105) | e4m3 activation quantizer (compiles on sm_89 too) | — | MEMRA_MMQ_F8F4 | mmq_ffi.rs:105-111 |
| mmq_fp8_blk.cu | `memra_mmq_fp8_blk`, `_act_bytes`, `_scale_rows`, `_scale_cols`, `_quantize_act`, `_grouped`, `memra_fp8_blk_count_nan` | FP8 E4M3 block-128, original f32 grid retained | sm_120a warp MMA; sm_100a dense `tcgen05`/TMEM twin plus legal plain-MMA grouped fallback; stub on 89/90a | native source default on 120a; explicit `MEMRA_FP8_MMQ=1` on 100a, `NativeReference` only after 0.173x fallback pp1483 | mmq_ffi.rs / fp8_ffi.rs |
| mmq_iq_experts.cu | `memra_mmq_iq_experts`, `memra_mmq_iq4xs_dense` (needs `in_f % 256 == 0`, :870), `_quantize_act`, `_fused_act_quant`, `_act_bytes` | IQ3_S, IQ4_XS (W4A8), 35B MoE prefill | smem guard vs sm_120a ~99KB (:849, :870) | MEMRA_MOE_MMA (mmq_ffi.rs:286), MEMRA_PP_IQMMQ (mmq_ffi.rs:502-506), MEMRA_IQ_FAST=0 kill (mmq_ffi.rs:574) | mmq_ffi.rs:288-345 |
| mmq_q8_0_f32acc.cu | `memra_accprobe_act_bytes`, `_gemm_s32`, `_gemm_f32` — "THE Q1 INSTRUMENT for the FP8-ST v3 gate (research-only)… never linked into a serving path" (:1; build.rs:439) | Q8_0, f32-acc probe | `#if __CUDA_ARCH__ >= 1000` (:193) | NONE (research) | mmq_ffi.rs:206-227 |

### Other static-lib TUs

| file | host entry symbols | purpose | arch guard | dispatch flag | FFI binding |
|---|---|---|---|---|---|
| f16_prefill.cu | `memra_f16_pp_gemm[_pre]`, `memra_f16_cvt`, `memra_{q8_0,q4_0,q6_K,q4_K,q5_K}_dequant_f16` | cuBLASLt FP16 TN prefill on resident f16 dequant mirror of quantized weights Per-device cuBLASLt handle slots since 2026-09-02 (lane glm5-b200): a handle created on one device returned CUBLAS_STATUS_EXECUTION_FAILED from the other PP stage on a 2x B200 pair; plan caches key on the device. | host cuBLASLt | MEMRA_PP_F16, MEMRA_PP_F16_BUDGET_MB, MEMRA_W8A8_SIM | f16_ffi.rs:22-62 (+build_*_raw wrappers f16_ffi.rs:725-816) |
| fp8_prefill.cu | `memra_fp8_pp_gemm` (:90) + `__global__` amax/scale/quant kernels | cuBLASLt FP8-E4M3 TN prefill + per-batch activation quantize Per-device cuBLASLt handle slots since 2026-09-02 (lane glm5-b200): a handle created on one device returned CUBLAS_STATUS_EXECUTION_FAILED from the other PP stage on a 2x B200 pair; plan caches key on the device. | — | MEMRA_PP_FP8, MEMRA_PP_FP8_BUDGET_MB, MEMRA_FP8_MMQ, MEMRA_ST_E4M3 (fp8_ffi.rs:27, 45-87, 239) | fp8_ffi.rs:27 |
| fp8_blk_dequant.cu | `memra_fp8_blk_q8_0_bytes` (:220), `memra_fp8_blk_dequant_q8_0` (:228) | device-side dequant of block-128 FP8 weights into GGUF Q8_0 blocks at model load | — | MEMRA_FP8_BLK_GPU (fp8_ffi.rs:476) | fp8_ffi.rs:458-474 |
| fa3_prefill.cu | `memra_fa3_prefill`, `memra_fa3_vl` (+stub twins rc=3, :19-23) | FA3 v10 engine shim, head_dim 256 only | wgmma/TMA sm_90a-only; `-DMEMRA_FA3_STUB` otherwise (build.rs:525-526) | promoted default-ON on hopper 2026-07-27, `MEMRA_FA3=0` reverts; engages only head_dim==256 causal t==t_kv (lib.rs:15399-15409; file header ":1-7 opt-in" is stale) | lib.rs:889-904 |
| moe_f16_grouped.cu | `memra_moe_m1_graph_splitk`; `moe_m1_graph_splitk_partial_kernel<1/2>`, `moe_m1_graph_splitk_reduce_kernel<1/2>` | Numeric class `moe_m1_graph_splitk_f32_fixed_order`: fixed maximum grid, device CSR slices, original half operands, ascending f32 reduction. **Not token-identical to sktail** because the reduction tree differs. Inactive rows `[CSR live, 6)` become +0: downstream projection excludes them through CSR and scatter ignores `pairs=-1`; component mask/poison assertions cover this contract. Drift row: 3/160 top-1 changes, mean/max KL(control to graph) 0.003669683/0.098539347, report-only. | M=1, 4096/2048, 6 slots, up to 16 slices | `MEMRA_DSV4_MOE_M1_SPLITK` defaults ON (unset/graph); explicit 0 is sktail rollback, seam decide-by: 2026-09-22 | Owner accepted 2026-09-08 with drift 3/160, mean KL 0.0037. Component/replay/refusal gates PASS; +10.226982%/+10.023951% sampled; [receipts #520](https://github.com/avifenesh/darklanes/pull/520), source `1cda33750`, binary `d5a8a69a` REACH (memra #454/#458): this family is only reachable on the matrix expert program, which the served path does not run, and on that program it requires the gate-only fused-GU arm, so an unarmed process refuses at load rather than failing every request. Declared in `crates/memra-engine/src/dsv4_doors.rs` and gated by `dsv4_doors::tests`. |
| moe_f16_grouped.cu | `moe_m1_splitk_fast_partial_kernel<1/2>`, `moe_m1_splitk_fast_reduce_kernel<1/2>` | Same `moe_m1_graph_splitk_f32_fixed_order` class, unchanged slice assignment and arithmetic order. Instrument + default-OFF door; candidate 1 uses one aligned 16-byte code load plus one 2-byte scale load per 32-value window. Original kernels, slice map, MMA order, reducer, 128-thread layout and one-block prefetch stay unchanged. Report modeled effective weight GB/s (unique codes + scales / partial time), with 1,792 GB/s a nominal peak. Live 0..6 cases are prefixes derived from one observed six-live CSR per rank/projection, plus an independently observed low-live case when seen in the bounded capture. | M=1, 4096/2048, 6 routes, device CSR live 0..6 | Default (unset). Explicit `MEMRA_DSV4_SPLITK_FAST=0` selects the base graph split-K entries; seam decide-by: 2026-09-23 | `dsv4_splitk_fast_gate`: real-route replay components, raw partial/output bits, canaries and modeled byte accounting. Harness-only `--diagnose-four-live` retains the r1 rank-0 GU route, dumps CSR/input/scales, checks the verbatim merged-wrapper control, current scaffold wrapper and separate kernels without events, then verifies separate retained partial/reducer graphs with timing events recorded outside capture on the replay stream; no timing rows in diagnostic mode. The 597969ee diagnostic localized the captured-event host synchronization failure. Each component graph must contain one kernel and zero events. Corrected baseline: 112 derived-prefix plus 16 observed-low rows pass raw-bit/canary/tail checks. Candidate 1 is qualified at `5d2cbc0d4`: component bit equality on all 128 rows with the partial interval improving in 58 of 64 warm/cold live cases (up to +14.93%), memcheck and synccheck 0 errors, four fresh full-model processes byte-identical against the eager oracle, and sampled ABBA +1.848008% forward and +1.702484% reversed with identical output. Occupancy inputs unchanged: 128 threads, 17,152 static shared bytes, 0 local, 5 CTAs/SM; GU registers 84 to 86. [Receipts, Darklanes #542](https://github.com/avifenesh/darklanes/pull/542). REACH (memra #454/#458): this family is only reachable on the matrix expert program, which the served path does not run, and on that program it requires the gate-only fused-GU arm, so an unarmed process refuses at load rather than failing every request. Declared in `crates/memra-engine/src/dsv4_doors.rs` and gated by `dsv4_doors::tests`. |
| moe_f16_grouped.cu | `memra_moe_m1_splitk`; `moe_m1_splitk_partial_kernel<1>` / `<2>` and `moe_m1_splitk_reduce_kernel<1>` / `<2>` | ModelOpt M=1 down/GU: adaptive 1-16 K slices, original half operands, fixed-order f32 reduction. Existing M1/half2 kernels remain the oracle. | 128-thread partial CTAs, 256-thread reduction; 4096/2048 geometry | Process-local `set_moe_m1_splitk_for_gate`, default OFF | mmq_ffi.rs; `research/dsv4f-moe-m1-splitk-20260907/RESULTS.md` |
| moe_f16_grouped.cu | `memra_moe_m1_splitk_component`, `memra_moe_m1_splitk_component_token` | Eight-token real-route component gate: per-rank GU/down ABBA, finite/repeat/canary checks, explicit tolerance and per-slot timing summary. | Same candidate and oracle launchers; diagnostic only | Process-local `set_moe_m1_splitk_component_for_gate`, default OFF | mmq_ffi.rs; private receipts named in `research/dsv4f-moe-m1-splitk-20260907/RESULTS.md` |
| moe_f16_grouped.cu | `memra_moe_f16g_{dequant,gemm,gemm_sk,gather_act,h2f,h2f_scaled,w_bytes,act_bytes}`, `memra_moe_kq_gemm_sk` | per-layer expert dequant to f16 + ONE grouped f16 GEMM per projection over CSR groups; per-qtype dequant kernels for Q4_0/IQ4_XS/IQ3_S/Q6_K/Q4_K/Q3_K (:109-246); "SASS portable across 89/90a/100a/120a" (:336) Per-device cuBLAS handle slots since 2026-09-02 (same B200 finding as f16_prefill.cu). | smem opt-in >48KB, 1 CTA/SM on sm_120a (:483-486) | MEMRA_MOE_F16G (=2 single-kernel, mmq_ffi.rs:394), MEMRA_F16G_SK/_TAIL/_DIRECT/_DEBUG | mmq_ffi.rs:348-424 |
| moe_f16_grouped.cu | `memra_moe_kq_gemm_sk_grid` | **REMOVED 2026-09-06**. The bounded persistent-grid arm was bit-identical and engaged on the target pair, but full-model A/B was flat/no-go at 256 and 8192 contexts. The generic `memra_moe_kq_gemm_sk` path remains the only DSV4 matrix visitor. | Historical component/sanitizer and full-model receipt retained; no serving default was promoted. | Removed door receipt: `research/dsv4f-2card-1m-20260904/grouped-grid-bound.md` plus private `grouped-grid-perf-20260906` receipt | — |
| cutlass_fp4_sm120.cu | `memra_cutlass_fp4_{workspace,sfa_size,sfb_size,fp4_gemm,repack_sfa,repack_sfb}`, `memra_nvfp4_{quant,dequant}_ref`, `memra_gguf_nvfp4_deinterleave` | CUTLASS 4.x NVFP4 W4A4 GEMM wrapper | hard sm_120a gencode (build.rs:604); build asserts 120a-only (build.rs:236-237) | MEMRA_CUTLASS (build env) | cutlass_ffi.rs:47-113 |
| mla_attn.cu (glm-dsa / glm5_next MLA-DSA forward, build.rs:497-504; portable CUDA C, default FMA contraction, NOT the -fmad=false class dsv4_gpu.cu below uses) | `memra_mla_{split_latent,append_latent}_f32` (latent cache write); `memra_mla_absorb_q_f32` + `_split_f32` (q_lat = W_uk^T . q_nope, one block per (query,head), output-range split twin bit-identical by construction, lane/glm5-decode-diet 2026-08-31); `memra_mla_decompress_v_f32` + `_split_f32` (same pattern, W_uv . o_lat); `memra_mla_attn_absorbed_f32` (dense absorbed-MQA online softmax over the contiguous cache, one block per (query,head), maxdiff-vs-CPU-oracle class per the file header, NOT bit-identity); `memra_mla_attn_gathered_f32` + NEW `_split_f32` (same online-softmax body over a per-query gathered DSA index list shared across heads; the split twin, added lane/b200-mla-decode-20260902, recomputes the shared score/softmax tile walk IN FULL per split block and restricts only the final per-l accumulate-and-write loop to an output range, bit-identical to the unsplit kernel by the same "which block writes it" argument as the absorb/decompress splits, NOT a slot-range split, see the file's "B200 (sm_100a) t<=8 decode arm" section header for why a slot-split was refused); NEW `memra_mla_attn_gathered_dsa_f32` (lane/b200-dsa-decode-20260902, `MEMRA_B200_DSA_DECODE>=1`: the same grid, the same MLA_WARPS-wide slot tiles, the same warp-per-slot lane stride and 5-step shuffle tree and the same online-softmax fold as `memra_mla_attn_gathered_f32`, with each tile's KV rows staged ONCE into shared memory by float4 loads and read back for BOTH the score dot and the PV accumulate -- one L2/SM crossing per row per head instead of two -- and the 8 tile exponentials hoisted into registers, 24 -> 8 per thread per tile; BIT-IDENTICAL by construction, barrier count per tile unchanged at 2 because `__syncwarp` covers the staging); NEW `memra_mla_dsa_attn_split_f32` = `memra_mla_dsa_attn_warp_kernel<J,JP>` + `memra_mla_dsa_attn_combine_kernel` (`MEMRA_B200_DSA_DECODE=2`, the WARP-ONLINE arm: one warp owns one (token, head, slot-chunk) and holds the whole kv_rank-wide accumulator in REGISTERS (`J = kv_rank/32` per lane), so every KV element is read from memory EXACTLY ONCE and consumed twice from registers, there is NO `__syncthreads` at all (warp-local online softmax, 5-step `__shfl_xor_sync` butterfly so every lane ends with the sum), and two `expf` per slot per warp replace ~196k per warp per layer; `J`/`JP` are template parameters because the accumulator must stay in registers, instantiated for kv_rank {512,1024} x d_rope {0,64} and returning 40023 otherwise. NUMERIC CLASS `dsa-warp-online-f32`, NOT bit-identical -- it folds PER SLOT where the shipped kernel folds in 8-slot tiles, then merges `chunks` partials -- held to an ARGMAX gate plus maxdiff/max-rel by `dsa-decode-gate`. B200 2026-09-03 (2x B200 SXM, N=5): 552.1 -> 54.3 us at t_q=1 / 1M with 32 chunks (**10.2x**), argmax MATCH at 128k/256k/1M; and 1 of 256 latent-row argmaxes MOVED at kv=131072 / t_q=4 for EVERY swept chunk count, which is why the class is confined to plain decode by `mla_ffi::MLA_DSA_NAMED_CLASS_T_MAX` = 1 and `mla_dsa_attn_arm_effective` demotes arm >= 2 above that width to the SHIPPED kernel in code. Serving A/B on the pair, vendor sampling, 256k prompt: door off 30.07 tok/s, `=1` 33.0, `=2` 43.04 (+43.1%), TTFT unchanged. Arm code per width from `mla_ffi::MLA_DSA_ATTN_ARM` (32 at t_q=1, 1 at t_q=4, 0 elsewhere); `memra_mla_dsa_attn_chunk_span` is the host-visible partition so Rust and CUDA cannot disagree. Companion finding: the BIT-IDENTICAL single-pass arm above is a measured LOSS on that rig, because nvcc had already hoisted the loop-invariant `expf` and the second `cache[]` pass hits L1/L2 -- bit identity is the binding constraint on this kernel, which is why the depth win needed a named class); `memra_mla_index_append_ring_f32` (indexer state ring write); `memra_mla_kpool_pool_keys_f32` (pool-key collapse, learned softmax over gate+APE); `memra_mla_kpool_score_ref_f32` / tiled `memra_mla_kpool_score_f32` (DSA pool scoring, six-step pinned-rounding contract, reference vs shipped-tiled twins, BIT-IDENTICAL by construction per the file's rounding-sequence proof); NEW `memra_mla_kpool_score_dsa_f32` (`memra_mla_kpool_score_dsa_kernel<H,RP,KC>`, lane/b200-dsa-decode-20260902, `MEMRA_B200_DSA_DECODE>=1`: the DECODE-shaped scorer, blocked on (head, pool) instead of (query, pool) because at t_q=1 the only reuse axis is heads -- one thread owns RP pools and ALL H heads with `dot[H][RP]` in registers, pool keys stream through smem in KC slabs stored transposed at row stride BP+1, and the q slab is read `float4` over four heads at a time as a block-wide broadcast, dropping shared loads per FFMA from 2.0 to 0.156 at H=32/RP=2; BIT-IDENTICAL to `_ref_f32` by construction -- `c`-ascending dot from +0.0f, `h`-ascending head mix inside ONE thread, all six rounding steps spelled with explicit intrinsics; H templated {16,32,64}, other geometries return 40023 and take the shipped dispatch); NEW `memra_mla_kpool_select_dsa_f32` = `memra_mla_kpool_select_{clear,hist,tie,count,emit}_kernel` (lane/b200-dsa-select-20260903, `MEMRA_B200_DSA_SELECT`): the EXACT multi-CTA selector. The shipped kernel grids `t_q` blocks, so plain decode selects on ONE CTA; this spreads the same work over `memra_mla_kpool_select_ctas(n_pools)` CTAs per query in six launches, each ending in a LAST-CTA epilogue (`atomicAdd` + `__threadfence()`) rather than a grid barrier or a cooperative launch, so no occupancy change can hang it. Only 32 bits take a radix descent (two 65536-bin passes): the key's low word is the unique pool index, so the threshold's index is the rank-r smallest index among the pools TYING at the threshold score, which is a count plus a scan. EXACT, not banded, by construction -- the emitted plane is a pure function of the `select_k`-th smallest order key, keys are distinct, and this computes the same key and runs the same `key(p) <= thr` test. `memra_mla_kpool_select_ws_ints` / `_ctas` are the host-visible workspace layout so Rust and CUDA cannot disagree; `memra_mla_kpool_select_dsa_redarm_f32` is the gate's RED ARM (threshold lowered by one, which always drops the threshold pool) and is never on a serving path. Gated by `dsa-select-gate`: byte-identical `idx` at {mixed, all-ties, sparse, empty} x n_pools {4096..262144} x t_q {1,4}, plus the red arm and an anchor against the reference selector; 5090 PASS 2026-09-03, 40/40 EXACT, 3.17x at 1M and 1.50x at 256k, a LOSS below 128k (hence `MLA_DSA_SELECT_MIN_POOLS` = 65536). NEW `memra_mla_kpool_score_tc_f32` (`memra_mla_kpool_score_tc_kernel`, lane/glm5-dsa-score-tc-20260907, `MEMRA_DSA_SCORE_TC=1|2`, prefill `t_q >= 8`: the scorer's contraction on bf16 `mma.sync.m16n8k16` with f32 accumulation (`=2` splits every operand hi/lo bf16 and takes three MMAs per product for f32-class error; `memra_mla_q_split_bf16_kernel` writes the query planes), one CTA per 128-pool tile with the keys as register-resident B fragments, bf16 query rows cp.async double-buffered through smem and read by `ldmatrix`, the f32 head-mix epilogue reduced over the 16 head rows of each m-tile by three xor-shuffles; NUMERIC CLASS bf16-mma, NOT bit-identical, admitted by served-prompt ids like `MEMRA_MLA_TC_PREFILL`; d=128 and heads%16 only, else 4003x refusal to the f32 dispatch); `memra_mla_kpool_select_f32` / `_ref_f32` (top-k pool selection, radix-select vs reference twins, exact order-key isomorphism) | f32 throughout | portable (no arch-specific intrinsics; no stub needed) | `MEMRA_MLA_DECODE_SPLIT` (absorb_q/decompress_v split, decode-diet lever 4, mla_ffi.rs); `MEMRA_MLA_TC_PREFILL` (t>=16 prefill chain lives in f16_prefill.cu and flash_attn.cu instead, see that row); `MEMRA_B200_MLA_DECODE_ARM` (sm_100a t<=8 arm covering absorb_q/decompress_v/attn_gathered splits; the split factor per kernel comes from the t_q-keyed tables `MLA_B200_{ABSORB_Q,DECOMPRESS_V,ATTN_GATHERED}_SPLIT` in mla_ffi.rs, a cell of 1 = the shipped launcher; B200-measured 2026-09-02, t_q=1 splits 4/4/2 and absorb_q split=4 at t_q=4, shipped everywhere else; gated by `mla-decode-arm-gate` (bit-identity at t_q in {1,2,4,8} x kv in {2k,32k,128k,256k} x split in {2,4,8}, cold-L2 scrub before every timed launch, plus a 5% regression bar on the table's arm at every (t_q, kv); kv axis from lane/b200-mla-depth-20260902, policy still t_q-keyed only); lane/b200-mla-decode-20260902, default OFF, FLAGS.md); `MEMRA_B200_DSA_DECODE=1\|2` (sm_100a t<=8 depth door: level 1 = the bit-identical single-pass gathered kernel + the head-blocked scorer, level 2 additionally allows the `dsa-warp-online-f32` warp-online arm; checked BEFORE `MEMRA_B200_MLA_DECODE_ARM` in `mla_attn_gathered`; scorer engages from `MLA_DSA_SCORE_MIN_POOLS`=4096 pools up; gated by `dsa-decode-gate` over context {2k,32k,128k,256k,1M} x t_q {1,4} with bit-identity, an argmax gate on the warp-online arm and a 5% regression bar on the policy-selected arm; B200 2026-09-03: scorer BIT-IDENTICAL and 3.97-7.64x, gathered 10.2x at t_q=1; the run's 4 ARGMAX failures at t_q=4 put the named-class width rule in the door; timing bar is hard only on an sm_100a build and DIAGNOSTIC elsewhere, correctness bars hard everywhere; lane/b200-dsa-decode-20260902, default OFF, FLAGS.md, roofline research/b200-dsa-decode-20260902/ROOFLINE.md); `MEMRA_DSA_SCORE_TC=1|2` (prefill scorer on bf16 tensor cores, 2 = hi/lo split for f32-class error, t_q >= 8, any arch with mma.sync bf16; lane/glm5-dsa-score-tc-20260907, default OFF, decide-by 2026-09-21, FLAGS.md); `MEMRA_B200_DSA_SELECT=1` (sm_100a exact multi-CTA k-pool selection; engages at t_q <= 8 and n_pools >= MLA_DSA_SELECT_MIN_POOLS = 65536, the measured crossover; gated by `dsa-select-gate` with a red arm that must fail first; lane/b200-dsa-select-20260903, default OFF, FLAGS.md) | mla_ffi.rs (all `memra_mla_*` extern decls + `Engine` wrappers `mla_absorb_q`/`mla_decompress_v`/`mla_attn_absorbed`/`mla_attn_gathered`/kpool wrappers) |
| `memra_mla_attn_absorbed_live_kernel`, `memra_mla_append_latent_live_kernel` (cu/mla_attn.cu, lane/mla-live-len-20260905) | live-length twins of the dense decode core and the latent append: `t_kv = pos_d[0] + t_q` / `slot = pos_d[0]` read from the decode-graph door's device position word instead of a launch scalar, so both have a fixed launch geometry and can sit inside a captured graph (the prerequisite for capturing the MLA middle at short context); one shared body with the scalar entries, bit-identical at the same length | f32 | none | no door (consumed by the middle-capture arc); wrappers `Engine::mla_attn_absorbed_live`, `Engine::mla_append_latent_live` | C launchers `memra_mla_attn_absorbed_live_f32`, `memra_mla_append_latent_live_f32`; gate `tests/mla_live_len_gpu.rs` |
| `memra_mla_kpool_score_dsa_live_kernel`, `memra_mla_kpool_score_ref_live_kernel`, `memra_mla_kpool_select_live_kernel`, `memra_mla_attn_gathered_live_kernel`, `memra_mla_dsa_attn_warp_live_kernel`, `memra_mla_index_append_ring_live_kernel`, `memra_mla_kpool_pool_keys_live_kernel` (cu/mla_attn.cu, lane/mla-mid-capture-20260905 + lane/glm5-kgraph-20260905) | live twins of the DSA k-pool middle reading the door's device position word: the scorer (head-blocked where instantiated, the reference kernel otherwise) at the capacity pool stride, the single-CTA selector at the capacity `index_width` with sentinel fill publishing the true width in a device word (`width_d`), the gathered and warp-online attention bodies walking exactly that width (explicit `idx_stride`), the indexer state append and the pool-key build over the pools a t-row append completes (grid.y = t/pool+1, exits past the live count). All t-row capable (the verify-walk shape, K-graph lane), bit-identical to the scalar launches by construction (same bodies, strides parameterized). Consumers: `HybridModel::mla_mid_post_live` (t=1, `MEMRA_GLM5_GRAPH_MLA_MID`), `mla_attn_cached_rows_live` (t rows). Gates: `tests/mla_kpool_live_gpu.rs` (t=1 and t=3 bitwise vs scalar at several counts under a capacity, red arms on the position and width words), `glm5_decode_graph_capture_gpu::{graph_door_mla_mid_match_eager_bitwise, mla_rows_live_matches_rows_exact_bitwise}` |
| `memra_mla_kpool_score_dsa_live_kernel`, `memra_mla_kpool_select_live_kernel` (cu/mla_attn.cu, lane/mla-kpool-live-20260905) | live-count twins of the DSA decode scorer (capacity grid, `n_pools` read from the door's device word; every block masks `p < n_pools`) and of the single-CTA selector (grid t_q, count from the device word): one shared body with the scalar entries each, bit-identical at the same count at t_q = 1; the third and fourth pieces of the MLA middle capture | f32 scores, i32 idx | none | no door yet; wrappers `Engine::mla_kpool_score_dsa_live`, `Engine::mla_kpool_select_live` | C launchers `memra_mla_kpool_score_dsa_live_f32`, `memra_mla_kpool_select_live_f32`; gate `tests/mla_kpool_live_gpu.rs` |
| `memra_mla_kpool_select_v2_kernel`, `memra_mla_kpool_select_live_v2_kernel` (cu/mla_attn.cu, lane/kpool-select-v2-20260906) | the single-CTA k-pool selector's v2 body (`memra_mla_kpool_select_body_v2`): the shipped 8-pass MSB-first radix select and key, with (a) the 256-bin rank walk done by warp 0 in one shuffle scan (`memra_kpool_warp_scan_incl`: each lane owns 8 bins) instead of thread 0's serial 256-bin loop, and (b) the membership count and emit done per WARP over a contiguous slice of the row, four coalesced 128 B lines in flight per lane, ballot + popc for the in-order slot, `sh_wsum` for the warp prefix, instead of a per-thread contiguous chunk plus thread 0's serial 256-thread scan. Same key, ranks, membership and emit order, so the emitted index lists are bit-identical to `memra_mla_kpool_select_kernel` / `_live_kernel`. Rig instrument (5090 laptop, ordering only, RUSTC_WRAPPER= build, two rounds each within 0.2 us): 61,323 pools random 114.4 vs 132.0 us, flat 191.7 vs 233.2, two-level 183.3 vs 213.0; ncu clock-locked on the same shape: global sectors 49.8k vs 154k. Two arms that lost on the way and are not in the code: warp-aggregated `__match_any_sync` histogram atomics (won flat, lost random) and a single coalesced load per emit iteration (long-scoreboard 7.5 vs 5.9 per issue, 468 vs 362 us). Door `MEMRA_KPOOL_SELECT_V2` (default OFF, decide-by 2026-09-20 on the box `selab` row at 256k). Gate: `mla_kpool_select_v2_gpu` (28 cells, both twins, ties, -inf, NaN, red arm). |

### Stub files (fail-closed twins)

- `mmq_fp4_stub.cu` — `memra_mmq_nvfp4{,_ex,_ex2,_act_bytes}` outside sm_120a/sm_100a.
- `mmq_nvfp4_w4a8_stub.cu` — full W4A8/F8F4 ABI on sm_89/90a; sm_100a compiles both real routes.
- `mmq_fp8_blk_stub.cu` — block-FP8 ABI outside sm_120a/sm_100a.

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

## HC24 split dots numeric class (owner accepted default ON, S16)

`cu/dsv4_dense_m1_exact_tail.cuh` adds `dsv4_hc_dot_split_partial_kernel<S>`
and `dsv4_hc_dot_split_reduce_kernel<S>` for S=8/16/32. Only the device HC-24
pre-attention/pre-FFN sites with F32 N=24,K=16384 dispatch this pair. Other
dots shapes retain their current kernel. Existing CUDA kernels are unchanged.

Each of 24*S blocks has 128 threads. A contiguous K slice retains increasing
eight-element per-lane multiply/add order and the exact-tail 128-leaf tree.
S=32 has 64 zero leaves because its slice has 512 elements. One 32-thread
second-stage block sums each row's partials in ascending slice order with
explicit round-to-nearest f32 adds. No atomics or fused multiply-add.
This is the **HC24 split dots** numeric class, not bit-identity with the
exact-tail dots class: restarting accumulators and summing slices changes
association. Each S is a distinct class and must be pinned in its receipt.

Scratch is 24*32 F32 elements per decode state and rank, allocated before
capture. Every call writes all partials it reads, both stages use the same
stream, and graphs retain stable scratch addresses. `MEMRA_DSV4_HC_DOT_SPLIT`
is ON when unset; exact `1` or `16` also selects the owner-chosen S=16.
Explicit `0` restores sequential dots after a fresh process/state capture.
S8 and S32 remain explicit opt-in classes; other strings select OFF.
Rollback seam decide-by: 2026-09-23, owner accepted 2026-09-09.
S32 was component-fastest but its 1.407% advantage over S16 did not justify
rebuilding and re-review; only S16 has the model campaign receipts.

[Darklanes #538](https://github.com/avifenesh/darklanes/pull/538) banks both-rank
S8/16/32 components and zero-error memcheck/synccheck, two fresh process
observations per arm with external token/logit/cache/hidden equality, AR epochs
and eight refusals per arm per process. ON census is 86 partial plus86 reducer
nodes per rank in each of the three forward variants; OFF and commit segments
have zero. Source `d42196214`, binary
`d9ca7ac0bb6417fcd99e176cd6e2244d9e9d8b6b837a08d73b27fc0a7bd2dc5c`.

KEEP default ON: sampled pooled +2.701174% forward / +2.591532% reverse,20 eligible
rows per order with first capture timed. All128 DRIFT-R3 input hashes match:
49/2048 top1 changes (2.392578%), KL OFF-to-ON mean/max
0.005672511/0.421308907 and ON-to-OFF0.005758277/0.483190648; greedy16/64
identical and2099/4096 matching tokens. This is a distinct numeric class,
not token-identical to the sequential dot. The owner accepted default ON
on 2026-09-09, also informed by [drift mechanism #534](https://github.com/avifenesh/darklanes/pull/534)
and [task accuracy #545](https://github.com/avifenesh/darklanes/pull/545):
control 177/300, split-K + HC 177/300, McNemar p=1.00.
`dsv4_hc_dot_split_gate --defaults` observes unset/0 without an HC override,
checks every default census, 256-step eager identity within the selected class,
eight refusals and five retained-graph sanity rows. Historical Rust and CUDA
controls pin HC OFF before discovery or model creation. Profile and default
engagement paths retain the environment policy.
The source rebase does not relabel the pinned binary receipts as a new build.

## DSV4 segmented replay component, 2026-09-08

| Source | Kernel | Contract / dispatch |
| --- | --- | --- |
| `cu/dsv4_replay_control.cuh` | `dsv4_replay_control_kernel` | Integer live token/position/uniform storage, ring slot, compressor cadence/offsets, indexer bounds. Same-device C4/C128 IF handles set on every execution. Standalone `tools/dsv4-full-token-control-gate.cu` only; no runtime FFI or model dispatch. Default OFF; decide-by 2026-09-22. `research/dsv4f-full-token-replay-20260908/DESIGN.md`. |
| `tools/dsv4-full-token-control-gate.cu` | `snapshot`, `emit4`, `emit128`, `producer`, `finish`, `rollback` | Integer payload/control fixtures around the existing `memra_tp_ar_1stage` transport and its live-fault entry. Not model compressor, attention, head or sampling kernels. No runtime dispatch. Same diagnostic lifetime as the control kernel. |
| `cu/dsv4_gpu.cu` | `dsv4_replay_input_kernel`, `dsv4_replay_tick_kernel` | Stable 24-byte request controls and per-rank segment counters. Gate-only full-token replay, default OFF, decide-by 2026-09-22; FFI in `dsv4_ffi.rs`. |
| `cu/dsv4_gpu.cu` | `dsv4_replay_copy_row_kernel`, `dsv4_replay_copy_if_kernel` | Live append/emission addresses and uniformly predicated pending shifts. Byte copies only; existing active compressor arithmetic/reduction order retained. Same diagnostic door. |
| `cu/dsv4_gpu.cu` | Existing compressor pool, f32 RMSNorm, RoPE-at, Hadamard and activation-quant kernels | Optional uniform whole-block emission predicate at entry, before barriers. Null preserves eager behavior. `memra_dsv4_replay_compressor_emit` composes the exact active program; no CUDA conditional body. Real-kernel byte/sanitizer gate: `tools/dsv4-replay-live-kernel-gate.cu`. |
| `cu/dsv4_gpu.cu` | Existing redirect, numeric top-k, f32 indexer-score and sink score/soft/out kernels | Optional live position/count/slot inputs; same loop bounds and reduction order as eager. No padded reduction replacement. Same default-OFF full-token door and component gate. |
| `cu/tp_ar.cu` | `memra_tp_ar_1stage_kernel` | Optional live rank/layer fault input before actual barriers. Existing eager entry passes null; start/end barriers and device epoch program retained. Replay FFI and epoch readback: `tp_ar.rs` / `dsv4_ep.rs`. |
| `cu/tp_ar.cu` | `memra_tp_ar_1stage_kernel` (phase instrument) | Gate-only `MEMRA_DSV4_AR_PHASE`, default OFF, decide-by 2026-09-24. Four `clock64` stamps and two `%globaltimer` stamps taken by block 0 thread 0 into a device ring, decomposing each of the 86 joins into peer wait / reduce / tail wait; the record carries its own cycles-to-nanoseconds calibration so no assumed clock can rescale a phase. Stamps go INSIDE the existing graph node because a per-AR `cudaEventRecord` inside stream capture becomes an event-record node (172 new nodes per rank per step) and Nsight graph-node tracing inflates the same rows by two orders of magnitude (darklanes #519). Every added branch is kernel-parameter-uniform and the reduce's operand order, indexing and arithmetic are untouched, so the instrumented arm is bit-identical to the product arm by construction, not by tolerance. Two further gate-only arms live behind the same door: the NULL arm (`=null`) reads this rank's own operand twice instead of one local and one peer operand, keeping launch shape, both barriers, epochs and refusal words, to bound transport against wait; and a red-arm `delay_ticks` spin keyed on one site and one rank, which the gate requires to appear in the OTHER rank's wait phase at that site and nowhere else. FFI and record: `tp_ar.rs`; allocation, siting and drain: `dsv4_ep.rs`. Gate: `dsv4-ar-phase-gate`. |
| `cu/dsv4_sampler.cu` | `dsv4_sample_draw` | Pointer-fed uniform in replay; existing by-value path remains. Same `device-f64-exp-tree-cdf-v1` program. Real changing-uniform equivalence gate: `tools/dsv4-replay-sampler-gate.cu`. |

## Known UNKNOWNs

Per-variant dispatch flags inside the four giant fatbin TUs (kernels/qmatvec/flash_attn/
hybrid) are family-level here; per-variant selection lives across ~411 `MEMRA_` read
sites in src/lib.rs and was not traced symbol-by-symbol. Rows say UNKNOWN where the
specific gate was not found. FLAGS.md is the authoritative flag catalog.

## DSV4 device sampler, 2026-09-07

| Translation unit | Kernels | Contract / gate |
| --- | --- | --- |
| `cu/dsv4_sampler.cu` | `dsv4_sample_prepare`, `dsv4_sample_merge`, `dsv4_sample_exp_scan`, `dsv4_sample_offsets`, `dsv4_sample_draw` | Head-stream f32 penalty/key preparation; stable unique-key merge chain; f64 exp and block prefix sums; block offsets; top-k/top-p inverse CDF. Request-owned scratch, one token u32 D2H. Numeric class `device-f64-exp-tree-cdf-v1`; finite inputs required. `MEMRA_DSV4_SAMPLER=device`, default OFF. Component tape and sampled ABBA: `dsv4_tp_ep_sampled_perf_gate --sampler-component` / `<model> <source> --sampler-abba`. Receipt: `research/dsv4f-gpu-sampler-20260907/RESULTS.md`. |

### Dense M=1 exact-tail transport (default ON, 2026-09-08)

`crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh`, included only by
`cu/dsv4_gpu.cu`, defines `dsv4_dense_exact_tail_fp8_kernel<1,false>` and
`dsv4_dense_exact_tail_dots_kernel<1>`. Original M=1 arithmetic bodies are
copied intact up to the reduction tail. The replacement replays the exact
128-leaf tree after one shared store/barrier, using guarded full-mask warp-0
shuffles. Original kernels and grouped/M>1 dispatch remain the control.
`memra_dsv4_dense_exact_tail_{fp8,dots}` refuse unsupported raw calls; the two
existing M-row launchers select them by default when admitted;
`MEMRA_DSV4_DENSE_EXACT_TAIL=0` retains the original kernels. Explicit host-thread
gate overrides still select immutable A/B capture functions. No expert or
compressor arithmetic changes. Dense independent orders +0.459578%/+0.397451%
are banked in [private Darklanes #507](https://github.com/avifenesh/darklanes/pull/507),
in that lane's model result report. Combined qualification
is tracked by `research/dsv4f-cadence-dense-default-on-20260908/DESIGN.md`.
The direct composition receipt is
[Darklanes #509](https://github.com/avifenesh/darklanes/pull/509), +1.87% forward /
+2.11% reverse with identity. The dense default affects admitted eager calls
as well as new captures; it is not a replay-only dispatch.

### Full-token cadence capture (default ON within admitted replay, 2026-09-08)

No new kernel: three retained forward graphs omit inactive compressor emission/
shift calls, retaining the existing active kernels, geometry and reduction order.
`MEMRA_DSV4_REPLAY_CADENCE=0` selects the original full-forward graph at fresh
request arming. Shared commit/head/sample and both refusal checks remain.
Cadence independent orders +1.007107%/+1.074545% are banked in
[private Darklanes #508](https://github.com/avifenesh/darklanes/pull/508),
in that lane's combined result report. Full replay
admission remains unchanged, including pos<512; no general eager graph fallback.
Both rollback seams have decide-by 2026-09-22 for removal review.
Composition confirmation is directly recorded in
[Darklanes #509](https://github.com/avifenesh/darklanes/pull/509), +1.87%/+2.11%
with identity, alongside the standalone cadence #508 and dense #507 receipts.

### KV RMSNorm and RoPE default with rollback, 2026-09-09

`dsv4_norm_rope_f32_fixed_order_kernel` is the default-ON (admitted TP/EP f32x)
`MEMRA_DSV4_NORM_FUSE` arm. It replaces the adjacent KV norm and rotary launches
in each t=1 device batch attention layer (SWA, CSA and HCA). The 128-thread
RMSNorm reduction is unchanged; only shared-memory transport replaces the
normalized f32 global-memory intermediate. The subsequent QAT is unchanged.
Retained census confirms 43 launches removed per rank per forward step,
with 43 fused nodes in each ON forward variant and zero in OFF.

Attention-entry norm feeds Q and KV projections and, in compressed layers,
f32 compressor/indexer projections. Q norm/pack is already fused by the diet.
MoE-entry norm feeds router logits before activation quantization and grouped
FP8-to-half gathering; shared experts also consume its BF16 pack. Those are
not one adjacent norm/gather/convert chain. Compressor emission norm feeds
RoPE then Hadamard/FP4 (indexer) or QAT (attention), but its replay wrappers
are outside this lane. Final norm feeds f32 head dots. No fusion of these
fan-out chains or modification of their reduction trees is proposed.

KEEP small, same numeric class: +0.607010% forward and +0.437574% reverse
pooled throughput on pinned DSV4F EP+TP2 2x RTX PRO 6000 with full replay,
cadence, dense exact-tail, graph split-K, device sampler and diet. A second
forward run confirms +0.499856%. Each order has 20 sampled rows with first
capture included. All 86 component sites pass raw-bit comparison, memcheck
and synccheck report zero errors, and every 256-step identity/census/reset
and 16-refusal invocation passes. Receipts: [private Darklanes #530](https://github.com/avifenesh/darklanes/pull/530),
the report and raw manifests linked there, source `511f0e663`,
binary `e36c98b0bd80cd8f1c6895f7193e68ebf9120327e437b5fab07c1945cad5761c`.
The composition with dense-fast is KEEP at +1.618979% / +1.609496%, with
40 identity-matched sampled rows, and is now the default in the admitted path.
Same numeric class, token-identical to the prior default. Rollback uses explicit
`MEMRA_DSV4_NORM_FUSE=0` with a fresh process/uncaptured state; unset is ON.
Rollback seam decide-by: 2026-09-23. Composition receipts: [private Darklanes #535](https://github.com/avifenesh/darklanes/pull/535). FFI entry: `memra_dsv4_norm_rope_f32_fixed_order` in
`src/dsv4_ffi.rs`, dispatched by the t=1 batch attention path in `src/dsv4_gpu.rs`.

### Dense-fast exact-tree kernels and qualification (default ON, 2026-09-09)

`cu/dsv4_dense_m1_exact_tail.cuh` adds `dsv4_dense_fast_fp8_kernel<2>`
and `dsv4_dense_fast_dots_kernel<1>`, selected in the existing raw exact-tail
launchers by `MEMRA_DSV4_DENSE_FAST`. FP8 uses 256 threads for two independent
rows sharing the identical E4M3 table; each row keeps 128 leaves. Dots retain
128 threads, existing 16-byte operand loads and four-iteration loop unrolling.
Leaf t consumes K positions 8*t+1024*j+[0..7] in ascending j/element order.
The final tree is ((p[t]+p[t+64])+(p[t+32]+p[t+96])) followed by guarded
16/8/4/2/1 warp shuffles, all F32 additions. FMAD stays disabled. Ragged tile
rows participate in barriers without out-of-range operand loads or stores.

`tools/dsv4-dense-fast-gate.cu` checks raw bits, guards, operand immutability
and actual retained graph functions. All 24 real rank/shape cases, 36 boundary
cases and two cancellation witnesses pass in normal, memcheck and synccheck;
both sanitizers report zero errors. Each real case has 50 warm and 50 cold
CUDA-event samples per arm; cold flushes 256 MiB. GB/s is modeled unique tensor
traffic, not measured DRAM bandwidth. Resource APIs report static occupancy
limits; disassembly reports static instructions. The captured operand callback
is null in normal work and verifies registered stream and allocation ownership.

`src/bin/dsv4_dense_fast_gate.rs` forces A OFF and B ON on the default
split-K/cadence/device sampler/diet program, checks 256 per-step identities,
retained resets, every forward variant's functions and 16 live refusals.
Its 20-row ON/OFF/OFF/ON and single reverse twin include each scored arm's first
capture. Pooled gains are +1.551526% and +1.459516%; all 40 rows are eligible
and share token/logit/cache/hidden identity. Composition with norm-fuse is KEEP
at +1.618979% / +1.609496% and defaults ON when unset. Explicit `0` is the
rollback with fresh uncaptured states; seam decide-by: 2026-09-23.
Same numeric class, token-identical to the prior default. Source `711165799`, model binary SHA256
`4be3e8084bb7d589abb8d2250c06f8c12f1edb713a66e2390bc90ed91821d5fd`.
Receipts: [private Darklanes #529](https://github.com/avifenesh/darklanes/pull/529).
Composition receipts: [private Darklanes #535](https://github.com/avifenesh/darklanes/pull/535).
`dsv4_densefast_normfuse_default_gate` checks real unset/0 selection before
capture, both function censuses, eager identity, refusals and five sanity rows.
Gate-only `memra_dsv4_dense_fast_restore_default_for_gate` restores the actual
environment policy after the eager OFF oracle. No kernel arithmetic changes.

### Remaining DSV4 activation packing (experimental, 2026-09-09)

`MEMRA_DSV4_NORM_FUSE2` is default OFF, decide-by: 2026-09-23.
`dsv4_norm2_pack_f32_fixed_order_kernel` uses the original 128-thread norm
reduction and f32 epilogue, emits BF16 RNE as well as retained f32, and permits
attention Q_a/KV to share one unchanged pack. FFN routing and quantization keep
the f32 row. `dsv4_norm2_swiglu_pack_kernel` preserves clamp, sigmoid and f32
multiply rounding before BF16 RNE. `dsv4_norm2_quant_half_kernel` uses four
64-thread teams to repeat the original per-128 FP8 max tree, power-of-two scale,
E4M3 rounding and zero-sign canonicalization, then the original 256-thread
row-scale tree and lossless half/status expressions. Intermediate codes/scales
are shared-memory transport. No expert GEMV, split-K, dense kernel, HC or rotary
kernel changes. Each site is geometry checked at dispatch and falls back to its unfused chain
outside the fused domain; the intermediate transport additionally requires the
half mirror's own row capacity. Counts per rank and forward variant: 86
norm/pack, 43 shared SwiGLU/pack, 43 quant/half, 387 launches gross for a net
215 removed. Commit has none. OFF has zero
new symbols. FFI: `src/dsv4_ffi.rs`; component capture and raw-bit gate:
`src/dsv4_norm2_component_gate.rs`; replay gate: `dsv4-norm-fuse2-gate`.
Evidence: [private Darklanes #560](https://github.com/avifenesh/darklanes/pull/560),
raw-bit identity at all 344 component sites, memcheck and synccheck zero
errors, +1.0926% pooled ABBA and +0.9075% pooled reverse on the sampled
default program. Default is ON when unset in the admitted TP/EP f32x topology; explicit `0` is
the rollback seam, decide-by: 2026-09-23.

### Wide DSV4 norm2 pack (default ON since 2026-09-10, door opened 2026-09-09)

`MEMRA_DSV4_NORM2_WIDE` is default ON under an admitted `MEMRA_DSV4_NORM_FUSE2`
(flipped 2026-09-10 on the model campaign receipts below); explicit `0` is the
rollback seam, and unset without the norm2 door degrades to OFF because the wide
pack has no other call site.

`dsv4_norm2_pack_f32_fixed_order_kernel` launches grid 1 / block 128 over one
4096-element f32 row. One CTA holds both the reduction and the whole epilogue:
14.426623 us per launch, 1.240690 ms/step on rank 0 at 86 launches, a provisional
0.221812% of 1,792 GB/s. That geometry was inherited unchanged from the separate
pack the norm2 door replaced, so it is not a regression the door introduced, but
it is the largest single kernel in the fused norm2 family.

`dsv4_norm2_pack_f32_fixed_order_wide_kernel` partitions the EPILOGUE COLUMNS
across `NORM2_WIDE_TILES` CTAs of 128 threads. Grid X is a column tile, not a
row: the pack domain is one row of 4096 and the launcher refuses anything else.
Every CTA repeats, byte for byte, the same eight-load accumulation order and the
same `dsv4_block_sum_f32` tree over the whole row, so `tot`, `mean` and `rsq` are
bit-identical in every CTA and identical to the single-CTA kernel. The written
value is a pure function of (column, rsq), so which CTA writes a column cannot
move a bit.

**Class: SAME.** The reduction order is the contract the `fixed_order` name
carries, and nothing here changes it, so this rewrite needs no numerical-drift
qualification: bit equality is a construction, and the component gate refuses on
the first differing bit rather than scoring a tolerance. Contrast the two rejected
shapes, both of which change the summation tree and would therefore be a NEW class
needing drift rows (Darklanes #534 showed a changed f32 tree flips near-tie
argmax): per-CTA partial sums combined in a second phase, and a single wider block.

The price is a redundant row read per CTA. At 4096 f32 that is 16 KB re-read
`tiles` times, which lands in L2 after the first CTA touches the row, against an
epilogue that becomes `1/tiles` as wide per CTA. Because every CTA still runs the
whole reduction, the sweep's asymptote as `tiles` grows was expected to BE the
reduction floor, with the gap between `tiles=1` and that floor the only thing this
door can buy. The component gate reports that sweep rather than assuming it, and
what the sweep reports is confirmed as the reduction floor by a
second card class, see the discussion under Evidence below. The real ceiling on the
door is the 1.240690 ms/step the kernel it replaces costs in the live model.

The launcher pins block 128 (the tree is the contract) and requires
`128 * tiles` to divide `n`, so no CTA is empty and every thread writes the same
number of columns. `tiles = 1` reproduces the original geometry through the wide
symbol and is the sweep's own red arm.

No launch count moves: 86 packs per rank and forward variant in both arms, one
symbol or the other, never both and never neither. No other kernel changes. FFI:
`src/dsv4_ffi.rs`; component gate: `src/dsv4_norm2_wide_component_gate.rs`;
replay and sampled gate: `dsv4-norm2-wide-gate`.

Evidence (2x RTX PRO 6000 Blackwell Max-Q dev pair, head d710438fb, binary
cb270f66, 2026-09-10): the component cell passed raw-bit equality at all 172 live
pack sites (13,588 / 13,588 comparisons, warm and cold, tiles 1..32; the warm sweep
on a sample site runs 12.383 us at tiles=1 down to 6.730 us at tiles=32, 4.63 GB/s
unique rising to 8.52 GB/s unique at 83.99 GB/s issued, the predicted L2-served
re-read). The model program on the pinned default shape (PRIME=256, OUTPUT=256):
wide 55.154288 tok/s vs narrow 52.054319 pooled ABBA (+5.955257%, mean +5.9542%)
and wide 55.192155 vs narrow 52.191374 pooled reverse (+5.749573%, mean +5.7550%),
steady ranges disjoint in both orders (3.019 and 2.946 tok/s separation), 40/40
rows eligible with zero looped rows, and every row in both orders carries the same
generated/logits/cache/hidden digests as the narrow arm (same-class identity held
at model scale, not just per-site). Four qualify processes (two per arm) matched
the same digests. Receipts banked in darklanes
`research/dsv4f-norm2-wide-20260909/`.

The sweep's asymptote IS the reduction floor this section predicted, and a first
reading that called it an instrument floor was withdrawn on 2026-09-10. The test was
to run the same sweep on a second card class: an instrument overhead is roughly
constant and would not scale, a computational floor does. It scales. Dev pair Max-Q
`tiles=1` 10.439-12.854 us with floor 6.084-6.909 us; prod-candidate Workstation 600 W
`tiles=1` 8.122-8.213 us with floor 3.941-4.087 us. Both absolutes move about a third
while `floor / tiles=1` stays at 0.5375-0.5883 and 0.4803-0.4987.

So the component number is a real prediction: 86 launches at a 5.653-5.945 us warm
saving is 0.486-0.511 ms/step (0.370 on the cold rows). The measured model return is
1.073033 ms/step forward and 1.057288 reverse, i.e. **2.1x to 2.9x the prediction**,
and below the 1.240690 ms/step the single-CTA pack costs in the live model. The factor
of two is UNEXPLAINED. The surviving hypothesis is that replay forward span is not
additive in isolated kernel durations, which Darklanes #562 measured in the other
direction on this same path: 0.164670 ms/step of kernel time removed returned
-0.026864 ms/step of replay forward. Isolated per-launch timing on this path predicts
the sign of a graph-replay change and not its size.

Sanitizers (2026-09-10, on MERGED main `4eaf708e7` rather than the lane head, binary
`f553bc452ad4398efc90a38216c1972f3254aaabe50c7d34cbb1b017b7830b58`, 2x RTX PRO 6000
Blackwell Workstation Edition prod-candidate box): `compute-sanitizer --tool memcheck`
and `--tool synccheck` each report `ERROR SUMMARY: 0 errors` under `--error-exitcode 99`,
each over 344 `COMPONENT` rows and 13,588 comparisons with the wide kernel engaged at
`tiles=32`. Non-vacuity is asserted in the controller, not assumed: each sanitizer log
must carry its own row count and `COMPONENT_PASS` line or the cell fails.

That cell also captured its OWN operands on that box instead of inheriting the
norm-fuse2 lane's, and reproduced
`COMPONENT_PASS sites=172 references=172 comparisons=13588 bits_equal=true class=same`
against them, so the same-class claim now rests on two independent captures taken on two
different card classes. All 172 captured operand descriptors read `4096 1`: the pack refuses
any other shape, so 2 sites x 43 layers x 2 ranks is the entire call-site domain rather
than a sample of it.

These rows are CORRECTNESS ONLY. That box is a different card class from the pair the
door was scored on, and no absolute timing from it enters this door's verdict.
