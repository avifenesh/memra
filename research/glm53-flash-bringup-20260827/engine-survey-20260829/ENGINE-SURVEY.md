# How vLLM and SGLang serve glm5_next (GLM-5.3-Flash): engine survey and transfer map

Lane: `lane/glm5-engine-survey` (2026-08-29/30, web + source survey, read-only, no engine changes).
Owner question, after the step37 sweep: the OpenRouter providers serve this model on vLLM and
SGLang at full context and high throughput; incoai benched a DFlash2 drafter on SGLang at
1.73x-2.79x. What do their glm5_next paths actually do, and what should memra copy?

Method: both engines' actual GLM-5.3-Flash support branches were fetched and read as source
(vLLM PR #53906 head, SGLang PR #36507 and #36708 heads, fla upstream `ad4af37f` of 2026-08-29),
plus the official recipes/cookbook pages, the zai model card, the incoai drafter card, and the
OpenRouter endpoint API. Every number below is quoted from its named source. Figures marked
ARITHMETIC are derived, not measured. Where a statement is a reading of code rather than a
measured behavior, the file path is the citation.

Support status, stated precisely: neither engine has glm5_next on `main` as of 2026-08-30.
vLLM support is PR #53906 ("[Model] add GLM-5.3-Flash support", OPEN, branch `glm-release`,
opened 2026-08-26, "This PR needs flashinfer v0.6.18rc10"). SGLang support is PR #36507
("GLM-5.3-Flash support", OPEN, support branch `xinyuan/glm-5.3-flash-support`, opened
2026-08-26); the DFlash2 capture PR #36708 is MERGED (2026-08-27) into that support branch, not
into main. Providers run these branch builds / vendor docker images (the vLLM recipe says "use
docker before the integration is included in the public repo"; the ROCm image is
`vllm/vllm-openai-rocm:glm53-flash`). "Working, optimized paths TODAY" is true, on pinned
branches.

Checkpoint geometry cross-checked against our banked `../glm-config.json` (identical to what
both engines consume): 45 layers, 34 KDA + 11 MLA/DSA (`kda_layers`/`full_attn_layers` lists),
288 routed experts + 1 shared, top-8, sigmoid + `noaux_tc`, `routed_scaling_factor 2.5`,
`swiglu_limit 10.0`, kv_lora 512, q_lora 1536, qk_nope 256, qk_rope 0 (`mla_use_nope: true`),
`index_topk 2048`, `index_kpool 4`, `index_n_heads 32`, `index_head_dim 128`, mHC `hc_mult 4`,
`hc_sinkhorn_iters 20`, KDA heads 64 x dim 128, conv 4, `gate_lower_bound -5.0`, 1 MTP layer,
1,048,576 max positions.

---

## 1. vLLM (PR #53906, branch `glm-release`)

Model package: `vllm/models/glm5next/` (nvidia/amd split), registered as
`Glm5NextForCausalLM` / `Glm5NextForConditionalGeneration` / `Glm5NextMTPModel`. Files:
`nvidia/model.py` (1,195 lines), `nvidia/kda.py` (604), `nvidia/attention.py` (590),
`nvidia/mtp.py` (429), plus shared layers `model_executor/layers/mhc.py` (665) and
`model_executor/layers/sparse_attn_indexer_kpool.py` (1,112).

### 1.1 KDA kernels: the fla library, vendored. Chunk size 64.

- Decode and spec-verify run `fused_recurrent_kda`; prefill runs `chunk_kda_with_fused_gate`.
  Both come from a vendored copy of the fla library:
  `vllm/third_party/flash_linear_attention/ops/kda.py` (1,723 lines, MIT, "Related files are
  modified and supported by the Moonshot AI Team" in upstream).
- Chunk size: `FLA_CHUNK_SIZE = 64` (`vllm/third_party/flash_linear_attention/ops/utils.py:31`),
  with intra-chunk sub-blocks `BC = min(16, BT)` (kda.py `chunk_kda_scaled_dot_kkt_fwd`).
- The chunked prefill pipeline (`_chunk_kda_fwd_with_cumulative_g`, kda.py:1360) is exactly the
  WY/Gcum decomposition: (1) `fused_kda_gate_chunk_cumsum` (gate from A_log/dt_bias fused with
  chunk-local cumsum, output pre-scaled by `RCP_LN2` so downstream kernels use exp2), (2)
  `chunk_kda_scaled_dot_kkt_fwd` (A = beta * K K^T with the per-channel cumulative gate `gk`
  applied; the intra-chunk Aqk is "kept in fp32; the computation has very marginal effect on the
  entire throughput", kda.py comment), (3) `solve_tril` (the UT/WY triangular inverse), (4)
  `recompute_w_u_fwd` (w, u, kg), (5) `chunk_gated_delta_rule_fwd_h` (cross-chunk state
  recurrence, `use_exp2=True`), (6) `chunk_gla_fwd_o_gk` (output; writes into the v buffer to
  save an allocation). Six kernel-launch stages per layer call, intra-chunk parallel, state
  carried across chunk boundaries.
- The safe gate is first-class: `safe_gate: bool` + `lower_bound` parameters ("GLM-5.3-Flash
  uses a bounded sigmoid gate instead of the default unbounded softplus gate",
  `glm5next/nvidia/kda.py:270-273`; checkpoint `gate_lower_bound -5.0`).
- Decode: `fused_recurrent_kda` is one Triton kernel per layer per step with the gate computed
  IN-KERNEL (`COMPUTE_GATE=True`: "replicates fused_kda_gate's arithmetic bit-for-bit and skips
  its launch + fp32 [n, H, D] intermediate per layer", kda.py model file comment), beta
  sigmoided in-kernel (`SIGMOID_BETA`), and on pure-decode / pure-verify steps the kernel writes
  straight into the layer output buffer (`out=` parameter), skipping the merge copy.
- State dtypes: conv state follows `mamba_cache_dtype` (model dtype), recurrent state is pinned
  `torch.float32` (`mamba_utils.py kda_state_dtype`, returns `(state_dtype, torch.float32)`).

### 1.2 Projection layout: merged GEMMs, and the KDA trunk is deliberately BF16

- All six KDA input projections (q, k, v, b, f_a, g_a) are merged into ONE GEMM:
  `in_proj_qkvbfg_a` ("Merge q, k, v, b, f_a, g_a projections into one GEMM (6 to 1 launches)",
  `glm5next/nvidia/kda.py:171`), with f_a/g_a shards replicated across TP ranks.
- The three short convolutions run as ONE merged `causal_conv1d` over q|k|v ("bit-identical to
  three calls", kda.py:377-383); the merged weight is built once and cached.
- KDA projections are BF16 by design: the layer is constructed with `quant_config = None`
  because "KDA projections remain BF16 because fp8 checkpoints omit their scales"
  (kda.py:154). They then run through vLLM's ordinary BF16 tensor-core GEMM path in BOTH
  phases. The FP8 checkpoint's q_a/kv_a/q_b/o_proj MLA projections are block-dequantized to
  BF16 at load (`_try_load_fp8_attn_proj`, model.py), and the vision tower likewise stays BF16.
  No engine dequantizes a BF16-resident weight to f32 at prefill; there is no f32 GEMM anywhere
  on this path.

### 1.3 Hybrid KDA+MLA cache and prefix caching

- The model implements `IsHybrid`/`HasInnerState`; KDA layers get (conv_state, recurrent_state)
  pools via `MambaStateShapeCalculator.kda_state_shape`, MLA layers get the paged latent cache,
  and vLLM's hybrid KV-cache manager auto-aligns mamba and attention block sizes (comment at
  `glm5next/nvidia/model.py:965-968`).
- The conv-state width includes `num_spec` "so the spec-decode conv update ... can slide the
  window across the draft-verify tokens without reading past the allocated width" (kda.py
  `get_state_shape` comment).
- Prefix caching over linear state is a three-mode policy, `vllm/config/cache.py:188`:
  `mamba_cache_mode` = "none" (prefix caching off), "all" ("cache the mamba state of all tokens
  at position i * block_size"), "align" ("only cache the mamba state of the last token of each
  scheduler step and when the token is at position i * block_size. This is the default when
  prefix caching is enabled"). "align" requires chunked prefill; `mamba_block_size` is aligned
  to the attention block size. Models opt into "all" via the
  `SupportsMambaPrefixCaching` interface (`interfaces.py:1100`, "currently experimental");
  Glm5Next does NOT declare it, so it serves prefix caching in "align" mode.
- Two further state-traffic tools exist in the same config block: `use_replayssm` (ReplaySSM:
  "cache recent SSM inputs and skip the per-step full-state store, writing the checkpoint back
  only on flush", default buffer 16) and `use_kda_recoverssm` (Kimi-K3 KDA speculative decode).

### 1.4 CUDA graphs: the whole hybrid decode step is captured, "breakable"

- All Glm5Next architectures are in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`
  (`vllm/config/vllm.py:75-93`), alongside DeepseekV32/V4, KimiK3/KimiLinear and MiniMaxM3.
- Breakable CUDA graphs (`vllm/compilation/breakable_cudagraph.py`) replace FX-graph
  pre-splitting with runtime stream-capture breaks: "a single capture context drives the whole
  forward and intercepts attention / kv-cache custom ops at the dispatcher to end the current
  stream capture, run the op eagerly, and resume capture. The captured artifact is a list of
  zero-arg callables". The docstring credits the idea to sgl-project/sglang#19102 ("Introduce
  CUDA graph debug mode with breakable CUDA graph", merged 2026-04-11).
- The KDA layer's host-side branching `_forward` is decorated `@eager_break_during_capture`
  (kda.py:326), so its prefill/decode dispatch runs eagerly between graph segments.
- The Sinkhorn is NOT a dynamic bit: mHC pre/post ops are TileLang kernels with the Sinkhorn
  loop unrolled in-kernel as a fixed-count serial loop
  (`vllm/model_executor/kernels/mhc/tilelang_kernels.py`, `for _ in T.serial(sinkhorn_repeat - 1)`
  at three sites; `mhc_sinkhorn_iterations` default 20 from config). Fixed iterations, one fused
  kernel per site, graph-safe by construction. Backends: tilelang (CUDA), triton fallback,
  aiter (ROCm), torch native.
- The DSA indexer runs under breakable-CG in eager segments, with its small-op clusters carved
  into `@torch.compile` leaves to recover fusion ("the MLA indexer runs under breakable-CG
  (CompilationMode.NONE), which blocks FX-graph fusion of the surrounding eager ops; carving
  each cluster into its own @torch.compile leaf still fuses them",
  `glm5next/nvidia/attention.py:41-50`).
- Separately, `MambaModelConfig` defaults hybrid/mamba models to `FULL_AND_PIECEWISE` CUDA graph
  mode ("required to get good performance for mamba layers in V1",
  `vllm/model_executor/models/config.py:608-617`).

### 1.5 mHC: expand once, contract once, fuse post+pre between layers

- `hc_expand` runs ONCE at layer 0 and `hc_contract` ONCE after the last mHC layer; in between,
  each layer's deferred `hc_post` is fused with the NEXT operation's `hc_pre` (+ the RMSNorm)
  into a single kernel, `MHCFusedPostPreOp` ("inter-layer fusion", model.py forward, lines
  440-510). The 4-stream state (`post`, `comb`) is carried between layers as tensors.
- The mix weights `hc_attn_fn`/`hc_ffn_fn` are `[(2+n)*n, n*hidden]` fp32 parameters; per layer
  there are exactly two fused mHC kernel sites (attn and ffn), not a per-site
  pre/post/expand/collapse chain.
- MTP layers and non-mHC configs skip mHC entirely (plain residual path).

### 1.6 DSA indexer + kpool: compressed fp8 pool cache, chunked prefill, short-prefill bypass

- The indexer K cache stores kpool-COMPRESSED entries: one fp8 entry (plus fp32 scale per 128
  elements) per `index_kpool` tokens, expressed as `tokens_per_state = index_kpool` on the KV
  spec so "vLLM's indexer metadata builder emit[s] pool-granular slot_mapping / seq_lens /
  cu_seq_lens / page_table for free, and shrinks the cache allocation" (Glm5NextIndexerCache
  docstring, attention.py:78-100). Pool content is a softmax-weighted sum driven by a learned
  gate (`index_kpool_compress_gate`) plus a learned APE (`index_kpool_compress_ape`).
- A separate paged TAIL cache (`Glm5NextTailCache`, `KpoolTailSpec`) holds the in-progress
  pool's raw bf16 K + gate score, one block per request, "overwritten in place by pos % kpool";
  prefill seeds it, PD-disaggregation transfers it, decode reads it to compress the boundary
  pool correctly.
- Indexer arithmetic: q is rotated by a fused fp32 FWHT-128 + fp8 quant kernel
  (`fwht128_quant_fp8`, "avoids an intermediate HBM round-trip and bf16 matrix-rounding bias");
  the head gate weights are computed in fp32 ("bf16 error can change near-tie pool rankings on
  long-context tasks", attention.py:322-333); wk and weights_proj are fused into one GEMM
  (`wk_weights_proj`); logits run on DeepGEMM `fp8_mqa_logits` / `fp8_fp4_paged_mqa_logits`
  (heads zero-padded up to the kernel's {32, 64} head requirement).
- Prefill is CHUNKED with a bounded transient: the logits workspace is capped by
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512, `vllm/envs.py:60`), K is gathered from the
  paged cache into a shared workspace per chunk (`cp_gather_indexer_k_quant_cache`), top-k runs
  as a custom CUDA op `top_k_per_row_prefill` selecting `select_k = topk_tokens // kpool` POOLS
  (512 pools for 2048/4), then one fused Triton kernel expands pools to token indices and
  appends the causal tail ("replaces ~25 elementwise ops",
  `sparse_attn_indexer_kpool.py:590-605`).
- SHORT-PREFILL BYPASS: when `max_prefill_seq_len <= topk_tokens` (2048), the indexer scoring is
  skipped entirely and the top-k buffer is filled with all causal token indices
  (`sparse_attn_indexer_kpool.py:436-468`). Prompts at or under 2,048 tokens never pay for the
  indexer.
- The top-k buffer is preallocated `[max_num_batched_tokens, topk + tail]` rounded up to
  128-column tiles ("Sparse MLA tiles top-k in 128 columns; padded slots remain masked",
  model.py:587-605).
- DeepGEMM's paged-MQA constrains `block_kv` to exactly 32 or 64, which forces
  `cache block_size` to be a multiple of `index_kpool * 32` (guarded up front in
  `get_kv_cache_spec` rather than "the opaque C++ assert").

### 1.7 MoE: the standard fused-MoE stack, 288 experts is not special

- `Glm5NextMoE` wraps `FusedMoEFactory` with `num_experts=288`, `top_k=8`, grouped top-k,
  sigmoid scoring, `e_score_correction_bias` for `noaux_tc`, `routed_scaling_factor 2.5`, and a
  PRE-CLAMPED SwiGLU (`SiluAndMulWithClamp`, `swiglu_limit 10.0`). The router GEMM
  (`GateLinear`) computes logits externally in a router dtype
  (`_get_moe_router_dtype`, reused from deepseek_v2). Nothing glm5-specific in the expert GEMM
  path: it rides the same fused_moe backends as DeepSeek-class models.
- EPLB is wired (redundant experts, `n_physical_experts`), plus sequence-parallel MoE mode
  (tokens sharded across TP ranks around the expert computation, DSv4 pattern: mHC runs on the
  SP shard, attention gathers, reduce-scatter after).

### 1.8 MTP / speculative decoding

- Native MTP: `vllm/config/speculative.py` maps `model_type glm5_next` to `glm5_next_mtp`
  (`architectures ["Glm5NextMTPModel"]`). The MTP layer is DeepSeek-style enorm/hnorm/eh_proj
  with a fused kernel (`fused_eh_norm`: "zero pos-0 embeds + enorm(embeds) + hnorm(prev) + cat")
  and an MLA+indexer mixer (with `skip_topk` / `compact_topk_indices` hooks for the draft loop).
  No mHC in the MTP layer.
- The official recipe serves `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`.
- KDA under spec verify: spec tokens are routed to `fused_recurrent_kda` with
  `spec_state_indices_tensor` where "Spec tokens carry num_spec+1 recurrent-state columns each
  and are advanced with num_accepted_tokens for rejection-sampling rollback" (kda.py:395-400).
  Rollback is therefore per-step state columns + accepted-length commit, not recompute.

### 1.9 Parallelism

- PP is GATED OFF for this model: "PP is gated off for GLM-5.3-Flash (no
  make_empty_intermediate_tensors)" and the deferred mHC post/comb state "are not propagated
  across PP ranks" (model.py:676-694). Parallelism is TP (heads divide evenly; recipe TP4), EP
  (+EPLB), sequence-parallel MoE, and PD disaggregation (NIXL/Mooncake connectors; recipe pins
  `VLLM_SSM_CONV_STATE_LAYOUT=DS` and `VLLM_KV_CACHE_LAYOUT=HND` identically on both pools, and
  notes the KDA conv-state and KV layouts "must be pinned identically").

---

## 2. SGLang (PR #36507 support branch; DFlash capture PR #36708 merged into it)

Model file: `python/sglang/srt/models/glm5_next.py` (1,921 lines) + `glm5_next_nextn.py`,
config `srt/configs/glm5_next.py`.

### 2.1 Maximal reuse: MLA is DeepSeek's, MoE is DeepSeek's

- The MLA/DSA layers ARE `DeepseekV2AttentionMLA` (`from ...deepseek_v2 import
  DeepseekV2AttentionMLA`, constructed with `skip_rope=True` for NoPE).
- The MoE IS DeepSeek's: `from sglang.srt.models.deepseek_v2 import DeepseekV2MoE as
  Glm5NextMoE` (glm5_next.py:107). GLM-5.3-Flash is also registered in the DeepSeek-family
  server-arg overrides (`arg_groups/overrides.py`, `_deepseek_family_overrides`), which default
  `attention_backend = "dsa"` and resolve KV dtype / MoE runner per device. The 288-expert MoE
  therefore inherits the whole DeepSeek stack: fused_moe triton, DeepEP/EPMoE, FlashInfer
  TRT-LLM MoE on Blackwell (the incoai bench ran "FlashInfer TRT-LLM MoE for the target"),
  two-batch overlap (TBO), single-batch overlap (SBO), EPLB, shared-expert fusion.
- Same projection fusions as vLLM, independently arrived at: `fused_qkvbfg_a_proj` (merged
  6-way input GEMM, when unquantized and TP-uniform), one merged qkv conv1d, and f_b/g_b as one
  BATCHED GEMM (`ColumnParallelBatchedLinear(2, head_dim, projection_size)`).

### 2.2 KDA kernel matrix: pluggable decode/prefill/verify backends, Triton (fla) default

`--linear-attn-backend` default is `"triton"` (server_args.py:2700). The dispatcher
(`layers/attention/linear/kda_backend.py`) selects independently per mode:

- decode: `triton` (fla fused_recurrent), `helion`, `cutedsl` (SM100), `flashinfer`
  (`recurrent_kda`, SM100). On SM100+ with `mamba_ssm_dtype bfloat16`, decode DEFAULTS to
  flashinfer (`server_args.py:6735-6756`). Note the state dtype: SM100 runs the KDA recurrent
  state in BF16; "SM90 uses float32".
- prefill (extend): `triton` (fla chunk kernel, chunk_size 64 in their vendored copy,
  `python/sglang/kernels/ops/attention/fla/kda.py:432`), `helion`, `flashkda` (a PREFILL-ONLY
  KDA kernel library; decode falls back to triton), `cutedsl` ("SM100 chunk prefill pipeline"),
  `nvidia_kda` (SM100), `ptx_kda` (SM103/GB300).
- verify: `triton` ("fused chain + tree (retrieve_parent_token) verify; the reference the KDA
  correctness tests assert against"), `flashinfer` (chain only), `nv_cutedsl`.
- "FlashInfer has no KDA chunk kernel" (kda_backend.py comment): even the kernel-library vendor
  does not ship a KDA prefill kernel; everyone's chunked prefill is the fla Triton form or an
  SM100-specific pipeline.

### 2.3 Hybrid cache: two pools, and the KDA pool is the concurrency limiter

- Two memory pools: the paged attention KV pool and a separate KDA state pool (`MambaPool` with
  per-layer conv + temporal states, `mem_cache/memory_pool.py:365`). The cookbook states it
  plainly: "GLM-5.3-Flash maintains a paged KV pool for attention and a separate KDA state
  pool. The KDA state pool can limit concurrency before the KV pool is full", tunable via
  `--mamba-full-memory-ratio` / `--max-mamba-cache-size`.
- ARITHMETIC on why: one KDA temporal state is 64 heads x 128 x 128 fp32 = 4 MiB per layer,
  x34 layers = 136 MiB per sequence (68 MiB in the SM100 bf16 configuration), plus conv states.

### 2.4 Prefix caching over linear state: SOLVED, on by default

This is the answer to our latent-plane question. SGLang has a mamba radix cache
(`mem_cache/mamba_radix_cache.py`: "The radix tree data structure for managing the hybrid (full
and Mamba) KV cache"), and `Glm5NextForConditionalGeneration` is enrolled in BOTH
`_MAMBA_RADIX_CACHE_ARCHS` and `_MAMBA_EXTRA_BUFFER_ARCHS` (`arg_groups/overrides.py:1781,1803`).

- Checkpoint granularity: state snapshots are named on a grid of
  `mamba_cache_chunk_size = max(FLA_CHUNK_SIZE (64) or the model's mamba chunk size, page_size)`
  (server_args.py:9929, runtime_context.py:1649).
- Storage diet: cached checkpoints live in an INT8 store with per-slot scales ("Why int8 (not
  fp8): a cached checkpoint is loaded ONCE on a cache hit, then decode continues from the
  dequantized state", `mem_cache/mamba_checkpoint_pool.py`), dequantized directly into the
  active pool on restore.
- Two strategies resolved automatically (`_mamba_radix_cache_resolution`): "extra_buffer"
  (MambaPool ping-pong plus "track-snapshot writes (decode + extend) so donated slots hold real
  states for prefix-cache restores"; requires the triton linear backend) and "no_buffer" (which
  sets `disable_overlap_schedule = True`).
- ReplaySSM interacts correctly with it: "the decode kernel force-flushes the ring into
  temporal[slot] on the radix track boundary seq_lens % mamba_track_interval == 0"
  (server_args.py:6830 area).
- The cookbook instruction is unambiguous: "Keep the prefix cache enabled for every strategy."

### 2.5 DSA/kpool, KV dtype pairing

- Same kpool concept as vLLM (their DSA stack: `layers/attention/dsa/dsa_indexer_kpool.py`,
  `kpool_fp8_index.py`, TileLang kernels, `kpool_topk_transform` JIT kernels).
- Blackwell pairing, measured: "the recipes default to an FP8 KV cache with TRT-LLM DSA: on
  GB300 this pairing measured 2.9-5.7% higher throughput and about 1.8x the KV token capacity
  at identical pool bytes, with GSM8K accuracy within noise of BF16" (cookbook). H100/H200
  default BF16 KV + TileLang DSA; "TileLang DSA with FP8 KV is not a valid CUDA combination".

### 2.6 mHC

- Fused `hc_pre` / `hc_post` ops (`sglang/kernels/ops/layernorm/mhc.py`), integrated through a
  dedicated `MHCLayerCommunicator` (plus `MHCHybridDSACPLayerCommunicator` for prefill context
  parallelism); an AMD fused post+pre boundary op exists
  (`deepseek_common/amd/deepseek_v4_fused_mhc.apply_mhc_post_pre_boundary`). Sinkhorn iterations
  are passed as a constant (`sinkhorn_iters=self.config.hc_sinkhorn_iters`); `hc_expand` /
  `hc_contract` bracket the stack, as in vLLM.
- The cookbook warns: "do not override linear_lower_bound" (the KDA bounded-gate parameter).

### 2.7 Speculative decoding: adaptive MTP by default, DFlash2 as the swap-in

- MTP is served as `--speculative-algorithm EAGLE` ("upstream folds the older NEXTN spelling
  into EAGLE"), and the recommended low-latency recipe is "adaptive MTP 5/1/6" (adaptive draft
  depth: "Adaptive MTP changes the draft depth as acceptance changes, reducing unnecessary
  draft work when the server is busy", cookbook).
- DFLASH is a first-class speculative algorithm (`srt/speculative/dflash_worker_v2.py` etc.).
  The draft is a block-diffusion model: "It predicts a whole block of tokens in a single pass
  and keeps the top candidates at every position. A lightweight selector then traces one
  coherent path through them" (incoai model card). Block layout: position 0 is the anchor
  token, gamma = block_size - 1 draft positions (dflash_worker_v2.py); the drafter runs on
  FlashAttention 4 (`--speculative-draft-attention-backend fa4`), not the target's DSA
  backends.
- The VERIFY is one target forward over the whole block: "The draft proposes a whole block per
  step and the target verifies it in one forward pass, so output quality stays the target's"
  (cookbook). T-parallel by construction; chain layout (topk <= 1) for the mamba path.
- KDA state on rejection, both spec families: the target runs verify over all draft positions;
  afterwards `update_mamba_state_after_mtp_verify` commits the state for the LAST ACCEPTED
  position via "a fused gather-scatter kernel" (`hybrid_linear_attn_backend.py:1233`,
  `scatter_mamba_states_after_mtp_verify`); with ReplaySSM-KDA "the accepted drafts live in the
  per-slot ring (written during verify); no intermediate_ssm is allocated. Replay the accepted
  prefix into temporal instead of scattering an intermediate state"
  (`commit_kda_replayssm_after_verify`, chain-only). Rollback is checkpoint-and-commit, never
  recompute-from-scratch.
- PR #36708 (merged) is only the capture seam: a standard `set_dflash_layers_to_capture` hook
  plus mHC contraction of the aux hidden states ("GLM-5.3-Flash keeps its inter-layer mHC state
  at hc_mult * hidden_size, while a DFLASH draft consumes one hidden_size stream per selected
  target layer", PR body; `_prepare_aux_hidden_state` applies `hc_contract`).
- Constraints they publish: neither MTP nor DFlash2 runs with DP-Attention; spec does not start
  under PD disaggregation at the current cut; DFlash2 needs the support branch and the drafter
  is access-gated CC BY-NC-ND.

### 2.8 Parallelism

- TP4/EP4 on 4x GB300 is the validated NVIDIA shape; TP8 single-node on AMD. PP metadata exists
  for the model (`test_glm5_next_pp_metadata.py`; pp_proxy carries hidden_states + residual).
  Decode context parallelism DCP4 is validated on 4x GB300 for long context; DSA prefill
  context-parallel communicators exist (`communicator_mhc_hybrid_cp.py`). PD disaggregation
  "moves both the paged DSA KV and the KDA recurrent state" and its validated arm "used the
  triton MoE runner; deep_gemm under PD is untested".

---

## 3. The fla library itself (fla-org/flash-linear-attention @ ad4af37f, 2026-08-29)

- `fla/ops/kda/`: `chunk.py` (autograd wrapper, `chunk_size: int = 64` default), `chunk_fwd.py`
  (same stage list as the vLLM vendored copy; `use_gate_in_kernel` fuses the gate into the
  cumsum), `chunk_intra.py` and `chunk_intra_token_parallel.py` (an intra-chunk token-parallel
  variant), `wy_fast.py` (the WY recompute), `fused_recurrent.py` (decode), `gate.py`,
  `naive.py` (reference). There is also `precond_kda/`.
- The kernel API already carries serving affordances: varlen `cu_seqlens`, `initial_state` /
  `output_final_state` (chunk-boundary carry), `return_intermediate_states` (checkpointing for
  prefix caches / spec), `ssm_state_indices` + `num_accepted_tokens` (paged state + spec
  rollback in the recurrent kernel), `safe_gate` / `lower_bound` (the GLM bounded gate), and a
  `cp_context: FLACPContext` (context parallelism inside the kernel wrapper).
- Numeric class: intra-chunk attention matrix fp32, decays evaluated with exp2 after RCP_LN2
  scaling, output/state in model dtype. Upstream treats the chunked form as the training and
  prefill kernel and the fused recurrent as the decode kernel; nobody bit-matches chunked
  against sequential.
- Relation to our L3 (`crates/memra-engine/src/kda.rs`, commit e69ed0600): same algebra, same
  default chunk 64 (ours: `MEMRA_KDA_CHUNK` default 64, clamped to multiples of 32 in [32,128]),
  same per-channel Gcum WY form, same "not bit-identical to sequential, band-gated" stance.
  Ours is 5 launches per layer call (cumgate, attn, solve, state, output) vs fla's 6-7 (their
  kkt runs as two intra/inter kernels); fla folds the output gate differently (we fold the
  inter-chunk output gate into K5 q staging). Same family, independently converged.

---

## 4. Performance anchors

All numbers quoted exactly from their sources.

### 4.1 incoai DFlash2 card (huggingface.co/incoai/GLM-5.3-Flash-DFlash2, fetched 2026-08-30)

Runtime: "SGLang on four NVIDIA GB300 GPUs (TP4), with TRT-LLM DSA and FlashInfer TRT-LLM MoE
for the target, FP8 target KV cache, and FlashAttention 4 for DFlash 2 draft attention"; block
size 8 (7 draft tokens per verification step); sampling temperature 1.0 / top-p 0.95, reasoning
effort Max.

- Autoregressive decode, concurrency 1: 146.8 (GSM8K), 157.5 (MATH-500), 166.6 (HumanEval),
  168.2 (MBPP), 169.3 (MT-Bench) output tok/s.
- Native MTP at 7 drafts: 282.6 (1.93x), 323.2 (2.05x), 323.5 (1.94x), 299.5 (1.78x),
  231.1 (1.36x).
- DFlash2: 355.4 (2.42x), 438.9 (2.79x), 436.8 (2.62x), 402.2 (2.39x), 293.2 (1.73x).
- Acceptance length (completion tokens per verify step): MTP 5.06 / 4.95 / 4.70 / 4.26 / 3.71;
  DFlash2 5.78 / 5.86 / 5.32 / 4.85 / 4.03.
- Concurrency 32: autoregressive 917.3-2,102.0 tok/s; DFlash2 1,318.3-4,198.4 tok/s
  (1.44x-2.01x).

### 4.2 SGLang MI355X recipe (sgl-project/sglang PR #36732, measured, FP8, TP8/EP1, 8x MI355X)

ISL/OSL 1,024/1,024, radix cache disabled, full decode graphs at bs 1 and 32:

- Concurrency 1: aggregate 287.8493 tok/s total (35.9812 tok/s/GPU), median TTFT 84.8435 ms,
  median TPOT 6.8675 ms, median interactivity 145.6126 tok/s/user.
- Concurrency 32: aggregate 5,731.3649 tok/s (716.4206 tok/s/GPU), median TTFT 514.8845 ms,
  median TPOT 10.6696 ms, 93.7246 tok/s/user.
- "Against the recorded 8x B200 baseline, MI355X reached 90.18% / 89.11% of B200 total
  throughput / median interactivity at concurrency 1 and 101.07% / 98.35% at concurrency 32."
- Accuracy on the same stack: GSM8K 1,288/1,319 = 97.65%.
- ARITHMETIC: TTFT 84.8 ms on a 1,024-token prompt is a node-level prefill class of roughly
  12,000 prompt tok/s (8 GPUs), i.e. ~1,500 tok/s per GPU on that card class, WITH the 1k
  prompt fitting entirely under the indexer's short-prefill regime.

### 4.3 vLLM recipe (recipes.vllm.ai/zai-org/GLM-5.3-Flash, fetched 2026-08-30)

- Weights "about 306 GiB for the default native FP8 checkpoint"; BF16 roughly twice that.
  A RedHatAI/GLM-5.3-Flash-NVFP4 variant exists whose "MoE experts are quantized to 4-bit
  (NVFP4), and it requires NVIDIA Blackwell GPUs" (same weight format family as our mint).
- Reference launch: TP4 on one GB200 tray, `--kv-cache-dtype fp8`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`.
- MI355X at TP4: "the FP8 checkpoint reports a 14.92M-token KV pool and ~113.81x max
  concurrency at 128K context; the BF16 checkpoint reports an 8.87M-token KV pool".
- FlashInfer 0.6.17+ required for NoPE sparse MLA (the PR itself pins v0.6.18rc10).

### 4.4 OpenRouter (provider listing from the public API, fetched 2026-08-30)

20 provider rows (Z.AI, Novita, DeepInfra, GMICloud, Modal, Parasail, Reka, Together,
Wafer, DigitalOcean, Morph, SiliconFlow, Friendli, Phala, Fireworks, Cloudflare, Io Net,
BaseTen, Venice, Relace). Nearly all serve context_length 1,048,576 at quantization fp8
(exceptions: Reka and Io Net 262,144; Cloudflare lists 1,310,720). Pricing clusters at
$0.075-0.15 per M input, $0.25-0.50 per M output, $0.015-0.03 per M cache read. The API
snapshot exposes no per-provider throughput fields (null in this response), so no tok/s claims
from OpenRouter are quoted here.

### 4.5 What the anchors mean against our numbers

Our banked figures: decode 21.3 tok/s on 4x RTX PRO 6000 Blackwell (dflash2-probe RECEIPTS,
03009909d; decode attribution: 17.1 ms/token launch-structure term above a 15.9 ms roofline,
`../decode-attribution-receipts/ATTRIBUTION.txt`), prefill 616-639 tok/s with L1 grouped
prefill ON (commit 1e0e12723: "7.2-7.4x TTFD (85 -> 616-639 tok/s prefill)").

- The GB300 (~8 TB/s HBM3e) autoregressive 146.8-169.3 tok/s at TP4 is roughly 4.5x our card's
  1.79 TB/s bandwidth. Scaled naively by bandwidth that is a ~33-38 tok/s class on our silicon;
  our BF16_MMV roofline is 99 tok/s and f32 is 63 (ATTRIBUTION.txt). ARITHMETIC, direction
  only: upstream decode sits ON its bandwidth roofline; ours sits on a launch-structure wall
  under the roofline. Closing X and running the BF16 trunk is worth more than any kernel swap.
- 22 -> 90 decode decomposes on engine evidence as: BF16 trunk residency (banked ceiling 63 ->
  99 tok/s), launch-diet + graph capture (upstream has no analog of our 17.1 ms X), then MTP
  multiplies whatever base rate remains by a measured 1.36x-2.05x (and DFlash2-class drafting
  by 1.73x-2.79x).

---

## 5. What memra should copy (ranked)

Ranked by expected value against decode 22 -> 90 tok/s and prefill 630 -> thousands.

### C1. Decode launch-diet on the KDA trunk: merge projections, fuse the gate in-kernel, write into the output buffer

Both engines independently converged on the same three moves: (a) one merged GEMM for
q|k|v|b|f_a|g_a (vLLM `in_proj_qkvbfg_a`, "6 to 1 launches"; SGLang `fused_qkvbfg_a_proj` plus
a batched f_b/g_b pair), (b) one merged causal conv over q|k|v ("bit-identical to three
calls"), (c) the KDA gate (A_log/dt_bias/sigmoid/lower-bound) and beta sigmoid computed INSIDE
the recurrent kernel, which "skips its launch + fp32 [n, H, D] intermediate per layer", with
the kernel writing directly into the layer output buffer on pure decode steps. Our decode
attribution proved X = 17.1 ms/token of launch structure, invariant across 5.2x transport and
2x residency; these are exactly the launches it is made of, plus the graph capture in C2.
Expected value: this is the decode blocker's own term. Target: pull decode off the launch wall
toward the banked BF16 roofline (99 tok/s), then let MTP multiply. Extends L2 (which currently
covers prefill GEMM class only) and the decode arc; contradicts nothing.

### C2. Whole-step graph capture with eager breaks (the breakable-graph pattern)

vLLM captures the ENTIRE hybrid decode step (34 KDA + 11 MLA/DSA + mHC + MoE) as one breakable
CUDA graph: runtime stream-capture that breaks out only at attention/kv custom ops, replaying
"a list of zero-arg callables". Default ON for glm5_next
(`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`). The dynamic bits are made static the same way we
would: Sinkhorn is a fixed 20-iteration serial loop inside one fused kernel; the indexer's
host-side branching runs in the eager segments. SGLang runs "full decode graphs at batch sizes
1 and 32" in its measured recipe. For memra this is the generalization of the launch-gap
solution: capture the per-step launch sequence once, replay it, keep the KDA/kpool metadata
updates in eager segments. Expected value: co-equal with C1 (it removes the same X term where
C1 cannot fuse further); it is also the piece we cannot get by kernel work alone.

### C3. Serve the native MTP head with per-step state columns and post-verify commit

The checkpoint ships an MTP layer we load and ignore ("The MTP layer is unused",
ATTRIBUTION.txt). Upstream, it is the single biggest measured decode multiplier: 1.36x-2.05x at
concurrency 1 (incoai card), acceptance 3.71-5.06 tokens per verify at 7 drafts, recipe default
in vLLM (num_speculative_tokens 5) and SGLang (adaptive 5/1/6). The KDA-state mechanics to copy
are concrete and small: verify runs all draft positions in one target forward (T-parallel,
chain layout); per-draft-step recurrent states are kept (num_spec+1 state columns in vLLM, a
per-slot ring in SGLang's ReplaySSM variant); after verify, one fused gather-scatter commits
the state of the last accepted position; the conv state is allocated num_spec wider so the
window slides across draft tokens. Expected value: a measured ~1.4x-2x multiplier on top of
C1+C2's base rate; 45 tok/s base x 1.9 clears 85; 50 clears 90. This validates the T-parallel
verify arc GO (03009909d) and supersedes its external-drafter dependency: the native head is
license-clean, and upstream's MTP acceptance (3.71-5.06) is measured HIGHER than our
teacher-forced DFlash2 probe cycles (3.06 all / 4.66 tool-wire), so the verify arc should serve
MTP first and treat block-diffusion drafting as a later upgrade.

### C4. Prefix caching over the latent planes: 64-grid state checkpoints in the radix/prefix tree

SGLang enrolled glm5_next in the mamba radix cache on day 0 and instructs "Keep the prefix
cache enabled for every strategy". The design to copy: snapshot the KDA recurrent state on a
fixed grid (max(kda chunk 64, page size)), store checkpoints INT8-quantized with per-slot
scales (loaded once per hit, dequantized into the active pool), take "track-snapshot writes"
during both decode and extend, and truncate prefix hits to the deepest checkpointed boundary
(vLLM's "align" mode is the same idea stated as policy: cache the state only when the last
token of a step lands on a block boundary). Our prefix cache currently refuses on glm5's
latent planes (`MEMRA_PREFIX_CACHE_MB=0`, restore defect); this gives L5 its target design and
its memory diet. Expected value: for our agentic multi-turn ICP it removes WHOLE prefills;
multiplies every prefill win; also the enabling condition for cheap multi-turn TTFT at 1M
context.

### C5. Indexer short-prefill bypass and pool-granular top-k

Two cheap, immediately checkable moves on the DSA side: (a) when max prompt position <=
index_topk (2048), skip indexer scoring entirely and fill causal indices (vLLM
`sparse_attn_indexer_kpool.py` short-prefill path); (b) score and select POOLS
(select_k = index_topk / index_kpool = 512 rows) and expand to token indices in one fused
kernel, with the tail pool always appended, and cap the prefill logits transient with an
explicit workspace budget (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, default 512). Also copy the
compressed indexer-K cache framing (one fp8 entry per kpool tokens plus a raw tail buffer) when
the 1M-context indexer memory starts to matter. Expected value: small now (our kpool bench is
ms-class), but it hardens the 1M lane: indexer transient memory becomes O(budget), not O(ctx),
and short agent turns never touch the indexer.

Runners-up, recorded not ranked: BF16 KDA recurrent state on Blackwell-class cards (SGLang SM100
default with flashinfer decode; halves state traffic; needs a numerics gate since vLLM keeps
fp32), FP8-KV + faster-DSA pairing (SGLang measured "2.9-5.7% higher throughput and about 1.8x
the KV token capacity"), and mHC inter-layer post+pre fusion (vLLM MHCFusedPostPreOp; our
hyper.rs already batches per-chunk, the deferred-post trick is the remaining ~2x on mHC launch
count, second-order by our own attribution).

---

## 6. What NOT to copy, and why

- **Cross-rank pipeline parallelism for this model.** vLLM gates PP off outright (no
  make_empty_intermediate_tensors; the deferred mHC post/comb state does not cross rank
  boundaries and would either stall the fusion or change numerics). SGLang carries PP metadata
  but its validated recipes are TP/EP only. Our single-process ppN placement is the right
  shape; do not spend a lane on mHC-aware PP.
- **The Python kernel-stack dependencies (DeepGEMM, TileLang, FlashInfer, Triton-JIT) as
  dependencies.** Copy designs (pool-granular cache, fused post+pre, breakable capture), not
  the stacks. Memra is the only runtime engine (house law); every design above is expressible
  in our CUDA/Rust seams.
- **The DFlash2 drafter as a serving component.** CC BY-NC-ND 4.0, access-gated, probe-only
  (already our law, RECEIPTS.md); the cookbook itself marks the combination "not yet measured
  on the cookbook hardware". The native MTP head is the production path (C3).
- **Softmax-router batched arms for this arch.** Both engines route sigmoid/noaux_tc through
  dedicated grouped-topk with the correction bias; our M3 gate-MISMATCH lesson (74602-vs-92)
  already codified this. L1 did it right.
- **EPLB / redundant experts / DeepEP wide-EP.** Datacenter EP-scale machinery; at our
  single-box TP4 scale it buys nothing and adds a rebalance state machine. Revisit only with a
  multi-node lane.
- **FP8 KV cache as a default flip.** Upstream's pairing rules are strict (TRT-LLM DSA + FP8 KV
  valid, TileLang + FP8 invalid; Hopper unsupported). On our card class it is unmeasured; per
  the flag-default law it would need its own gated cell with sampled twins, not a copy.
- **vLLM's "align"-mode subtlety as an excuse to skip decode-time snapshots.** SGLang's
  extra_buffer strategy exists precisely because align-style caching alone starves hits under
  overlap scheduling and paging; if we build L5, build the snapshot writes on both decode and
  extend from the start.

---

## 7. Three most surprising things

1. **Nobody wrote a bespoke production KDA kernel.** Both engines vendor the fla library's
   Triton chunk kernel (chunk 64, WY/Gcum, exp2, fp32 intra scores) for prefill and its fused
   recurrent for decode; FlashInfer, the kernel vendor of record, "has no KDA chunk kernel"
   (SGLang kda_backend.py). The most exotic layer in the model is served by library code, and
   the engineering went into seams instead: merged projections, in-kernel gates, state pools,
   graph capture. Our L3 kernel is not behind the state of the art; it IS the state of the art
   form, minus their backend menu.
2. **Prefix caching over recurrent state is solved and ON upstream.** SGLang snapshots the KDA
   state on a 64-token grid into an int8 checkpoint pool inside the radix tree and tells
   operators to keep it enabled for every strategy; vLLM ships three mamba cache modes with
   block-aligned state checkpoints. The "linear attention cannot prefix-cache" assumption our
   pinned-off cache reflects is dead upstream.
3. **The model is running around an unused 2x we already possess.** The single biggest measured
   decode multiplier on GLM-5.3-Flash is its own MTP head (1.36x-2.05x at c=1, acceptance
   3.71-5.06 at 7 drafts), served by default in both engines' recipes, with the KDA rollback
   problem reduced to per-step state columns plus one post-verify scatter. We load that layer
   and ignore it.

Honorable mention: vLLM turning OFF pipeline parallelism for a flagship 320B model because mHC
made the inter-layer seam stateful, and borrowing SGLang's breakable-graph idea (sglang#19102)
to graph-capture around it.

---

## 8. Cross-reference against our lanes: validates / contradicts / extends

| Finding | Lane | Verdict |
|---|---|---|
| All engines run MoE prefill as grouped/batched expert GEMMs (FusedMoE stack; DeepseekV2MoE alias) | L1 grouped prefill (ON, 616-639 tok/s, dd7f1d11d) | VALIDATES; nothing upstream suggests a further prefill dispatch class beyond grouping + batching across requests |
| vLLM keeps every KDA projection BF16 by construction and serves both phases through BF16 tensor-core GEMMs; FP8 checkpoint MLA projections are dequantized to BF16 at load; no f32 GEMM exists on the path | L2 tc-trunk (pending box A/B, 88c4841a4) | VALIDATES the direction and the acceptance stance: the BF16 numeric class is the industry-serving class for this trunk; the near-tie argmax flip blocking MEMRA_PP_BF16 should be adjudicated by the logit-delta cell / owner acceptance named in the FLAGS row, not by byte identity. EXTENDS: merge the 6 KDA input projections and the 3 convs (C1), which L2 does not yet cover |
| fla chunk kernel: chunk 64 default, per-channel Gcum WY form, solve_tril, fp32 intra scores, exp2 decays, band-not-bit numerics, state carried across chunks | L3 chunked KDA scan (pending throughput A/B, e69ed0600, MEMRA_KDA_CHUNK default 64) | VALIDATES exactly, including our chunk-size default and numeric-class stance; the only remaining question is our throughput A/B. EXTENDS: fla ships an intra-chunk token-parallel variant (`chunk_intra_token_parallel.py`) if the A/B shows intra-chunk parallelism is the binding term |
| Verify = one target forward over the draft block; per-step KDA state columns / ring; post-verify gather-scatter commit of last accepted; conv state widened by num_spec | T-parallel verify arc (GO, 03009909d) | VALIDATES the GO; EXTENDS with the concrete rollback design and two upstream costs we had not scoped: the num_spec+1 state-column memory (or the ReplaySSM ring to avoid it) and the wider conv state |
| Native MTP head default-on in both engines, measured 1.36x-2.05x; acceptance 3.71-5.06 at 7 drafts; adaptive depth 5/1/6 | Verify arc sequencing | EXTENDS/CORRECTS the arc's drafter assumption: serve the native MTP head first (license-clean, higher measured acceptance than our external-drafter probe's 3.06 all-traffic), keep DFlash2-class drafting as the later upgrade |
| SGLang mamba radix cache: 64-grid int8 state checkpoints, snapshot writes on decode+extend, prefix cache ON for every strategy; vLLM mamba_cache_mode all/align | L5 prefix-cache re-enable (blocked on latent-plane restore defect) | VALIDATES that the lane is worth building and EXTENDS it with the full design (grid, int8 store, dual-phase snapshot writes, hit truncation to checkpointed depth) |
| Indexer short-prefill bypass (<= 2048 causal), pool-granular top-k, MB-capped chunked prefill workspace, compressed fp8 pool cache + raw tail buffer | kpool/1M lanes (kpool bench banked ms-class) | VALIDATES that MLA/DSA is not the wall today; EXTENDS the 1M lane with O(budget) transient memory and the short-turn bypass |
| Sinkhorn = fixed-iteration serial loop inside one fused kernel; mHC = expand once, contract once, fused post+pre per site | mHC (second-order per our attribution) | VALIDATES our per-chunk batching; EXTENDS with the deferred-post inter-layer fusion if mHC ever becomes visible post-L1/L2/L3 |
| vLLM PP gated off for glm5_next (mHC state does not cross ranks) | Our single-process ppN placement | VALIDATES; do not build cross-rank PP for this model |
| GB300 TP4 autoregressive decode 146.8-169.3 tok/s; MI355X x8 145.6 tok/s/user TPOT 6.87 ms; both ride their bandwidth rooflines | Decode attribution (X = 17.1 ms/token; rooflines 63/99 tok/s) | VALIDATES the attribution's claim that our wall is launch structure, not model cost: upstream's decode sits on the roofline our banked arithmetic predicts once X is removed and the trunk is BF16 |

### Sequencing changes recommended (for the owner and the next lane cuts)

1. **L2 gains a decode twin and rises to co-equal priority with L3.** Upstream evidence makes
   the BF16 trunk the serving class in both phases; L2's box A/B should keep its three
   factorized arms but the flip decision should also read the decode-side MEMRA_BF16_MMV
   ceiling (99 tok/s) as part of the same acceptance, and the near-tie argmax blocker should go
   to the logit-delta cell already named in the FLAGS row.
2. **L3 is confirmed as-built; its A/B is a pure throughput question.** No redesign indicated.
   If the A/B disappoints, the named follow-up is fla's intra-chunk token-parallel variant, not
   a different algebra.
3. **The verify arc reorders: native MTP head before external drafter.** New first milestone:
   MTP-head forward + T-parallel verify + per-step state columns + post-verify scatter commit,
   gated with our existing byte-identity decode gates on the non-spec path (vLLM keeps decode
   and verify on the same recurrent kernel, which is also our L3 seam's stance).
4. **A decode launch-diet + step-capture lane (C1+C2) should be cut before or alongside the
   verify arc**, because the spec multiplier applies to its output: 21.3 x 1.9 = 40 tok/s, but
   (post C1/C2 base ~45-60) x 1.9 clears the 90 target.
5. **L5 (prefix cache) inherits a concrete upstream design** and stops being blocked on
   "design unknown"; it remains blocked on the latent-plane restore defect lane, which now has
   a template for what restore must produce.

---

## Source index

| source | pin |
|---|---|
| vLLM GLM-5.3-Flash support | PR vllm-project/vllm#53906, OPEN, branch `glm-release`, fetched 2026-08-30; files cited: `vllm/models/glm5next/nvidia/{model,kda,attention,mtp}.py`, `vllm/model_executor/layers/{mhc.py,sparse_attn_indexer_kpool.py}`, `vllm/model_executor/kernels/mhc/tilelang_kernels.py`, `vllm/third_party/flash_linear_attention/ops/{kda.py,utils.py}`, `vllm/compilation/breakable_cudagraph.py`, `vllm/config/{vllm.py,cache.py,speculative.py}`, `vllm/model_executor/models/{config.py,interfaces.py}`, `vllm/transformers_utils/configs/glm5_next.py`, `vllm/envs.py` |
| SGLang GLM-5.3-Flash support | PR sgl-project/sglang#36507, OPEN, branch `xinyuan/glm-5.3-flash-support`, fetched 2026-08-30; files cited: `python/sglang/srt/models/glm5_next.py`, `srt/layers/attention/linear/{kda_backend.py,kernels/*}`, `srt/layers/attention/hybrid_linear_attn_backend.py`, `srt/mem_cache/{mamba_radix_cache.py,mamba_checkpoint_pool.py,memory_pool.py}`, `srt/arg_groups/overrides.py`, `srt/server_args.py`, `python/sglang/kernels/ops/attention/fla/kda.py` |
| SGLang DFlash capture | PR sgl-project/sglang#36708, MERGED 2026-08-27 into the support branch; `srt/speculative/dflash_worker_v2.py` |
| SGLang breakable-graph origin | PR sgl-project/sglang#19102, MERGED 2026-04-11 |
| SGLang MI355X measured recipe | PR sgl-project/sglang#36732 (body, with engine/AITER revisions pinned in it) |
| fla library | github.com/fla-org/flash-linear-attention @ `ad4af37f` (2026-08-29), `fla/ops/kda/` |
| vLLM recipe | recipes.vllm.ai/zai-org/GLM-5.3-Flash (fetched 2026-08-30) |
| SGLang cookbook | docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash (fetched 2026-08-30) |
| Model card | huggingface.co/zai-org/GLM-5.3-Flash; tech report arXiv 2602.15763; upstream reference `src/transformers/models/glm5_next/modular_glm5_next.py` (our banked copy: `../modular_glm5_next-ref.py`) |
| DFlash2 drafter + bench | huggingface.co/incoai/GLM-5.3-Flash-DFlash2 (fetched 2026-08-30); inco.ai/blog/dflash2; github.com/z-lab/dflash |
| OpenRouter provider list | openrouter.ai/api/v1/models/z-ai/glm-5.3-flash (public listing API, fetched 2026-08-30) |
| Our lanes | `../prefill-gap-20260829/PREFILL-GAP.md`; `../decode-attribution-receipts/ATTRIBUTION.txt`; L1 commits 1e0e12723 / dd7f1d11d; L2 branch `lane/glm5-tc-trunk-prefill` @ cb6600490; L3 branch `lane/glm5-kda-chunk-scan` @ e69ed0600; DFlash2 probe branch `lane/glm5-dflash2-probe` @ 03009909d |
