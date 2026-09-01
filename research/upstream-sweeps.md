
## Sweep 2026-07-15T07:30:04Z (since 2026-07-15T00:00:00Z)

### llama.cpp commits (decode-relevant, CUDA)
- (none)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-07-27T06:17:01Z (since 2026-07-15T07:30:04Z)

### llama.cpp commits (decode-relevant, CUDA)
- cohere2 moe template parser: enforce JSON schema for text responses if a response schema is provided (#26018)
- common : auto-download dflash- and eagle3- HF sidecars (#25811)
- conversion: fix non-MoE NomicBert GGUF conversion error (#25996)
- convert : fix dflash target tokenizer mismatch during conversion (#25733)
- cuda: add sqrt_softplus in topk-moe for dsv4 (#25896)
- cuda : CUDA GGML_OP_LIGHTNING_INDEXER implementation (generic vector kernel + wmma kernel) (#25545)
- CUDA: dedup MoE gate/up activation quantization (#25441)
- cuda: extract Q1_0 elements via __byte_perm (#25628)
- CUDA: fix external compilation of q1_0 MMQ (#25778)
- cuda: GET_ROWS quants (#25962)
- CUDA: Improve NVFP4 W4A4 activation quantization (#25730)
- cuda : relax tensor contiguity requirements for quantized concat (#25678)
- CUDA: Support CUDA Virtual Devices (#25228)
- CUDA: tighter MMQ src1 buffer size for native fp4 (#25613)
- CUDA: vectorize same-type get_rows with int4 copy (#25929)
- DeepseekV4: Add fused hyper-connection ops (#25585)
- DeepseekV4: reduce graph splits (#25702)
- Enable CUDA graphs on volta+turing (#25749)
- metal: fuse snake activation (mul, sin, sqr, mul, add) (#25459)
- model: rotate injected K/V cache for DFlash (#25823)
- mtmd : use align_corners for qwen3vl vision position embedding interpolation (#25781)

### vllm-project/vllm releases
#### v0.26.0 (2026-07-27T01:06:58Z)
- * **New Inkling model family** with a full support stack: base modeling (#48799), piecewise CUDA graph support (#48822), Hopper FA4 relative attention (#48858), MTP=1 speculative decoding (#48869), LoRA (#48884), and standard ModelOpt NVFP4 quantization (#48990).
- * **DeepSeek-V4 performance push** across vendors: a specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` (1.5–2x kernel, #47463), and redundant repeat/copy removal (1.8% E2E TPOT, #48137), plus ROCm two-stage compressor for HCA prefill (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), and DSpark speculative decoding on AMD (#47419) and XPU (#47677).
- * **Flexible attention backends**: the attention backend can now be selected per KV-cache group (#48012), and sliding-window support is now an explicit backend capability (#48011) — improving support for hybrid models.
- * **KV offloading & tiered secondary storage** matured substantially: offloading metrics (#45958, #47666, #47679), tier-owned event handling (#46544, #47923), object-store secondary tier with workload identity (#47063, #47274, #48150), DP-replica-aware tiering (#47987), and encoder-cache (EC) connectors including CPU offloading (#42433, #47423).
- * New models: Inkling family (#48799, #48822, #48858, #48869, #48884, #48990), BertForMaskedLM (#48463), RobertaForTokenClassification / XLMRobertaForTokenClassification (#47991), LongCat-Flash-Lite n-gram embedding (#47857), Cosmos3 Edge Reasoner (#48291) and Cosmos3-Super registration (#48211), TranslateGemma-12b-it (#41599).
- * GLM5.2: migrate MoE sequence-parallel support to the non-torch-compiled path (#47881).
- * LoRA: FlashInfer MoE LoRA for BF16 models (#48632), LoRA for tower/connector in LlavaNextVideo (#48594), fp32 `lm_head` on the LoRA path (#48525), optimized `TrtLlmLoRAExperts` (#48759).
- * fp32 `lm_head` for generation models via `head_dtype` (#48390); lower memory for capturing large CUDA graph sizes (#48483); opt-in persistence and reuse of the memory-profiling result across boots (#47388); improved InstantTensor loading (#46868).
- * Attention: select a different attention backend per KV-cache group (#48012); sliding-window as an explicit backend capability (#48011); KV-cache layout refactor packing K/V into the content dim across backends (#44455); MRV2 virtual-batch PCP for MLA (#46570).
- * Speculative decoding: runtime draft weight update (#46725), hybrid (SWA + full attention) DFlash drafters (#47914), SWA support for qwen-eagle3 (#47568), Gemma4-12B DSpark draft model (#47216), DSv4 DSpark on AMD (#47419), separate `kv_cache_dtype` for `speculative_config` (#48787).
- * KV offloading: basic offloading metrics (#45958), split CPU cache usage into read/write gauges (#47666) and tiering-lookup-delay into sync/async histograms (#47679), tier-owned event handling and BlockStored events (#46544, #47923), object-store secondary tier with workload identity (#47063, #47274, #48150), DP-replica-aware tiering (#47987), `blocks_per_chunk` config for heterogeneous KV groups (#48878), P2P default host/port env vars (#47636).
- * DeepSeek-V4: specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` 1.5–2x (#47463), redundant repeat/copy removal (1.8% TPOT, #48137).
- * MoE router GEMMs: BF16x3 router GEMM (#47973), FP32 router GEMV (#48335), generic CuteDSL LL BF16 router GEMM (#42562); TRTLLM BF16 MoE modular kernel (#45182); write FlashInfer combine into final output (#47156).
- * Qwen: fuse more RMSNorm + all-reduce in Qwen3.5 (#46998), replace MoE all-reduce with reduce-scatter (#47006), Qwen3.5 H20 optimization (#48350), expand Triton warmup coverage (#47546).
- * MLA: dense MHA path for short sparse-MLA sequences (#47327); MiniMax-M3 long-context decode indexer on sm100 (#48582).
- * Kernels: CUDA kernel for ReLUSquaredActivation / relu^2 (#39058), Helion kernel lazy registration (#48264), vectorize `_copy_mamba_state_block` to uint64 (#48110), stop upcasting logits to fp32 in the sampler (#48641).
- * ROCm: fp32 `head_dtype` `torch.mm` fast path (#48688), DSv4 two-stage compressor kernel (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), DSv3.2 sparse MLA KV-split heuristic (#46832) and MTP CUDA-graph mode (#45149), MXFP8 GEMM for MiniMax-M3 (#46117), AITER sparse paged attention + spec decode for MiniMax-M3 (#47287, #47984), MiniMax-M2 fused QK-norm + all-reduce via AITER (#44849), HybridW4A16 linear kernel (#40977), Qwen3-30B-A3B QK-Norm+RoPE+KV runtime fusion (#42749).
- * XPU: batch-invariant kernels (#41934), HND KV layout support (#47975), DSpark spec decode for DSv4 (#47677), nightly/release image publishing (#47880, #48126).
- * CPU: DFlash speculative decoding for GDN models on CPU (#46090), s390x NUMA topology (#40714), native macOS arm64 CPU wheel builds (#48289); POWER VSX math function optimization (#47321) and IBM Power docker builds using prebuilt wheels (#46017).
- * Distributed fusion: FlashInfer MNNVL all-reduce RMS quant fusion (#48064).
- * Build/autotune: arm64 Blackwell SM10x/SM110 image builds (#48041); skip CuTeDSL fp4_gemm autotuning by default (#48268).
- * Decode Context Parallel (DCP): hybrid attention support (#40996), DCP + Eagle for Tokenspeed MLA backends (#48180).
- * Humming w[2-7]a[4,8] weight-only inference with compressed-tensors (#46390); int4 quantization for the emulation MoE backend (#48451); INT2 XPU weight-only quant linear (#47521).
- * NVFP4/MXFP4: `nvfp4_per_token` online MoE quantization (#48538), CuTe-DSL FlashInfer MXFP4 quantization (#48417); bounded peak memory when repacking FP4 MoE weights for Marlin (#47851) and for NVFP4 MoE weight loading (#46276).
- * MLA: `kv_cache_dtype_skip_layers` support (#47309).
- * Transformers 5.13.0 (#47867), FlashInfer 0.6.14 (#47669), NIXL 1.3.1 (#47559), tpu-inference v0.24.0 (#47835), nvidia-cutlass-dsl 4.6.0 (#47442), vllm_xpu_kernels v0.1.11.1 (#48942).
- * FlashAttention 3 pinned to the torch stable-ABI commit (#47995); ABI-stable FlashMLA build (#48174).

### sgl-project/sglang releases
#### v0.5.16 (2026-07-25T00:13:18Z)
- **DSpark: confidence-driven speculative decoding**: A new speculative algorithm. It drafts semi-autoregressively in blocks, then sizes each verify window from the draft's own confidence instead of a fixed draft length. Reaches **383.7 tok/s at accept length ~5** on DeepSeek-V4-Pro, TP8 on B300 (bs=1). Enable with `--speculative-algorithm DSPARK` and `SGLANG_RAGGED_VERIFY_MODE=compact`; tune the block with `--speculative-dspark-block-size` ([#30261](https://github.com/sgl-project/sglang/pull/30261), [#31434](https://github.com/sgl-project/sglang/pull/31434), [blog](https://www.lmsys.org/blog/2026-07-06-dspark-sglang)).
- **Inkling support**: A 975B-parameter multimodal MoE with a 1M-token context. It mixes sliding-window, full and Mamba2 linear attention, and adds an NVFP4 MoE, optional vision/audio towers and native MTP. On Blackwell it reaches up to **71.7k tok/s input** and **171.0 tok/s per-user decode**. Verified on Blackwell TP4/TP8, H200 and AMD MI350X / MI355X ([#31681](https://github.com/sgl-project/sglang/pull/31681), [blog](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support), [cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling)).
- **Other new models added**: [LongCat 2.0 FP8](https://docs.sglang.io/cookbook/autoregressive/Meituan/LongCat-2.0), JetBrains Mellum v2, [Pi0.5](https://docs.sglang.io/cookbook/vla/OpenPI/Pi0.5), plus diffusion support for [LongLive 2.0](https://docs.sglang.io/cookbook/diffusion/LongLive/LongLive-2.0).
- **GLM-5.2 DSA cache layer split under prefill CP**: KV and indexer cache layers are sharded across CP ranks. Each rank owns a disjoint layer range instead of all layers. That cuts per-rank KV memory by **~74%** (0.77 to 0.20 GB/rank) at 8192 tokens on GLM-5.2-FP8, 78 layers, cp_size=4. Enable with `--enable-dsa-cache-layer-split`, which needs `--enable-prefill-cp --cp-strategy interleave` ([#29421](https://github.com/sgl-project/sglang/pull/29421)).
- **ReplaySSM Ring Spec-Verify (GDN)**: Drops the per-draft SSM snapshot. Speculative scratch goes from **11.5 GB to 1.8 GB per GPU (6.4x smaller)** on Qwen3.5-35B-A3B at TP1, at accuracy and throughput parity. Opt in with `--enable-gdn-replayssm-spec` (default off; GDN with a linear draft chain only, `--speculative-eagle-topk` in {None, 1}), and tune the ring via `--linear-replayssm-cache-len` ([#28695](https://github.com/sgl-project/sglang/pull/28695)).
- **Linear attention on Blackwell (SM100)**: The first correct KDA MTP path. Its `recurrent_kda` decode kernel runs at **29.6 us vs 36.8 us** for Triton (ncu, B=64). The full decode path reaches parity by B=128 and **1.35x at B=256**, and is slower below that ([#30113](https://github.com/sgl-project/sglang/pull/30113)). Separately, GDN/KDA CuteDSL prefill fuses state I/O into the chunk-h kernel ([#30169](https://github.com/sgl-project/sglang/pull/30169)).
- **QServe and FBGEMM FP8 quantization are removed**: the experimental QServe (QoQ) W4A8 and FBGEMM FP8 paths are gone. `--fp4-gemm-backend cutlass` goes too, along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer ([#31109](https://github.com/sgl-project/sglang/pull/31109), [#30448](https://github.com/sgl-project/sglang/pull/30448)).
- **Dependencies**: flashinfer 0.6.14 ([#29910](https://github.com/sgl-project/sglang/pull/29910)), CuTe DSL 4.6.0 ([#31714](https://github.com/sgl-project/sglang/pull/31714)), sgl-kernel 0.4.5 ([#31496](https://github.com/sgl-project/sglang/pull/31496)), llguidance 1.7.6 ([#31484](https://github.com/sgl-project/sglang/pull/31484)).
- * **`--fp4-gemm-backend cutlass` is removed** along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer. Use `auto`, which picks `flashinfer_cutedsl` on SM100 and `flashinfer_cutlass` on SM120: [#30448](https://github.com/sgl-project/sglang/pull/30448)
- * **The SGLang-Diffusion post-training rollout endpoint now returns `application/msgpack`** instead of JSON, with tensors as raw msgpack bytes rather than base64 (`tensor_to_base64` / `base64_to_tensor` become `tensor_to_bytes` / `bytes_to_tensor`), so RL rollout consumers must be upgraded in lockstep with the server: [#31565](https://github.com/sgl-project/sglang/pull/31565)
- * **Temperature-0 nondeterminism under DP attention with breakable prefill CUDA graph.** On the DSV4-Flash FP4 recipe, the idle-rank dummy extend introduced by [#30898](https://github.com/sgl-project/sglang/pull/30898) perturbs real requests' logits, so identical temperature-0 requests can diverge. The guarding determinism test is disabled as a stopgap rather than fixed ([#31125](https://github.com/sgl-project/sglang/pull/31125)); not enabling breakable prefill CUDA graph avoids the path.
- * A bump to **flashinfer 0.6.15** was landed and reverted this cycle; this release pins **0.6.14** ([#31502](https://github.com/sgl-project/sglang/pull/31502), [#31625](https://github.com/sgl-project/sglang/pull/31625)).
- * **CPU AMX optimizations for diffusion** were reverted ([#28527](https://github.com/sgl-project/sglang/pull/28527), [#30716](https://github.com/sgl-project/sglang/pull/30716)).
- | **LongLive 2.0** | diffusion | [#27639](https://github.com/sgl-project/sglang/pull/27639) | [link](https://docs.sglang.io/cookbook/diffusion/LongLive/LongLive-2.0) |
- * [Docs] Inkling cookbook: LoRA cells require --disable-prefill-cuda-graph: [#31418](https://github.com/sgl-project/sglang/pull/31418)
- * [Spec] fix inkling multi layer mtp draft extend cuda graph: [#32254](https://github.com/sgl-project/sglang/pull/32254) (cherry-picked as [#32260](https://github.com/sgl-project/sglang/pull/32260))
- * [Fix] Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay: [#30839](https://github.com/sgl-project/sglang/pull/30839)
- * [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly: [#28416](https://github.com/sgl-project/sglang/pull/28416)
- * Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing): [#31001](https://github.com/sgl-project/sglang/pull/31001)
- * [DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill into the DSA Backend (DeepSeek-V3.2): [#30514](https://github.com/sgl-project/sglang/pull/30514)
- * Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2: [#30272](https://github.com/sgl-project/sglang/pull/30272)
- * [DSA] Fix top-k v2 emitting invalid indices under tie overflow / inf scores (IMA in FA3 sparse decode): [#30645](https://github.com/sgl-project/sglang/pull/30645)
- * [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph: [#31705](https://github.com/sgl-project/sglang/pull/31705)
- * Fix nvfp4 online scale with pcg: [#32246](https://github.com/sgl-project/sglang/pull/32246) (cherry-picked as [#32259](https://github.com/sgl-project/sglang/pull/32259))
- * Fix stale flashinfer-MLA fallback poisoning spec verify capture (trtllm_mla + tc_piecewise): [#32288](https://github.com/sgl-project/sglang/pull/32288) (cherry-picked as [#32346](https://github.com/sgl-project/sglang/pull/32346))
- * flashmla: sync-free spec via device-side draft-extend: [#31090](https://github.com/sgl-project/sglang/pull/31090)
- * [Spec] DFlash: remove per-step host syncs so the CPU runs a full step ahead (spec-v2 overlap): [#31468](https://github.com/sgl-project/sglang/pull/31468)
## Piecewise & Breakable CUDA Graph
- * Enable breakable prefill CUDA graph for DP attention: [#30898](https://github.com/sgl-project/sglang/pull/30898)
- * feat: enable piecewise prefill graph for Kimi K2.5/K2.7: [#30889](https://github.com/sgl-project/sglang/pull/30889)
- * [Diffusion] Enable breakable CUDA graph (BCG) for diffusion DiTs: [#27436](https://github.com/sgl-project/sglang/pull/27436)
- * [KDA] Add FlashInfer SM100 KDA decode + MTP (target_verify) backend: [#30113](https://github.com/sgl-project/sglang/pull/30113) ⭐
- * [GDN/KDA] Fuse SM100 CuteDSL prefill state I/O into the chunk h kernel: [#30169](https://github.com/sgl-project/sglang/pull/30169) ⭐
- * [GDN] Auto-select FlashInfer GDN prefill on validated SM100 configs: [#29734](https://github.com/sgl-project/sglang/pull/29734)
- * [Feature] Add FP4 KV Cache Design and support SM120 GPUs: [#21601](https://github.com/sgl-project/sglang/pull/21601)
- * Fuse the preprocess kernels of trtllm-gen attention: [#29690](https://github.com/sgl-project/sglang/pull/29690)
## MoE & Expert Parallelism
- * Support Waterfill with MegaMoE backend: [#27350](https://github.com/sgl-project/sglang/pull/27350)
- * Support Flashinfer one-sided A2A + CuteDSL MoE for Nemotron Ultra: [#28309](https://github.com/sgl-project/sglang/pull/28309)
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-03T21:57:39Z (since 2026-07-27T06:17:01Z)

### llama.cpp commits (decode-relevant, CUDA)
- add rdna3.5, and 3 to mmq configs so they can be tuned independently. (#26199)
- chat : add qwen3 specialized parser (#26252)
- conversion: fix Qwen2.5-Omni mmproj conversion regression (#26262)
- CUDA: Add backend sampler for penalties sampler (#25262)
- CUDA: add Q2_0 support (#25707)
- cuda: extract Q2_0 elements via __byte_perm (#25603)
- CUDA: Fix data-races when reusing SMEM in block_reduce (#26385)
- DeepseekV4 MTP + DSpark (#25784)
- ggml-cuda: add chunked SSD matmul for Mamba-2 prefill acceleration (#22675)
- ggml-cuda: Allow transpose-free gemmv computation (#26171)
- ggml-cuda : disable MMQ on devices with less than 48 KiB shared memory (#26141)
- ggml: use dynamic allocation for split graph inputs (#22789)
- graph : fix unused input tensors in minimax m3 graph (#26519)
- model: MTP support for Qwen3-Next (#25589)
- model : support MTP in GLM-4.7-Flash (#24868)
- Remove custom cpu op from the M3 graph, express with stock ops (#26297)
- Support rotated kv cache quant (#26180)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-05T12:20:53Z (since 2026-08-03T21:57:39Z)

_Frame: self-competition (llama benching stopped 2026-08-03); upstreams are idea mines._
_Filter: sm_120/Blackwell, FP8/FP4 (FP8-ST program = priority 1), MTP/spec-decode_
_(MTP-p2 lane), chunked-prefill determinism (open chunk-order reduction-stability_
_finding), session/prefix affinity, batch-invariance._

### llama.cpp commits (decode-relevant, CUDA)
- #26605 — fit: include NextN/MTP layers in auto VRAM fitting (alloc crash with
  `--n-cpu-moe=0`). CARE: MTP-p2 must account for draft-layer VRAM in memra's budget
  fitting; check our accounting before the lane merges. lever-queue: no (checklist item).
- #26389 — server: spec-decode counters on /metrics, incl. per-draft-position accepted
  counter (vLLM schema). CARE: dogfood found short-ctx sampled acceptance 0.55 vs 0.73 —
  per-position acceptance telemetry in memra-serve /metrics turns that from a posthoc dig
  into a live gauge. lever-queue: YES (serve, small).
- #26510 — speculative: refactor common_speculative_init config dedup. SKIP: code hygiene only.
- #26524 — samplers: drop "full-context window" from history-based penalizers (backend
  sampler init-order constraint). SKIP for the mechanism; NOTE: llama continues moving
  samplers GPU-side (#25262 family) — same direction as our lsampler.
- #26531/#26577 — loader: allow tensor reshape during load + dflash wo_a fix. SKIP: loader flexibility.
- #7bd8282c3 #26510 / #26618 / #26254 / mtmd batch — SKIP: conversion, TTS, multimodal.
- PR #24364 (OPEN, updated 2026-08-05) — force NVFP4 W4A8 path for W4A16_NVFP4 layers on
  Blackwell via new `GGML_HINT_NO_QUANT_SRC1` hint (GGUF metadata-driven, mul_mat +
  mul_mat_id); measured quality gain vs native W4A4, cites W4A4-hard-for-small-LLMs.
  CARE: converges exactly with FP8-ST — FP4 weights x FP8 activations as the
  quality-safe alternative to W4A4; bears directly on nvfp4-strict lane arm design.
  lever-queue: YES (priority — track the PR, steal the W4A8 framing).

### vllm-project/vllm commits
- #40372 (merged) — batch-invariant NVFP4 MoE via cutlass: invariance pinned with static
  asserts in the .cu + a dedicated bs1-vs-bsN test so nobody breaks it silently. CARE:
  the *pattern* — treat batch-invariance as a gated property with asserts + tests, not a
  posthoc observation. Adopt for memra's chunk-order stability work. lever-queue: YES.
- #38561 (closed for rebase, not rejected) — batch-invariant chunked prefill for
  mamba/hybrid: force chunk splits ONLY at mamba chunk boundaries + align prefix-cache
  entries to the same grain. CARE: DIRECT hit on our open chunk-order
  reduction-stability finding — vLLM independently hit reduction-order instability
  across chunk sizes and fixed it by constraining split points to a fixed grain.
  lever-queue: YES (top candidate).
- #45683 (open) — deterministic MoE combine under VLLM_BATCH_INVARIANT: fixed-root
  reduce+scatter instead of routing-dependent reduce_scatterv. CARE: more evidence
  reduction-order instability is a live upstream problem (cross-rank flavor); memra is
  single-GPU so the fix itself is N/A. lever-queue: no (evidence note).
- #50992 — KV-offload ARC batch eviction was quadratic (rescan from head per evicted
  block); monotonic iterators + deferred mutations = 16-81x on eviction batches. CARE:
  dogfood found F5 evict-realloc on 12/12 requests — audit memra's host-cache/SLRU
  eviction for the same rescan shape while fixing F5. lever-queue: YES (pairs with F5).
- #48048 — first-class request-level `session_id` (HTTP X-Session-ID + engine plumbing),
  explicitly built for KV/prefix affinity consumers. CARE: session/prefix affinity for
  memra-serve (owner dogfoods multi-turn agentic); a session key is the cheap
  prerequisite for prefix-cache pinning. lever-queue: YES (serve, medium).
- #49969 — DSpark top-k Markov projection: select top-k from base logits once, apply the
  sequential per-step draft bias only to those candidates, scatter into dense row,
  sampling path unchanged. CARE: draft-cost pattern for MTP-p2 — restrict per-step
  draft-side correction to a candidate set instead of full vocab. lever-queue: YES.
- #50911 — fused non-causal TokenSpeed MLA for DSpark verify. SKIP: MLA-specific.
- #49792 — SM100 CuTeDSL fused query kernel (DSA fused_q). SKIP: model/arch-specific.
- #48861 — NVFP4 quant out_dtype must match model dtype, not torch default. SKIP: their-stack bug.
- #50323 — CI: fail evals when NaNs appear in logits. NOTE: cheap evidence-discipline
  gate, matches our battery philosophy. lever-queue: no.

### sgl-project/sglang commits
- #33063 — trtllm_mha decode: stop allocating per-layer scratch inside the decode CUDA
  graph — 3 fill launches/layer baked into every replay, AND they sat between PDL-linked
  kernels so the PDL dependent-launch chain signalled a FillFunctor instead of the real
  consumer. CARE: memra graph-decode + PDL on sm_120 — audit capture for allocs/fills
  inside the graph and for PDL chain breakage. lever-queue: YES (top candidate).
- #33306 — avoid TRTLLM prefill output copy: kernel writes directly into the
  preallocated piecewise-graph output buffer via `out=`. CARE: same audit for memra
  prefill outputs — any epilogue copy into a graph buffer is free meat. lever-queue: yes (small).
- #32575 — build empty-prefix last_loc sentinel on-device: inline `torch.tensor([-1])`
  per fresh request = pageable H2D + stream sync draining the in-flight forward. CARE:
  audit memra-serve batch-prep for host-materialized scalars/masks; the F5/prime-chunk
  work touches exactly this region. lever-queue: yes (audit item).
- #33545 — optimistic prefill with L2 hierarchical cache + write-back. NOTE: prefix-cache
  tiering direction; memra single-box, low priority. lever-queue: no.
- #16072-adjacent SGLang #33427/#33598 — post-capture KV sizing + clearer reservation
  logs. SKIP: parity, memra sizes before capture.
- #30206 (merged this window) — capture legal multi-request prefill CUDA graph batches.
  SKIP: their piecewise-prefill bug.
- #81c7a54ec — sm_100f (family) instead of sm_100a for sgl-kernel. NOTE: family-arch
  flag now load-bearing upstream; memra stays sm_120a per doctrine. SKIP.

### flashinfer-ai/flashinfer commits
- #4210 — remove NVFP4 TMA input padding copy: TMA tensor map describes physical M rows,
  G2S TMA out-of-bounds zero-fill covers padded scale-layout tiles — kills the
  non-128-aligned-M copy cliff (212µs -> 59µs at M~32k, B200). CARE: FP8-ST/NVFP4
  prefill activation-quant path + the prime-chunk/prefill wall — odd-M chunks are
  exactly where padded staging copies bite; the OOB-zero-fill trick is HW-portable to
  sm_120 TMA. lever-queue: YES (top candidate).
- #4263 — 64-bit row addressing in per-token NVFP4 quantizer (int32 overflow at large
  M*K). CARE: one-line audit of memra quantizer index math at 27B prefill shapes. lever-queue: audit.
- #3523 — sm120 groupwise GEMM called cudaGetDeviceProperties (~1.7ms, synchronous) per
  invocation with dead results. CARE: grep memra host launch paths for per-call device
  queries. lever-queue: audit.
- #4202 — fp8 e5m2 output in rmsnorm_quant / fused_add_rmsnorm_quant. NOTE: fused
  norm->FP8-quant epilogue is the FP8-ST activation-side lever; they now cover both
  e4m3/e5m2. lever-queue: yes (FP8-ST reference impl).
- cake_kda / delta-rule / MxInt4 MoE family — SKIP: model- or SM100-specific.

### NVIDIA/TensorRT-LLM commits
- #17234 — opt-in pinned staging for weight-load H2D: pageable H2D degrades to a ~260µs
  per-work-item driver polling crawl on GB300 (36min-4h loads). PARITY: memra spill
  doctrine already mandates bounded pinned host buffers — this is the receipt for why.
  lever-queue: no (doctrine confirmed).
- #16072 — pre-allocate CUDA-graph padding dummies during warmup; lazy alloc under KV
  saturation silently dropped graph coverage for the process lifetime. CARE: verify
  memra graph-decode preallocates padding resources at warmup, and that a failed graph
  path is loud, not a silent eager fallback. lever-queue: yes (gate item).
- #17159 — reject NaN top_p/min_p/temperature in SamplingParams. CARE: fold NaN
  validation into the lsampler top_p/min_p truncation bugfix already queued from
  dogfood. lever-queue: yes (rides the existing fix).
- #17172 — per-request seed in TorchSampler. PARITY: memra sampled-regime gates already
  seeded. SKIP.
- #16957 — TriAttention KV-cache compression (training-free decode-time eviction, ICML
  2026): score evictable region every beta tokens, keep budget, compact in place. NOTE:
  research file for long-context serve; quality-affecting, not a near lever. lever-queue: no.
- #17106 — enable KV block reuse for flashinfer backend. SKIP: parity.
- #16558 — DSA indexer k-cache only for owning layers. SKIP: model-specific.

### Ranked shortlist (lever candidates, with receipts)
1. **Chunk-boundary-pinned prefill invariance** (vLLM #38561 + #45683 + #40372): two
   independent vLLM lanes hit reduction-order instability (chunked prefill across chunk
   sizes; MoE combine across routing) and both fixes are the same shape — constrain the
   reduction segmentation to a fixed grain, then pin the property with static asserts +
   a bs/chunk-invariance test. Direct answer to our open chunk-order
   reduction-stability finding: align memra chunk split points to a fixed grain and add
   the invariance gate to the battery.
2. **Decode-graph scratch + PDL-chain audit** (SGLang #33063 + TRT-LLM #16072): fill
   kernels allocated inside graph capture replay forever AND break PDL dependent-launch
   chains; padding dummies allocated lazily silently kill graph coverage under load.
   memra runs graph-decode + PDL on sm_120 — one afternoon audit, high odds of found
   money on the felt decode path (the dogfood deficit).
3. **TMA OOB-zero-fill for odd-M prefill quant** (FlashInfer #4210): 3.6x on
   non-128-aligned M by deleting the padded staging copy and letting G2S TMA zero-fill.
   Priority-1 filter match (FP8/FP4 prefill); attacks the prime-chunk/prefill wall
   named in the dogfood head-to-head.
4. **W4A8 as the quality-safe NVFP4 activation path** (llama.cpp PR #24364, open): FP4
   weights x FP8 activations beats native W4A4 on quality with receipts; converges with
   FP8-ST (we already have e4m3 activation kernels) and reframes an nvfp4-strict arm.
5. **Spec-decode acceptance telemetry + draft-cost top-k** (llama.cpp #26389 + vLLM
   #49969): per-position acceptance counters on /metrics make the 0.55-vs-0.73 short-ctx
   acceptance finding a live gauge; DSpark's restrict-draft-correction-to-top-k pattern
   is portable to MTP-p2's per-step draft cost.

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-07T03:15:50Z (since 2026-08-05T12:20:53Z)

_Frame: self-competition; upstreams are idea mines. Filter: chunked-prefill/prime_
_invariance + reduction-order exactness (step35 just killed two segmentation axes —_
_chunk-boundary determinism is gold), spec/MTP acceptance + scheduling (concurrency-gated_
_spec live; PP-2+spec crash under debug), PP 2-stage on workstation GPUs, sm_120 FP8/NVFP4_
_MMA + block-scale, MoE residency/spill, serve QoS (admission/isolation/prefix-cache)._

### llama.cpp commits (decode-relevant, CUDA)

- #26672 — model-loader: fix quantized reshaped tensor strides — the #26531 reshape-on-load
  path built `nb` without accounting for quantized block types. CARE: memra loads GGUF; if
  we ever reshape quantized tensors at load, stride math must be block-aware. Audit-grade
  only — memra doesn't use llama's loader. lever-queue: no (their-stack bug).
- PR #26675 (OPEN, ggerganov) — **ggml_prec spec rework**: explicit per-op declaration of
  accumulator type AND src1 on-the-fly quantization type
  (`ggml_mul_mat_set_prec_acc(z, GGML_PREC_F16)`, `_set_prec_src1(z, GGML_PREC_Q8)`),
  with GGUF-carried per-op precision policy planned. This is the gate ggerganov set for
  W4A8 PR #24364 ("we need to agree on that before proceeding"). CARE: llama is
  formalizing exactly the thing memra's FP8-ST program hand-rolls — a per-op contract for
  activation-side quantization and accumulator precision. Two takes: (1) W4A8-on-Blackwell
  is now on llama's critical path, endorsed at the architecture level — more receipts for
  FP8-ST as THE prod direction; (2) the GGUF metadata-driven precision-policy idea is a
  clean pattern if memra ever wants artifact-pinned per-layer activation formats.
  lever-queue: YES (track both PRs; FP8-ST lane).
- #26649 — ggml_build_forward_order: expand-as-ordering-hint marked unselected branches
  for compute, running ops on never-uploaded inputs. SKIP: their graph-API footgun.
- #26645/#26656/#26660/#26665/#26613 — mtmd, server cors, conversion, grammar. SKIP.
- W4A8 PR #24364 — no code movement this window, but the #26675 dependency (above) is the
  first maintainer-driven unblock since it opened. Still the FP8-ST convergence signal.

### vllm-project/vllm commits

- #51113 (merged) — **mamba "align" chunking: prefix-cache poisoning past
  `last_cache_position`**. Their invariant: block-table slot p may only be hashed as
  state@(p+1)*block_size; chunk-end alignment was enforced only *below*
  last_cache_position, so under CONCURRENCY one request's mid-block chunk end left a
  slot holding state@364 that later got hashed as state@1600 — silent, persistent,
  poisoned every resume; single requests were accidentally safe. Fix: gate end-alignment
  on prefill_end (only the final chunk may end unaligned) + unconditional mid-block
  realign stop for off-grid resumes. CARE: DIRECT sequel to step35 — this is the
  *prefix-cache* flavor of the same disease (chunk segmentation must land on the state
  grain, at every boundary, including resume-from-cache starts). memra's chunk-fix keyed
  the SWA prefill arm on seq_end; the vLLM lesson is the *cache-entry* side: any state
  snapshot keyed by position must only ever be published at grain-aligned chunk ends,
  and the unaligned-START (resume off-grid) is a second hole. Audit memra prefix/session
  reuse for both when serve prefix-cache lands. Also note their test shape: a dedicated
  regression test on the split function. lever-queue: YES (top candidate).
- #48341 (merged) — async scheduling now DEFAULT-ON for draft-model spec decode (was
  classified unsupported by the default-resolution path; explicit enable already worked).
  CARE: spec + async scheduling co-existence is table stakes upstream now; memra's
  concurrency-gated spec (spec only when idle) is the conservative cousin. When MTP-p2
  revisits the gate, the receipt that draft-model spec runs under overlapped scheduling
  upstream is the bar. lever-queue: yes (MTP-p2 design input).
- #50183 (merged) — NaN in target logits → `tl.argmax` on all-NaN block returns
  out-of-range index → OOB read → IMA in the rejection sampler (hit during DSpark warmup).
  Fix: NaN→-inf before argmax + clamp index. CARE: memra's verify path does argmax over
  target logits; a NaN row must not turn into an OOB index. Fold a NaN-poisoned-logits
  case into the lsampler top_p/min_p bugfix battery (dogfood already found sampler
  truncation injecting low-id tokens — same neighborhood). lever-queue: yes (rides
  existing lsampler fix).
- #50939 (merged) — -1 placeholder draft-token ids reached block verification and caused
  OOB (greedy path was guarded, block-verify path was not). CARE: if memra's verify ever
  pads draft slots, the pad id must be rejected in EVERY verify flavor, not just the one
  that was tested. Checklist item for MTP-p2 verify. lever-queue: audit.
- #49206 (merged) — PRIORITY preemption silently skipped the next request for a whole
  scheduling step (req_index bookkeeping off-by-one when the preempted request sat before
  the cursor). CARE: serve QoS — admission/preemption bookkeeping bugs are silent
  starvation, exactly what memra-serve's isolation work must gate with a
  sustained-KV-pressure regression test, not eyeballs. lever-queue: yes (serve QoS test
  pattern).
- #51089 (merged) — request priority via HTTP header (`X-Request-Priority`-style parse into
  engine priority). NOTE: pairs with #48048 session_id from last sweep — the serve QoS
  ingress surface upstream is converging on header-carried per-request knobs.
  lever-queue: no (fold into the existing serve session/QoS item).
- #50507 (merged) — partial-tail prefix reuse for hybrid models: preserve/lookup/restore a
  fine-grained prefix boundary *inside* a physical block (block large due to mamba state),
  copy-on-write append after restore; 784→896 cached tokens on a 900-token prompt. CARE:
  same grain-vs-block tension memra will hit if prefix cache lands with large KV pages.
  lever-queue: no (design note for serve prefix-cache).
- #50230 (merged) — PDL for DSA decode kernels: Triton kernels get a USE_PDL constexpr with
  `gdc_wait()`/`gdc_launch_dependents()`; NVFP4 quant kernels switch to
  `cudaLaunchKernelEx` + programmatic stream serialization; +2.8% output tok/s at MTP=5.
  PARITY-ish: memra already runs PDL on sm_120; the news is they're threading PDL through
  the *spec-decode draft* path specifically. Cross-check MTP-p2 draft kernels are inside
  memra's PDL chain, not breaking it (last sweep's SGLang #33063 showed fills breaking
  chains). lever-queue: audit (extends existing graph/PDL audit item).
- #50904 (merged) — MTP draft loop skip_topk: reuse the step-0 top-k buffer across draft
  steps instead of recomputing indexer top-k per step (2x kernel-level). CARE: same
  draft-cost family as #49969 last sweep — compute once on the target pass, reuse across
  draft steps. Portable pattern for MTP-p2 per-step cost. lever-queue: yes (merge with
  the existing draft-cost item).
- #50029/#49764 (merged) — online NVFP4 expert packing: quantize each expert from the
  ORIGINAL tensor + FP32 global scale directly (the old path folded scale into BF16 and
  recast — an extra rounding step before group-16 scale selection); share online weight
  scales across TP. CARE: nvfp4-strict lane — if memra's NVFP4 repack ever stages through
  a scaled BF16 temporary, it's leaving quality on the floor; quantize from source with
  the scale applied in FP32. One-line audit of tools/repack paths. lever-queue: yes
  (nvfp4-strict audit).
- #50276 (merged) — packed KV block zeroing stride bug. SKIP: their layout.
- #50613 (merged) — per-request scheduling for MLA chunked context. SKIP: MLA-specific.
- #51304 — NaN-in-logits counts copied to host asynchronously (follow-up to the NaN CI
  gate from last sweep). NOTE: they made the NaN gauge cheap enough to leave on in prod.
  lever-queue: no.

### sgl-project/sglang commits

- #33587 (merged) — **WAR fences aligned with CUDA-graph metadata reads**: the overlap
  scheduler rewrites shared request/attention metadata while the previous forward still
  runs on another stream; the read_done fence could only be published before/after a whole
  graph replay, so prefill fell back to a coarse whole-forward barrier AND TRTLLM SWA
  cache writes re-read the live full-to-SWA mapping after an early fence instead of the
  snapshot. Fix: lift read_done after CG metadata init; snapshot SWA write locations
  during metadata prep. CARE: memra-serve's overlap between batch-prep and in-flight
  forward has the identical hazard shape — any host-side rewrite of metadata a replaying
  graph still reads is a silent racer. The *snapshot-at-prep, consume-snapshot-at-write*
  discipline is the portable fix. lever-queue: YES (serve overlap audit).
- #33253 (merged) — breakable-graph padding: padded `positions` reached the attention
  backend, and DCP used positions for KV ownership → virtual padded tokens competed with
  real tokens for physical KV slots, silently corrupting cache (GSM8K-verified fix).
  CARE: graph-decode padding hygiene — every tensor the backend consumes must be narrowed
  to the real token count, not just Q/K/V. memra pads decode batches to graph buckets;
  audit which per-token side tensors (positions, slot maps) flow in un-narrowed.
  lever-queue: YES (pairs with #33587).
- #33666 (merged) — PP: mamba pool sized per whole model instead of per stage → per-slot
  cost overestimated ~pp_size×, clamping max_running_requests to 6 on K3 PP8. Fix scales
  by the HEAVIEST stage's layer share so derived limits stay uniform across ranks without
  a collective (own-share sizing diverged in 95/145 budget sweeps). CARE: memra PP-2 — any
  per-stage resource (KV pool, spec scratch, draft KV) must be budgeted on the stage's own
  layer slice, and cross-stage-derived limits (micro-batch size) must be computed
  identically on both stages. Direct checklist for the PP-2 lane. lever-queue: YES.
- #32700 (merged) — SWA chunk-cap escape hatch fired under *transient* SWA pressure, not
  just true head-of-line livelock: admitting shrunken prefill chunks into headroom that
  running decodes needed collapsed the evictable cushion → retraction storm, 40%+ of
  prefill compute was re-prefill rework at 0.9 pool usage. Fix: hatch only when budget ≥
  whole pool (can never fit); transient pressure WAITS. CARE: serve admission — memra's
  admission gate must distinguish "can't fit now" (wait, protect decode cushion) from
  "can never fit" (act). The B<R / R≤B<C / B≥C three-case table is the cleanest admission
  framing seen upstream. lever-queue: YES (serve QoS design).
- #33794 (merged) — paged SWA retraction resume: align full+SWA preallocation to physical
  allocator pages; keep remaining SWA budget consistent when multiple retracted requests
  resume. NOTE: retraction-resume accounting is a bug-farm; memra-serve doesn't retract
  yet. lever-queue: no.
- #30393 (merged) — HiCache restores draft-side caches (packed tail-layers or sidecar
  pools) so an L2/L3 prefix hit doesn't silently drop spec acceptance because the draft
  KV/indexer state is missing. CARE: when memra prefix-cache + MTP coexist, a "cache hit"
  that restores only target KV will quietly halve acceptance — the failure is invisible
  without per-position acceptance telemetry (last sweep's #26389 item). lever-queue: yes
  (design constraint, serve prefix-cache × MTP-p2).
- #33788 (merged) — inference-mode mismatch in FlashInfer warmup. SKIP.
- #33663/#33527 (merged) — serving benchmark: post-warmup /flush_cache raced the scheduler
  (400 aborts); fix waits for idle with a timeout. NOTE: memra bench scripts flush between
  phases — same race shape if serve goes async. lever-queue: no (bench hygiene note).
- #33618 (merged) — MoE deferred finalize default-on. SKIP: their kernel plumbing.
- #33115/#33621 (merged) — ModelOpt FP4 online MoE weight quant + pinned 4over6 settings.
  NOTE: online-quant settings are getting pinned as correctness surface upstream;
  nvfp4-strict already treats them as locked. SKIP (parity).

### flashinfer-ai/flashinfer commits (v0.6.18 tagged this window)

- #4165 (merged) — **gate cuDNN out of SM12x bmm_fp8 auto**: on RTX 5090 the cuDNN FP8
  path without override_shape (a) builds a fresh execution graph per distinct (M,K,N) —
  measured 236ms host-side PER SHAPE, unbounded under serving load; (b) raises
  cudaErrorMisalignedAddress ASYNCHRONOUSLY, uncatchable at the call site. CARE: two
  general laws for the FP8-ST lane on sm_120: per-shape host-side compilation is a
  serving hazard (memra's fixed-shape graph doctrine already avoids this — receipt), and
  async faults escape try/except-style guards, so backend-selection gates must be
  capability-checked up front, not caught. Also: evidence gathered on a 5090 with an
  FP8-dense/FP4-expert mixed model — someone is running memra's exact rig shape through
  vLLM. lever-queue: yes (doctrine receipt + FP8 backend-gate audit).
- #3984 (merged) — autotuner nearest-profile cache keyed on objects whose hash excluded
  but equality included per-call closures → same-hash-never-equal entries, permanent
  LRU churn + GIL contention, ~4% TTFT. CARE: hash/eq contract violations in hot host
  paths are invisible until py-spy; memra's Rust host side is largely immune, but any
  Python tooling on the serve path (bench drivers) can hit this. lever-queue: no
  (evidence note).
- #4295 (merged) — top-k tie-break: deterministic *selection* decoupled from deterministic
  output *ordering* — tie-break modes keep deterministic filtered selection at the
  boundary but skip the index sort unless deterministic=True. CARE: exactness doctrine
  vocabulary — memra's top-k (router + sampler) should state which of the two properties
  each caller needs; chunk-invariance gates need deterministic selection, not sorted
  output. lever-queue: yes (small; exactness lane).
- #4352 (merged) — FP8 block-scale + BF16 routed MoE now accept unpacked (topk_ids,
  topk_weights) so FP32 routing weights reach the combine un-truncated (the packed form
  crushed them to bf16). CARE: router-weight precision through the combine is a quality
  axis memra controls; verify memra's expert-combine keeps router weights at FP32.
  lever-queue: audit (one-liner).
- #4266 (merged) — Blackwell CuTeDSL BF16 split-k dense GEMM. NOTE: split-k on Blackwell
  dense is upstream now; memra's decode GEMMs already split-k where profitable. SKIP.
- #4186/#4027 (merged) — CuTe DSL finalize tail handling + MoE monokernel barrier removal.
  SKIP: their kernels.

### NVIDIA/TensorRT-LLM commits

- #16170 (merged) — **PP sample-state relay deadlock**: a background relay thread
  (last rank→0→1→...) needed the GIL; when the executor thread blocked in a GIL-holding
  native call waiting on GPU progress (DeepGEMM JIT cold load), the relay starved, the
  downstream rank never launched its forward, and the in-flight NCCL p2p kernel deadlocked
  the ring. Fix: relay INLINE in the executor loop (default), drain pending isends before
  entering any forward, and fail loudly on a missing relayed batch. Verified 0/20 hangs vs
  ~25% before. CARE: PRIME suspect shape for memra's PP-2+spec crash-under-debug — debug
  builds change timing exactly where a cross-stage relay/side-channel can starve. Laws to
  port: no cross-stage delivery on a thread that competes with the executor for a lock;
  drain outstanding sends before blocking in compute; missing-relay = loud error, never
  infinite wait. lever-queue: YES (top candidate, PP-2 lane).
- #17162 (merged) — spec warmup under-counted blocks (ignored num_extra_kv_tokens +
  draft-token reserve) AND add_dummy_requests leaked registered sequences on failure →
  LLM() startup hang on mamba-hybrid + spec. CARE: same family as last sweep's #16072
  (warmup preallocation must mirror real allocation EXACTLY, spec tokens included);
  extends the existing gate item — memra warmup accounting must include MTP draft
  tokens. lever-queue: yes (extends existing gate item).
- #16925 (merged) — MTP acceptance-rate regression at large batch: stale token→request map
  in the one-model draft loop (built for max_draft_len+1 tokens/req, reused after layout
  changed to 1 token/req) corrupted sparse-index reads AND indexer-K writes across
  requests. CARE: MTP-p2 — any per-token→request map must be rebuilt (or indexed
  layout-invariantly) between target and draft phases; corruption showed up as an
  ACCEPTANCE-RATE gap, not a crash — another receipt that per-position acceptance
  telemetry is the cheap detector. lever-queue: yes (MTP-p2 checklist + telemetry
  receipt).
- #17323 (merged) — Kimi identity-RoPE table sized to 65536 but chunked-context indexes by
  absolute position → OOB on longer prefills; radix-tree stale walk destroyed chains
  holding live pages of in-flight sequences. CARE: position-indexed tables must be sized
  to max_position, and cache eviction must check liveness across ALL lifecycles.
  lever-queue: audit (rope-table sizing one-liner).
- #16603 (merged) — reject MNNVL on split NVLink topology. SKIP: multi-node.
- #15138 (merged) — CuTeDSL FP8/FP16 MLA decode fmha lib. SKIP: MLA/SM100.
- #17277 (merged) — BREAKING block-reuse policy rename + tests. SKIP: their API.

### Ranked shortlist (lever candidates, with receipts)

1. **PP relay/side-channel starvation laws** (TRT-LLM #16170 + SGLang #33666): the PP
   deadlock anatomy — cross-stage delivery starved by a lock the executor holds, sends
   not drained before blocking compute, silent infinite wait — plus per-stage resource
   budgeting with rank-uniform derived limits. Both are direct checklist items for the
   PP-2 lane and #16170 is the best candidate mechanism yet for the PP-2+spec
   crash-under-debug (debug timing widens exactly that starvation window).
2. **Grain-aligned state publication, including resume** (vLLM #51113): the prefix-cache
   sequel to step35 — a state snapshot keyed by position may only be published at
   grain-aligned chunk ends, only the FINAL chunk may end unaligned, and the off-grid
   *start* (resume from cache/connector) is a second hole. Feeds the chunk-invariance
   gate and becomes a hard design constraint the day serve prefix-cache lands.
3. **Overlap-scheduler write-after-read hygiene** (SGLang #33587 + #33253): metadata
   rewritten by batch-prep while a graph replay still reads it, and padded per-token side
   tensors (positions/slot maps) leaking into backends. Snapshot-at-prep +
   narrow-everything are one audit of memra-serve's overlap path; high found-money odds
   given F5/prime-chunk already touch this region.
4. **Admission: wait-vs-never-fits three-case gate** (SGLang #32700 + vLLM #49206):
   transient pressure must WAIT (protect the decode cushion; the escape hatch caused a
   40%-rework retraction storm), only provably-never-fits acts; and preemption
   bookkeeping needs a sustained-pressure regression test because its failure mode is
   silent starvation. Direct design input for memra-serve QoS/admission.
5. **FP8/NVFP4 quality + backend-gate receipts** (llama.cpp #26675 + vLLM #50029 +
   FlashInfer #4165): ggml_prec formalizes per-op activation-quant/accumulator contracts
   (the W4A8 unblock — FP8-ST direction endorsed upstream); online NVFP4 packing must
   quantize from source with FP32 scale (no BF16 fold-recast round-trip — audit memra
   repack); and SM12x FP8 backend selection must be capability-gated up front because
   async faults are uncatchable. All three feed nvfp4-strict/FP8-ST.

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-10T06:17:01Z (since 2026-08-07T03:15:50Z)

### llama.cpp commits (decode-relevant, CUDA)
- CUDA: fix thread/block count in quantized cpy kernel launches (#26731)
- cuda: fix warnings for unused variable/function (#26688)
- CUDA: fuse rms_norm + mul + rope (+ view + set_rows) (#26767)
- mtmd: stop feeding the text stream again during Qwen3-TTS generation (#26706)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
#### v0.5.17 (2026-08-08T00:19:16Z)
- **Kimi K3 day-0 support**: A 2.8T-parameter multimodal LatentMoE (896 experts, top-16, routed in a 3584-dim latent space) with a 1M-token context, 69 KDA linear-attention layers interleaved with 24 MLA layers, and a MoonViT3d vision tower, shipping as a native MXFP4 checkpoint. SGLang serves it from day 0 with DCP, DSpark speculative decoding, chunked-prefill PP with TP decode, KDA-aware prefix caching, HiCache L2 over DCP, LoRA on the quantized weights, and reasoning, tool-call and OpenAI-compatible serving, verified on NVIDIA GB300 and AMD MI35x ([#32541](https://github.com/sgl-project/sglang/pull/32541), [#32828](https://github.com/sgl-project/sglang/pull/32828), [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33025](https://github.com/sgl-project/sglang/pull/33025), [#33112](https://github.com/sgl-project/sglang/pull/33112), [blog](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support), [cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3), [roadmap](https://github.com/sgl-project/sglang/issues/32607)).
- **MiniMax-H3 day-0 support**: MiniMax's video generation model that produces a video and a synchronized stereo audio track in one request, served natively on SGLang-Diffusion across all three public task profiles: text-to-video-and-audio (`t2va`), first/last-frame conditioning (`fl2va`), and image/video/audio reference conditioning (`ref2va`, which also covers video-to-video). Verified on B200 (TP2 + Ulysses4), H100 (TP2 + Ulysses2), and 2x RTX 5090 with layerwise offload ([#33275](https://github.com/sgl-project/sglang/pull/33275), [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)).
- **Other new models added**: [EmbeddingGemma](https://docs.sglang.io/cookbook/autoregressive/Google/EmbeddingGemma) and [LFM2.5](https://docs.sglang.io/cookbook/autoregressive/LiquidAI/LFM2.5) embedding models, nvidia/MiniMax-M3-NVFP4, plus cookbook recipes for Poolside's [Laguna-S-2.1](https://docs.sglang.io/cookbook/autoregressive/Poolside/Laguna-S-2.1) family and [Inkling-Small](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small).
- **DCP communication backends and q-replicate (Helix)**: The DeepSeek-MLA decode context-parallel path gains pluggable comm backends. `a2a` exchanges packed attention output plus fp32 LSE in a single NCCL collective per layer, with fp8 KV carried as uint8 byte transport; `fi_a2a` delegates the cross-rank exchange to the FlashInfer MNNVL kernel on GB200. `--dcp-replicate-q-proj` projects full-head Q locally and skips the per-layer Q head-dim all-gather. Select with `--dcp-comm-backend {ag_rs, a2a, fi_a2a}` ([#21637](https://github.com/sgl-project/sglang/pull/21637)).
- **DWDP for MoE prefill**: A new prefill parallelism strategy that prefetches peer expert weights over NVLink P2P and computes all experts locally, removing EP all-to-all token dispatch. On 4x B200 with gpt-oss-120b, prefill-only, DWDP4 reaches **1.92x over DEP4** at MNT 32K / ISL 32K, and **506K vs 329K tok/s (1.54x)** at saturation (CONC=128, ISL=8K). Enable with `--dwdp-size`; the authors mark it early-development ([#29778](https://github.com/sgl-project/sglang/pull/29778)).
- **SM90 FP8 MegaMoE for DeepSeek-V4**: Adds the DeepGEMM MegaMoE A2A path on SM90 for DeepSeek-V4-Flash/Pro FP8, including the pre-dispatch JIT kernel and FP8 expert weight preparation. Guarded behind `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` ([#29016](https://github.com/sgl-project/sglang/pull/29016)).
- **Faster engine recovery**: Large-model restarts cost 3 to 6+ minutes today, about 6.5 minutes for Qwen3-235B FP8 on 4 GPUs, because weights reload from storage and CUDA graphs recapture. A weight-cache daemon holds weights per GPU so a restarting engine can recover from cache instead ([#27139](https://github.com/sgl-project/sglang/pull/27139)).
- **Lower host overhead in hybrid-linear MTP decode**: Under spec-v2 overlap scheduling each decode step runs draft, verify and extend CUDA graphs, and the eager seams between them become GPU idle time at low concurrency. This trims that host work so the host stays off the critical path ([#32219](https://github.com/sgl-project/sglang/pull/32219)).
- **Dependencies**: flashinfer 0.6.15.post1 ([#31927](https://github.com/sgl-project/sglang/pull/31927)), sgl-deep-gemm 0.1.5.post1 ([#32345](https://github.com/sgl-project/sglang/pull/32345), [#33143](https://github.com/sgl-project/sglang/pull/33143)), helion 1.4 ([#32562](https://github.com/sgl-project/sglang/pull/32562)), mooncake 0.3.12.post1 ([#32302](https://github.com/sgl-project/sglang/pull/32302)), dynamo-tokenizers 1.7.0 ([#32981](https://github.com/sgl-project/sglang/pull/32981)). PyTorch stays at 2.11.0 and the CUDA base image at 13.0.1.
- | MiniMax-H3 | Diffusion | [#33275](https://github.com/sgl-project/sglang/pull/33275) | [link](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) |
- | MiniMax-M3-NVFP4 | Autoregressive | [#31989](https://github.com/sgl-project/sglang/pull/31989) | |
- | EmbeddingGemma | Autoregressive (Embedding) | [#32375](https://github.com/sgl-project/sglang/pull/32375), [#32383](https://github.com/sgl-project/sglang/pull/32383) | [link](https://docs.sglang.io/cookbook/autoregressive/Google/EmbeddingGemma) |
- * [MTP] Cut spec-v2 host-seam overhead in hybrid-linear MTP decode: [#32219](https://github.com/sgl-project/sglang/pull/32219) ⭐
- * [DFLASH] Support grammar-constrained decoding in speculative verify: [#30096](https://github.com/sgl-project/sglang/pull/30096)
- * Overlap grammar (constrained decoding) with speculative decode verify: [#31488](https://github.com/sgl-project/sglang/pull/31488)
- * [Spec] Support sampling in the DSPARK graph-folded draft proposal: [#33298](https://github.com/sgl-project/sglang/pull/33298)
- * [Spec] Add `trtllm_mha` support for Gemma 4 MTP draft attention backend: [#25545](https://github.com/sgl-project/sglang/pull/25545)
- * [Perf] Fold dspark dense draft embedding into the draft graph via forward_embed: [#31985](https://github.com/sgl-project/sglang/pull/31985)
- * [Perf] Stack dspark dense draft per-layer ctx KV projection into one GEMM: [#31986](https://github.com/sgl-project/sglang/pull/31986)
- * [gdn] fused replayssm ring write into flashinfer gdn mtp verify kernel: [#33102](https://github.com/sgl-project/sglang/pull/33102)
- * [EAGLE] Handle NaNs in fused top-k=1: [#32396](https://github.com/sgl-project/sglang/pull/32396)
- * Support SGLANG_SIMULATE_ACC_LEN for DFLASH: [#32595](https://github.com/sgl-project/sglang/pull/32595)
- * Fix DSpark loading for hybrid DSV4 NVFP4: [#33276](https://github.com/sgl-project/sglang/pull/33276)
## Piecewise & Breakable CUDA Graph
- * Turn on breakable prefill cuda graph for dp attention by default: [#31682](https://github.com/sgl-project/sglang/pull/31682)
- * [BCG][4/N] Enable bcg on megamoe & flashinfer a2a backend: [#33150](https://github.com/sgl-project/sglang/pull/33150)
- * [CUDA Graph] Allow custom decode graph runners: [#33553](https://github.com/sgl-project/sglang/pull/33553)
- * [cuda_graph] Gate breakable-CG capture_inputs retention to DP-gather paths: [#32678](https://github.com/sgl-project/sglang/pull/32678)
- * Enable post-capture KV sizing with DP attention: [#33427](https://github.com/sgl-project/sglang/pull/33427)
- * fix(server): capture legal multi-request prefill CUDA graph batches: [#30206](https://github.com/sgl-project/sglang/pull/30206)
- * [Attention Backend] Extend hpc_ops dynamic-scheduled decode to bf16: [#32304](https://github.com/sgl-project/sglang/pull/32304)
- * [DSA] Q8KV8 FP8 Sparse Prefill on GLM-5.2 & DeepSeek-V3.2: Q8-Path & Shared-Path Optimizations: [#31888](https://github.com/sgl-project/sglang/pull/31888)
- * [GDN] Support FlashInfer GDN prefill with extra-buffer radix cache: [#29735](https://github.com/sgl-project/sglang/pull/29735)
- * [Kernel] Fuse KV-cache writes for asymmetric K/V (head_dim != v_head_dim): [#32813](https://github.com/sgl-project/sglang/pull/32813)
- * [Perf][DSA] Pass topk_length to flash_mla_sparse_fwd in the sparse attention path: [#31128](https://github.com/sgl-project/sglang/pull/31128)
- * [Perf] Skip page-table columns past kv length in DSA draft-extend metadata kernel: [#31981](https://github.com/sgl-project/sglang/pull/31981)
- * Support a same-size mixed q dtype in the fused RoPE kernels: [#31834](https://github.com/sgl-project/sglang/pull/31834)
- * [Fix] Route asymmetric-KV models to fa4 on SM100 and pin MiMoV2 FP8 MoE to flashinfer_trtllm: [#32818](https://github.com/sgl-project/sglang/pull/32818)
- * [Fix] Fix trtllm_mla backend + fp8 kv cache without rope: [#32181](https://github.com/sgl-project/sglang/pull/32181)
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-08-24T06:17:02Z (since 2026-08-10T06:17:01Z)

### llama.cpp commits (decode-relevant, CUDA)
- chat : tighten bare function parsing for Qwen models (#26793)
- ci: Add support for CUDA 13.4 ARM64 builds for Windows (#26650)
- ci: add Windows ARM64 CUDA support to the manual workflow (#27300)
- convert : handle per_layer_config in Gemma4 (transformers 5.15) (#26882)
- CUDA: adding switch points per HW and quant type to tune the mvq->MMQ decode crossover (#26079)
- cuda : add POOL_1D support (#27573)
- cuda : add warp-per-row wkv7 kernel for single-token decode (#26111)
- CUDA: MMVQ nwarps=8 for bs=1 for dense models on DGX Spark (#26843)
- CUDA: only disable CUDA graphs when mul_mat_id actually needs a stream sync (#26802)
- DeepseekV4: fix rollback with multi-seq (#26756)
- dflash : clarify output logging of target_layer_ids (#27013)
- Dflash support for nemotron-3.5 (#26905)
- ggml-cpu/ops: vectorize flash-attention V-cache F16 to F32 conversion (#26947)
- ggml-cuda: provide static workspace for cuBLAS handles (#26574)
- ggml : require contiguous src for ROLL on CUDA and Metal (#25928)
- graph : create V as a view of K in the k_iswa build_attn (#27392)
- kleidiai : add SME2 F32 GEMV kernel support (#26891)
- metal : dequantize quantized KV to F16 before flash attention (#27390)
- metal : dequant kv cache only for large batches (#27438)
- model : BailingMoE3 Support (#26608)
- model : disallow integer dflash sliding_window_pattern (#26900)
- model : GraniteSWAForCausalLM / GraniteMoeSWAForCausalLM (#25505)
- model : support DSpark for bailingmoe3 (#27508)
- OpenVINO: Qwen3.5, memory optimization, and test-recurrent-state-rollback (#26952)
- RPC: populate use_count to enable fusion inside backends (#27142)
- server: allow accessing /metrics and /slots during llama_decode() (#27041)
- spec: enable backend sampling for both dflash & dspark (#26958)
- TP: enable tensor split for LFM2/LFM2MOE (#26993)

### vllm-project/vllm releases
#### v0.27.1 (2026-08-11T10:47:49Z)
#### v0.27.0 (2026-08-10T21:18:11Z)
- * **Kimi K3 support** with a full stack landing in one release: core model files and kernels (#50089, #50000), Python (#50093) and Rust (#50104) frontends, AttnRes kernels (#50090), DeepGEMM support (#50458), compressed-tensors quantized checkpoints (#50500), DSpark AR fusion (#50242), and an option to shard the shared expert instead of replicating it (#50656).
- * **More new models**: Qwen3.5 text-only dense and MoE models (#50210) with EVS video token pruning (#48912), K-EXAONE-2.0-750B-A37B (#50524), VaultGemma via the Transformers modeling backend (#49803), and jina-embeddings-v5-text-nano (#50688).
- * **FlashAttention 4 integration deepens on SM100**: FP8 KV cache support (#42569) and headdim-256 support (#42669), backed by a new JIT warmup infrastructure (#47451) and runner-owned Triton kernel warmup (#49903) that remove first-request compilation stalls.
- * **DeepSeek-V4 performance push**: sequence parallelism (#46789), ~2x kernel improvement by skipping empty c128 launches (#48957), 3.4% E2E TTFT from skipping unneeded topk/router (#49486), 3.9% E2E TTFT from workspace reuse (#49236), 1.88x kernel from removing a redundant full kernel (#50298), adaptive topk width (1.0% E2E, #50004), 448 MiB GPU memory saved in the PP buffer (#50312), a compact MXFP4 indexer KV cache (#48993), and removal of sparse-MLA q-head padding on FlashInfer >= 0.6.14 (#48047).
- * **Disaggregation for hybrid models**: NIXL P/D for hybrid MLA+SSM models (#49762), heterogeneous P/D block sizes for hybrid models (#49612), and MoRIIO heterogeneous TP<->DP prefill/decode read routing (#46116).
- * **Rust frontend grows a gRPC control plane**: engine-aware health reporting (#48992), abort control (#49255), server and model discovery (#49491), KV event source discovery (#50033), plus `vllm-bench` integrated into the `vllm` CLI (#48930).
- * Kimi K3: new model (#50000) with model files and kernels (#50089), Python frontend (#50093), Rust frontend (#50104), AttnRes kernels (#50090), DeepGEMM support (#50458), DSpark AR fusion (#50242), and optional shared-expert sharding (#50656).
- * New models: Qwen3.5 text-only dense and MoE (#50210), K-EXAONE-2.0-750B-A37B (#50524), VaultGemma via Transformers backend (#49803), jina-embeddings-v5-text-nano with EuroBERT encoder backbone (#50688).
- * Inkling: llm-compressor NVFP4 weights (#49258) and compressed-tensors dynamic FP8 (#48876).
- * Multimodal: VidCom2 video token pruning (#47750), EVS for Qwen3.5 (#48912), ViT CUDA graph for Gemma-4 (#46837), Cosmos3 FP8 ModelOpt/Diffusers remapping (#48952), MiniMax-M3 MSA speculative decode verification (#50032) and default video processor (#50305), DeepSeek-OCR-2 TTFT optimization (#49531), longer max audio duration for MOSS-TD (#49403).
- * Diffusion models: top_k and top_p sampling for DiffusionGemma (#45429).
- * Transformers modeling backend: audio model support (#39330), improved `fx` tracer (#49957), fused residual-add + RMSNorm compilation pass (#48757), and fixes for MLA padding + grouped topk routing (#49982), MQA with TP (#49987), and Qwen3-VL M-RoPE (#49292).
- * Attention: FlashAttention 4 SM100 FP8 KV cache (#42569) and headdim-256 (#42669); query replication for MLA decode under DCP for DeepSeek-V2/R1 and Kimi-K2.5 (#45964); masked MHA for sparse MLA prefills (#48770); skip sparse indexer scoring for short dense prefills (#48407); FlexAttention epilogue hook (#45841) and encoder block-mask compile explosion avoided (#50339); attention backends stay eligible for text-only serving of prefix-LM models (#48796); merge-attention context count as a runtime argument (#48739); unified multi-path encoder CUDA graph support (#49934); encoder cache extension hooks (#48218).
- * KV offloading: generic P2P secondary tier with peer lookup/serving (#48021), per-request tier filtering with TierFilter/TierMatcher (#48123), self-describing KV events with TieringOffloadingSpec (#48679), pluggable eviction policies via CachePolicyFactory (#49114), deduplicated replicated MLA KV in the shared CPU region (#48906), single-copy MLA layout for CPUOffloadingSpec (#50301), CPUOffloadingSpec moved onto SharedOffloadRegion (#50094), TP-independent compact secondary identity (#49858), batched C store/load for filesystem offload (#49152), reliable partial-tail offload for sub-block prompts (#49502), per-layer canonical KV page mappings for parallelism-agnostic offload (#48408).
- * Mamba/hybrid: ReplaySSM caching for faster Mamba2 standard decode (#48018), fused align-mode DS-conv state migration with num_accepted_tokens > 1 (#49291), FlashInfer Mamba SSU algorithm selection (#50157), fixed `/wake_up` crash on hybrid models (#41602).
- * Spec decode: DSpark Markov head replicated across TP ranks (#49731), `sample_from_anchor` loaded from speculators config (#48639), earliest-completing stop string selected (#49391).
- * Structured outputs: grammar advanced across the reasoning boundary with spec decode (#44993).
- * RL: weight version tagging for RL rollouts (#49040), stateful trainer-send IPC (#48981), vLLM config set during weight reload (#45989), router replay output from the FlashInfer monolithic MoE kernel (#44214).
- * DeepSeek-V4: sequence parallelism (#46789), ~2x kernel skipping empty c128 launches (#48957), 3.4% E2E TTFT skipping topk/router in decode (#49486), 3.9% E2E TTFT workspace reuse (#49236), 1.88x kernel removing a redundant full kernel (#50298), adaptive topk width 1.0% E2E (#50004), 448 MiB GPU memory saved (#50312), compact MXFP4 indexer KV cache (#48993), sparse-MLA q-head padding removed for FlashInfer >= 0.6.14 (#48047).
- * Kernels: RMSNorm uncontiguous support with 1.2–3.1x kernel improvement (#49750), MoE `reduce_scatter` regression fix restoring 5% E2E throughput (#48763), non-grouped bias-less topk routing dispatched to the fused path (#49618), tuned LL BF16 router GEMM (#48774) with warmup skipped for non-MoE models (#49659), Triton tensor-descriptor path for fused MoE via `VLLM_TRITON_USE_TD` (#42436), cudagraph/DP padding skipped in topk (#48979), coalesced HBM access in the Marlin INT4-FP8 AWQ preprocess kernel (#47268).
- * NVIDIA next-gen: `sm_107` for Rubin (#49387), NVLink all-reduce paths on SM107 (#49647), fixed CUDA arch detection producing kernel-less builds on SM121 (#49904).
- * ROCm: gfx1250 architecture enabled (#46516), AITER FP8 ViT encoder attention (#49937), fused shared expert for Quark DeepSeek-V4 checkpoints (#48044), Quark GLM-5.2 checkpoint inference fixes (#48886), DSv3.2 per-decode FillFunctor launches eliminated in the sparse-MLA hot loop (#44527), B-preshuffled attention FP8 projections for DSv4 (#46720), TML Inkling enabled (#48841), tuned selective_state_update float16 config for MI325X (#50006), GPT-J-style MRoPE fixed and optimized (#49906), quickreduce accuracy fix in cudagraph mode (#46913), cached fp32 upcast of static e8m0 weight scales (#47773), batch DMA for CPU KV cache loads (#49843), elastic EP scaling accuracy fix (#47206).
- * XPU: QK Norm + RoPE fusion pass (#49394), FP8 o_proj with fp8_bmm and load-time scale transpose (#48334), DeepSeek-V4 fuse_index_q SYCL kernel path (#45991), TD operand loads for batched MoE GEMM (#46340), RMSNorm kernels unified with vllm_c (#46981).
- * CPU: INT8 fused MoE kernel for Arm CPUs (#48637), s390x inference optimization with oneDNN INT8 GEMM (#50219), GDN conv path optimized for speculative decoding (#48577), granite-4 enabled (#47641), FAST_EXP for Power (#49571), CPU kernels bumped to the latest version (#50387), macOS build fixes (#49021, #50915).
- * P/D disaggregation: NIXL P/D for hybrid MLA+SSM models (#49762), NIXL heterogeneous P/D block sizes for hybrid models (#49612), MoRIIO heterogeneous TP<->DP prefill/decode read routing (#46116), optional lookup disable on PD decode (#50498), prefill token ids reused on the decode chat path (#48145), detokenization streaming derender (#47301), NixlPush skips an extra handshake step in D->P (#49345).
- * Fixes: P/D preemption race (#50297), KV lease deadlines rebased onto the worker clock (#50326), NIXL hybrid MLA+mamba heterogeneous TP (#49297), internal LB load-balancing (#49204).
- * Mooncake: vectorized `prepare_value` on the KV load path (#48531), full external hits re-derived on stored boundaries (#49481).
- * Communicators: process-checkpoint lifecycle hooks, starting with FlashInfer (#46877).
- * New capabilities: FP4 Qutlass integration for compressed-tensors (#43229), CuTeDSL MoE for ReLU2 NVFP4 (#49580), MXFP8 linear support in INC (#47514), AutoRound W4A16 MoE and MXFP4 linear/MoE on XPU (#47124), KV quant mode for TurboQuant (#50533), ModelOpt FP8 emulation on SM80 (#50019), `--linear-backend` honored for ModelOpt W4A16 (#50273).
- * Checkpoints: compressed-tensors support for DeepSeek-V4 (#41276) and Kimi-K3 (#50500); `find_matched_target` prioritizes fused-name matches (#49483).
- * MoE refactor: FusedMoE renamed to FusedMoEFactory (#44941), MoeWNA16 migrated to the MK oracle scheme (#44120), Quark w8a8-int8 (#46765) and MXFP4 `aiter`/`emulation` backends (#49348, #48949) moved to kernel abstractions, CT WNA16 Marlin/MoE methods merged (#44570), Quark W4A8 (INT4-FP8) MoE CI coverage (#48050).
- * Rust frontend: gRPC control plane with engine-aware health reporting (#48992), abort control RPC (#49255), server and model discovery (#49491), and KV event source discovery (#50033); `vllm-bench` integrated into `vllm-rs` and the `vllm` CLI (#48930) with opt-in Rust delegation for `vllm bench serve` (#50081); zero-copy multimodal tensor slicing (#48781), multimodal tensors in auxiliary frames (#49341), `--limit-mm-per-prompt` (#49604), ordinary-text tokenizer encoding (#49992).
- * Multimodal: mm hash algorithm selection via CLI (#49686), configurable PyNvVideoCodec decoder concurrency (#49753), RFC 2397 parameters accepted in base64 data URLs (#48973).
- * Transformers 5.14.1 (#49223), FlashInfer 0.6.15 (#48914) then 0.6.16.post3 (#50892), AITER 0.1.16.post5 (#48683) then 0.1.19 (#49361), NCCL 2.30.7 enabling DeepEPv2 in the vllm/vllm-openai image (#45321), tpu-inference v0.25.0 (#49431) then v0.26.0 (#50522), Helion 1.4.0 (#50307), NIXL and UCX upgraded on ROCm (#49251).
- * Build: vllm-flash-attn bumped to a C++20-compatible commit for torch-nightly (#49326), ABI-stable FA2 build pin (#50474).

### sgl-project/sglang releases
#### v0.5.18 (2026-08-22T00:09:15Z)
- | SANA-Video | Diffusion | [#32921](https://github.com/sgl-project/sglang/pull/32921) | [link](https://docs.sglang.io/cookbook/diffusion/SANA-Video/SANA-Video) |
- | LingBot-Video-MoE | Diffusion | [#32341](https://github.com/sgl-project/sglang/pull/32341) | [link](https://docs.sglang.io/cookbook/diffusion/LingBot-Video/LingBot-Video-MoE) |
- | LTX-2.5 | Diffusion | [#34471](https://github.com/sgl-project/sglang/pull/34471) | [link](https://docs.sglang.io/cookbook/diffusion/LTX/LTX2.5) |
- | Cosmos3 Edge & Distilled | Diffusion | [#31590](https://github.com/sgl-project/sglang/pull/31590) | [link](https://docs.sglang.io/cookbook/diffusion/Cosmos/Cosmos3) |
- | LongCat-Image | Diffusion | [#23274](https://github.com/sgl-project/sglang/pull/23274) | |
- Plus cookbook recipes for the [Qwen3.8 family](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8), [Ling-3.0](https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ling-3.0-flash), [Nemotron 3.5 Lightning](https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Lightning), [Dots3-Note](https://docs.sglang.io/cookbook/autoregressive/RedNote/Dots3-Note), and DeepSeek-V4-Pro-0813 ([#34809](https://github.com/sgl-project/sglang/pull/34809)).
- **Overlapped checkpoint staging at startup**: Checkpoint pages now stage from storage while CUDA graphs capture. Qwen3-32B on H100 starts **8.6-11.7% faster** than serial with prefetch, and **2.38x faster (35.6s vs 84.8s)** than the plain default. Opt in with `--startup-weight-load-mode overlap` ([#32017](https://github.com/sgl-project/sglang/pull/32017)).
- **TP LMHead with All-to-All**: The TP LMHead's allgather + scatter becomes a single all-to-all for pure-DP dp-attention. On DeepSeek-V4-Pro B200 decode, LMHead time drops **320us to 169us** and TPOT improves 36.97ms to 35.67ms ([#32313](https://github.com/sgl-project/sglang/pull/32313)).
- **FlashInfer MNNVL for pure allreduce**: Non-fused allreduce sites now reuse the FlashInfer MNNVL workspace instead of falling back to NCCL. DeepSeek-V4-Flash TP4 decode on Blackwell gains **up to +6.9% at small batches**. Auto-enabled for DeepSeek-V3/V3.2/V4; elsewhere `--enable-flashinfer-pure-allreduce` ([#30700](https://github.com/sgl-project/sglang/pull/30700)).
- **One compiled-kernel cache directory**: Triton, FlashInfer, Inductor, DeepGEMM, and CUDA driver caches all move under `SGLANG_CACHE_DIR`. The first launch after upgrading recompiles once; see Breaking Changes ([#32434](https://github.com/sgl-project/sglang/pull/32434)).
- **Dependencies**: torch 2.13.0 with triton 3.7.1 ([#28836](https://github.com/sgl-project/sglang/pull/28836)), flashinfer 0.6.17 ([#33997](https://github.com/sgl-project/sglang/pull/33997)), CuTeDSL 4.6.2, fixing an FA4 startup regression on Blackwell ([#34372](https://github.com/sgl-project/sglang/pull/34372)), DeepEP now installed from released `sgl-deep-ep` wheels ([#33932](https://github.com/sgl-project/sglang/pull/33932)), and sgl-kernel 0.4.6.post1 ([#33842](https://github.com/sgl-project/sglang/pull/33842)).
- * [mm] rust-server: native multimodal processing for Qwen VL (integrate sglang-mm, e2e): [#32365](https://github.com/sgl-project/sglang/pull/32365)
- * [Spec] Support logprobs with DFlash: [#33459](https://github.com/sgl-project/sglang/pull/33459)
- * [Spec] Wire DFLASH aux-hidden capture into the Qwen3.5 text-only wrapper: [#34771](https://github.com/sgl-project/sglang/pull/34771)
- * [Spec] Support mamba-radix-cache-strategy extra_buffer_lazy with DFLASH: [#34763](https://github.com/sgl-project/sglang/pull/34763)
- * [Spec] Support MegaMoE for DSpark under dp attention: [#34844](https://github.com/sgl-project/sglang/pull/34844)
- * Fix DFlash sliding attention causality defaults: [#34524](https://github.com/sgl-project/sglang/pull/34524)
- * [Spec] Budget the DFLASH draft KV pool from its own attention geometry: [#34234](https://github.com/sgl-project/sglang/pull/34234)
- * fix(dflash): account for DCP in draft KV pool sizing: [#33912](https://github.com/sgl-project/sglang/pull/33912)
- * [DSV4] Fix silent KV corruption when speculative draft tokens > 4: [#34189](https://github.com/sgl-project/sglang/pull/34189)
- * [DSpark] Fix EP1 decode performance regression: [#34759](https://github.com/sgl-project/sglang/pull/34759)
## Piecewise & Breakable CUDA Graph
- * [BCG][6/N] Allow prefill breakable CUDA graph for the Kimi archs: [#34245](https://github.com/sgl-project/sglang/pull/34245)
- * fix: always capture default prefill CUDA graph: [#33352](https://github.com/sgl-project/sglang/pull/33352)
- * Fix padded positions in breakable CUDA Graph attention: [#33253](https://github.com/sgl-project/sglang/pull/33253)
- * fix: avoid piecewise prefill graph for trtllm_mla: [#32785](https://github.com/sgl-project/sglang/pull/32785)
- * Reenable breakable CUDA graph for NemotronH: [#34538](https://github.com/sgl-project/sglang/pull/34538)
- * Fix prefill CP graph overflow with larger bucket search: [#33906](https://github.com/sgl-project/sglang/pull/33906)
- * Fix stale track rows corrupting conv checkpoints under the prefill graph: [#34184](https://github.com/sgl-project/sglang/pull/34184)
- * Fix sconv track refresh on graph capture: [#35042](https://github.com/sgl-project/sglang/pull/35042)
- * Increase post-capture decode memory reserve: [#34996](https://github.com/sgl-project/sglang/pull/34996)
- * fix: support FA4 backend for GLM4.7-flash: [#33436](https://github.com/sgl-project/sglang/pull/33436)
- * feat: Add flashinfer mHC fusion for DSV4: [#33616](https://github.com/sgl-project/sglang/pull/33616)
- * [DSV4] Turn on mhc post pre fusion by default: [#35214](https://github.com/sgl-project/sglang/pull/35214)
- * [SM12x] Default the fused MHC post+pre path on: [#34019](https://github.com/sgl-project/sglang/pull/34019)
- * [trtllm_mha] perf: Stop allocating per-layer scratch inside the decode CUDA graph: [#33063](https://github.com/sgl-project/sglang/pull/33063)
- * add flashinfer cute-dsl backend for mxfp8 gemm: [#34042](https://github.com/sgl-project/sglang/pull/34042)
- * fix(dsa): use FlashInfer fused top-k for packed PAGED rows: [#33006](https://github.com/sgl-project/sglang/pull/33006)
- * [DSA] Fix top-k v2 dropping non-primary ranks' output on CUDA 13.1+ (root cause for #33835): [#34167](https://github.com/sgl-project/sglang/pull/34167)
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._
