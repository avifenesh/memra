# The GLM-5.3-Flash prefill gap: attribution and plan (2026-08-29)

Owner directive: "learn from other engines, why we have so bad prefill."

Measured (ring-sizing ctxprobe, 4x RTX PRO 6000 Blackwell Server 96 GB, 2026-08-29,
`../ring-sizing-20260828/box-ctxprobe/BOXPROBE.md`): streamed TTFD 57.4 s at 4,630 prompt
tokens, 67.1 s at 5,550, 78.8 s at 6,470. That is 80.7 / 82.7 / 82.1 tok/s of prefill,
a FLAT ~12.2-12.4 ms per prompt token. vLLM / SGLang / TensorRT-LLM prefill MoEs of this
class at thousands to tens of thousands of tok/s on comparable hardware (receipts in §2).
The gap is 100x-class and this document attributes it from source, receipts, and the
literature, then ranks the levers that close it.

The one-sentence verdict, stated plainly up front: **glm5_next prefill runs every prompt
token through the decode program.** The routed-MoE FFN dispatches each of the 4,096 tokens
of a prime chunk one at a time through per-expert matvec kernels (49 launches per
token-layer, 2,058 per token, ~8.4M per chunk), re-reading the routed expert weights from
VRAM for every token instead of batching tokens by expert into a handful of tensor-core
GEMMs per layer. Prefill's per-token cost is therefore decode's per-token cost, and the
measurement says exactly that: cost per prompt token is constant in prompt length, which
is the signature of per-token dispatch, not of a compute-bound GEMM prefill.

House note on numbers: every measured figure below is quoted from its banked receipt with
a pointer; figures marked ARITHMETIC are derived from source structure or config geometry
and are estimates, not measurements. The profile script that turns the estimates into
receipts is `profile-prime-phases.sh` in this directory, named in §4 as the first action
of the next box window.

---

## 1. Source audit: where a 4,096-token prime chunk actually goes

Path walked (tree `9e4b197bf4`, origin/lane/glm53-flash-bringup):
`prime_cache_hyper` -> `hyper_prime_ranges` (chunks of at most `PRIME_CHUNK_MAX_TOKENS =
4096`, `crates/memra-kv/src/lib.rs:185`) -> `prime_chunk_hyper`
(`crates/memra-engine/src/hybrid_forward.rs:1867`) -> per layer: `hyper::pre` -> mixer
(KDA scan or MLA+kpool) -> `hyper::post` -> `hyper::pre` -> `hyper_ffn_branch`
(`hybrid_forward.rs:1188`) -> `hyper::post`.

### 1.1 The MoE FFN prefills one token at a time. This is the wall.

`hyper_ffn_branch` routes prefill to `moe_ffn_il_prefill` (`hybrid_forward.rs:6516`),
whose comment says the quiet part: "Step35 promotes expert-grouped dispatch by default
while decode/spec callers keep `moe_ffn_il`". But inside `moe_ffn_inner`, every grouped
arm is denied for glm5_next:

- The Step EP/TP grouped-GEMM prime (`hybrid_forward.rs:6787`, the arm behind
  `MEMRA_STEP_GEMM_PRIME`, measured 170-270 TFLOP/s and family-default ON for step37
  since 2026-08-28, `lib.rs:681`) requires `m.step_ep`/`m.step_tp`, which glm5's
  single-process ppN deployment does not build. Not reachable.
- `sigmoid_resident_dev_eligible` (`hybrid_forward.rs:7358` call site) requires the
  SlidingGatedMoe decode-batch program, which glm5_next's plan is not. Denied.
- The A2 expert-grouped prefill (`hybrid_forward.rs:7369`) is behind `MEMRA_MOE_GROUPED`,
  default OFF ("the local 5090 transfer gate rejected the default flip",
  `hybrid_forward.rs:382`). Not engaged.
- Fall-through: `moe_ffn_sequential_zq8` (`hybrid_forward.rs:7407` -> `:7642`).

Inside `moe_ffn_sequential_zq8`, the two batched arms that DO serve other archs at
prefill are both denied by predicate for glm5_next:

- `moe_ffn_pairs` (the "MoE PREFILL PAIR-BATCH (2026-07-06, the 16x pp hole)" arm,
  `hybrid_forward.rs:7733`, one launch per projection covering ALL (token,expert)
  pairs) carries `cfg.sigmoid_router().is_none()` (`hybrid_forward.rs:7759`) -
  glm5_next is a
  sigmoid-router arch (noaux_tc), so it is excluded, deliberately: the pairs arm routes
  through the fused SOFTMAX router and would silently pick wrong experts (the M3
  gate-MISMATCH 74602-vs-92 lesson).
- `moe_ffn_dev` same denial, plus `t <= MOE_DEV_MAX_T`.

So each of the chunk's 4,096 tokens enters the per-token loop at
`hybrid_forward.rs:8098` (`for tok in 0..t`), and per token per MoE layer the shipped
default runs the sequential per-expert program: 1 z-quantize + 8 x {gate matvec, up
matvec, pre-clamped SwiGLU, act quantize, down matvec, axpy} = **49 kernel launches per
token-layer**. That count is not an estimate: the fused-epilogue lane profiled the
structure with nsys (counts only, rig exactness law honored) and banked it in
`../moe-epilogue-receipts/nsys-launch-counts.md`: "At the real model's n_used = 8 the
same structure gives 49 -> 4 per token-layer ... i.e. 2058 -> 168 launches per token
across 42 MoE layers."

The fused MoE epilogue (`MEMRA_MOE_FUSED_EPI`, FLAGS.md:474) collapses 49 to 4 but is
DEFAULT OFF, deliberately, because it has no throughput receipt yet (the flag-default
law); the ctxprobe arms did not set it, so the measured 80-83 tok/s is the 49-launch arm.
Even at 4 launches per token-layer the dispatch class is still per-token matvec.

Now the arithmetic of what per-token dispatch costs (geometry from
`../decode-attribution-receipts/ATTRIBUTION.txt`, which measured decode's identical
structure):

- **Launches**: 2,058 MoE launches/token x 4,096 tokens = **8.43M launches per chunk**
  (trunk adds only ~1-2k per chunk: its kernels are chunk-wide). The decode attribution
  solved the launch/latency term of the SAME loop structure as an invariant **17.1
  ms/token above the bandwidth roofline, unchanged by 5.2x transport and 2x residency
  changes** - "the evidence that it is launch structure, not bytes." Prefill's measured
  12.2-12.4 ms/token sits in the same class (slightly below 17.1 because prefill's SLRU
  hit pattern within a chunk is friendlier and some decode-only host syncs are absent).
  ARITHMETIC: at 8.43M launches per 50.4 s chunk, the whole wall is consistent with
  ~6 us of launch+gap per kernel; per-token MoE dispatch is sufficient to explain
  essentially the entire prefill wall by itself.
- **Weight re-reads**: per token, 8 routed experts x 3 projections x 4.72 MB NVFP4 =
  113 MB/layer, x42 layers = **4.76 GB of expert weights read from VRAM per token**
  (the measured decode row: "MoE experts read from VRAM, 1008 x 4.72 MB -> 2.7 ms" at
  ~1.79 TB/s). A grouped prefill reads each expert ONCE per layer per chunk:
  288 x 3 x 4.72 MB = 4.1 GB/layer, 172 GB/chunk = ~96 ms/chunk = **0.023 ms/token**,
  a ~115x reduction of the MoE weight-traffic term (ARITHMETIC).
- **Host router drains**: routing is exact and m-invariant (`moe_router_logits`,
  `hybrid_forward.rs:7473`), good; but `moe_route_sigmoid_cfg` (`hybrid_forward.rs:9114`)
  reads sel/w back to the HOST - one D2H readback + pipeline drain per MoE layer per
  chunk, 42 per chunk. The step37 lane measured this exact structure at m=4092 and wrote
  the warning into the source: "sigmoid + top-8 over 288 experts for every one of 4096
  tokens, per layer ... a D2H copy and a full pipeline drain 42 times per chunk.
  Attribute it before optimizing the kernel it sits in front of"
  (`hybrid_forward.rs:6795-6798`).

Cross-model receipt that dispatch class is the variable: memra prefills qwen3.8 (softmax
router, so the pairs batch arm ENGAGES) at roughly 1,000-2,500 tok/s on production-class
hosts (internal serving receipts, 2026-08-29: max servable cold prompt under the 90 s
first-token deadline ~= prefill_rate x 90 s ~= 95k-225k tokens depending on host class).
Same engine, same trunk machinery: the arch that gets batched expert dispatch prefills
12-30x faster than the arch that is denied it.

### 1.2 Trunk projections: the BF16 trunk rides an f32, non-tensor-core GEMM at prefill

glm5_next keeps ALL 34 KDA layers' projections in BF16 (census: "All KDA projections are
BF16"), plus kv_b, mHC, embeddings and lm_head. At prefill m >= 16, `Engine::matmul`
(`lib.rs:12312`) has real tensor-core arms for quantized weights (FP8 cuBLASLt TN at
620-795 TFLOP/s, NVFP4/MMQ W4A8 default-on) - the minted NVFP4 tensors (MLA q_a/q_b,
o_proj, dense MLPs, shared experts) are fine. But a BF16-resident tensor has no Q8_0/f16
mirror, so every prefill projection **dequants the full weight to f32 and runs an f32,
non-tensor-core GEMM plus a fresh 2x-weight f32 buffer per call** - the exact cliff the
`MEMRA_PP_BF16` FLAGS row (docs/FLAGS.md:864) measured on the step37 prime: "norm+qkv
507 ms + o_proj 311 ms at ~15-20 TFLOP/s/rank on a 250+ TFLOP/s card." The tensor-core
BF16 door exists (`memra_bf16_pp_gemm`) and is OFF because it failed the first-token
argmax gate on one of four real prompts; re-admission needs a logit-delta cell or an
owner acceptance decision, both named in that row.

ARITHMETIC for glm5: KDA q/k/v [8192,4096] x3 + wo [4096,8192] + low-rank pairs per
layer, x34 layers, x4096 tokens ~= 37 TFLOP per chunk. At 15-20 TFLOP/s that is
~1.9-2.5 s/chunk (~0.5 ms/token); at tensor-core rates ~0.2-0.4 s. Real, engine-generic,
and second-order next to §1.1.

### 1.3 KDA scan: sequential over tokens, by declared deliberate increment

`crates/memra-engine/src/kda.rs:13-23` states it outright: "PREFILL DISPATCH -
SEQUENTIAL SCAN, not the chunked UT transform (deliberate, this increment).
`memra_kda_scan_s128` runs prefill and decode alike ... The chunked twin is a
tuning-phase follow-up," and notes KDA's per-channel decay needs a per-channel
cumulative gate `Gcum[t][i]` (the banked `chunk_kimi_delta_attention` reference in
`../modular_glm5_next-ref.py`) where GDN gets away with a scalar per (token, head).
The launch (`kda.rs:586-595`) is one kernel of grid (64 heads x 32 column-blocks) whose
blocks each iterate all `t` timesteps serially: one launch (good), serial arithmetic in
t (the thing chunked WY kernels exist to fix; the GDN twin already ships behind
`MEMRA_GDN_CHUNKED`). x34 layers per chunk. Cost per chunk not yet isolated on this
card class - that is one of the two numbers `profile-prime-phases.sh` returns.

### 1.4 MLA + DSA kpool: chunk-wide and already cheap; f32 kernel class noted

`mla_attn_cached` -> `mla_attn_core` (`hybrid_forward.rs:5927`) processes the whole
chunk per call: projections via `Engine::matmul` (tensor-core for the NVFP4 members),
one `mla_attn_gathered`/`mla_attn_absorbed` launch over the chunk. The kernels are
custom f32 (not flash-class, not tensor-core: `mla_ffi.rs:353,648`), but DSA caps
attended keys at 2048+tail and only 11 main-stack layers are MLA, and the banked bench
says this whole subsystem is milliseconds: `../kpool-bench-Frankfurt-crossover.txt`
has, at t_q=512, score 0.80-41 ms and attend ~5 ms per layer per call across ctx 4k-1M
(pool-key build is incremental and flat ~0.01 ms/step). Even scaled to t_q=4096 this is
sub-second per chunk across all 11 layers. NOT the wall. (The ring-OFF 2.4x prefill
slowdown found by the ctxprobe is an open attribution flagged in BOXPROBE.md, keyed to
the flat-plane scorer path; the ring is ON by default.)

### 1.5 mHC: per-chunk batched kernels, second-order

`hyper.rs` runs, per layer per site, one cuBLASLt f32 mixes GEMM (`Engine::linear`) plus
batched `memra_dsv4_{rowsq_scale, hc_sinkhorn_m, hc_collapse, hc_post}` kernels over the
whole chunk - sinkhorn is per (token, site) semantically but batched in one `_m` kernel
launch, ~8-10 launches per layer per chunk, not per token. Cost: the stream state is
`[t, 4, hidden]` f32 (4x the serial trunk's transient, the named cost of the simple
form, `hybrid_forward.rs:1286-1295`), and the hc GEMMs are small. Second-order; the
profile buckets it to confirm.

### 1.6 Launch-count estimate per 4,096-token chunk (ARITHMETIC, from source structure)

| term | launches |
|---|---|
| MoE routed experts, 49/token-layer x 42 layers x 4096 tokens | ~8.43M |
| MoE shared expert + router, per layer chunk-wide | ~500 |
| Trunk mixers (34 KDA + 11 MLA), chunk-wide kernels | ~1.5k |
| mHC pre/post/expand/collapse, chunk-wide | ~1k |
| **Total** | **~8.4M, of which >99.9% is per-token MoE dispatch** |

At the ~6-8 us launch+gap class the decode attribution solved on this box family, the
launch term alone reproduces the measured wall. The MoE FFN is the wall; everything else
is a follow-on lever.

---

## 2. What the fast engines do, mechanism by mechanism

(Web survey 2026-08-29, primary sources only: project source files, PRs, official docs
and blogs, each with version or date. For each mechanism: why it is fast, and whether
memra already has the ingredient.)

### 2.1 vLLM

- **Fused MoE: sort tokens by expert, then expert-batched GEMM.** The
  `fused_moe` Triton kernels (`vllm/model_executor/layers/fused_moe/fused_moe.py`,
  main) consume `sorted_token_ids` - "sorted indices of tokens, repeated topk times and
  arranged by the expert index ... padding ensures divisibility by BLOCK_SIZE_M" -
  produced by `moe_align_block_size` (CUDA kernel
  `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`, cub BlockScan prefix sums),
  which "pads the number of tokens that each expert needs to process so that it is
  divisible by block_size". Quantized/EP backends are grouped GEMM proper:
  `cutlass_moe.py` (FP8/NVFP4 CUTLASS grouped), `deep_gemm_moe.py` (DeepGEMM,
  `VLLM_USE_DEEP_GEMM` added in v0.8.3 PR #13932, 2025-04-01, default ON on main),
  `marlin_moe.py` (PR #7527/#7766, Aug 2024); matrix at
  docs.vllm.ai/en/latest/design/moe_kernel_features/ (2026-08-18). Why fast: each
  expert's weights stream through tensor cores ONCE per layer per batch over all its
  tokens, instead of once per (token, expert). Ingredient in memra: partially, three
  disjoint forms - `moe_ffn_pairs` (batched pairs but matvec-class and
  softmax-router-only), A2 `moe_ffn_grouped` (host grouping, `MEMRA_MOE_GROUPED` off),
  and the step37 NVFP4 grouped f16 GEMM prime (170-270 TFLOP/s measured,
  `MEMRA_STEP_GEMM_PRIME`, family-default ON for step37, TP-runtime only). NONE
  reachable by glm5_next today (§1.1).
- **Chunked prefill inside the continuous batch.** V1 default: a fixed token budget per
  step mixes compute-bound prefill chunks with memory-bound decodes;
  `max_num_batched_tokens` defaults are hardware-keyed (16384 on >=160 GB cards, 8192
  server on H100-class; `vllm/engine/arg_utils.py get_batch_defaults()`, main; V1 alpha
  blog 2025-01-27). memra: has the chunking (`hyper_prime_ranges`, 4096 cap), lacks the
  batched inner loop that makes a chunk cheap - our chunk is a scheduling unit wrapped
  around a per-token program.
- **Varlen flash prefill attention.** One fused ragged-batch kernel
  (`flash_attn_varlen_func` with `cu_seqlens`, e.g. `v1/attention/backends/mla/common.py`
  v0.8.4); FA3/FA4 by arch, FlashInfer first on Blackwell
  (docs.vllm.ai/en/latest/design/attention_backends/, 2026-08-28). memra glm5: MLA/DSA
  is already chunk-wide and ms-class (§1.4); adequate for now.
- **CUDA graphs are decode-only, by stated reasoning.** v0.6.0
  `worker/model_runner.py capture_model` docstring: "CUDA graph's performance gain is
  negligible if number of batched tokens are larger than 200 ... vLLM only captures
  decoding requests." That is the norm our prefill inverts: our prefill launches are
  decode-sized, so we pay the launch tax 4,096 times per chunk where vLLM's prefill
  kernels are big enough not to need graphs at all.
- **Published prefill class**: DeepSeek-R1 NVFP4 22,476 prefill tok/GPU/s on GB300
  (vllm.ai/blog 2026-02-13); 26.2k prefill tok/GPU/s disaggregated on GB200
  (2026-02-03); ~2.2k tok/s/GPU wide-EP on H200 (2025-12-17); GLM-4.5-FP8 355B on
  4x H200: mean TTFT 2.1 s at 8k-token prompts, burst 16 (vllm-project/recipes GLM.md).

### 2.2 SGLang

- **Same grouped-GEMM MoE class, with the prefill/decode split made explicit.**
  DeepGEMM grouped GEMM for FP8 MoE shipped v0.4.4 (2025-03-13), default-on for
  Hopper/Blackwell; the LMSYS 96xH100 deployment blog (2025-05-05) states DeepGEMM's
  contiguous grouped-GEMM layout "is ideal for the prefill phase" while the masked
  layout pairs with decode, and DeepEP's normal dispatch is "optimized for handling
  long input sequences, such as during the prefill phase". Two-batch overlap gave
  +27-35% prefill throughput; EPLB +1.49x prefill at scale. Published prefill:
  52.3k input tok/s per 8-GPU H100 node (~6.5k/GPU) at 2k-token prompts (2025-05-05);
  26,156 input tok/GPU/s on GB200 NVL72 (2025-09-25).
- **RadixAttention prefix caching** (arXiv 2312.07104; lmsys.org blog 2024-01-17):
  KV of shared prefixes lives in a radix tree (`RadixCache.match_prefix`); a hit
  removes that prefix's prefill entirely and only the suffix is prefilled; cache-aware
  routing took hit rates 20% -> 75% (v0.4 blog, 2024-12-04). memra: the latent-plane
  prefix cache EXISTS but is pinned OFF (`MEMRA_PREFIX_CACHE_MB=0` in every ctxprobe
  arm) due to the restore defect. For agentic multi-turn traffic (our buyer profile)
  this lever multiplies every kernel win. §3 L5.
- GLM-4.5-Air fused-MoE tuning configs for RTX PRO 6000 Blackwell exist upstream
  (PRs #10243 Sept 2025, #13711 Nov 2025) - the model class runs grouped-GEMM prefill
  on exactly our card class - but no published throughput row.

### 2.3 TensorRT-LLM

- **CUTLASS grouped GEMM MoE, one launch for all experts.** Tokens radix-sorted by
  expert (`fusedBuildExpertMapsSortFirstToken`), gathered expert-contiguous
  (`expandInputRowsKernelLauncher`), then ONE CUTLASS grouped GEMM runs every expert's
  FC1, again for FC2, and `finalizeMoeRoutingKernelLauncher` un-permutes and does the
  top-k weighted reduction (`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`,
  `include/moe_gemm_kernels.h`, main, verified 2026-08-29; FP8xFP4, FP4xFP4, BF16xFP4
  instantiations with SM100 block-scaled layouts). NVIDIA: "Grouped GEMM helps
  accelerate MoE models by offering a more efficient way to perform multiple expert
  computations in parallel" (world-record blog, Mar 2025).
- **In-flight batching + chunked context**: context-phase tokens pack into the same
  iteration as generation tokens; chunked context (default `max_num_tokens` 8192 since
  v0.11) is "always enable it" guidance (nvidia.github.io/TensorRT-LLM
  paged-attention-ifb-scheduler docs). Chunked prefill is supported for Glm4Moe,
  DeepseekV3, Qwen3Moe (supported-models.md, main).
- On OUR card class: RTX PRO 6000 (SM120) is supported since v0.20.0, and
  `CuteDslB12xFusedMoE` is documented as "NVFP4 hybrid CUTLASS-prefill / FlashInfer
  NVFP4 MoE decode - best perf on RTX PRO 6000" (MOE_DEVELOPER_GUIDE.md) - i.e. the
  vendor's answer for this exact card is still grouped-GEMM prefill over NVFP4 banks,
  the same weight format we mint.

### 2.4 llama.cpp (closest cousin: GGUF-class quant, single box, no batching server)

- **`ggml_mul_mat_id` batches expert matmuls over the ubatch.** Op introduced with
  Mixtral (PR #4406, 2023-12-13); the decisive change is PR #6505 (2024-04-18), "group
  all experts in a single ggml_mul_mat_id": Mixtral Q3_K_S on one RTX 3090 Ti went
  pp512 387.6 -> 1226.1 t/s (3.16x). Later the per-expert column info moved inside the
  MMQ kernel (PR #13199, 2025-04-30: DeepSeek-V2-Lite pp2048 3,491 -> 7,603 t/s) and
  id-compaction moved fully on-device with no host sync (PR #15525, 2025-08-25:
  GLM-4.5-Air Q4_0 pp512 1,800 -> 2,680 t/s on 3x RTX 4090) - note that last receipt is
  our architectural cousin's cousin: GLM-Air MoE prefill at 2.7k t/s on consumer cards.
- **Quantized prefill without whole-model dequant**: dispatch in
  `ggml-cuda.cu ggml_cuda_mul_mat_id` sends batch <= 8 to fused expert MATVECS (MMVQ)
  and batch > 8 to int8 tensor-core MMQ GEMMs consuming the ids directly; MMQ became
  the default over dequant+cuBLAS in PR #8075 (2024-06-24). This is precisely the
  pp-vs-tg split memra already implements for its OWN trunk matmuls at
  `GEMM_M_THRESHOLD = 16` (`lib.rs:12331`) and then does not apply to the routed-expert
  loop on sigmoid-router archs: we run the "batch <= 8" arm at batch 4096.
- **pp:tg ratios on big MoEs, published llama-bench rows**: Mixtral 8x7B 24x
  (PR #6505); Qwen3-235B-A22B Q8_0 21x (discussion #14174, Jun 2025); DeepSeek-R1 671B
  IQ1_S 23x (issue #11474); and the single-card anchor for OUR hardware class:
  **gpt-oss-120b MXFP4 on ONE RTX PRO 6000 Blackwell 96 GB: pp2048 5,518 t/s vs tg128
  196.3 = 28x** (official guide, discussion #15396, Aug 2025). memra glm5's ratio is
  ~1.9x against the best decode arm (82 pp vs 43 ms/token A3 decode) and ~1.0x against
  A1. A prefill:decode ratio near 1 on a tensor-core GPU is by itself diagnostic of
  per-token prefill dispatch.

### 2.5 DeepSeek FlashMLA / MLA prefill practice

- **FlashMLA** (2025-02-24 release): decode-side MLA kernels for Hopper, "up to 3000
  GB/s in memory-bound ... and 580 TFLOPS in computation-bound configuration on H800"
  (README at first release; 660 TFLOPS after the 2025-04-22 kernel). Current main
  (2026-01-16): dense DECODING is SM90-only MQA-form; dense PREFILL exists only as an
  SM100 standard-MHA kernel (NVIDIA-contributed, PR #76); sparse (DSA) prefill kernels
  reach 640 TFlops H800 / 1450 TFlops B200. There is NO dense absorbed-form (576/512)
  prefill kernel - nobody runs absorbed MLA at prefill.
- **The dual-form rule, stated by every engine**: prefill runs the MATERIALIZED MHA
  form (compute-friendly, 192/128 head dims), decode runs the ABSORBED latent MQA form
  (data-movement-friendly, 576/512). Canonical text is vLLM's
  `v1/attention/backends/mla/common.py` module docstring (PR #13789, 2025-02-27): "we
  generally want to use the compute friendly approach for 'prefill' ... and the
  data-movement friendly approach for 'decode'"; FlashInfer PR #765 (2025-02-01) and
  TRT-LLM tech blog 03 ("non absorbed version is beneficial for the prefill phase with
  input length 256 or larger") say the same. (Survey correction for future citation
  hygiene: neither a vLLM nor a FlashInfer MLA blog post exists; cite the docstrings
  and PRs.) memra runs the absorbed f32 form for both phases (§1.4). Because DSA caps
  attended keys at 2048+tail, this is currently ms-class and NOT part of the 100x; it
  becomes the binding term only after L1-L3 land, and the dual-form rewrite is filed as
  a later lever, not this arc.

### 2.6 The anchor for "what this card class should do"

No engine has published GLM-5-class prefill on RTX PRO 6000 Blackwell. The honest
anchors: llama.cpp does 5.5k t/s prefill for a 120B MoE on ONE card of our class
(§2.4); SGLang/vLLM do 22-26k prefill tok/GPU/s for DeepSeek-class MoEs on GB200/GB300
datacenter parts and ~6.5k/GPU on H100 at scale; TRT-LLM ships a grouped-GEMM prefill
MoE path named for our exact card. Against llama.cpp's single-card 5.5k, our 82 tok/s
is a **67x gap on like hardware**; against datacenter grouped-GEMM deployments it is
100-300x. "100x-class" is fair.

---

## 3. The plan: levers ranked by estimated share of the gap

Estimated shares are of the measured 12.3 ms/token prefill wall; "target" is the
per-token cost after the lever, ARITHMETIC unless receipted. Gates follow house law:
routing/selection is exactness-gated; grouped expert GEMM output is reference-band
gated (the grouped GEMM prime is measured non-bit-stable run to run,
`hybrid_forward.rs:6812-6824`, so byte identity is not the honest bar there), plus the
sampled vendor-default post-deploy probe per serving law.

### L1. Expert-grouped MoE prefill for sigmoid-router archs (glm5_next) - THE LEVER

- **Share: ~75-90% of the wall** (2,058 of ~2,060 launches/token; 4.76 GB/token of
  VRAM weight re-reads -> 0.023 ms/token equivalent).
- Build: a grouped prefill arm reachable from `moe_ffn_inner` for
  `cfg.sigmoid_router().is_some()` at `t >= PRIME_MIN_T`, composed of ingredients that
  all exist: exact m-invariant router logits (`moe_router_logits`, keep), device
  grouping of (token,expert) pairs by expert (counting sort, the `moe_align_block_size`
  idea; kill the per-layer host sel/w readback at prefill while keeping the host oracle
  as the gate arm), then per-expert tensor-core GEMM over the NVFP4 banks - the step37
  `run_tensor_parallel_routes_nvfp4_prime_grouped` kernel class (measured 170-270
  TFLOP/s) generalized off the TP runtime to single-device slab/SLRU provenance, with
  the three glm5 specifics the fused epilogue already solved and gated once: sigmoid
  noaux_tc sel/w, PRE-clamped SwiGLU epilogue, per-expert NVFP4 `weight_scale_2` fold
  (`../moe-epilogue-receipts/`, 13-test gate with five red arms - reuse those arms).
- Accepts: reference-band vs `memra-reference` on the NVFP4 macro fixture (band
  calibrated by the MEMRA_MOE_DETERM jitter probe), routing exactness (sel/w
  byte-identical to host oracle), run-gen argmax gate on real prompts, then the
  interleaved x5 TTFD A/B on the serving card class and the sampled vendor-default twin.
- Target: MoE term ~0.1-0.4 ms/token; whole-wall 12.3 -> ~2-3 ms/token (4-6x) before
  L2/L3. Engine work generic in the kernel, glm5-specific in the wiring.
- Agent-time: ~1-2 agent-days of code; the schedule is hardware time - one bench-box
  window for the calibration band + A/B, one for the qualification battery.

### L2. Tensor-core prefill for BF16-resident trunk (engine-generic)

- **Share: ~4-8%** now (~0.5 ms/token, 34 KDA layers' f32 GEMMs at 15-20 TFLOP/s);
  becomes ~25% of the post-L1 wall.
- Build: re-open `MEMRA_PP_BF16` (`memra_bf16_pp_gemm` exists) under its own FLAGS-row
  terms: a logit-delta cell showing the flipped position is a near-tie, or a shape
  restriction avoiding the flip, or the owner accepting argmax movement; alternatively
  a Q8_0 mirror minted at load for BF16 trunk tensors rides the existing MMQ W4A8 path
  with no new numeric door.
- Accepts: run-gen argmax gate + logit-maxdiff class stated (the `MEMRA_BF16_MMV`
  acceptance class), boot battery.
- Agent-time: days-class code either way; one box window of gates.

### L3. Chunked KDA prefill scan (glm5-specific; generic to KDA archs)

- **Share: unknown, bounded** - the serial-in-t term the profile must size (§4). The
  GDN chunked WY twin next door is the template; KDA needs the per-channel `Gcum`
  algebra, whose reference is already banked (`chunk_kimi_delta_attention`).
- Accepts: bit-gate vs the sequential scan on the WY grid boundaries (the GDN
  grid-alignment law already generalizes: `align_prime_ranges_to_gdn`), fixture gate vs
  `memra_reference::kimi_delta_net`.
- Agent-time: the kernel is the largest single kernel job in this plan (~2-3 agent-days)
  - sequence it after the profile proves its share.

### L4. Prefill host-sync diet (generic)

- **Share: small now, visible post-L1**: 42 per-layer sel/w D2H drains per chunk plus
  per-layer trace/record hooks. L1's device grouping removes the structural need; keep
  the host oracle behind an env for gates.

### L5. Prefix cache re-enable (product lever, not kernel work)

- Not part of the 12.3 ms/token attribution, but the SGLang comparison is a reminder
  that for agentic multi-turn traffic (our ICP) prefix reuse removes WHOLE prefills.
  Blocked on the latent-plane restore defect that forced `MEMRA_PREFIX_CACHE_MB=0`;
  that defect has its own lane and its fix multiplies every win above for warm traffic.

### Sequencing and the honest ceiling

L1 alone takes glm5 prefill from ~82 tok/s to an estimated 400-800 tok/s
(2-3 ms/token); L1+L2+L3 to an estimated 1,500-3,500 tok/s (0.3-0.8 ms/token), i.e.
TTFD ~2-3 s at 4.6k tokens instead of 57 s, inside the 90 s platform deadline out to
the low hundreds of thousands of tokens. The remaining distance to the biggest
published engine numbers is batching across requests, flash-class MLA prefill forms,
and multi-card expert parallelism - real, later, and not needed to close the product
blocker (the deadline wall and the 1M-context claim's serving viability).

---

## 4. The one profiling question the next box window answers first

**Of the 12.3 ms/token, what is the exact split between (a) per-token MoE dispatch,
(b) the KDA sequential scan, (c) f32 trunk GEMMs, (d) router host drains, (e) rest?**
§1 argues (a) dominates from launch counts and the decode X-term; the plan's L1-before-L3
sequencing depends on it. The script is written and in this directory:
`profile-prime-phases.sh` - one nsys pass over a single cold 4,096-token prime with the
real NVFP4 artifact on a bench box, kernels bucketed by phase family, memcpy reported
separately, wall-minus-GPU-time reported as the launch-gap term (the decode-attribution
method applied to prefill). Run it BEFORE building L1's kernel so the A/B has its
baseline attribution, and bank the phase table beside this file.

---

## Receipts index

| claim | receipt |
|---|---|
| TTFD 57.4/67.1/78.8 s at 4.6/5.5/6.5k tokens | `../ring-sizing-20260828/box-ctxprobe/BOXPROBE.md`, `05/06-*-timing.txt` |
| 49 launches per token-layer; 2058/token across 42 layers | `../moe-epilogue-receipts/nsys-launch-counts.md` |
| decode split: 17.1 ms/token invariant launch term; 4.76 GB/token expert VRAM reads; 171.2 GB expert bank | `../decode-attribution-receipts/ATTRIBUTION.txt` |
| BF16 trunk prefill at 15-20 TFLOP/s, f32 dequant per call | `docs/FLAGS.md:864` (`MEMRA_PP_BF16`) |
| grouped NVFP4 GEMM prime 170-270 TFLOP/s; per-token routes 240 s at m=4092; 3.5-4.9 s vs 29 s+ fallback (step37) | `crates/memra-engine/src/lib.rs:675-686`, `hybrid_forward.rs:6777-6798` |
| KDA sequential-scan prefill is deliberate; chunked twin named follow-up | `crates/memra-engine/src/kda.rs:13-23` |
| kpool/MLA subsystem is ms-class per layer per call | `../kpool-bench-Frankfurt-crossover.txt` |
| fused epilogue exactness + five red arms, both provenances | `../moe-epilogue-receipts/README.md`, `AB-PLAN-RESIDENCY.md` |
| qwen3.8 prefill ~1-2.5k tok/s on the same engine (batched pairs arm engaged) | internal serving receipts, 2026-08-29 (private ops repo) |
