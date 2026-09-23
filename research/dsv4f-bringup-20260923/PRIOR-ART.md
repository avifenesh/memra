# DSv4-Flash prior art: published batch-1 anchors and kernel designs (survey 2026-09-23)

Read-only survey of public DeepSeek-V4 serving code and published numbers, done before the next
tuning lanes so they adopt what already exists instead of inventing it. Sources are pinned:
VL = vllm-project/vllm@eb2e91d3bac8, SG = sgl-project/sglang@86cb4a0f9796, HF =
huggingface.co/deepseek-ai/DeepSeek-V4-Flash `inference/`. Other repos are cited by path at their
default branch on 2026-09-23. Nothing here is a runtime dependency: memra reads these designs and
reimplements the ones that keep its numeric program.

## Published batch-1 decode

| setup | tok/s | spec decode | source |
|---|---|---|---|
| 2x RTX PRO 6000, TP=2, vLLM + b12x kernels | 130.8 (smoke 132..133) | none | local-inference-lab/rtx6kpro `models/ds4-flash-v6.md` |
| 2x RTX PRO 6000, TP=2, vLLM FlashInfer sparse MLA | 123.9 (CUTLASS MoE), 122.7 (default MoE) | none | same file |
| 2x RTX PRO 6000, TP=2, vLLM + b12x | 214.6 | MTP-2 | same file |
| 2x RTX PRO 6000, TP=2, V4-Flash-0731, b12x r16 | 212.45 (c1, n=3, 256 tokens) | DSpark K5 | hikarioyama/dsv4-flash-0731-sm120 `data/verified_r16_baseline.json` |
| H200, V4-Flash TP=4, 30K prefix | 266 at 4K ctx, 240 at 900K | EAGLE, accept about 2.5 | LMSYS blog `blog/2026-04-25-deepseek-v4.md` (read off the figure) |
| B200, V4-Pro TP=8, 30K prefix | 199 -> 180 | EAGLE | same blog |

The rtx6kpro rows are ctx 0k, fp8 KV, 30 s cells. These replace the vLLM TP=2 109.3 tok/s floor
of `BASELINE.md` as the plain anchor on this card shape: **130.8 tok/s plain, 212 tok/s with
DSpark**. No batch-1 V4-Flash number without speculative decoding was found for Hopper or
datacenter Blackwell.

## Reference program (HF `inference/kernel.py`, `model.py`)

- `hc_split_sinkhorn_kernel`: one CTA per token, 64 threads. Row softmax plus eps, column
  normalize, then 19 more row/column rounds, each with +eps.
- `hc_pre`: fp32 `F.linear(x, fn)` times rsqrt of the mean square, Sinkhorn, then
  `y = sum(pre * x)` stored as bf16. `hc_post`: `post * x + sum_j comb[j,k] * res[j]`, stored bf16.
- RMSNorm: plain fp32 torch, not fused.
- Gate: fp32 linear, `softplus().sqrt()`, bias added for selection only, top-6, gather the
  unbiased scores, `/ sum`, `* 1.5`. The first 3 layers are hash-routed.
- wo_a: bf16 einsum `bsgd,grd->bsgr` (the code comments "could do FP8 einsum").
- `sparse_attn_kernel`: grid (m, b), 256 threads, 64-token blocks, online softmax; the sink is
  added at the end (`sum_exp += exp(sink - max)`) and P is cast to bf16 before PV.

## Kernel designs by component

**mHC.**
- VL `vllm/model_executor/kernels/mhc/tilelang.py`: 2 kernels up to 32 tokens; below 16 tokens
  tile_n=2, n_splits=8, 128 threads, grid (1,12,8), split-K partials then a serial split sum.
  Deterministic, but squares and GEMV read the unrounded fp32 new residual.
- SG `python/sglang/kernels/ops/layernorm/mhc.py`: the same TileLang design.
- b12x (`local-inference-lab/b12x`, Apache-2.0, CuTe DSL for SM120/121) `b12x/norm/mhc/`: decode
  post+pre in 2 launches, no atomics. Partial kernel: 128 threads, one hidden element per thread,
  32 tiles x 7 mix groups = 224 CTAs, computes `o = post*x + comb^T r` and **rounds it to bf16
  before reuse, as the reference does**; 5-step xor butterfly, serial 4-warp sum, per-tile partials
  to global, plus the 4x4 Gram of the residual. Finalize: 1024 threads per 1024-element tile, sums
  the 32 partials, runs Sinkhorn per CTA, and takes the RMS as `pre^T G pre`. The Gram RMS is not
  the reference's mean of squares of bf16 y; the `lagged_mix` variant sums squares of the bf16 y
  directly. PDL off by default.
- TensorRT-LLM `cpp/tensorrt_llm/kernels/mhcKernels/`: ksplit variant with fp32 residual; the
  all-in-one variant uses float `atomicAdd` and last-CTA election (nondeterministic, noted in
  `mhcFusedHcKernel.cu`).
- DeepSeek TileKernels `tile_kernels/mhc/sinkhorn_kernel.py` matches the reference;
  `pre_big_fuse_kernel.py` is 96 threads without RMSNorm.

**Router.**
- VL `fused_moe/router/dsv4_topk.py`: Triton, one warp, PDL; computes `w * (scale / sum)`.
- SG `moe_fused_gate.cuh`: softplus as `fmaxf(x,0) + log1pf(expf(-|x|))`; the Triton twin
  open-codes log1p. Argmax by `shfl_down`, sum by warp reduce.
- TileKernels `tile_kernels/moe/scoring.py`: threshold-20 softplus (torch's form),
  `topk_gate_kernel.py` 32 threads with lowest-index ties, `normalize_weight_kernel.py` a serial
  sum started at 1e-20.
- TensorRT-LLM `noAuxTcKernels.cu`: sigmoid routers only, `score*scale/(sum+1e-20)`.

**wo_a.**
- VL `models/deepseek_v4/nvidia/ops/o_proj.py`: fused inverse RoPE + FP8 quant, then one DeepGEMM
  `fp8_einsum` over all 8 groups.
- SG `ops/attention/dsv4/wo_a.py`: GEMV grid (1024, 2), and a small-batch split-K 8x512 with fp32
  partials reduced by `tl.sum` (a tree); used only at the TP=4 shape.
- b12x `gemm/_shared/wo_mxfp8.py`: inverse RoPE + MXFP8 per-32 quant, weights repacked to MXFP8,
  one grouped GEMM. A different weight and activation program.

**RMSNorm.** FlashInfer `include/flashinfer/norm.cuh` for d=4096 bf16: 512 threads, 8 serial
squares per thread, xor butterfly, a warp-0 butterfly over 16 partials; `griddepcontrol.wait` at
start and `launch_dependents` at the end (PDL).

**Sparse attention on SM120.** FlashInfer `include/flashinfer/attention/sparse_mla_sm120/`
(PR #4380, top-k 192/256, 8..128 heads): decode grid (tokens, h_blocks, splits), a fixed-order
sequential split merge with grid (tokens, heads) and 64 threads; the sink folds as
`gmax = max(gmax, sink*LOG2E)`, `total += exp2f(sink_log2 - gmax)`. Split outputs are bf16.
FlashMLA sparse decode is SM90/SM100 only; its combine adds the sink as
`global_lse += log2f(1 + exp2f(sink*L2E - global_lse))`.

**Expert offsets.** VL `moe_align_sum_kernels.cu`: `cub::BlockScan`, integer and exact.

**Other.** The vLLM recipe (`vllm-project/recipes` `DeepSeek-V4-Flash.yaml`) notes sm_120 lacks
the FP4 indexer cache and the mega-MoE kernel, so it forces an fp8 indexer KV. The LMSYS blog's
radix-select top-k with cluster launch cuts batch-1 top-k at 1M context from over 100 us to about
15 us. The TensorRT-LLM V4 blog reports router GEMM 8 -> 3 us and top-k 384 -> 112 us, with no
batch-1 tok/s.

## Where prior art leaves the reference numeric program

memra's DSv4 kernels are bit-faithful to its CPU oracle (`-fmad=false`, fixed orders). A design
from this list is adopted only in a form that keeps that program.

1. vLLM, SGLang and TensorRT-LLM mHC use an unrounded fp32 new residual; the reference and b12x
   round to bf16.
2. b12x's Gram RMS is not the reference mean of squares.
3. Router normalization: vLLM `scale/sum`; TensorRT-LLM and TileKernels add 1e-20.
4. Softplus has three forms across torch/TileKernels, SGLang CUDA and SGLang Triton.
5. Atomics in the TensorRT-LLM all-in-one mHC; DeepGEMM's `use_deterministic_algorithms()` now
   defaults to False.
6. FlashInfer SM120 merges bf16 split partials in the exp2 domain; FlashMLA uses another sink
   LSE formula.
7. Tree reductions: SGLang `tl.sum`, the FlashInfer RMSNorm tree, DeepGEMM's TF32 prenorm GEMM.
8. FP8/MXFP8 activation quantization on wo_a (vLLM, b12x), which the no-format-shortcuts rule
   excludes as a deliverable.

## Adoption list

- **mHC in 2 launches**, on the b12x grid (32 tiles x 7 groups, per-tile partials, fixed
  cross-tile order), keeping the bf16 rounding of the new residual and taking RMS from the squares
  of the bf16 y, with PDL.
- **One warp router**: threshold-20 softplus, sqrt, bias, 6 argmax rounds with lowest-id ties, a
  serial 6-element sum, `w / sum * 1.5`.
- **RMSNorm** with the FlashInfer launch shape and PDL, with the reduction order made explicit.
- **wo_a as one grouped launch** (rows x 8 groups) on bf16 activations instead of 8 GEMVs.
- **Sparse attention** built on the FlashInfer `sparse_mla_sm120` structure and fixed-order merge,
  with fp32 split partials.
- **Expert offsets at batch 1** as 6-entry single-warp logic instead of a block prefix scan.
- **PDL** (`griddepcontrol`) across the decode chain.
