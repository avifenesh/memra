# VISION-LANE.md — image+video input for Qwen3.8-27B serving (2026-08-15)

Owner directive: serve vision (image AND video), one-day lane, back to perf after.

## Config truths (read from the checkpoints, 2026-08-15)

- vision_config: qwen3_5_vision ViT — depth 27, hidden 1152, heads 16 (head_dim 72),
  intermediate 4304, gelu_pytorch_tanh, patch 16, spatial_merge 2, temporal_patch 2,
  num_position_embeddings 2304 (LEARNED absolute pos, 48x48 grid, bilinear-interp per
  image grid — NOT rope), out_hidden 5120, deepstack []. dtype bf16, ~700M params.
- text side rope_scaling: **null** — NO M-RoPE. Vision tokens take standard sequential
  positions in the TRUNK. The TOWER, however, carries its own 2D vision rope
  (Qwen3_5VisionRotaryEmbedding, theta 10000, head_dim/2 angles split h/w 18+18,
  neox rotate-half) applied to q/k in every block ON TOP of the learned pos table —
  discovered at parity-bisect time, 2026-08-15.
- pos-table interpolation: bilinear with align_corners=TRUE (linspace 0..47), not the
  0.5-offset align_corners=False mapping.
- `Engine::sdpa_naive`'s doc comment claims [head_dim, n_head, T] layout; the KERNEL
  actually indexes token-major [T, n_head, head_dim] == the raw qkv GEMM row layout.
  (Parity bisect receipt: pre-blocks cos 1.000000, blk0 0.94 with the dim-major
  permute, post-blocks 0.13; token-major + rope -> merger min-cos 0.9997 PASS x3.)

## STATUS 2026-08-15: image path CODE-COMPLETE on lane/vision
- Tower parity gate: PASS on 3 structured images (mean_cos >= 0.999995, min 0.9997)
  vs HF Qwen3_5VisionModel f32 CPU oracle (box, tools receipt /tmp/vis).
- Serving: image_url data URIs end-to-end (parse -> patchify -> pad-run render ->
  admit alignment -> tower overlay -> prime_cache_overlaid). Vision sessions: no
  reuse/prefix/affinity/park/spec/batch-prime; pads never hit decode_step.
- Remaining to prod: box battery (text paths unchanged) + VQA smoke + models.toml
  modality flip + MEMRA_VISION_DIR in the launch script. Video: next (frame sampling
  <= 16, pairs fill temporal_patch 2; video_url currently 400s).
- preprocessor: mean/std 0.5, patch 16, merge 2, pixel budget shortest 65536 /
  longest 16777216 (area), Qwen2VLImageProcessorFast semantics.
- **Weights location**: the ct-NVFP4 checkpoint has ZERO visual tensors (recipe strips
  the tower). The official FP8 checkpoint carries all 333 `model.visual.*` tensors in
  ONE shard: `outside.safetensors` (box:
  /data/memra/models/Qwen3.8-27B-FP8-.../outside.safetensors). Tensor names:
  blocks.N.attn.{qkv,proj}.{weight,bias}, blocks.N.mlp.linear_fc{1,2}.{weight,bias},
  norms per block, patch_embed + pos_embed + merger (verify exact names on load).

## Build order (v1 = correctness on cuBLAS; quantize later if it ever matters)

1. `MEMRA_VISION_DIR=<dir-with-outside.safetensors>`: loader module reads visual.*,
   bf16 -> f32 GpuTensors (~2.8 GB VRAM; fine).
2. Host preprocess: base64 image decode (image crate), resize to grid (multiples of
   32 px, area within budget), normalize, patchify [N, 3*2*16*16=1536] (image repeats
   its frame for temporal_patch 2).
3. ViT forward: patch_embed GEMM + interp pos_embed; 27x (LN -> qkv GEMM -> full
   bidirectional attention (cuBLAS QK^T -> row softmax -> V) -> proj -> LN -> MLP
   gelu_tanh); no KV cache, no causal mask.
4. Merger: 2x2 spatial merge concat -> LN -> fc1 -> gelu -> fc2 -> [N/4, 5120].
5. Splice: chat template expands <|image_pad|> x n_tokens between
   <|vision_start|>/<|vision_end|>; prime the trunk with MIXED embeddings (text token
   embeds + image embeds at pad positions) — the `_h`/embd_dev prime seams exist.
6. Video: uniform frame sampling (cap ~16), consecutive-frame pairs fill
   temporal_patch 2, per-frame grids concatenated; template timestamps per Qwen3VL
   processor. v1 ships images first, video behind the same tower next.
7. API: chat content arrays — {"type":"image_url","image_url":{"url":"data:..."}};
   http(s) URL fetch gated behind MEMRA_FETCH_URLS=1 (SSRF posture: off default).
8. Gates: HF transformers reference on 3 images — image-token embedding parity
   (cosine > 0.999 per token vs reference merger output), then end-to-end VQA smoke
   (describe-this-image correctness), then serve battery unchanged (text paths
   untouched), modality flip in models.toml (input_modalities += image) + pricing
   per-image = input tokens (image tokens bill as prompt tokens).

Trunk source note: vision requests can serve on ANY trunk (ct-NVFP4 ST, GGUF, FP8) —
the tower output is just embeddings; the FP8 dir is only the tower's weight source.

## Tensor census (outside.safetensors, verified)

- patch_embed.proj.weight [1152, 3, 2, 16, 16] + bias — flatten to Linear 1536->1152.
- pos_embed.weight [2304, 1152] — 48x48 learned grid, bilinear-interp to each image grid.
- blocks.0..26: norm1/norm2 = **LayerNorm WITH bias** (not RMS), attn.qkv [3456,1152]+bias,
  attn.proj [1152,1152]+bias, mlp.linear_fc1 [4304,1152]+bias, fc2 [1152,4304]+bias.
- merger: norm (LN 1152 + bias), linear_fc1 [4608,4608]+bias (4608 = 1152*4 spatial
  concat), linear_fc2 [5120,4608]+bias.
- Same shard also carries bf16 lm_head/embed_tokens/final norm (FP8's unquantized parts).

Missing engine primitives for the tower (all tiny): LayerNorm(+bias), gelu_pytorch_tanh,
bidirectional row-softmax attention (cuBLAS QK^T/V around a row-softmax kernel). All GEMMs
ride cuBLAS f32 (`linear_f32`). Everything else exists.
