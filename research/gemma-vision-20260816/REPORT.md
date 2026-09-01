# gemma-4 vision input — derived laws + tower parity (lane/gemma-vision, 2026-08-16)

Doctrine: gemma-4 is its own semantic program. Every law below was derived fresh from
the family's reference implementation (llama.cpp `clip_graph_gemma4v` + `mtmd`, the
vendor-blessed consumer of the official `google/gemma-4-*-qat-q4_0-gguf` packaging) and
the mmproj tensor census. Nothing was assumed from the qwen3_5 lane; where the families
differ the difference is called out, because those are exactly the spots an analogy
would have shipped a silent wrongness.

## 1. Checkpoint + census (gemma-4-31B-it mmproj, `general.type = mmproj`, projector `gemma4v`)

- ViT: 27 blocks, hidden 1152, 16 heads (head_dim 72), ffn 4304, patch 16, BF16
  weights + F32 norms, 356 tensors. `clip.vision.image_size = 224` is vestigial — the
  family is NATIVE RESOLUTION (see §2).
- NOT SigLIP, despite the dims: RMS norms everywhere (no LayerNorm, no biases),
  per-head RMS q/k norms (`attn_{q,k}_norm.weight`, 72), sandwich post-norms
  (`attn_post_norm` / `ffn_post_norm`), GATED ffn (`ffn_gate/up/down`).
- Factored position tables: `v.position_embd.weight` logical [2, 10240, 1152] — one
  x-table, one y-table, 10240 rows each (native-res headroom), ADDED to patch embeds.
- Head tensors: `v.std_bias` / `v.std_scale` [1152], `mm.input_projection` 1152→5376.
  No pre_ln / post_ln. No ClippableLinear clamp scalars in this file.
- `image_mean = [0,0,0]`, `image_std = [1,1,1]` — normalization is NOT in the
  preprocessor for this family.

## 2. Derived preprocessing law

- Native resolution, smart-resize: round each side to the 48 grid (patch 16 × merge 3),
  then floor/ceil-rescale into the pixel budget; budget = OUTPUT tokens 40..280 where
  one token = one 48×48 px block (llama.cpp `set_limit_image_tokens(40, 280)`;
  the 40 floor is a quality bump, "performs quite poor with small images").
- Bilinear resample; pixels to [0,1]; the GRAPH applies `2x − 1` (no mean/std).
- Patch rows are (c, ky, kx)-ordered 768-float vectors (bias-less conv16-as-linear).

## 3. Derived tower law (per block; differences vs qwen3_5 flagged)

```
res = x
h   = rms(x, ln1)                                   # qwen: LayerNorm+bias
q,k,v = h @ Wq, h @ Wk, h @ Wv                      # separate, no biases (qwen: fused qkv + bias)
q,k = per-head rms(., q_norm/k_norm)                # qwen: none
v   = per-head WEIGHTLESS rms(v)                    # gemma4v-only quirk
q,k = 2D rope: dims 0..36 rotate by pos_x, 36..72 by pos_y   # qwen: y first — ORDER FLIPPED
      neox pairs (d, d+18) per half, theta = 100.0  # qwen theta: 10000
attn = sdpa(q, k, v, scale = 1.0, full)             # UNSCALED — qwen: 1/sqrt(d)
o   = attn @ Wo
x   = res + rms(o, attn_post_norm)                  # post-norm BEFORE residual (qwen: none)
res = x
h   = rms(x, ln2)
ffn = (gelu_quick(h @ Wgate) ⊙ (h @ Wup)) @ Wdown   # gelu_quick = x·σ(1.702x); qwen: plain MLP gelu_tanh
x   = res + rms(ffn, ffn_post_norm)
```

Head: 3×3 avg-pool over the grid (qwen: 2×2 concat+MLP merger) → ×sqrt(1152) →
`(x − std_bias) · std_scale` → weightless rms → project 1152→5376.

Positions feed the tower twice: additive factored tables at input AND per-layer rope —
both x = column, y = row, row-major grid.

## 4. Parity oracle — PASS at 1.000000, with one real finding

Independent NumPy reference (`gemma_vision_ref.py` — separate code path, same mmproj
weights, law from §3) vs the memra CUDA tower (`vision_gemma.rs`), per-token cosine on
a deterministic 384×384 synthetic gradient (24×24 grid → 64 output tokens):

| stage        | min_cos   | mean_cos  |
|--------------|-----------|-----------|
| pre_blocks   | 1.000000  | 1.000000  |
| blk0         | 1.000000  | 1.000000  |
| post_blocks  | 1.000000  | 1.000000  |
| pre_proj     | 1.000000  | 1.000000  |
| projected    | 1.000000  | 1.000000  |

(qwen lane bar was min-cos 0.9997; this lane lands exactly-1.0 at f32 print precision.)

**TF32 finding:** under the engine's default f32 cuBLASLt path the same tower reads
min_cos 0.9878 / mean 0.9998 at post_blocks — the UNSCALED attention (kq_scale 1.0)
amplifies TF32's 10-bit mantissa across 27 blocks. `NVIDIA_TF32_OVERRIDE=0` restores
exact parity. The qwen tower shares the same GEMM path but its 1/sqrt(d)-scaled
attention masked the loss. Serving decision for this family: run the tower with TF32
off (env or a non-TF32 GEMM arm) — the delta is measurable and the doctrine does not
ship "probably benign".

Reproduce:
```
gemma_vision_oracle <mmproj.gguf> <out_dir> [image]      # memra tower + stage dumps
GGUF_PY=<llama.cpp>/gguf-py python3 gemma_vision_ref.py <mmproj.gguf> <out_dir>
```

## 5. Derived SERVING laws — why this lane stops before the serve wire-up

1. **Non-causal image spans.** `mtmd_decode_use_non_causal` returns TRUE for gemma4v:
   the LM decodes image-token spans with BIDIRECTIONAL attention during prefill.
   memra's prime path is causal-only; priming gemma image embeddings through it would
   be exactly the fluent-hallucination failure class the vision doctrine exists to
   prevent — plausible answers, silently wrong attention. The qwen overlay could ship
   on the causal path because qwen3_5 image pads ARE causal; gemma's are not. Wiring
   the overlay before a masked-prefill arm exists would pass every smoke and fail the
   truth, so the serving gate must REFUSE gemma+images until that arm lands.
2. Token layout: `<|image>` (255999) + N soft positions (`<|image|>`, 258880) +
   `<image|>` (258882), N = (W/16)(H/16)/9 per image, 40 ≤ N ≤ 280.
3. Session laws (re-derived, not copied): the 31B LM is full-attention +
   sliding-window hybrid (5:1) — no recurrent state, so the qwen GDN cache-boundary
   law does NOT carry over. But any prefix-cache entry whose boundary lands INSIDE a
   bidirectional image span is invalid by construction (its KV rows were computed
   under a mask no causal continuation can reproduce), so: no cache split/seed inside
   image spans; entries must end strictly before or after a whole span. Spec + vision
   stays OFF at v1 for the same reason the qwen lane chose: the draft plane never saw
   the overlay, and here additionally never saw the non-causal mask.

## 6. What is NOT supported / not done (explicit)

- **Serving wire-up: NOT DONE, by derived law** (§5.1). The tower + oracle are the
  deliverable; the masked-prefill (bidirectional-island) arm is the prerequisite work
  item for serving, sized separately.
- Video: no video law exists for gemma4v in the reference (image-only projector);
  E2B/E4B audio (`GEMMA4A/UA` projectors) out of scope.
- Pan-and-scan / tiling: the reference serves gemma4v single-crop native-res; no tile
  law was derived and none is implemented.
- The 12B "encoder-free" variant is a DIFFERENT program (linear input projections, no
  ViT) — nothing here applies to it.
- e2e behavioral probe via `llama-mtmd-cli` (blue triangle): blocked by a crash in the
  pinned llama.cpp's `mtmd_init_from_file` on this rig (their tooling, reproduced on a
  fresh rebuild of cli + libmtmd). The tower parity gate does not depend on it; the
  serving lane should re-run the decisive probe end-to-end through memra once the
  masked-prefill arm exists.

## 7. Files

- `crates/memra-engine/src/vision_gemma.rs` — tower + preprocessing (laws in the
  module doc), `GemmaVisionTower::load/forward`, refuses non-gemma4v projectors.
- `crates/memra-engine/src/bin/gemma_vision_oracle.rs` — oracle harness + stage dumps.
- `research/gemma-vision-20260816/gemma_vision_ref.py` — independent NumPy reference.

---

# Masked-prefill (bidirectional-island) arm — 2026-08-16, same lane

## Mask law (pinned to reference code, boundary semantics answered)

Source: llama.cpp `mtmd-helper.cpp:290-333` — each image chunk decodes as its own batch
wrapped in `llama_set_causal_attn(false)`; text chunks decode causally.

- **Does the island see preceding text?** YES — the non-causal batch attends the whole
  KV cache, which holds everything before the image.
- **Does text after the image attend to the island?** YES, causally, like any KV — the
  island's K/V rows are mask-independent; only island QUERIES differ.
- **Do two islands see each other bidirectionally?** NO — each image is its own
  non-causal batch (their TODO requires one image per ubatch); a later island sees an
  earlier one causally only. Implemented as a span-id per position: visible iff
  `same_island || (causal && within SWA window)`.
- **SWA interaction:** islands are ≤280 tokens < window 1024, so island-internal
  visibility and the R6 window never conflict; the window applies unchanged to
  everything non-island.
- **Embedding scale:** `gemma4.cpp:182` — `ubatch.token ? sqrtf(n_embd) : 1.0f`. Image
  embeddings enter the trunk UNSCALED; memra splices tower rows after the text
  sqrt(5376) scale, raw.

## Implementation

- `sdpa_naive_island_f32` kernel + `Engine::sdpa_naive_island` (span-id mask,
  causal+window law preserved for text).
- `gemma4_prime` accepts the overlay: splice-after-scale + span-id build; island primes
  route every layer through the mask-capable naive kernel (correctness-first v1, same
  posture as the tower). `prime_cache_overlaid` routes gemma4 overlays in; E4B still
  refuses. Text-only primes take the UNTOUCHED pre-existing branches (island=None) —
  structural byte-identity, no behavior change without an overlay.
- `MEMRA_GV_FORCE_CAUSAL=1` — wrong-arm probe seam only (splice without islands).
- `gemma_vision_e2e` harness: image → tower → overlay → masked prime → greedy decode,
  first-step TOP8 print, and a RAW: prompt mode that scores llama-server's exact
  rendered prompt for stream-identical cross-engine comparison.

## Gates

**Behavioral (decisive probes, temp 0, greedy, memra masked arm):**
- blue triangle: "The image contains one shape: - A blue triangle." — matches
  llama-mtmd-cli reference ("The image contains one shape: a blue triangle.").
- missed-detail tell (triangle + red circle): "- a blue triangle / - a red circle" —
  both objects, matches reference. (Reference stack itself unblocked: the earlier
  mtmd crash was VRAM — `--no-mmproj-offload`; plus `--jinja` for the gemma4 template.)
- The WRONG arm (causal) also answers these tiny probes correctly — behavioral probes
  cannot discriminate the mask law (the qwen lesson, again). The numeric gate below is
  the real discriminator.

**Numeric (same Q4_0 weights, stream-identical prompt via /apply-template):**
first generated position after the image boundary, llama-server top-10 logprobs vs
memra TOP8 logits:
- top-1 IDENTICAL (id 100, logprob −0.0000 both sides);
- top-3 identical in order (100, 236820, 101);
- 8/8 memra top-8 ids inside llama's top-10; rank swaps only between pairs separated
  by <0.25 logprob in the reference;
- distribution structure agrees: Δ(top1→top2) 12.26 (memra) vs 12.50 (llama).
- **Mask discriminator:** the forced-causal arm shifts logits up to ~0.8 and pulls
  ids 95246/54204/919 into its top-8 — none of which appear in the reference top-10.
  Masked arm = reference-consistent; causal arm = measurably off-reference.

**TF32 law:** all gates above ran `NVIDIA_TF32_OVERRIDE=0`. Tower: required (unscaled
attention amplifies TF32; §4). LM-side prime: the q4_0 trunk path is quantized-kernel
dominated (not cuBLASLt-f32), so TF32 is not the LM's noise driver — but the PARITY
RUNS keep it off end-to-end so the tower stays exact. Serving decision documented:
tower TF32-off is mandatory; LM path unaffected.

## Launch remainder for gemma vision SERVING (not in this change)

1. memra-server wire-up behind `MEMRA_GEMMA_VISION` (default off, refuse gemma+images):
   family switch in content_to_text_vision (gemma prep + `<|image>`/soft/`<image|>`
   placeholder render), Request.images generalization (qwen VisionUnit → family enum),
   vision_spans on soft id 258880, build_vision_overlay → GemmaVisionTower, worker
   spawn-time tower load. Admission laws to keep: vision_req captured before take,
   spec-burst backstop, pads-never-decode guard.
2. Session laws to enforce at serve time: no prefix-cache boundary inside an island
   (v0 moot — gemma4 prime is fresh-prompt-only and refuses continuation), spec-off for
   vision sessions, no park/seed/fanout (family-agnostic s.vision gates already exist).
3. Text-only byte-identity battery: gemma serve gate re-run with the seam off AND on,
   no-image requests must stream byte-identical (engine-side is structurally identical;
   the battery receipt is still owed).
4. Decisive probe pair through memra-server itself (this report's harness answers are
   the target strings).
5. Perf posture: island primes run the naive kernel every layer — fine for ≤280-token
   islands at v1; FA island support is a later arm.
