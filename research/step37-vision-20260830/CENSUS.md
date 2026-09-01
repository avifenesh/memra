# step37 (Step-3.7-Flash) IMAGE INPUT: census + plan (lane/step37-vision, 2026-08-30)

Doctrine: step37 is its own semantic program. Nothing below is inherited from the
qwen3_5 or gemma4 vision lanes by analogy; every law was derived fresh from the pinned
artifact (HF `stepfun-ai/Step-3.7-Flash-NVFP4` @ `4275532ffd9a9496ff36b7a2dc4a9db1048da438`):
its `config.json` (`vision_config`, `model_type: perception_encoder`), the vendor
reference code shipped inside the checkpoint (`vision_encoder.py`,
`processing_step3.py`, `modeling_step3p7.py`, `configuration_step3p7.py`,
`chat_template.jinja`), and the safetensors shard headers. Where the family differs
from the two shipped vision programs the difference is flagged, because those are
exactly the spots an analogy would ship a silent wrongness.

## 1. Tensor census (from the shard headers at the pinned rev)

667 tensors, ALL BF16, unquantized inside the NVFP4 artifact (the modelopt
`quantization_config.ignore` list carries `model.vision_model*` and
`model.vit_large_projector`). 666 tower tensors live in `model-00001-of-00013`,
the projector in `model-00013-of-00013`.

| count | dtype | shape | name pattern |
|---|---|---|---|
| 1 | BF16 | [1536, 3, 14, 14] | model.vision_model.conv1.weight (patch embed, NO bias) |
| 2 | BF16 | [1536] | model.vision_model.ln_pre.{weight,bias} |
| 1 | BF16 | [2704, 1536] | model.vision_model.positional_embedding (52x52 grid) |
| 47 | BF16 | [4608, 1536] / [4608] | resblocks.N.attn.in_proj_{weight,bias} (fused qkv) |
| 47 | BF16 | [1536, 1536] / [1536] | resblocks.N.attn.out_proj.{weight,bias} |
| 47x2 | BF16 | [1536] x2 | resblocks.N.ln_1.{weight,bias} |
| 47x2 | BF16 | [1536] x2 | resblocks.N.ln_2.{weight,bias} |
| 47 | BF16 | [1536] | resblocks.N.ls_1.gamma (LayerScale, attn branch) |
| 47 | BF16 | [1536] | resblocks.N.ls_2.gamma (LayerScale, mlp branch) |
| 47 | BF16 | [8960, 1536] / [8960] | resblocks.N.mlp.c_fc.{weight,bias} |
| 47 | BF16 | [1536, 8960] / [1536] | resblocks.N.mlp.c_proj.{weight,bias} |
| 1 | BF16 | [3072, 1536, 3, 3] / [3072] | model.vision_model.vit_downsampler1.{weight,bias} |
| 1 | BF16 | [6144, 3072, 3, 3] / [6144] | model.vision_model.vit_downsampler2.{weight,bias} |
| 1 | BF16 | [4096, 6144] | model.vit_large_projector.weight (NO bias, `projector_bias: false`) |

Absent, by design (`vision_config`): no `class_embedding` (`use_cls_token: false`),
no `ln_post` (`use_ln_post: false`), no q/k norms, no gate projections. This is CLIP
lineage (LayerNorm with biases, fused in_proj, quick_gelu MLP), NOT the gemma4 RMS
program and NOT the qwen3_5 merger program. The census decides: LayerScale gammas and
2D rope make it a distinct third program.

Geometry: width 1536, heads 16, head_dim 96, mlp intermediate 8960
(`int(1536 * 5.8333...)`), depth 47, patch 14, image_size 728 (52x52 = 2704 patches),
`layer_norm_eps 1e-5`, `hidden_act quick_gelu`, `ls_init_value 0.1` (the checkpoint
gammas are trained values, not the init).

## 2. Derived tower law (per block; differences vs the shipped programs flagged)

```
x   = conv1(pixels)              # 14x14 stride-14, no bias == Linear over (c,ky,kx) 588-float rows
x  += abs_posemb                 # [2704,1536] for the 52 grid; other grids: bilinear
                                 # F.interpolate align_corners=FALSE   <- qwen3_5 is align_corners=TRUE
x   = LayerNorm(x, ln_pre)       # gemma4 has no pre-LN; qwen has none either
47 x resblock:
  res = x
  h   = LayerNorm(x, ln_1)                        # gemma: RMS
  qkv = h @ in_proj_w.T + in_proj_b               # fused, chunk(3) order q,k,v
  q,k = rope2d(q, k)                              # see below; v untouched
  a   = sdpa(q, k, v, scale=1/sqrt(96), full)     # gemma: UNSCALED; qwen: same scaling as here
  o   = a @ out_proj.T + out_proj_b
  x   = res + ls_1.gamma * o                      # LayerScale <- NEITHER shipped program has this
  res = x
  h   = LayerNorm(x, ln_2)
  f   = quick_gelu(h @ c_fc.T + b) @ c_proj.T + b # quick_gelu = x*sigmoid(1.702x)
                                                  # qwen: gelu_tanh; gemma: gated GEGLU-quick
  x   = res + ls_2.gamma * f
NO ln_post (use_ln_post false; the vendor class defaults true, the config overrides)
```

2D rope (EncoderRope2D, theta 10000, dim = head_dim = 96):
`inv_freq[i] = 10000^(-2i/48)` for i in 0..24; per token at grid (row, col) the
48-dim half-tables are `col * inv_freq` (FIRST half of head_dim) and `row * inv_freq`
(SECOND half), each `repeat_interleave(2)`d to 48 dims. Rotation is INTERLEAVED
(GPT-J style, `rotate_half` pairs (2i, 2i+1) sharing one angle), applied over the
FULL 96 dims of q and k, every head identically.

Flag the deltas: qwen3_5 rotates NeoX-paired (d, d+half) with y-first tables and
covers head_dim 72; gemma4 rotates NeoX pairs (d, d+18) inside x-first halves with
theta 100. step37 is x-first like gemma but INTERLEAVED-paired like neither, over 96
dims. Wrong pairing produces a fluent, wrong tower; this is oracle-gated per stage.

Positions in the rope are frame-local grid (row, col) of the ViT input tile. The rope
cache is built on the 52x52 max grid; a 36x36 tile (504 crop) indexes positions
`row*52+col`, which reduces to the same `(row, col)` angles. No temporal axis: no
video for this family (the reference processor has no video path).

Attention span law: each ViT input (the 728 main image, or ONE 504 crop) is its own
[B] batch entry in the reference; attention never crosses tiles or images. So one
tile = one forward = one attention segment. (The vision.rs "joint clip attention"
question is moot here: there is no multi-frame grouping at all.)

## 3. Head: downsamplers + projector (the "169 tokens" arithmetic)

After the 47 blocks the [n, 1536] grid reshapes to [1536, g, g] and runs TWO
overlapping 3x3 stride-2 pad-1 convs (these live in the vision_model but execute in
the LM wrapper's `_process_image_features`):

- vit_downsampler1: 1536 -> 3072, g 52 -> 26 (or 36 -> 18)
- vit_downsampler2: 3072 -> 6144, g 26 -> 13 (or 18 -> 9)

then row-major flatten to [g'*g', 6144] and `vit_large_projector` (6144 -> 4096, no
bias). 52-grid main image => 169 rows of trunk width 4096; 36-grid crop => 81 rows.
`config.image_token_len 169`, `patch_token_len 81`, `hidden_size 4096` all confirm.
Unlike qwen (2x2 concat + MLP merger) and gemma (3x3 avg-pool + std_bias/scale + RMS),
the step37 head is real convolution: overlapping windows, zero padding. Implemented
as host im2col + device GEMM (weights [C_out, C_in*9] with (c,ky,kx) inner order,
exactly the PyTorch conv weight flatten).

## 4. Preprocessing law (processing_step3.py, pinned rev)

Normalization: CLIP mean/std, `mean [0.48145466, 0.4578275, 0.40821073]`,
`std [0.26862954, 0.26130258, 0.27577711]`. The vendor Compose order is
ToTensor -> Normalize -> Resize (normalize BEFORE resize); per-channel affine
commutes with the linear resample up to fp rounding, memra resizes first then
normalizes, and the fixed-pixel oracle plus e2e gates arbitrate.

Resize: torchvision `Resize((728,728))` for the main image, `(504,504)` for crops,
`InterpolationMode.BILINEAR` + `antialias=True` (the Step3VLProcessor passes
"bilinear"; the "bicubic" default is dead code at this rev). Every ViT input is a
SQUARE resize, aspect ratio destroyed deliberately for the main view; the crops
preserve local geometry.

Tiling (ImagePatcher), in order:
1. pad-to-square (paste at 0,0, black fill) ONLY if `min(w,h) < 32` and aspect > 4.
2. cap the long side at MAX_IMAGE_SIZE 3024 (PIL bilinear resize, aspect kept).
3. `determine_window_size(long, short)`: long <= 728 => `short if long/short > 1.5
   else 0`; long > 728 => `min(short, 504) if long/short > 4 else 504`.
4. window 0 => NO crops (the common small/square case: exactly one 169-token unit).
5. else snap w and h to whole windows (`get_image_size_for_crop`: ratios with a 0.2
   fractional-overflow rule), PIL-bilinear resize to that snapped size, slide a
   non-overlapping window_size grid (x_num columns per row), crop each tile.
6. newline mask: after each full row of tiles EXCEPT a trailing row-final tile
   (the reference pops the last newline).

## 5. Template + token-expansion contract

`chat_template.jinja` (pinned rev) renders an image content part as the literal
`<im_patch>` at its position inside the message; adjacent TEXT parts are joined with
one space, and an image part resets that separator (text directly after an image gets
no space). The PROCESSOR then replaces each single `<im_patch>` occurrence with the
full per-image expansion, crops FIRST, then the main view:

```
for each crop tile i:  <patch_start> + 81 x <im_patch> + <patch_end>
                       (+ <patch_newline> when newline_mask[i])
then:                  <im_start> + 169 x <im_patch> + <im_end>
```

Token ids: `<im_start>` 128000, `<im_patch>` 128001, `<im_end>` 128002,
`<patch_start>` 128003, `<patch_newline>` 128004, `<patch_end>` 128005.

Embedding merge (modeling_step3p7.py `get_input_embeddings` +
`merge_multimodal_embeddings`): image feature rows replace the embedding rows at
`input_ids == 128001` positions IN ORDER; delimiter tokens (128000/2/3/4/5) keep
their ordinary text embeddings. Rows per image are concatenated crops-then-main,
matching the token layout. Image spans are CAUSAL: the text model builds standard
`create_causal_mask` / `create_sliding_window_causal_mask` with no image-span
special-casing (unlike gemma4's bidirectional islands, like qwen3_5). Embeddings
enter the trunk UNSCALED. Images per request: the template supports any number
(one `<im_patch>` per image part); memra keeps its `VISION_MAX_IMAGES` (8) envelope.

Per-image prompt-token count: `169 + 2 + n_tiles*(81 + 2) + n_newlines`, i.e. 171
for the no-tiling case. All of it bills as ordinary prompt tokens.

## 6. Implementation shape (decided)

Sibling module `crates/memra-engine/src/vision_step.rs` (the vision_gemma.rs
pattern), NOT a weight-map onto vision.rs: LayerScale, quick_gelu, interleaved rope,
align_corners=false posemb, conv downsampler head and the tiling preprocessor are
each program-level differences; only `EmbedOverlay` (family-agnostic) is shared.

- Loading: IN-CHECKPOINT. `StModel::open(model_dir)` routes names through
  `model.safetensors.index.json`; the tower reads its 667 BF16 tensors from the same
  artifact the trunk serves from. No side file, no MEMRA_VISION_DIR twin. New env
  seam `MEMRA_STEP_VISION=1` (default OFF, deliberate: the flag row lands with the
  receipts) selects the step arm in content parsing exactly as MEMRA_GEMMA_VISION
  does for gemma; the worker loads the tower at spawn from the model dir when set.
- HTTP: a step arm in `content_to_text_vision` that (a) plans each image
  (header dims -> decode budget -> tiling plan) pre-decode, (b) emits the exact
  expansion string of section 5, (c) reproduces the template's text-separator law
  for the message it renders into. Data URIs only, same SSRF posture.
- Admission: `step_vision_spans` walks `<im_patch>` (128001) runs and consumes them
  against each unit's expected run layout ([81] x n_tiles then [169]); any mismatch
  400s. Existing family-agnostic session laws apply unchanged via `vision_req`:
  vision sessions are NOT spec-eligible (admit gate + spec-round backstop), bypass
  every token-keyed reuse tier, never seed the prefix cache, never replay after OOM,
  prime alone. step37 serves with MTP spec ON for text; an image request drops that
  session to the plain path by the existing law, and the gates prove both sides.
- Tower forward: correctness-first v1 (f32 GEMMs via Engine::linear, sdpa_naive
  scale 1/sqrt(96) non-causal, host rope/LayerScale/quick_gelu/im2col), staged
  MEMRA_VISION_DEBUG dumps at pre_blocks / blk0 / post_blocks / downsampled /
  projected. 2704-token segments clear the sdpa_naive 12288 ceiling.
- TF32 law: parity runs pin NVIDIA_TF32_OVERRIDE=0 end-to-end (gemma lane finding;
  step37's attention is scaled, so the amplification is expected milder, but the
  doctrine does not ship "probably benign": measured both ways in the oracle).

## 7. Oracle + gates (all banked in this lane)

1. Staged parity: `step_vision_oracle` (memra tower, stage dumps) vs an INDEPENDENT
   NumPy reference (`step_vision_ref.py`, same safetensors weights, the law of
   section 2 reimplemented) AND vs the vendor torch code (`vision_encoder.py` +
   downsamplers + projector, run offline). Per-token cosine per stage; bar: min-cos
   >= 0.9997 on the projected rows (qwen bar 0.9997, ornith 0.99983, gemma landed
   1.000000; state the measured number, whatever it is).
2. Decisive can't-hallucinate probes through the REAL `/v1/chat/completions` with
   `image_url` data-URI parts, vendor-default sampling: freshly generated images
   whose content is unguessable from text (random strings / digits / colors),
   answers must name the content. Both the no-tiling shape (<=728 square) and a
   tiling shape (e.g. 1600x900, 4 tiles + main) probe the layout law.
3. Multi-image request; usage.prompt_tokens accounts the full expansion per image.
4. Spec interaction: text request on the same binary keeps MTP spec engaged
   (usage/receipts show K>0); the image request serves plain by the admit law and
   the spec-round backstop never fires. Mid-conversation image after text turns
   covered.
5. Text-only regression: greedy byte-identity of a text request through the
   vision-enabled binary vs the same binary with the seam off (MEMRA_STEP_VISION
   unset). Zero ILLEGAL / #87 / panics across the battery.

## 8. RESULTS (2026-08-30, all gates banked under receipts/)

Environment: dev box, 2x RTX PRO 6000 Blackwell; artifact
`/data/models/step37-flash-nvfp4` verified against the pin by shard sha256
(`f02b63dc...` / `2d81bee6...` == the HF LFS oids at rev 4275532f); memra commit
04b3c771e; parity runs NVIDIA_TF32_OVERRIDE=0. Invocations and full outputs:
`receipts/receipts-parity.txt`, `receipts/receipts-e2e.txt`,
`receipts/receipts-byte-identity.txt`, `receipts/box-receipts.txt`; drivers
`e2e_gates.py`, `receipts/byte_probe.py`, `receipts/boot-server.sh`.

| gate | verdict | numbers |
|---|---|---|
| staged parity, 52-grid (728 main view), memra tower vs independent NumPy ref | PASS | projected min_cos 1.000000 (pre_blocks 1.000000, blk0 1.000000, post_blocks 0.999999, downsampled 1.000000); bar 0.9997 (qwen 0.9997 / ornith 0.99983 / gemma 1.000000 precedent) |
| staged parity, 36-grid (504 crop tile) vs NumPy ref | PASS | every stage min_cos 1.000000 |
| staged parity, both grids, vs the checkpoint's OWN torch code (vision_encoder.py + downsamplers + projector) | PASS | projected min_cos 1.000000 both grids (post_blocks 0.999998 / 0.999999) |
| e2e can't-hallucinate, single image (640 square, no tiling), real /v1/chat/completions, image_url data URI, vendor-default sampling | PASS | fresh randomized shape/color named exactly, 3 independent sweeps |
| e2e can't-hallucinate under TILING (1600x900 -> 6 crops + main) | PASS | content named exactly; layout law live |
| e2e multi-image (2 images, one request) | PASS | both contents named, numbered correctly |
| e2e image mid-conversation (after text turns) | PASS | content named exactly |
| usage.prompt_tokens accounting | PASS | exactly +171 per plain image, +670 for the 1600x900 tiling, multi = 2x171 + text |
| faked pad tokens in user text | PASS | 400: "prompt carries 2 step image pad run(s) but the request's units need 1" |
| http(s) image URL / video part | PASS | 400 both (SSRF off; no video for this family) |
| spec x vision (MTP spec-configured binary, MEMRA_SPEC_K=3 + MEMRA_SERVE_SPEC=1) | PASS | every vision request admits `[spec-k] K=0 source=eligibility-fallback` (plain path — an image span can never meet a draft walk); text requests on the same binary keep `K=3 source=operator-pin` with live spec-acc bursts (cum accept ~0.80); the spec-round backstop never fired |
| text-only byte identity (greedy, vision-enabled boot vs seam-off boot) | PASS | content + reasoning_content + finish_reason + usage token counts byte-identical (311 completion tokens); only usage.elapsed_s (timing) differs |
| hygiene, both boots | PASS | ILLEGAL=0, #87=0, panic=0 |

Support state reached: the step37 IMAGE surface is implemented, parity-gated against
two independent offline references, and serve-gated end-to-end on the pinned NVFP4
artifact. The flag ships default OFF; arming a customer deployment is a serving
decision that belongs to the deployment's own rollout lane (staging, sampled
post-deploy battery per the serving laws) — not made here.

Serving recipe (what a deploy needs on top of the model's existing text recipe):
`MEMRA_STEP_VISION_DIR=<model dir>` on the same artifact the trunk serves
(tower is in-checkpoint; ~8 GB extra f32-resident VRAM; MEMRA_STEP_VISION=0 is the
kill switch). No other flag changes; vision requests self-select the plain decode
path under the existing admission law.

Open risks / explicit non-claims:
- No perf claim: the tower is v1 correctness-first (f32 GEMMs, host rope/gelu/im2col);
  the 728 main view forwards in ~9.3s, a 504 tile in ~3.8s on this card class. A
  latency-focused rewrite is a separate, receipts-gated lane.
- Resize kernels: memra uses the image crate's Triangle (bilinear) where the vendor
  stack mixes PIL and torchvision bilinear; the fixed-pixel oracle gates the tower
  independently of resampling, and the e2e probes bound the end-to-end effect. A
  bit-exact preprocessing comparison against Step3VLProcessor was not run.
- The vendor template's space-separator law for adjacent text parts applies only when
  the step seam is armed; text-only deployments keep the pre-lane concat bytes.
- Prefix-cache interaction: vision requests bypass every token-keyed reuse tier
  (family-agnostic law); no cache-split-inside-span hazard exists on this family's
  causal spans, but no vision-specific cache tier was built either.

Audit note: the box receipts cite memra commit 04b3c771e — the pre-amend twin of the
landed ef8357e7a. `git diff 04b3c771e ef8357e7a` touches only
research/step37-vision-20260830/e2e_gates.py and research/tune-data/perf-ci.jsonl
(zero crates/ changes), so the gated binary's source tree is byte-identical to the
landed engine code.
