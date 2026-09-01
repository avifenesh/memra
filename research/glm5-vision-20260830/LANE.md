# lane/glm5-vision: GLM-5.3-Flash (glm5_next) vision bring-up

Owner directive: "if glm has vision we need to add support for it as well."
Upstream is image-text-to-text; the config carries vision_config (24-layer ViT) and
image/video token ids; the published NVFP4 mint carries all 347 `model.visual.*`
tensors. Engine bring-up only, no artifact work. Video: census only, OUT OF SCOPE
for this lane (`video_url` refuses at intake; the video arms stay in the tensor
census and the template).

## Pins

- model: zai-org/GLM-5.3-Flash @ 04c4e9e95c5da8862dced7e5056455116f83a7e0
- tower shard: model-00062-of-00062.safetensors,
  sha256 d3087816db95f962a3a74c057b8398e1492458ecd99608f706a400f112a825c6
  (347 `model.visual.*` tensors, all BF16, + 4 trunk stragglers)
- upstream code: transformers 5.16.1 (first release carrying glm5_next); the vision
  classes were diffed byte-identical to transformers main at fetch time. torch
  2.13.0+cpu for the fixture capture (venv ~/venvs/glm5v-ref).
- processor constants: processor_config.json (CLIP mean/std, patch 14, temporal 2,
  merge 2, budget 16..8000 merged tokens). The repo ships NO preprocessor_config.json;
  the image-processor defaults live in code.

## Tower semantics (upstream citation per element)

All from transformers 5.16.1 `models/glm5_next/modeling_glm5_next.py` +
`image_processing_glm5_next.py` + `vision_utils.py`:

- ViT 24 blocks, hidden 1024, 16 heads (head_dim 64), ffn 4096, RMS norms eps 1e-5
  (weight-only). Block: `x += attn(rms(norm1, x)); x += mlp(rms(norm2, x))`.
- Attention: FUSED qkv `[3072, 1024]` WITH bias; per-head q/k RMS norms (width 64)
  applied BEFORE the rope (`Glm5NextVisionAttention.forward`); scaled `1/sqrt(64)`;
  non-causal; one segment per image (`get_vision_cu_seqlens`, merge_temporal=False).
- Positions: ROPE-ONLY. `GlmOcrVisionModel.__init__` DELETES `self.embeddings` and
  `post_conv_layernorm` — no learned position table, matching the hand-census.
  2D rope theta 10000 (the `Glm5NextVisionRotaryEmbedding.__init__` DEFAULT — not a
  config key): dim 32, `inv_freq[i] = theta^(-2i/32)` (16 values); position ids
  `(h, w)` block-major over 2x2 merge blocks (`get_vision_position_ids`:
  `reshape(h/m, m, w/m, m).transpose(1, 2)`); angle vector = `[h*f | w*f]` doubled
  (`cat((rotary, rotary), -1)`), rotate_half => NeoX pairs `(d, d+32)`.
- Block MLP: gate/up/down WITH biases; clamped SwiGLU
  `silu(min(gate, 10)) * clamp(up, -10, 10)` (swiglu_limit 10.0, `Glm5NextVisionMLP`).
- Head: post-encoder RMS norm; conv 2x2/stride-2 downsample `[4096, 1024, 2, 2]` +
  bias over each merge block (token groups of 4 in (in_row, in_col) order, the view/
  permute in `Glm5NextVisionModel.forward`); merger `Glm5NextVisionPatchMerger`:
  proj 4096->4096 (no bias) -> `nn.LayerNorm(4096)` (weight+bias, torch default eps
  1e-5) -> `nn.GELU()` (EXACT erf, not tanh) -> gate/up 4096->10240 (no bias, same
  clamps) -> down -> 4096 == trunk n_embd. Rows splice over `<|image|>` embeddings.
- Splice ids (config.json): begin 154830, image 154854, end 154831 (video
  154832/154855/154833). The template emits ONE `<|image|>` per image part
  (`emit_image()`); `Glm5NextProcessor.replace_image_token` expands it to
  grid/4 copies. memra renders the expanded run inline at intake (byte-equality
  proven by surface fixture 23, assert inside gen_surface_fixtures.py).
- Pixels: rescale 1/255 -> CLIP mean/std; smart_resize factor 28, temporal 2, budget
  16..8000 merged tokens, aspect-preserving content + ZERO-PAD canvas (not stretch);
  patchify rows `(c, t=2 duplicated frames, 14, 14)` in block-major token order.
  memra's port of smart_resize verified identical on 12 shapes incl. budget-capped,
  min-upscaled, extreme-aspect (this doc's generator + resize_check).

## Which precedent is reused per component (gemma vs q38 vs upstream)

The owner asked for the three-way comparison. Verdict: the q38 tower (vision.rs) is
the closer ENGINE SKELETON; the gemma tower (vision_gemma.rs) contributed two shapes;
wherever either precedent diverges from GLM's math, upstream wins (no-generic-support).

| component | glm5 semantics | q38 precedent | gemma precedent | reused |
|---|---|---|---|---|
| patch embed | conv3d->(c,t,ph,pw) linear, biased | same pattern (conv->linear flatten, biased) | conv2d, bias-less | q38 mechanics |
| positions | rope-only, block-major (h,w) | learned 48x48 table + 2D rope | factored additive tables + 2D rope theta 100 | NEITHER table; upstream rope only (theta 10000, pairs (d,d+32)) |
| attention | fused qkv+bias, per-head q/k RMS, scaled | fused qkv+bias, NO qk norms, scaled | split q/k/v unbiased, q/k/v RMS norms, UNSCALED | q38 skeleton (fused qkv split, sdpa_naive); gemma's per-head-RMS host helper shape; upstream decides scaled + no v-norm |
| block MLP | biased clamped SwiGLU (silu) | 2-linear gelu_tanh | geglu-quick, unbiased | upstream math, house host-activation pattern |
| norms | RMS eps 1e-5 | LayerNorm + bias | RMS eps 1e-6 + sandwich post-norms | gemma's RMS (engine `rms_norm`), NO sandwich |
| downsample + merger | conv 2x2 channel-major gather -> gated clamped merger (LayerNorm + exact GELU) | token-major 2x2 concat -> 2-linear gelu_tanh merger | 3x3 avg-pool -> standardize -> single proj | NEITHER; upstream math (channel-major conv layout differs from q38's token-major concat) |
| splice | placeholder rows replaced BEFORE hc stream expansion; grid-derived counts | EmbedOverlay + spans + overlay prime | soft-token runs, masked-prefill arm | q38 mechanics verbatim (shared `EmbedOverlay`); new hyper-prime splice point |
| preprocessing | smart_resize + PAD canvas, CLIP norm, bicubic | smart_resize STRETCH, CLIP norm | 48-grid smart resize, no normalize, 2x-1 | vendor port; house plan-don't-decode admission (hermes decode-bomb law) from both |

## Defect found and fixed (pre-existing, live on the bring-up branch)

The generic `vision_config` parse read glm5's keys through gemma-shaped names
(`num_hidden_layers` vs `depth`, `num_attention_heads` vs `num_heads`,
`hidden_activation` vs `hidden_act`) and silently attached a WRONG vision plan to
every glm5_next ModelPlan: 16 layers, 12 heads, gelu_pytorch_tanh, rope theta 100.
Nothing had consumed it yet. Fixed by keying the parse on
`vision_config.model_type == "glm5_next_vision"` (required-field refusal style) and
splitting the plan type into `VisionPlan::{Factored, Glm5Fused}` — two semantic
programs, never approximated by one another.

## Gates (cert lines with invocations)

Reference-vs-upstream pin (CPU, 391.7s release):
```
MEMRA_GLM5_VISION_SHARD=~/models/glm53-vision/model-00062-of-00062.safetensors \
cargo test --release -p memra-reference --test glm5_vision_upstream -- --ignored --nocapture
det112:     post_blocks 5.81e-5 | downsample 2.46e-5 | merger 1.14e-6   PASS
det448x224: post_blocks 7.24e-2 | downsample 2.53e-2 | merger 6.88e-4   PASS
text448:    post_blocks 9.28e-1 | downsample 2.30e-1 | merger 4.40e-3   PASS (merger-only gate)
```

Engine GPU gate (rig 5090, flock /tmp/memra-5090.lock, NVIDIA_TF32_OVERRIDE=0,
correctness-only, 2.9s):
```
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
MEMRA_GLM5_VISION_SHARD=~/models/glm53-vision/model-00062-of-00062.safetensors \
cargo test --release -p memra-engine --test glm5_vision_gpu -- --ignored --nocapture
det112:     patch 8.58e-6 | blk0 1.14e-5 | post 4.65e-5 | down 1.65e-5 | merger 5.14e-7  PASS
det448x224: merger 6.85e-4  PASS   text448: merger 4.45e-3  PASS
```

MINT-BYTES twin (2026-08-30, rented 4x RTX PRO 6000 Blackwell Server box, card 2 via
CUDA_VISIBLE_DEVICES=2,3, TF32 off; same test, MEMRA_GLM5_VISION_SHARD pointed at the
published NVFP4 mint DIRECTORY — StModel resolves model.visual.* through the mint's
own index): the deployment artifact's visual tensors pin to the same upstream truth.
```
tower loaded from /root/models/glm53-nvfp4 (24 blocks, out_width 4096, f32-resident)
det112:     patch 8.58e-6 | blk0 1.34e-5 | post 4.54e-5 | down 1.67e-5 | merger 6.35e-7  PASS
det448x224: merger 6.85e-4  PASS   text448: merger 4.22e-3  PASS   (4.3s)
```
(box receipt row in that box's /root/BOX-QUEUE.md; build dir removed, nothing left
running.)

Red-proofs (one sabotage per run, reverted after each; det112, first stage caught):
```
patch-embed transposed      patch 1.08e1    RED
rope h/w axes swapped       blk0  1.89e-1   RED
raster-order positions      blk0  1.29e-1   RED
k_norm dropped              blk0  2.20e0    RED
uniform +1 position shift   GREEN BY MATH — rope attention is translation-invariant;
                            a uniform shift is not a defect class for a rope-only
                            tower (it would be for a learned-table tower). The real
                            position classes are the axis swap and the order mixup,
                            both red above.
```

Tiny parity (reference determinism + multimodal splice oracle):
```
cargo run -p memra-cli -- model verify tiny --against glm5_next --out <dir>
  -> reference-vision-oracle.tsv + reference-multimodal-oracle.tsv (new for glm5_next)
```

Template image arm: surface fixture `23-image-message-16tok`
(glm53-flash-bringup-20260827/surface-fixtures) — generation-time assert proves
memra's inline splice == template typed-part arm + processor expansion;
`glm5_fixtures_match_the_vendor_jinja` passes. Closes the surface lane's
"image arms not expressible yet" note for IMAGES (video/audio still not expressible).

Preprocess pin: `glm5_vision_decode_is_deferred_and_grid_pinned` (memra-server) —
plan/decode grid agreement + desync refusal. Pixel pipeline vs torchvision: 1-2 ulp
(2.4e-7) on identity-resize images; resize kernels (CatmullRom vs torchvision
bicubic) may differ by a hair, which is why parity fixtures are identity-resize and
the tower oracle feeds FIXED pixels (gemma-vision lane law).

## Fixtures (fixtures/, generator gen_upstream_fixture.py)

| fixture | grid | tokens | role | committed |
|---|---|---|---|---|
| det112 | 8x8 | 16 | tight pin (upstream fully deterministic on it: fresh-vs-banked 0.0) | yes (1.9 MB) |
| det448x224 | 32x16 | 128 | large + non-square rope pin; upstream self-delta post 7.2e-2 / merger 6.8e-4 | meta+ignore (regenerable, 15 MB) |
| text448 | 32x32 | 256 | can't-hallucinate probe image, ground truth "ZK5465 QV4655 XR0818" | meta+png (regenerable, 29 MB) |

FINDING (measurement law material): torch CPU f32 differs from ITSELF fresh-vs-banked
on >=512-token grids (kernel/reduction-order variation, chaotic growth through 24
blocks) — post_blocks 7.2e-2 on det448x224 and ~1.0 on text448, whose mostly-white
canvas makes ~1k near-identical tokens and softmax ties amplify reduction noise. The
MERGER output (what the trunk consumes) stays 1e-3-class. Bands are therefore
per-fixture with the class stated; a stage-level 1e-4 band on a big grid would be
gating torch's thread scheduler, not our math. bf16-activation rows (artifact dtype
class) are banked in each meta.json: det112 4.4e-3, det448x224 3.0e-2, text448
3.5e-2 — input to a future engine-dtype decision (v1 is f32-resident, house posture).

## Serving intake (DEFAULT ON since 2026-08-30 — see the flip record below)

Default ON: a glm5_next DIRECTORY checkpoint whose own safetensors index carries
`model.visual.*` loads its tower from itself at boot (probe tensor
`model.visual.patch_embed.proj.weight`; the mint ships all 347 tensors — MINT-BYTES
twin above). Absent tensors = text-only artifact, vision off, no flag needed.
`MEMRA_GLM5_VISION=0` is the rollback seam; `MEMRA_GLM5_VISION_DIR` overrides the
tower source (split artifacts). The HTTP intake keys on the WORKER's actual tower
decision (`GLM5_VISION_SERVING`), never on a raw env read. Path: image_url data-URI ->
plan-don't-decode admission -> `<|begin_of_image|>` + n x `<|image|>` +
`<|end_of_image|>` inline -> post-budget decode -> placeholder-run span alignment
(fail-loud) -> tower forward at first prefill tick -> EmbedOverlay rows replace
placeholder embeddings BEFORE hyper stream expansion (`EmbedOverlay::splice_into`;
the exact splice point the reference uses — execute_multimodal replaces rows before
hc_expand). Refusals (Err-not-assert): vision off / tower missing / trunk-width
mismatch / video_url / vision special ids in ANY message text (the intake guard
below) / grid desync. Serving cap 3072 merged tokens/image (v1 sdpa ceiling 12288
patches; vendor budget arm downsizes into the cap — fidelity knob, not a refusal;
vendor default 8000).

ppN + overlay (blocker 2 of the flip, CLOSED 2026-08-30): the splice is genuinely
STAGE-0-ONLY — `prime_cache_hyper_ppn` embeds tokens on stage 0's engine and every
later stage receives only the already-expanded `[t, streams, hidden]` boundary
payload, so the overlay splices at stage-0 embedding intake (both the streams-on and
the MEMRA_PP_STREAMS=0 arms) and no other stage carries overlay arithmetic. Overlay
rows are primary-resident (built by the worker's tower forward); a placement that
moves stage 0 OFF the primary device refuses loudly rather than peer-reading. The
serial PP-2 pipelined prime (non-hyper models) and gemma4 E4B still refuse. Gate:
glm5-hyper-ppn-gate arm 5 (OVERLAY TWIN) — truth by SUBSTITUTION (overlay rows =
embed() rows of known substitute tokens, so the overlaid walk must be BIT-IDENTICAL
to prime_cache over the substituted prompt): door-OFF serial, door-ON monolithic,
door-ON two-call windowed (the prefill-tick shape), each + decode continuation, all
BIT-IDENTICAL across n2/n3/n4 stages, splits 1|3|1,3|even, MEMRA_PP_STREAMS=0, and
P=16; red arm MEMRA_GLM5V_GATE_RED=span-shift bites 9/9 on all three overlay arms
(exit 1). Logs: ppn-overlay-gate/ in this lane dir.

## What the box window must cover (end-to-end, real artifact)

1. Boot the native server on the mint with MEMRA_GLM5_VISION=1 +
   MEMRA_GLM5_VISION_DIR=<mint dir>; startup log shows the tower loaded (f32,
   ~2.3 GB) next to the trunk.
2. The can't-hallucinate probe: POST /v1/chat/completions with text448.png as a
   data-URI image_url + "Transcribe the text in this image exactly."; REQUIRE the
   exact string "ZK5465 QV4655 XR0818" in the reply (unguessable from the prompt;
   fluent hallucination cannot pass). Cold session, greedy for the byte-pin arm AND
   the vendor-default sampled shape (serving law: no sampling params, spec-engagement
   receipt from the server log).
3. Splice-integrity twins: (a) two different images in one prompt, ask which
   contains which string (order pin); (b) det112 + "describe the colors" (red square
   on gradient — content pin); (c) refusal battery: image with flag off,
   oversized data URI, literal <|image|> in text, video_url.
4. Multi-turn: image in turn 1, text-only turn 2 (session reuse is bypassed for
   vision requests — verify the bypass holds and turn 2 still answers about the
   image from context).
5. Overlay-vs-text-prime discipline: server log receipt that the prefill ran the
   overlay prime (never decode_step over placeholder tokens), and the ppN/PP
   position per the note above.
6. Receipts: request/response JSON, server log, binary + bundle hashes, banked in
   this lane dir; only then may FLAGS default or product surfaces move (owner call).

## FLIP RECORD — default ON (lane/glm5-vision-default-on, 2026-08-30)

Owner order: "image should be default on, queue it" (2026-08-30). Per the new-flags
law the flip ships with its blockers closed, receipts attached, and the FLAGS.md row
rewritten in the same lane:

**Blocker 1 — the injection finding (vision-cell arms 20/22), CLOSED.** A TEXT-ONLY
request containing literal `<|image|>` (and the begin/end forms; ids
154830/154854/154831) tokenized to the special id (`st_partition` splits special
literals out of raw text before BPE) and was SERVED — the raw placeholder embedding
reached the trunk, fluent and invisible. The run/unit alignment only ever ran for
requests carrying image parts. Fix at intake: the codebase's existing hygiene is
reject-by-name (all three families' span alignment already refuses "literal <|image|>
tokens in message text" — when units exist), so the fix EXTENDS that refusal to every
prompt source: `vision_special_id_guard` at the `prepare_request` render/tokenize
waist counts each guarded special id in the ENCODED prompt (ids, never prose) against
the count this request's own image parts render — zero for text-only; begin/end
exactly one per glm5 unit; video ids always zero. Guard ids resolve from the model's
own tokenizer (`id_of` + the new `token_is_special`), so ordinary vocab entries that
merely look like markers are never policed. Gates (worker.rs
vision_special_guard_tests, all on token ids): red text-only refusal for all three
id forms; red smuggled-extra beside a real run; red lone begin with the run intact;
green exact render budget; hermetic fixture-tokenizer test proving raw text ->
special id -> refusal AND that the rendered run still admits. The 23 surface
fixtures stay byte-identical (`glm5_fixtures_match_the_vendor_jinja` PASS) — the
guard reads the token stream, it never rewrites the render. NOTE the deliberate
trade, stated: an honest text-only prompt that legitimately QUOTES `<|image|>` now
400s with a named error instead of serving a desynced special token; the refusal
names the rule so the client can escape by rephrasing.

**Blocker 2 — the ppN overlay arm, CLOSED.** Splice seam truth verified in code and
gated; see "ppN + overlay" above. The serving-shape note from the OFF era (single
engine / MEMRA_PRIME_PP=0) is obsolete: the 3-stage ppN prime now carries the
overlay natively.

**The flip.** Default ON keyed on artifact truth: glm5_next dir checkpoint +
`model.visual.*` present (probe `glm5_visual_tensors_present`, gated present-case on
the real shard inside the GPU gate + hermetic absent-case test) -> tower auto-loads
from the checkpoint itself; absent -> no vision, no flag. `MEMRA_GLM5_VISION=0` =
rollback seam. VRAM cost MEASURED at load, printed on the boot line: **2.12 GiB
device delta** (5090 gate run 2026-08-30 — the ~2.3 GB figure elsewhere in this doc
is the parameter-count estimate). FLAGS.md row rewritten same lane. Flip-day gate
re-runs, this branch: reference pin det112 post 5.81e-5 | down 2.46e-5 | merger
1.14e-6 PASS; engine GPU gate det112 patch 8.58e-6 | blk0 1.14e-5 | post 4.65e-5 |
down 1.65e-5 | merger 5.14e-7 PASS (banked-fixture numbers reproduced exactly);
surface fixtures byte-identical; server unit suite 472 PASS + 4 new guard tests.

**What this flip does NOT claim.** No serving-shape probe receipts: the
can't-hallucinate cell (greedy + vendor-default sampled, spec-engagement receipt)
re-runs on the 3-card box under its own queued window — that receipt is the
remaining precondition for exposing vision to CUSTOMERS (product surfaces move only
after it, per the box-cell section above and the issues-close-on-serving law). The
card3 1-card probe receipts (PASS, exact codes, both decode shapes) are banked at
research/glm53-flash-bringup-20260827/acceptance-probe-20260830/vision-cell/ on
lane/glm5-card3-acceptance-probe.

## Out of scope, stated

- Video serving (census + template arms exist; `video_url` refuses; needs its own
  lane: per-frame-pair segments, timestamp text, budget law).
- BF16 tower residency, cuBLASLt bias epilogues, device-side rope/activations
  (tuning lane; v1 is the correctness posture).
- ppN hyper prime overlay arm (refuses loudly; needed only if the serving shape
  requires PP during vision prefill — see the box-cell note).
