# Ornith-1.5-35B-A3B — ST NVFP4 bring-up plan (2026-08-19)

Target: checkpoint-faithful safetensors NVFP4 serving of `ornith-ai/Ornith-1.5-35B-A3B` on one
RTX PRO 6000 Blackwell 96GB, MTP spec decode with FR-Spec masked head, exactness-gated end to
end. GGUF is not the deliverable (the family's GGUF path already exists via Qwen3.6-35B-A3B);
the official BF16 safetensors at pinned rev `fbb995a79eedd569a5edc5f2af9644c0fa1124fc` is the
single semantic source. Nothing below is a support claim until the gates are green.

## 1. Census (byte-read from the artifact, 2026-08-19)

`config.json` + `model.safetensors.index.json` (banked in this dir): 1811 tensors, 71.90 GB
BF16, arch `qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`, transformers 5.8.1).

| block | shape facts |
|---|---|
| trunk | 40 layers; GDN hybrid 3:1 — 30 `linear_attn` (A_log, conv k=4, dt_bias, in_proj a/b/qkv/z, norm, out_proj) + 10 `self_attn` (16 heads, 2 KV heads, head_dim 256, q/k_norm, partial rotary 0.25, mrope interleaved [11,11,10], θ=1e7) |
| MoE | 256 experts top-8 per layer, moe_intermediate 512, shared expert 512 + shared_expert_gate; **experts FUSED 3D** (`mlp.experts.{gate_up,down}_proj`) ≈ 32.2B params ≈ 90% of bytes |
| MTP | 1 layer, shares embed/lm_head (`mtp_use_dedicated_embeddings: false`); full MoE block with **UNFUSED per-expert tensors** (256 × gate/up/down) + one full-attn block + fc + pre_fc norms |
| vision | 27-block ViT (1152 hidden, patch 16, merger → 2048), image+video tokens; bias tensors present (trunk has none) |
| embed/head | vocab 248,320, untied; ~508M params each |
| ctx | 262,144 native; YaRN factor-4 → ~1M (vendor-validated) |

Loader notes (verified against `crates/memra-gguf/src/hf_mapping.rs`): fused 3D experts are
sliced source-side (`:774`), MTP names map (`:570`), `Arch::Qwen35Moe` registered
(`config.rs:12`). **Check**: tensor names carry the multimodal `model.language_model.` prefix —
confirm the wrapper-strip path (Qwen3.8-27B precedent) covers the MoE + linear_attn names.

Template: `chat_template.jinja` ships in-repo (mint-trap does not apply, but the ST artifact we
publish must carry it). Tools branch present — `<function=...><parameter=...>` XML dialect
(vLLM `qwen3_xml`); reasoning `<think>` open-by-default with `enable_thinking=false` escape.
Both must hit the template byte-parity + tools round-trip gates.

## 1a. Pace pivot (owner, 2026-08-19 evening)

The GGUF NVFP4-MTP artifact + card ship FIRST — first-mover on the NVFP4 GGUF slot is the win,
and at the check hour (2026-08-19 ~20:00Z) that slot was still empty: official GGUF is
BF16/Q4–Q8 only, one community imatrix K-quant mint, an MLX mint wave (some with split MTP
weights), two empty MXFP8 placeholder repos, **no NVFP4 GGUF anywhere**. The family is
already on the engine's GGUF path (`qwen35moe`; official BF16 GGUF byte-checked: block_count
41 = 40 trunk + 1 NextN, `nextn_predict_layers=1`, glue tensors present), so this is
measure → touch-ups → gate → release. **The masked head and the after-mask MTP quant are
in scope for this release, not follow-ups** (owner): own-gen ranks → trim → hqmtp requant
(`tools/make-trimmed-draft.sh`: head NVFP4 + Q4_K_M block), ranks shipped as `.txt` for the
ST load-time trim path too. The checkpoint-faithful ST end-to-end program (§2) continues
behind it as the quality deliverable. NVFP4 ftype lives on the `nvfp4-imatrix-scale-search`
branch of the avifenesh/llama.cpp fork, not upstream.

## 2. Program stages (ST end-to-end — continues after the pace release)

### 2a. ST census update (2026-08-20): the official modelopt NVFP4 is the serving target

`ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (pinned rev `9660379a2f2c429c465eeed2f3a0f2433fc4381e`,
modelopt 0.45.0, MIXED_PRECISION): experts + shared_expert + lm_head = W4A16_NVFP4 (per-expert
UNFUSED tensors, `weight_scale` per-16 + `weight_scale_2` per-tensor); ALL attention + GDN
projections = FP8; embed / norms / router / vision / **the entire 785-tensor MTP head = BF16**.
memra's ST chain already consumes every class in code: `nvfp4_repack.rs` (modelopt→internal
NVFP4, verbatim code+scale copy), `source.rs` weight_scale handling, `model.rs` per-expert
`weight_scale_2` post-matmul macros, unfused `mlp.experts.{e}.{proj}` name mapping
(`hf_mapping.rs`). So stage 3's own-mint NVFP4 is NOT needed for the ST deliverable:
checkpoint-faithful support = serve the OFFICIAL NVFP4 artifact, gated. Our own-mint stays a
fallback only if the official artifact fails a gate for artifact (not engine) reasons.
Convergence: the MTP head being BF16 in the official NVFP4 means the head-training lane's
patched `mtp.*` (../ornith15-mtp-train-20260820/) drops into BOTH deliverables — the GGUF
remint and an ST head-patch — one training run, two artifacts. Staged to box3
`~/models/ornith15/nvfp4-official/`.

1. **Oracle before kernels.** Official BF16 through transformers ==5.8.1 as the
   can't-hallucinate oracle (CPU acceptable for short probes; box3 GPU for longer). Fixed probe
   set: argmax transcripts, tokenizer round-trips, template renders (with/without tools, with
   `enable_thinking` both ways), one image probe for the vision path. Explicit ruling required
   on the official GGUF as a secondary oracle (expect NO — quantized; imatrix on the 397B).
   vLLM ≥0.19.1 allowed as research control only.
2. **ST BF16 load + argmax gate.** `run-safetensors` path: load pinned BF16, `run-gen` argmax
   MATCH vs oracle on the probe set. This proves census + loader before any quantization.
3. **NVFP4 mint (the deliverable artifact).** Quantize from pinned BF16 only. Tier decision
   re-derived for A3B (do NOT copy the gemma attn-out-of-NVFP4 ruling): experts → NVFP4
   (~17.1 GB); attention/GDN/norms/router/shared-expert/vision/MTP-attn stay high precision
   (BF16 first, FP8 as a measured follow-up). Never a silently-retained or substituted tensor
   class — fail closed on any unmapped name.
4. **Masked head.** Extract MTP draft (`tools/extract_mtp_draft.py` — extend for the unfused
   256-expert MTP MoE block), generate own-gen FR-Spec ranks (foreign ranks cost −12 acceptance
   pts, Qwen 27B precedent), bake trim via `tools/trim_draft_head.py` hqmtp order (trim first,
   then NVFP4 the retained rows). Safetensors side uses `MEMRA_FRSPEC_TRIM` at serve.
5. **Gates (all green before any publish or serving claim):** kernel-check; `run-gen` argmax
   MATCH (NVFP4 arm vs its own reference program); `run-spec` K=1..8 self-consistency;
   template byte-parity vs `chat_template.jinja` incl. tools branch and think-block handling;
   tool-call round-trip through the XML dialect parser (gemma-4 dialect-parser precedent);
   vision probe; determinism ×2; 262k long-ctx cell. One numeric program per request — the MTP
   spec arm proves token-identity vs plain greedy at every K.
6. **Serving battery + bank.** Serving-shape gates on a quiet PRO 6000; the pre-Step window on
   the 4-card RTX PRO 6000 box is the qualification window. Then co-residency cells WITH Step-3.7
   loaded on the other 3 cards (solo vs contended, both orders, N≥5 interleaved) before any
   published number. Deep-prefix-cache config from day one (~70 GB headroom; the 3.1x lever,
   `research/canonflip-20260813/`).
7. **Publish (after 5–6, not before).** Two HF repos under `Avifenesh`:
   `Ornith-1.5-35B-A3B-NVFP4-MTP` (safetensors + template + index + masked-head artifact) and
   `Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF` (Qwen3.8 precedent). Card drafts in this dir; cards
   carry mechanism claims + memra / inference.tiyuvta.ai links, no product voice, no prices.

## 3. Rig plan

- **box3 (GPU0 granted by owner 2026-08-19; live-verified both GPUs idle, 219G free):** BF16
  snapshot staged at `~/models/ornith15/bf16` (pinned rev), census probes, oracle transcripts,
  BF16 load gate, mint, correctness cells. pkill ban and PID-only stops per the box3 handoff;
  single card; all perf numbers contended-by-construction → dev ratios only.
- **A quiet 4-card RTX PRO 6000 window (pre-Step):** sealed serving battery + bank numbers,
  then co-residency cells with Step-3.7 loaded on the other three cards.
- **Local 5090:** development iteration battery per house rules; not a release gate here unless
  a generic 5090-facing default moves.

## 4. Economics

Business side (market board, pricing, unit economics) lives in the private repo:
darklanes `research/ornith15-prep-20260819/ECONOMY.md`. This repo carries none of it.
