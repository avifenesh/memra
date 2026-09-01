# Gemma-4-31B serving launch proposal (lane/gemma-vision, 2026-08-16)

Owner GO on gemma-4-31B as the next serving SKU (OR demand 514B tok/wk, highest-demand
open model that fits one card; speed + vision are the two wedges). This is the measured
launch basis. **No pricing, models.toml, site, or prod change is made here — owner gates
all of those.** Numbers below are the input to those decisions.

## 1. Measured serving numbers (Japan RTX PRO 6000, 450W cap, current lane binary)

Artifact under test: **google gemma-4-31B-it QAT Q4_0 GGUF** — the vendor quality artifact,
used here as an INTERIM REFERENCE. **This is NOT the product artifact** (see §3).

| cell | measured | OR board bar |
|---|---|---|
| plain decode, single-stream (c1) | **72.4 tok/s** (3 reps 72.4–72.5, dead flat) | floor-best 55 tps / top 127 tps |
| TTFT p50 (c1) | **71 ms** | Cerebras 0.32 s class — we are far under |
| aggregate c8 | ~69 tok/s — **does not scale past c1** | — |
| plain decode smoke | 72.4 tok/s, coherent ("### What is Binary Search?") | — |

Two hard reads:
- **Plain decode already beats the OR floor-best (55 tps) by ~32%** even at the 450W cap
  and on the un-tuned Q4_0 trunk. A 600W card (the eventual serving card may differ; the
  450W cap costs little on membw-bound decode per the 2026-08-16 fleet law) and the NVFP4
  product artifact both move this up.
- **The c8 aggregate does NOT scale** (69 ≈ c1's 70). That is the trunk-quant signal: the
  Q4_0 `qmatvec_q4_0_mmvq` path is memory-bound and does not batch. This is exactly the
  kat-anomaly lesson and the reason the product artifact must be NVFP4-class (§3), where
  q38 showed 228–233 tok/s aggregate on the fused NVFP4 kernels.

## 2. Vision input — the second wedge, gated GREEN end-to-end

- Tower parity 1.000000 per-token cosine vs an independent NumPy reference (REPORT.md §4).
- Masked-prefill bidirectional-island arm: reference-exact numeric parity at the image
  boundary; forced-causal wrong arm measurably off-reference (REPORT.md, masked section).
- **Decisive probe pair THROUGH memra-server** (MEMRA_GEMMA_VISION=1, TF32-off, this run):
  - blue triangle → "There is one blue triangle in this image."
  - missed-detail tell (triangle + red circle) → "A blue triangle / A red circle" (both).
- Text-only request on the seam-ON server → "391" (17×23), byte-identical path (no-image
  requests never construct gemma_images, so they never touch the new mask path — structural
  guarantee; a full byte-identity battery is on the not-done list).
- Most OR floor providers skip image input even though gemma-4 supports it — this is a
  live differentiator, working now behind the seam.

## 3. Artifact recipe (item 6, per the safetensors-native addendum)

**Product artifact = official `google/gemma-4-31b-it` SAFETENSORS → memra rig-native repack**
(the README doctrine; the source of record, not GGUF). Per the trunk-quant law the repack
target is NVFP4-class trunk matvecs (the fused kernels that gave q38 its 140 tps / 228 agg),
NOT Q4_0 (which §1 just showed does not batch).

**The native LM path does NOT support gemma-4 today — this is the launch-critical arc.**
Findings on current main (receipts = line refs in the repo):
- `SafetensorsSource` + `find_nvfp4_native` + `find_fp8_native` + `Hy3RepackSource` exist
  and generalize at the SOURCE level (the Step lane's native-FP8 migration pioneers this).
- `HybridModel::load` already reads gemma-4 tensors from a `TensorSource` (GGUF today):
  K=V global dedup, fused gate/up split, per-layer geometry all handled.
- BUT `hf_mapping.rs::hf_from_ggml` has arms for hy3/minimax/qwen35 only — **no Arch::Gemma4
  arm**. Loading gemma-4 from safetensors would (a) fail to map the sandwich post-norms
  (`attn_post_norm`/`ffn_post_norm` — no HF name) and (b) NOT apply the gemma +1 norm fold
  (only qwen35 + minimax's `use_gemma_norm` get it; GGUF works only because llama.cpp
  pre-folds +1 into the GGUF weight). Raw safetensors norms would be silently wrong by +1.
- NVFP4-native gemma tensor discovery (`find_nvfp4_native` name coverage for gemma) is
  likewise unmapped.

**Sized arc for native gemma-4 (launch-critical, top of the remainder):**
1. `Arch::Gemma4` name-map arm in `hf_mapping.rs`: post-norms, per-layer geometry, fused
   expert names, K=V globals (no v_proj in HF either — confirm), embed/lm_head.
2. Gemma +1 norm fold on EVERY norm for `Arch::Gemma4` (the `NormPlusOne` transform, gated
   on the arch — same fold qwen35/minimax already use).
3. NVFP4-native tensor discovery for gemma + repack path (reuse Step-lane machinery; do NOT
   touch the RunPod pod — read `migration-66513c6fa` for the pattern).
4. Parity gate: native-loaded gemma-4 logits vs the GGUF twin, byte/near-exact (the same
   gate the Step lane runs).
Interim: measure/serve GGUF explicitly labeled not-the-product until native lands.

## 4. Drafter / speed-tier story (the wedge that earns the premium)

- memra `gemma_spec.rs` is the most advanced round machinery in the engine (draft-chain
  graphs, verify-stream, device-accept, burst) — the speed tier is our structural strength.
- BUT the on-disk `gemma-4-31B-it-Q4_0-MTP.gguf` drafter is **n_embd 1024 ≠ trunk 5376** —
  a distilled small-backbone draft memra's shared-embedding MTP path cannot attach
  (`draft n_embd != model n_embd`, hard assert). So SPEC IS UNMEASURED.
- The speed premium (72.4 → 127+ tps to rival Cerebras) depends on a MATCHING MTP head or
  the retrained draft (`Hikari07jp/DSpark-Gemma-4-31B-draft`, safetensors, on disk as
  `dspark-gemma4-31b-draft`). Confirming acceptance + spec-vs-plain interleaved ×5 on the
  right drafter is the pre-pricing measurement for the premium tier.

## 5. VRAM at serving ctx

Q4_0 trunk + gemma tower + 8k ctx: **50.3 GB** resident on the 96 GB card (17.6 GB trunk +
~1.8 GB f32 tower + KV + a 7.4 GB prefix cache). Huge headroom: full 262k ctx KV fits, or
the Japan pair runs two replicas, or gemma + a second SKU. NVFP4 trunk (~16 GB) is smaller.

## 6. Suggested price position (owner decides)

Do NOT launch at the floor ($0.08/$0.35). The demand receipt shows the market pays 3–7×
floor for throughput + features, and we clear the floor's speed on plain decode already,
with a vision wedge most floor providers lack. Two-phase:
- **Launch (plain, GGUF-or-native)**: mid-tier, ~Together ($0.39/$0.97) — justified by
  72 tps > 55 floor + working image input. Honest capacity_tpm (one card).
- **Speed-premium tier (fast-follow)**: once NVFP4 native + a matching drafter land and the
  aggregate + spec numbers confirm the 127-tps-class throughput, move toward Cerebras-tier
  ($0.99/$1.49). The economics sketch (h=0.673, 10:1) puts a served-Mout unit at ~$2.6
  (Together) to ~$6.4 (Cerebras) — q38-class unit economics.
- Cache-read at the q38 $0.20/M convention.

## 7. Honest not-done list (ranked)

1. **Native safetensors gemma-4 load path** (§3) — launch-critical; product artifact
   depends on it. Everything else can proceed on the interim GGUF.
2. **Spec drafter** — matching MTP head or dspark draft; the speed-premium tier's basis
   is unmeasured until this attaches (§4).
3. **NVFP4 product artifact** decode + aggregate battery (expected to fix the c8 non-scaling
   of §1; the pricing basis for the premium tier).
4. Full text-only byte-identity battery (seam off AND on) — structural guarantee holds;
   the empirical receipt is owed.
5. Clean single-tenant box time: this battery ran on a shared box that cleared the trunk
   mid-run once; the numbers above are from the clean re-run, but the box is not isolated.
6. Server surgery shipped behind MEMRA_GEMMA_VISION (default off) — merge-to-main battery
   (servegate + text byte-identity) is the release gate the owner controls.

## 8. What IS launch-ready now (behind seams, owner-gated to flip)

- Gemma-4 vision serving through memra-server: tower + masked island prime + decisive probe
  green end-to-end (committed on lane/gemma-vision, MEMRA_GEMMA_VISION seam).
- Gemma-4 text serving on GGUF: 72.4 tok/s plain, beats the OR floor, coherent.
