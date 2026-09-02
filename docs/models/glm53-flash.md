# GLM-5.3-Flash

Support state: **NativeReference**. Bring-up is gated on the bench box's two-card RTX PRO
6000 Blackwell Server receipts; NativeQualified admission is the next gate.

- Source: `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0` (FP8 e4m3, MIT).
- Architecture, read from the modeling source rather than the card: 45 decoder layers + 1
  MTP — 34 KDA linear-attention and 11 DSA (MLA + sparse indexer); MoE on 42+MTP layers
  (288 routed sigmoid + 1 shared) plus 3 dense layers; 4-stream hyper-connections; no
  rotary anywhere in the text stack. Trained `context_length` is 1,048,576 — a training
  fact, not a serving claim: the resident shape OOMs the 1M prime
  (`research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md`).
- Engine surface: the `glm5_next` model plan, a hand-written multi-card TP path
  (`crates/memra-engine/src/glm5_tp.rs`), MTP spec (`glm_spec.rs`), and vision
  (`vision_glm5.rs`). The template's always-rendered `Reasoning Effort: {Low|High|Max}`
  line and unconditional `<think>` tail are probed and documented in
  [serving](../SERVING.md).
- Bring-up receipts: `research/glm53-flash-bringup-20260827/` (tensor census, 262k
  two-card, batched decode gate); vision and multi-card batteries under
  `research/glm5-vision-20260830/` and `research/glm53-vision-ppn-20260901/`.

Checkpoint-, quantization-, and hardware-specific: none of this transfers to GLM-5/5.2
(`glm_dsa`, loader-only) or to any other family by analogy.
