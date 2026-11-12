# GLM-5.3-Flash

Z.ai's GLM-5.3-Flash (FP8 e4m3 source, MIT) on the hand-written `glm5_next` path, with
multi-card tensor parallelism, vision and MTP speculative decode gated there. The `glm5_next` pack
runs in the reference executor; those GPU gates are not pack gates.

| | |
|---|---|
| **Serving route** | NVFP4 mint on RTX PRO 6000 Blackwell: the batched-decode gate (2026-08-28), the two-card bring-up at 262k context (2026-08-30), vision and multi-card batteries (2026-08-30 to 2026-09-01). Multi-card serving is `MEMRA_GLM5_TP` TP-2 or TP-4, plain decode only. Standing gate: none |
| **Pack** | `glm5_next`, **NativeReference** (record `glm5_next-reference`: Config, TokenizerTemplate, TensorCensus and TinyParity passed; CheckpointParity, RewriteParity and Serve pending) <!-- support: glm5_next-reference --> |
| **Artifacts** | source `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`; NVFP4 mint [`tiyuvta/GLM-5.3-Flash-NVFP4`](https://huggingface.co/tiyuvta/GLM-5.3-Flash-NVFP4) (20 shards, 190,750,154,058 bytes as published 2026-08-28; the receipts name its earlier path `Avifenesh/GLM-5.3-Flash-NVFP4`) |
| **Drafters** | the embedded MTP layer (`glm_spec.rs`), off by default: `MEMRA_GLM5_MTP=1` loads it and `MEMRA_GLM5_SPEC=1` arms speculative decode; gated on single-engine and PP-N routes, refused on TP |
| **Hardware measured** | RTX PRO 6000 Blackwell, one card and pairs; B200 lanes in September |
| **Input** | text; image when the artifact carries `model.visual.*` (default on, `MEMRA_GLM5_VISION=0` disables) |

NativeQualified needs pack checkpoint parity and serve first. <!-- support: none; not NativeQualified -->

## Architecture

Read from the modeling source rather than the card: 45 decoder layers plus 1 MTP layer, 34 KDA
linear-attention and 11 DSA (MLA plus a sparse indexer); MoE on 42 layers plus MTP (288 routed
sigmoid experts and 1 shared) and 3 dense layers; 4-stream hyper-connections; no rotary anywhere in
the text stack. The trained `context_length` of 1,048,576 is a training fact, not a serving
claim: the resident shape runs out of memory at the 1M prime
([`research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md`](../../research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md)).

## Run it

No cookbook block. Two things decide the route:

- `MEMRA_GLM5_TP` takes TP-2 or TP-4 (`all@0,1` or `all@0,1,2,3`; TP-3 refuses by name). On a
  serving worker it serves plain decode only and refuses to boot unless `MEMRA_SERVE_SPEC=0` states
  that; spec programs, the batched hyper-connection walk (`MEMRA_HYPER_BATCH=1`) and DFlash2 are
  refused by name. Spec runs on single-engine and PP-N routes. Flag detail: [FLAGS.md](../FLAGS.md).
- The chat dialect is GLM's own, not ChatML: `[gMASK]<sop>` framing, `<|observation|>` tool turns,
  an unconditional `<think>` tail and an always-rendered `Reasoning Effort: {Low|High|Max}` line
  (default Max). An explicit reasoning-off request is a named 400, and `response_format` refuses
  by name. The effort ladder and template behavior are in [serving](../SERVING.md).

## Measured

Pass/fail gates with their receipts; no throughput number here has a current receipt.

- Template byte parity against the checkpoint's own `chat_template.jinja`: 20 generated fixtures
  replayed through the real request pipeline on all three wire formats
  (`research/glm53-flash-bringup-20260827/gen_surface_fixtures.py`, `surface-fixtures/`).
- Batched-decode gate (2026-08-28):
  [`batched-decode-gate/`](../../research/glm53-flash-bringup-20260827/batched-decode-gate/); two-card
  262k context (2026-08-30): [`262k-2card-20260830/`](../../research/glm53-flash-bringup-20260827/262k-2card-20260830/LANE.md).
- Vision and multi-card batteries: [`research/glm5-vision-20260830/`](../../research/glm5-vision-20260830/),
  [`research/glm53-vision-ppn-20260901/`](../../research/glm53-vision-ppn-20260901/).

## Limits

- None of this transfers to GLM-5 or GLM-5.2: those load through the `glm_dsa` pack, which is
  NativeReference on a tiny fixture only (record `glm_dsa-reference`). <!-- support: glm_dsa-reference -->
- B200 (`sm_100a`) doors were decided on 2026-11-04 after the box retired, and B200 is best effort
  ([decision](../decisions/GLM5-B200-DOORS.md)); the sm_100a TP-2 plain route samples on the device
  ([decision](../decisions/GLM5-TP-DEVICE-SAMPLER-DEFAULT.md)).

## Deeper sources

- Bring-up lane: [`research/glm53-flash-bringup-20260827/`](../../research/glm53-flash-bringup-20260827/BRINGUP.md)
  (census, placement, HF publish receipts).
- Engine: `crates/memra-engine/src/glm5_tp.rs`, `glm_spec.rs`, `vision_glm5.rs`; pack
  `crates/memra-gguf/src/model_packs/glm5_next/`.
- The dialect bring-up section and the earlier card lines: [history](../history/MODELS.md#glm-53-flash-serves-its-own-chat-dialect-not-chatml-laneglm53-flash-bringup-20260827).
