# Models

Which models memra runs, on which artifact and card, and in which state. This page is the index for
readers choosing a model: each row links the model's card, and the commands are in the
[cookbook](COOKBOOK.md).

## Two axes

Every claim about a model sits on one of two axes, stated separately.

- **Pack state**: the ModelPlan support state of the model's pack, backed by a record in
  [support-records.toml](support-records.toml). The three states are `NativeReference`, `NativeQualified` and `NativeTuned`, defined in [AGENTS.md](../AGENTS.md). <!-- support: none; not NativeReference NativeQualified NativeTuned -->
  Every doc line that names a state carries a `<!-- support: <record-id> -->` marker, checked by
  `tools/check-support-states.py`. No record today is NativeQualified or NativeTuned. <!-- support: none; not NativeQualified NativeTuned -->
- **Serving route**: a measured configuration (artifact, card, topology) with the gates it passed,
  the version and date it was last gated, and whether a standing gate covers it. Only the two
  `tools/release-roster.tsv` checkpoints have a standing gate: the tag battery on every release and
  the daily per-device regression run Ornith-1.5-35B-A3B and Qwen3.8-27B. Every other route was
  last measured on the date in its row.

The axes are independent. A model can serve through a hand-written path with no pack state
(Step-3.7-Flash, DeepSeek-V4-Flash), and a pack can be NativeReference with nothing serving it <!-- support: deepseek_v41-reference -->
(DeepSeek-V4.1-Flash). These docs say "measured configuration" or "gated route" for a serving
route; "qualified" and "tuned" belong to pack states.

## Roster

| Model | Pack | Pack state | Serving route, last gated or measured | Standing gate |
|---|---|---|---|---|
| [Qwen3.8-27B](models/qwen38-27b.md) | `qwen35` | NativeReference for the B200 block-FP8 artifact only <!-- support: qwen38-27b-fp8-b200 --> | NVFP4+Q5_K MTP GGUF, one card; tag battery v0.141.1, 2026-11-11 | yes, roster `vendor` |
| [Ornith-1.5-35B-A3B](models/ornith15-35b-a3b.md) | `qwen35_moe` | NativeReference through a record that names a Qwen3.6 artifact; none for this one <!-- support: qwen35_moe-reference --> | NVFP4+Q5_K MTP GGUF, one card; tag battery v0.141.1, 2026-11-11 | yes, roster `own` |
| [Qwen3.5-9B](models/qwen35-9b.md) | `qwen35` | NativeReference (CPU and one-B200 records) <!-- support: qwen35-9b-nvfp4-gguf,qwen35-9b-nvfp4-b200-w4a8 --> | NVFP4 MTP GGUF, RTX 5090 Laptop; serving cells 2026-10-02 | no |
| [Qwen3.6-27B](models/qwen36-27b.md) | `qwen35` | no record | NVFP4 / Q4_K_M MTP GGUF, RTX 5090 Laptop; board 2026-08-02 | no |
| [Qwen3.6-35B-A3B](models/qwen36-35b-a3b.md) | `qwen35_moe` | NativeReference <!-- support: qwen35_moe-reference --> | UD-IQ4_XS MTP GGUF, RTX 5090 Laptop; board 2026-08-02, GB10 cells 2026-11-10 | no |
| [Qwen-AgentWorld-35B-A3B](models/qwen-agentworld-35b-a3b.md) | `qwen35_moe` | no record | UD-IQ4_XS GGUF plus own drafter, RTX 5090 Laptop; 2026-08-02 | no |
| [Ornith-1.0-9B](models/ornith10-9b.md) | `qwen35` | no record | Q8_0 GGUF plus own drafter, RTX 5090 Laptop; 2026-08-02 | no |
| [Ornith-1.0-35B](models/ornith10-35b.md) | `qwen35_moe` | no record | Q4_K_M GGUF plus own drafter, RTX 5090 Laptop; 2026-08-02 | no |
| [Gemma-4 12B](models/gemma4-12b.md) | `gemma4_dense` | NativeReference on a tiny fixture <!-- support: gemma4_dense-reference --> | QAT Q4_0 GGUF plus assistant drafter, RTX 5090 Laptop; CLI cells July 2026 | no |
| [Gemma-4 26B-A4B](models/gemma4-26b-a4b.md) | `gemma4_moe` | NativeReference on a tiny fixture <!-- support: gemma4_moe-reference --> | QAT Q4_0 GGUF plus assistant drafter, RTX 5090 Laptop; CLI cells July 2026 | no |
| [Gemma-4 31B](models/gemma4-31b.md) | `gemma4_dense` | NativeReference on a tiny fixture <!-- support: gemma4_dense-reference --> | QAT Q4_0 GGUF, RTX 5090 Laptop, CLI cells July 2026; Q6_K trunk with vision on RTX PRO 6000, no speed receipt | no |
| [Gemma-4 E4B](models/gemma4-e4b.md) | `gemma4_e4b` | NativeReference <!-- support: gemma4_e4b-reference --> | none gated; engine cells on RTX 5090 Laptop 2026-11-04 | no |
| [Step-3.7-Flash 196B-A11B](models/step37-flash.md) | `step35` | none (pack support `None`) | IQ4_XS GGUF, PP-2 on 2x RTX PRO 6000; 2026-09-20 | no |
| [DeepSeek-V4-Flash](models/deepseek-v4-flash.md) | `deepseek_v4` | none; its drafter pack is NativeReference on a tiny fixture <!-- support: deepseek_v4_dspark-reference --> | NVFP4 checkpoint, TP/EP on 2x RTX PRO 6000; 2026-09-28 | no |
| [GLM-5.3-Flash](models/glm53-flash.md) | `glm5_next` | NativeReference <!-- support: glm5_next-reference --> | NVFP4 mint on RTX PRO 6000, TP-2/TP-4 plain; September 2026 | no |
| [Hy3](models/hy3.md) | `hy3`, `hy3_nvfp4` | NativeReference for both <!-- support: hy3-bf16-reference,hy3_nvfp4-modelopt-w4a16 --> | W4A16 mint, EP-4 on 4x RTX PRO 6000; reported 2026-08-30, receipts not tracked | no |
| [DeepSeek-V4.1-Flash](models/deepseek-v41-flash.md) | `deepseek_v41` | NativeReference <!-- support: deepseek_v41-reference --> | none (reference executor on CPU) | no |
| [Qwen3.8-Flash-Next](models/qwen38-flash-next.md) | `qwen4_exp` | none (pack support `None`) | none (bring-up gate path) | no |

Packs with no model card: `qwen3`, `qwen3_moe`, `llama_dense` and `glm_dsa` (GLM-5 and 5.2) are NativeReference on tiny fixtures; `olmoe`, `minimax_m3` and the `mimo_v2` profiles declare no state. <!-- support: qwen3-tiny-random,qwen3_moe-reference,llama_dense-reference,glm_dsa-reference -->

Speech: the Whisper large-v3, Nemotron RNNT and Qwen3-TTS packs bind their checkpoints and execute
nothing natively. The `/v1/audio/*` routes exist: transcription and session open answer 503
`engine_unbound`, since no speech operation has a CUDA kernel, the session list answers 200, and
frames or close answer 404 `unknown_session`. The program is [SPEECH.md](SPEECH.md).

## Generated board

`tools/update-perf-board.py` writes the block below from `research/tune-data/current-board.json`;
edit the board, not this file. Pack states and serving routes are in the roster above and on the
model cards.

<!-- PERF-MODELS:START (generated by tools/update-perf-board.py; do not hand-edit, edit research/tune-data/current-board.json instead) -->
| Model | Class | Quant | Drafter | First served in |
|---|---|---|---|---|
| Qwen3.5-9B | dense hybrid (GDN + gated attention) | NVFP4 (5090), Q8_0 (H100) | MTP + own-gen trimmed draft | v0.1.0 |
| Qwen3.8-27B | dense hybrid (GDN + gated attention) | NVFP4+Q5_K MTP GGUF · NVFP4 safetensors; performance receipts do not transfer between the two | MTP + own-gen FR-Spec masked ranks (safetensors: MEMRA_FRSPEC_TRIM; GGUF: pre-trimmed head, HF tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF) · DFlash2 block-diffusion drafter, opt-in (MEMRA_DSPARK_SPEC=1 + MEMRA_DSPARK_DRAFT, q4 default + FR-Spec trim, HF tiyuvta/Qwen3.8-27B-DFlash2-memra) | v0.82.2 (MTP), v0.113.0 (DFlash2) |
| Qwen3.6-27B | dense hybrid (GDN + gated attention) | NVFP4, Q4_K_M MTP-baked | MTP + own-gen trimmed draft | v0.1.0 |
| Qwen3.6-35B-A3B | MoE hybrid (GDN + gated attention) | IQ4_XS | MTP + own-gen trimmed draft | v0.1.0 |
| Gemma-4 26B-A4B | MoE | QAT Q4_0 | Gemma assistant drafter (MEMRA_DRAFT), served since 2026-08-17 at K=5; the cards' spec numbers are gemma-gate CLI cells | v0.23.0 |
| Gemma-4 31B | dense | QAT Q4_0 GGUF (RTX 5090 Laptop CLI cells) · Q6_K GGUF with image input (MEMRA_GEMMA_VISION=1 + MEMRA_GEMMA_MMPROJ, RTX PRO 6000) | Gemma assistant drafter (MEMRA_DRAFT), served since 2026-08-17 at K=5; the cards' spec numbers are gemma-gate CLI cells | v0.35.0 |
| Gemma-4 E4B | dense | QAT Q4_0 | Gemma assistant drafter through gemma-gate only; memra-server's drafter attach skips E4B, so it serves without a drafter | v0.35.0 |
| Gemma-4 12B | dense | QAT Q4_0 | Gemma assistant drafter (MEMRA_DRAFT), served since 2026-08-17 at K=5; the cards' spec numbers are gemma-gate CLI cells | v0.40.0 |
| Ornith-1.0-9B | dense hybrid (GDN + gated attention) | Q8_0 | own-gen donor-block draft | v0.63.0 |
| Ornith-1.0-35B | MoE hybrid (GDN + gated attention) | Q4_K_M | own-gen donor-block draft | v0.64.0 |
| Ornith-1.5-35B-A3B | MoE hybrid (GDN + gated attention) | NVFP4+Q5_K MTP GGUF (HF tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF) · image input through the checkpoint ViT (MEMRA_VISION_DIR, parity-gated) | continued-trained MTP head + FR-Spec ranks self-trim (MEMRA_FRSPEC_TRIM); since v0.105 cached-long prompts run at K=5 with a trimmed head, and since v0.108 the verify trunk replays as a captured graph by default (+19.7% on a current-generation host, less sensitive to the host CPU) | v0.97.0 |
| Qwen-AgentWorld-35B-A3B | MoE hybrid (GDN + gated attention) | UD-IQ4_XS (not UD-Q4_K_M: its Q5_K expert mix sits outside fast-path coverage) | own-gen drafter | v0.66.0 |
| Step-3.7-Flash 196B-A11B | MoE | IQ4_XS GGUF + Q8_0 MTP head (two-card PP-2) · NVFP4 safetensors, with image input through the checkpoint perception_encoder ViT (MEMRA_STEP_VISION_DIR, parity- and serve-gated, default off) · official FP8 (gated as PP-3, 2026-09-20) | Q8_0 MTP head; PP-2 serves plain batched decode by default (MEMRA_SPEC_GATE=0 restores spec) | v0.73.1 |
<!-- PERF-MODELS:END -->

## Capture models

Prefill-only models served through the capture surfaces, with no drafter and no board row.

| Model | Surface | Artifact | Serving route | Standing gate |
|---|---|---|---|---|
| [Qwen3-Embedding-8B](models/qwen3-embedding-8b.md) | `POST /v1/embeddings` | Q8_0 GGUF | since v0.116.0; gate figures reported, no tracked receipt | no |
| [Qwen3-Reranker-8B](models/qwen3-reranker-8b.md) | `POST /v1/rerank` | Q8_0 GGUF (f16 then `llama-quantize`) | since v0.116.0; gate figures reported, no tracked receipt | no |

## Hardware targets

| Target | Cards | Push-gated | Prebuilt | Card |
|---|---|---|---|---|
| `sm_120a` | RTX 5090 and RTX 50-series, RTX PRO 6000 Blackwell | yes | yes | [RTX 5090](rigs/rtx-5090.md), [RTX PRO 6000](rigs/rtx-pro-6000-blackwell.md) |
| `sm_121a` | GB10 (DGX Spark) | yes | no | [DGX Spark](rigs/dgx-spark.md) |
| `sm_100a` | B200, best effort | no | no | [B200](rigs/b200.md) |
| `sm_90a` | H100 | no | no | [H100](rigs/h100.md) |
| `sm_89` | Ada | no | no | [Ada](rigs/ada.md) |

The build auto-detects the target from the card's compute capability. All cards:
[rigs/](rigs/README.md); request shapes: [workloads/](workloads/README.md).

## Formats

Official safetensors checkpoints (config, tokenizer and template, quantization metadata, auxiliary
tensors) are the preferred semantic source for native work, repacked once into rig-native layouts.
GGUF stays a supported import format, and its gates must not regress
([SAFETENSORS-DECISION](decisions/SAFETENSORS-DECISION.md), [FORMAT-DECISION](decisions/FORMAT-DECISION.md)).

## Adding a model

Onboarding compiles a pack, not a new engine path. The procedure is [ONBOARDING.md](ONBOARDING.md),
the rules are the [model onboarding section](../AGENTS.md#model-onboarding-compile-a-native-plan-not-a-new-engine-path-owner-2026-08-22)
of AGENTS.md, and receipts go under `research/modelplan-onboarding-*`. To propose a model without
onboarding it, open an issue with the checkpoint ID, quantization, target GPU and use case
([CONTRIBUTING.md](../CONTRIBUTING.md)). A proposal is not a support commitment.

Earlier sections of this page (the format direction, the per-family table, lane notes, the
in-detail sections and the chat-dialect contract) are kept in [history/MODELS.md](history/MODELS.md).
