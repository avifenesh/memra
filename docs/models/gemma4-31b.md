# Gemma-4 31B

Google's Gemma 4 31B dense model, with the separate Gemma assistant drafter and optional image
input, on one card.

| | |
|---|---|
| **Serving route** | Two configurations. On the RTX 5090 Laptop: the QAT Q4_0 GGUF with the assistant drafter, `gemma-gate` CLI cells from July 2026. On RTX PRO 6000 Blackwell: the cookbook's Q6_K trunk with the official-lineage NVFP4 assistant drafter and the Gemma vision tower; the served-route receipt ran unpublished builds of the same weights and a ranks-trimmed drafter head ([earlier cookbook block](../history/COOKBOOK.md#gemma-4-31b)), and no speed receipt covers the published files. Standing gate: none; `tools/local-ci.sh` runs its `gemma-gate` stream-agreement cells on demand |
| **Pack** | `gemma4_dense`, NativeReference on a tiny fixture (record `gemma4_dense-reference`); no record names this checkpoint <!-- support: gemma4_dense-reference --> |
| **Artifacts** | trunks: `google/gemma-4-31B-it-qat-q4_0-gguf` (file `gemma-4-31B_q4_0-it.gguf`) for the CLI cells, `unsloth/gemma-4-31B-it-GGUF` (file `gemma-4-31B-it-Q6_K.gguf`) for the cookbook block; no revision pin is tracked for either. Vision projector `gemma-4-31B-it-mmproj.gguf` from the Google QAT repo. Drafter repo [`tiyuvta/gemma-4-31B-it-assistant-MTP-GGUF`](https://huggingface.co/tiyuvta/gemma-4-31B-it-assistant-MTP-GGUF): `gemma-4-31B-it-official-NVFP4-MTP.gguf` and `gemma-4-31B-it-official-Q8_0-MTP.gguf` |
| **Drafters** | the Gemma assistant drafter (a separate small model, not a NextN/MTP head) |
| **Hardware measured** | RTX 5090 Laptop, RTX PRO 6000 Blackwell |
| **Input** | text; image through `MEMRA_GEMMA_VISION=1` and `MEMRA_GEMMA_MMPROJ`, run with `NVIDIA_TF32_OVERRIDE=0` (see Limits) |

## Run it

Commands are in the [cookbook](../COOKBOOK.md#gemma-4-31b). The Gemma vision seam is
`MEMRA_GEMMA_VISION=1` plus `MEMRA_GEMMA_MMPROJ`; do not set `MEMRA_VISION_DIR`, which is the Qwen
tower, since one vision family runs per process. `MEMRA_DRAFT` attaches the assistant and
`MEMRA_GEMMA4_SPEC` unset arms served speculative decode at K=5.

## Measured

RTX 5090 Laptop, QAT Q4_0, `gemma-gate` CLI cells: speculative K=5 with the own-generated trimmed
drafter 124.3 tok/s on the chat prompt (N=3, re-paired 2026-07-26), K=6 97.3 tok/s at a 1.7k-token
context (2026-07-15 re-audit). The frozen llama.cpp reference values are in the
[Gemma-4 history table](../history/PERFORMANCE.md#gemma-4-qat-q4_0) and in `extra_card_rows` of
[`research/tune-data/current-board.json`](../../research/tune-data/current-board.json).

## Limits

- The speculative numbers are CLI measurements, not serving ones. The RTX PRO 6000 configuration
  has no tracked speed receipt.
- A drafter must come from its trunk's weights. On the QAT Q4_0 trunk the official-lineage head
  stays exact but measured about half the acceptance
  ([receipt](../../research/gemma-vision-20260816/ASSISTANT-ARM-RESULTS.md)).
- The vision tower's parity gates ran with TF32 off (`NVIDIA_TF32_OVERRIDE=0`). Under the default
  f32 cuBLASLt path, which allows TF32, the same tower read min-cos 0.9878 against the reference,
  because its unscaled attention amplifies TF32 rounding over 27 blocks
  ([`research/gemma-vision-20260816/`](../../research/gemma-vision-20260816/REPORT.md)). No code
  turns TF32 off for the tower, so the cookbook block sets `NVIDIA_TF32_OVERRIDE=0` for image input.
- `MEMRA_DRAFT` also re-keys three engine defaults by its presence (the windowed KV format, the
  f16 P/V prefill class and the stream-k GEMM entry on large dense models); see its row in
  [FLAGS.md](../FLAGS.md).

## Deeper sources

- Bring-up receipts: [`research/gemma4-bringup/`](../../research/gemma4-bringup/).
- Vision: [`research/gemma-vision-20260816/`](../../research/gemma-vision-20260816/REPORT.md).
- Pack: `crates/memra-gguf/src/model_packs/gemma4_dense/`.
