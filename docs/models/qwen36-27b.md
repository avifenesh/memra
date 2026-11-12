# Qwen3.6-27B

Qwen's 27B dense hybrid (Gated DeltaNet plus gated attention), from an NVFP4 or Q4_K_M GGUF that
carries its MTP tensors, on one card. It sits on the August decode board and has no newer
measurement.

| | |
|---|---|
| **Serving route** | NVFP4 / Q4_K_M MTP GGUF on one RTX 5090 Laptop, last measured on the 2026-08-02 board. Standing gate: none |
| **Pack** | `qwen35`; no pack record names this checkpoint |
| **Artifact** | no pinned revision is tracked; the receipts name the local file `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` |
| **Drafters** | the embedded MTP head; an own-generated trimmed draft |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## Run it

No cookbook block. Serve it like any GGUF through `MEMRA_MODELS="name=/path.gguf"` (see
[serving](../SERVING.md)); the MTP head in the file drafts with no flags.

## Measured

RTX 5090 Laptop, from the decode board in [PERFORMANCE.md](../PERFORMANCE.md#rtx-5090-laptop-board-2026-08-02):

- Plain tg128: 47.6 tok/s at a 512-token context, 46.2 at 6,257 tokens (2026-08-02, after commit
  `d14d7d8d`, N=5 interleaved; [raw runs](../../research/board-remeasure-20260802/)).
- Speculative K=3 with the own-generated trimmed draft: 116.4 / 101.2 / 86.0 tok/s on the
  short-code and medium-code (greedy) and long-agentic (sampled) prompts (paired 2026-07-18).

## Limits

- No gate has run on this checkpoint since the board. A change that touches the `qwen35` class is
  gated on Qwen3.8-27B, not on this model.
- Treat the exact file as part of the claim: another quantization of the same weights has no
  measurement here.

## Deeper sources

- Lanes: [`research/q27-deepdive-20260805/`](../../research/q27-deepdive-20260805/),
  [`research/q27-mtp-20260801/`](../../research/q27-mtp-20260801/) (raw only).
- Pack: `crates/memra-gguf/src/model_packs/qwen35/`.
