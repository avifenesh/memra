# Gemma-4 26B-A4B

Google's Gemma 4 26B-A4B mixture-of-experts from the official QAT Q4_0 GGUF, with the separate
Gemma assistant drafter, on one card.

| | |
|---|---|
| **Serving route** | QAT Q4_0 GGUF on one RTX 5090 Laptop. The assistant drafter serves through the `gspec` attach since 2026-08-17; the board's speculative numbers are `gemma-gate` CLI cells from July 2026. Standing gate: none |
| **Pack** | `gemma4_moe`, NativeReference on a tiny fixture (record `gemma4_moe-reference`); no record names this checkpoint <!-- support: gemma4_moe-reference --> |
| **Artifact** | Google's QAT Q4_0 GGUF, file `gemma-4-26B_q4_0-it.gguf` from `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (no revision pin is tracked) |
| **Drafters** | the Gemma assistant drafter with the own-generated ranks trim (`MEMRA_GEMMA_DRAFT_RANKS`, adopted for this model in July) |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## Run it

No cookbook block; the [Gemma-4 31B block](../COOKBOOK.md#gemma-4-31b) shows the attach. Attach the
assistant with `MEMRA_DRAFT=<assistant.gguf>`; `MEMRA_GEMMA4_SPEC` unset then arms served
speculative decode at K=5 (`0` is plain). Flags: [FLAGS.md](../FLAGS.md).

## Measured

RTX 5090 Laptop, `gemma-gate` CLI cells from the 2026-07-15 re-audit: plain decode 199.7 tok/s;
speculative K=6 with the trimmed assistant drafter 330.4 tok/s at a 1.7k-token context. The
frozen llama.cpp reference values for these two cells are in `extra_card_rows` of
[`research/tune-data/current-board.json`](../../research/tune-data/current-board.json); the
[Gemma-4 history table](../history/PERFORMANCE.md#gemma-4-qat-q4_0) holds other 26B cells.

## Limits

- The speculative numbers are CLI measurements, not serving ones.
- Since 2026-11-11 every MoE layer runs one program whether its experts sit in the resident slab or
  in the SLRU cache ([decision](../decisions/GEMMA-MOE-PLACEMENT-INDEPENDENT.md)); no board number
  has been re-measured since.

## Deeper sources

- Bring-up receipts: [`research/gemma4-bringup/`](../../research/gemma4-bringup/).
- Placement lane: [`research/gemma-moe-placement-20261111/`](../../research/gemma-moe-placement-20261111/).
- Pack: `crates/memra-gguf/src/model_packs/gemma4_moe/`.
