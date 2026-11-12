# Gemma-4 12B

Google's Gemma 4 12B dense model from the official QAT Q4_0 GGUF, with the separate Gemma
assistant drafter, on one card.

| | |
|---|---|
| **Serving route** | QAT Q4_0 GGUF on one RTX 5090 Laptop. The assistant drafter serves through the `gspec` attach since 2026-08-17; the board's speculative numbers are `gemma-gate` CLI cells from July 2026. Standing gate: none |
| **Pack** | `gemma4_dense`, NativeReference on a tiny fixture (record `gemma4_dense-reference`); no record names this checkpoint <!-- support: gemma4_dense-reference --> |
| **Artifact** | `google/gemma-4-12B-it-qat-q4_0-gguf@29d097773436b69ff9feafd636ab4cf873786537`, file `gemma-4-12b-it-qat-q4_0.gguf`, 6,975,879,296 bytes, sha256 `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` ([lock](../../research/mtp-learned-depth-20260920/gemma-artifacts.lock.json)) |
| **Drafters** | the Gemma assistant drafter (a separate small model, not a NextN/MTP head) with an own-generated ranks trim |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## Run it

No cookbook block; the [Gemma-4 31B block](../COOKBOOK.md#gemma-4-31b) shows the attach. Serve the
trunk through `MEMRA_MODELS` and attach the assistant with `MEMRA_DRAFT=<assistant.gguf>`;
`MEMRA_GEMMA4_SPEC` unset then arms served speculative decode at K=5 (`0` is plain). Vendor-sampled,
unpenalized requests take the drafter too (`MEMRA_GEMMA_SPEC_SAMPLED`, on by default). Flags:
[FLAGS.md](../FLAGS.md).

## Measured

RTX 5090 Laptop, `gemma-gate` CLI cells: plain decode 92.55 tok/s (2026-07-24, N=5 cell stamp,
[`research/gemma4-bringup/g12tg-cellstamp.log`](../../research/gemma4-bringup/g12tg-cellstamp.log));
speculative K=4 with the trimmed assistant drafter 240.9 tok/s on the chat prompt (N=3,
re-paired 2026-07-26) and 269.3 at a 1.7k-token context (2026-07-15 re-audit). The frozen
llama.cpp reference values are in `extra_card_rows` of
[`research/tune-data/current-board.json`](../../research/tune-data/current-board.json) and in the
[Gemma-4 history table](../history/PERFORMANCE.md#gemma-4-qat-q4_0).

## Limits

- The speculative numbers are CLI measurements, not serving ones.
- Gemma 4 E4B is a different program (per-layer inputs, shared KV) with [its own card](gemma4-e4b.md).

## Deeper sources

- Bring-up receipts: [`research/gemma4-bringup/`](../../research/gemma4-bringup/).
- Serving gaps closed in August: [`research/gemma4-serve-20260807/`](../../research/gemma4-serve-20260807/PROGRESS.md),
  tools in [`research/gemma4-tools-20260817/`](../../research/gemma4-tools-20260817/).
- Pack: `crates/memra-gguf/src/model_packs/gemma4_dense/`.
