# Qwen3.6-35B-A3B

Qwen's 35B-A3B mixture-of-experts on the Qwen3.5 MoE stack (Gated DeltaNet plus gated attention,
routed experts), from an IQ4_XS GGUF with its MTP head, resident on one card. Its file is the
artifact of the `qwen35_moe` pack record and a real-weight oracle file for `kernel-check`.

| | |
|---|---|
| **Serving route** | UD-IQ4_XS MTP GGUF on one RTX 5090 Laptop, last on the 2026-08-02 board; measured on GB10 on 2026-11-10 for a device default. Standing gate: none at model level; `kernel-check` in the tag battery reads this file for its real-weight oracle cells, which is kernel-level evidence only |
| **Pack** | `qwen35_moe`, NativeReference through record `qwen35_moe-reference` (CPU; Config, TokenizerTemplate, TensorCensus and TinyParity passed) <!-- support: qwen35_moe-reference --> |
| **Artifact** | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 17,730,509,792 bytes, sha256 `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3` (identity from the v0.141.1 release record). Source: `unsloth/Qwen3.6-35B-A3B-GGUF` at revision `a483e9e6cbd595906af30beda3187c2663a1118c` (same size and sha256 on Hugging Face, checked 2026-11-12). The support record still lists its artifact as local and unpinned |
| **Drafters** | the embedded MTP head; an own-generated trimmed draft |
| **Hardware measured** | RTX 5090 Laptop, DGX Spark (GB10) |
| **Input** | text |

## Run it

No cookbook block. Serve it like any GGUF through `MEMRA_MODELS="name=/path.gguf"` (see
[serving](../SERVING.md)); the MTP head in the file drafts with no flags.

## Measured

- **RTX 5090 Laptop decode board** ([PERFORMANCE.md](../PERFORMANCE.md#rtx-5090-laptop-board-2026-08-02)): plain tg128 187.0 tok/s
  at a 512-token context and 177.1 at 6,257 tokens (2026-08-02, after commit `d14d7d8d`, N=5
  interleaved); speculative K=2 with the own-generated trimmed draft 302.4 / 253.0 / 270.7 tok/s on
  the short-code and medium-code (greedy) and long-agentic (sampled) prompts (2026-08-02, N=3,
  [`research/f16g-default-rearb-20260802/`](../../research/f16g-default-rearb-20260802/)).
- **GB10, four penalised sampled requests at once** (2026-11-10, three boots per arm, the on arm
  ahead in nine of nine boot pairs): aggregate 88.97
  tok/s with the host penalty path and 131.41 with the device penalty arm, now the GB10 default;
  one request at a time is flat. [Receipt](../../research/devpenalty-gb10-20261110/README.md),
  [decision](../decisions/GB10-DEVICE-PENALTY-DEFAULT.md).

## Limits

- The pack record's CheckpointParity, RewriteParity and Serve gates are pending.
- Treat the exact file as part of the claim: another quantization of the same weights has no
  measurement here.

## Deeper sources

- Pack inspect receipt: [`research/spill-c-20260919/pro-single-day11/inspect/`](../../research/spill-c-20260919/pro-single-day11/inspect/).
- Pack: `crates/memra-gguf/src/model_packs/qwen35_moe/`.
