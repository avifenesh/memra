# Qwen3.5-9B

Qwen's 9B dense hybrid (Gated DeltaNet plus gated attention) from a community NVFP4 GGUF with an
MTP head, on one card. It is the pinned artifact of two pack records and a real-weight oracle file
for `kernel-check`.

| | |
|---|---|
| **Serving route** | NVFP4 MTP GGUF on one RTX 5090 Laptop: five scoped serving cells (streaming, cache, concurrency 4, cancellation, a 6,261-token prompt) passed on 2026-10-02 at `2873dd4ca3`. Standing gate: none at model level; `kernel-check` in the tag battery reads this file for its real-weight oracle cells, which is kernel-level evidence only |
| **Pack** | `qwen35`, NativeReference: record `qwen35-9b-nvfp4-gguf` (CPU; Config, TokenizerTemplate, TensorCensus, TinyParity and RewriteParity passed, CheckpointParity and Serve pending) and record `qwen35-9b-nvfp4-b200-w4a8` (one B200) <!-- support: qwen35-9b-nvfp4-gguf,qwen35-9b-nvfp4-b200-w4a8 --> |
| **Artifact** | `michaelw9999/Qwen3.5-9B-NVFP4-MTP-GGUF@640ced996727af23f39fd0946b6d2bbabad58cd2`, 5,657,607,424 bytes, sha256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` |
| **Drafters** | the embedded MTP head; an own-generated trimmed draft on the August board |
| **Hardware measured** | RTX 5090 Laptop; B200 (record above). The H100 Q8_0 lane is retired |
| **Input** | text |

## Run it

Commands are in the [cookbook](../COOKBOOK.md#qwen35-9b). The measured fast path is the default; no
flag is needed for speed. `MEMRA_SPEC_K` pins K (`0` is plain decode).

## Measured

- **Serving cells, RTX 5090 Laptop** (2026-10-02, context 8,192, temperature 0, 64-token budgets):
  serial short requests ran MTP K=3, four concurrent clients each matched their serial control on
  the plain batched route, and a warm hit on the 6,261-token prompt restored 6,240 tokens with the
  same decoded bytes as cold. [Receipt](../../research/modelplan-onboarding-20261002-qwen35-9b/RESULTS.md).
- **Decode board, RTX 5090 Laptop**: plain tg128 137.3 tok/s at a 512-token context and 131.2 at
  6,257 tokens (2026-08-02, after commit `d14d7d8d`, N=5 interleaved); speculative K=3 with the
  own-generated trimmed draft 281.0 / 211.7 / 187.1 tok/s on the short-code and medium-code
  (greedy) and long-agentic (sampled) prompts (paired 2026-07-18). Board and llama.cpp reference columns:
  [PERFORMANCE.md](../PERFORMANCE.md#rtx-5090-laptop-board-2026-08-02); raw runs in
  [`research/board-remeasure-20260802/`](../../research/board-remeasure-20260802/).
- **B200** (2026-09-01): default W4A8 passed the model manifest, K=1..8, sampled serving,
  concurrency 4, admission and rollback cells. See the [B200 card](../rigs/b200.md).

## Limits

- The serving record covers this artifact, this card and an 8,192-token envelope. It says
  nothing about another artifact, the full checkpoint context or another numerical program.
- CheckpointParity has no receipt against an external oracle, so no record advances past
  NativeReference. <!-- support: qwen35-9b-nvfp4-gguf -->

## Deeper sources

- Onboarding records: [`research/modelplan-onboarding-20261002-qwen35-9b/`](../../research/modelplan-onboarding-20261002-qwen35-9b/),
  [`research/modelplan-onboarding-20260822/qwen35/`](../../research/modelplan-onboarding-20260822/qwen35/).
- Pack: `crates/memra-gguf/src/model_packs/qwen35/`.
