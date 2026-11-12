# Ornith-1.0-9B

Ornith's first 9B, a dense hybrid on the Qwen3.5 stack (Gated DeltaNet plus gated attention),
from the vendor Q8_0 GGUF with an own-generated drafter, on one card. It ships no MTP head.

| | |
|---|---|
| **Serving route** | Q8_0 GGUF plus the own-generated donor-block drafter on one RTX 5090 Laptop, measured 2026-08-02 (spec self-consistency 9/9). Standing gate: none |
| **Pack** | `qwen35`; no pack record names this checkpoint |
| **Artifact** | `ornith-ai/Ornith-1.0-9B-GGUF@3296bc7a404871a72ac3f1903f561459c09b5c17` (the receipts name it `deepreinforce-ai/Ornith-1.0-9B-GGUF`, which now redirects there), file `ornith-1.0-9b-Q8_0.gguf`, 9,527,500,992 bytes, sha256 `d0e4bebaa8b3450c62090df1408f2ee5ccb2094f9c610ffde564a654483d4f37` |
| **Drafters** | own-generated donor-block draft ([`research/ornith-drafters-20260801/`](../../research/ornith-drafters-20260801/)); the GGUF has no NextN/MTP block (427 tensors against 668 for Qwen3.5-9B) |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## Run it

No cookbook block. Serve the trunk with its draft attached through
`MEMRA_MODELS="name=/trunk.gguf+/draft.gguf"` (see [serving](../SERVING.md)). A missing draft
file is a boot refusal, not a silent fallback.

## Measured

RTX 5090 Laptop, 2026-08-02, N=3 interleaved in one session, speculative K=3 against plain:

| prompt class | decode, spec K=3 | decode, plain | acceptance | e2e for 256 tokens |
|---|---:|---:|---:|---:|
| short code (27 tokens) | 188.1 tok/s | 87.0 tok/s | 61.1% | 1.395 s |
| medium code (1,845 tokens) | 150.4 tok/s | 85.0 tok/s | 47.0% | 2.050 s |
| long agentic (6,257 tokens) | 144.8 tok/s | 84.9 tok/s | 47.8% | 2.984 s |

[Receipt](../../research/ornith-bar-20260802/RESULTS.md) (also the llama.cpp reference arm).

## Limits

- Nothing here transfers to Ornith-1.5; it has [its own card](ornith15-35b-a3b.md).
- No gate has run on this checkpoint since 2026-08-02.

## Deeper sources

- Onboarding: [`research/onboard-ornith-20260801/`](../../research/onboard-ornith-20260801/README.md)
  (config identity against Qwen3.5-9B, field by field).
- Serving lane: [`research/ornith-serve-20260801/`](../../research/ornith-serve-20260801/).
