# Ornith-1.0-35B

Ornith's first 35B-A3B mixture-of-experts on the Qwen3.5 MoE stack (the same `qwen35moe`
Gated DeltaNet plus routed-expert layout as Qwen3.6-35B-A3B), from the vendor Q4_K_M GGUF with an
own-generated drafter, on one card. It ships no MTP head.

| | |
|---|---|
| **Serving route** | Q4_K_M GGUF plus the own-generated donor-block drafter on one RTX 5090 Laptop, measured 2026-08-02. Standing gate: none |
| **Pack** | `qwen35_moe`; no pack record names this checkpoint |
| **Artifact** | `ornith-ai/Ornith-1.0-35B-GGUF@383064f72a1ef3087b779f268d3ca117eb989aac` (the receipts name it `deepreinforce-ai/Ornith-1.0-35B-GGUF`, which now redirects there), file `ornith-1.0-35b-Q4_K_M.gguf`, 21,166,757,760 bytes, sha256 `ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002` |
| **Drafters** | own-generated donor-block draft ([`research/ornith-drafters-20260801/`](../../research/ornith-drafters-20260801/)); the GGUF has no NextN/MTP block |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## Run it

No cookbook block. Serve the trunk with its draft attached through
`MEMRA_MODELS="name=/trunk.gguf+/draft.gguf"` (see [serving](../SERVING.md)).

## Measured

RTX 5090 Laptop, 2026-08-02, N=3 interleaved, idle rig: plain decode tg128 139.1 tok/s (argmax
MATCH 3/3), prefill pp512 495.7 tok/s, 1.953 s for 512 prompt plus 128 generated tokens. The
adopted K=2 drafter's spec-over-plain ratios (1.38x / 1.09x / 1.05x on the short-code, medium-code
and long-agentic classes) put decode at about 192 / 152 / 146 tok/s, figures the receipt computes
from the plain row and those ratios rather than times. [Receipt](../../research/ornith-bar-20260802/RESULTS.md) (also the llama.cpp
reference arm).

## Limits

- Prefill on this Q4_K_M expert mix was the slow part in August (495.7 tok/s at pp512). The
  vendor repo publishes no IQ4_XS file, the quantization the Qwen3.6-35B-A3B route uses.
- Nothing here transfers to Ornith-1.5; it has [its own card](ornith15-35b-a3b.md).
- No gate has run on this checkpoint since 2026-08-02.

## Deeper sources

- Onboarding: [`research/onboard-ornith-20260801/`](../../research/onboard-ornith-20260801/README.md).
- Serving lane: [`research/ornith-serve-20260801/`](../../research/ornith-serve-20260801/).
