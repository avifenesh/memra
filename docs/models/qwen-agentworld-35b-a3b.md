# Qwen-AgentWorld-35B-A3B

A 35B-A3B mixture-of-experts fine-tune on the Qwen3.5 MoE stack (the same `qwen35moe` layout as
Qwen3.6-35B-A3B), from the unsloth UD-IQ4_XS GGUF with an own-generated drafter, resident on one
card. The artifact choice is part of the claim: the UD-Q4_K_M file from the same revision was
measured and not adopted.

| | |
|---|---|
| **Serving route** | UD-IQ4_XS GGUF plus the own-generated drafter at K=2 on one RTX 5090 Laptop, measured 2026-08-02 (kernel-check, run-gen argmax and drafter gates passed). Standing gate: none |
| **Pack** | `qwen35_moe`; no pack record names this checkpoint |
| **Artifact** | `unsloth/Qwen-AgentWorld-35B-A3B-GGUF@3a305abf5cfd119ee999dfe929c433746edd8d63`, file `Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf`, 17,785,036,032 bytes, sha256 `ff4201b0c163950dc96aeaca033398543a1d62513ddc1c4030f9b94823764e06` |
| **Drafters** | own-generated drafter built against this model's ranks; the GGUF has no NextN/MTP block |
| **Hardware measured** | RTX 5090 Laptop (experts resident: 15.22 GB of experts plus a 2.56 GB trunk) |
| **Input** | text |

## Run it

No cookbook block. Serve the trunk with its draft attached through
`MEMRA_MODELS="name=/trunk.gguf+/draft.gguf"` (see [serving](../SERVING.md)).

## Measured

RTX 5090 Laptop, 2026-08-02, N=3 medians interleaved in one session:

| prompt class | decode, plain | decode, spec K=2 | acceptance | e2e for 256 tokens, spec |
|---|---:|---:|---:|---:|
| short code (27 tokens) | 180.96 tok/s | 259.25 tok/s | 65.9% | 1.021 s |
| medium code (1,845 tokens) | 174.02 tok/s | 273.67 tok/s | 80.6% | 1.278 s |
| long agentic (6,257 tokens) | 173.18 tok/s | 275.66 tok/s | 87.1% | 2.090 s |

[Receipt](../../research/agentworld-iq4xs-20260802/README.md) (also the llama.cpp reference arm).

## Limits

- Do not use the UD-Q4_K_M file. Its `ffn_down_exps` tensors are Q5_K in 37 of 40 layers, and no
  fast expert path takes Q5_K: single gate runs the same day decoded 69.9 tok/s against 180.96 for
  UD-IQ4_XS, and the file misses naked residency by 0.20 GB
  ([`research/agentworld-20260802/`](../../research/agentworld-20260802/README.md)).
- No gate has run on this checkpoint since 2026-08-02.

## Deeper sources

- Onboarding of the same family: [`research/onboard-ornith-20260801/`](../../research/onboard-ornith-20260801/README.md).
- Pack: `crates/memra-gguf/src/model_packs/qwen35_moe/`.
