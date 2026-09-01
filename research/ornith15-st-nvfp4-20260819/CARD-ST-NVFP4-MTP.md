<!-- DRAFT model card — RESHAPED 2026-08-20 after the ST census (PLAN.md §2a).
     The official ornith-ai modelopt NVFP4 ST already exists and memra serves it directly,
     so a plain requant mirror adds nothing. The publishable ST artifact is the HEAD-PATCHED
     checkpoint: official NVFP4 + the continued-trained MTP head from
     ../ornith15-mtp-train-20260820/ + own-gen FR-Spec ranks.
     DO NOT PUBLISH unless (1) the trained head WINS the serve-level A/B vs the vendor head,
     and (2) the §2.5 gates in PLAN.md are green on the patched artifact. Every [TBD-*] gets a
     measured, receipted number or its row dies. -->
---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B-NVFP4
base_model_relation: finetune
pipeline_tag: text-generation
tags:
  - nvfp4
  - safetensors
  - memra
  - speculative-decoding
  - mtp
  - conversational
  - blackwell
  - qwen3_5_moe
  - moe
---

# Ornith-1.5-35B-A3B NVFP4 — continued-trained MTP head (memra serving artifact)

The official [ornith-ai NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
checkpoint with ONE change: the multi-token-prediction head (`mtp.*`, BF16 in the official
artifact) is **continued-trained on the trunk's own generations**. Ornith-1.5's RL loop moved
the trunk; the shipped 1-layer MTP head lagged it — measured [TBD-vendor]% draft acceptance
as shipped vs [TBD-trained]% after continued training (K=[TBD], greedy, same probes, RTX PRO
6000 Blackwell). Trunk, embeddings, lm_head and every quantized tensor are byte-identical to
the official checkpoint; a draft head can never change output — the target verifies every
drafted token — it only moves acceptance and speculative speed.

Training: teacher-forced (h, next-token) pairs from the model's own generations at serving
sampling (T=0.6/top-p 0.95/top-k 20, think + nothink), frozen trunk, ~[TBD]M tokens,
receipts in the [memra repo](https://github.com/avifenesh/memra)
(`research/ornith15-mtp-train-20260820/`).

Also ships FR-Spec vocab ranks (`ornith15-ranks-*.txt`) from the same own-gen corpus — with
memra, `MEMRA_FRSPEC_TRIM=<ranks.txt>` masks the draft head's vocab at load for cheaper
head reads.

Built for [**memra**](https://github.com/avifenesh/memra), a from-scratch Rust + CUDA
inference engine for RTX Blackwell (sm_120a) with per-request exactness gates: speculative,
graphed, and batched serving gated byte-identical to plain decode. Serves this checkpoint's
mixed classes natively (experts/shared/lm_head NVFP4, attention + GDN FP8, rest BF16) at the
full 262,144-token context on one 96 GB card.

- Engine: https://github.com/avifenesh/memra (MIT, crates.io: `memra-server`)
- Context: 262,144 native; chat template embedded (XML tool calling + `<think>` reasoning)

## Provenance

| | |
|---|---|
| Base | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`, rev `9660379a2f2c429c465eeed2f3a0f2433fc4381e` (MIT, modelopt 0.45.0) |
| Changed | `mtp.*` only (785 tensors, BF16) — continued-trained, frozen trunk |
| Unchanged | every other tensor byte-identical to the official checkpoint |
| Verified | memra `kernel-check` / `run-gen` argmax / `run-spec` K=1..8 / serve-st-gate [TBD-receipts] |

## Serve with memra

```bash
# speculative decode with the trained head (this artifact's point)
memra-server --model <this-repo-dir>

# masked draft head (FR-Spec ranks, load-time trim)
MEMRA_FRSPEC_TRIM=ornith15-ranks-owngen-32768.txt memra-server --model <this-repo-dir>
```
