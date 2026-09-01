---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
tags:
  - nvfp4
  - gguf
  - memra
  - speculative-decoding
  - mtp
  - conversational
  - blackwell
  - qwen3_5_moe
  - moe
model-index:
  - name: Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF
    results:
      - task:
          type: text-generation
          name: MTP speculative decode K=3, greedy, 256-token serve probes (code + agentic), same-window interleaved A/B
        dataset:
          name: memra serve probes (deterministic, per-request usage.spec accounting)
          type: memra-probe
        metrics:
          - name: draft acceptance, embedded continued-trained head (this artifact)
            type: acceptance-rate
            value: 0.4309
          - name: draft acceptance, vendor head (same mint recipe, superseded revision)
            type: acceptance-rate
            value: 0.3523
          - name: draft acceptance, FR-Spec masked head (top-32768 own-gen ranks)
            type: acceptance-rate
            value: 0.393
        source:
          name: memra v0.94.0 serving A/B, RTX PRO 6000 Blackwell
          url: https://github.com/avifenesh/memra
---

# Ornith-1.5-35B-A3B — NVFP4 GGUF with continued-trained MTP head (memra serving artifact)

NVFP4 (4-bit e2m1, per-16 FP8-e4m3 scales) GGUF of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), quantized
from the official BF16 GGUF release (token embeddings and output head Q5_K, norms F32).
**71.9 GB BF16 → 20.2 GB.** First NVFP4 quantization of this model in either format at
original publication (2026-08-19, ~34.5 h after the model dropped).

**This revision ships a continued-trained MTP head.** The vendor's multi-token-prediction
head is functional at chain depth 1 and collapses deeper — measured next-token top-1 on
held-out own-generation data: depth 1 **0.80**, depth 2 **0.27**, depth 3 **0.13** (the trunk's
RL loop moved; the 1-layer head did not follow). We continued-trained the head on the trunk's
own generations with depth-3 chain-rollout (teacher-forced tokens, self-recursive hidden
carrier — the exact recurrence speculative decoding runs at serve time). Same held-out
measurement after training: **0.81 / 0.58 / 0.43**. Serve-level draft acceptance (K=3, greedy,
same-window interleaved A/B): **0.431 vs 0.352** for the vendor head; code-generation probes
0.540 vs 0.388. Trunk, embeddings, lm_head and every non-head tensor are unchanged official
bytes; a draft head can never change output — the target verifies every drafted token — it
only moves acceptance and speculative speed.

Built as a serving artifact for [**memra**](https://github.com/avifenesh/memra), a from-scratch
Rust + CUDA inference engine for RTX Blackwell (sm_120a) with per-request exactness gates:
speculative, graphed, and batched serving are gated byte-identical to plain decode. A single
96 GB Blackwell card fits these weights with room for the full 262,144-token native context.

NVFP4 is not an upstream llama.cpp tensor type: this file runs on memra and on the
[NVFP4 branch of avifenesh/llama.cpp](https://github.com/avifenesh/llama.cpp/tree/nvfp4-imatrix-scale-search).
For upstream-llama.cpp/Ollama use, the official
[Q4_K_M–Q8_0 GGUFs](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) are the right pick.

- Engine: https://github.com/avifenesh/memra (MIT, crates.io: `memra-server`)
- Context: 262,144 native; chat template embedded (XML tool calling + `<think>` reasoning)
- Vocabulary: 248,320 (text→text serving; the upstream vision tower is not in this artifact)

## Provenance and verification

| | |
|---|---|
| Base | `ornith-ai/Ornith-1.5-35B-A3B` (MIT), official BF16 GGUF (`qwen35moe`, 41 blocks incl. NextN) |
| Head training | `mtp.*` only, continued-trained on the model's own generations (4,044 prompts, ~2.1M sampled tokens at the vendor serving temperature), depth-3 chain-rollout, frozen trunk; recipe + receipts: [memra `research/ornith15-mtp-train-20260820/`](https://github.com/avifenesh/memra) |
| Quantization | `llama-quantize` NVFP4 ftype (imatrix-aware branch), `--output-tensor-type q5_k --token-embedding-type q5_k`; head quantized after training (NVFP4 head measured at zero acceptance cost on this pipeline) |
| Main file | `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`, 20,188,038,400 B, sha256 `72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3` |
| Exactness | memra batteries, 2026-08-20: `run-spec` K=1..8 self-consistency PASS (spec ≡ plain greedy, token-identical) · chat-templated generation probes coherent · serve-level A/B accounted per-request (`usage.spec`) |

## Masked MTP draft (FR-Spec)

`mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768.gguf` (0.9 GB, sha256 `46f0dd4c…73fe899`) —
standalone speculative-decode draft from the trained MTP block: lm head masked to the
top 32,768 of 248,320 rows, ranked by the model's **own generations** (external text used as
prompts only), then requantized in the mask-first order — NVFP4 head, Q4_K_M block. Ranks ship
as `ornith15-ranks-owngen-32768.txt` (drives memra's load-time head mask on safetensors trunks
via `MEMRA_FRSPEC_TRIM`) and as `.gguf`. A mask can never change output; it moves draft
acceptance (measured 0.393 vs 0.431 embedded at K=3) against a smaller head read.

## Serving posture

At these acceptance rates speculative decode still trails this model's very fast plain A3B
decode end-to-end (plain ~196 tok/s vs spec ~84 on the measurement card, single-stream), so
memra serves Ornith-1.5 spec-off by default; the trained head narrows the gap (+29% relative
acceptance, +6% spec throughput vs the vendor head) and the draft artifacts are published for
engines and workloads where the trade differs.

Serve with [memra](https://github.com/avifenesh/memra):

```bash
# plain (default serving mode for this model)
memra-server --model Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf

# speculative, masked draft head
MEMRA_MTP_DRAFT=mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768.gguf \
memra-server --model Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
```

Built with `tools/make-trimmed-draft.sh` + `frspec-owngen` from the memra repo.
