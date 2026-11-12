# Qwen3.8-Flash-Next

Qwen's 3.8 Flash Next mixture-of-experts, in bring-up on a hand-written gate path. It has no
serving surface.

| | |
|---|---|
| **Serving route** | none |
| **Pack** | `qwen4_exp`, support `None`: its tiny plan executes in the reference executor, but no record claims the family |
| **Artifacts** | source `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`; NVFP4 mint [`tiyuvta/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/tiyuvta/Qwen3.8-Flash-Next-NVFP4) (the receipts name its earlier path `Avifenesh/Qwen3.8-Flash-Next-NVFP4`; no revision is pinned in them) |
| **Hardware** | RTX PRO 6000 Blackwell for the gate binaries |
| **Input** | text |

Support state: bring-up only, not yet NativeReference. <!-- support: none; not NativeReference -->
A serving route needs the NativeQualified gate set. <!-- support: none; not NativeQualified -->

## What is gated

- Source geometry, read from checkpoint headers rather than by analogy to a sibling: 48 text layers
  (36 GDN to 12 QSA at interval 4), a 512-expert softmax top-10 MoE plus a sigmoid-gated shared
  expert on every layer, a 51B n-gram/PLE block, a 1-layer QSA MTP head, vocab 248,320 and a
  262,144-token native window (1M by YaRN)
  ([`research/qwen4exp-bringup-20260829/ARCH.md`](../../research/qwen4exp-bringup-20260829/ARCH.md)).
- The NVFP4 mint is experts-only (9 shards, 174 GB) with the BF16 MTP head grafted back in. The
  upstream modeling code carries no MTP module, so a PTQ pass without the graft ships no draft head.
- Real-checkpoint gate
  ([`REAL-CHECKPOINT-GATE.md`](../../research/qwen4exp-bringup-20260829/REAL-CHECKPOINT-GATE.md)):
  the 360 GB BF16 and the NVFP4 mint both run final-logits argmax 10/10 against independent
  goldens, and the 64-token greedy chains match the HF bf16 chains 64/64 on prompt 0 (both arms) and
  on prompt 2 (NVFP4), with the remaining divergences late.
- Engine path: hand-written `crates/memra-engine/src/qwen4exp_gpu.rs` (gate binaries
  `qwen4exp_real_gate`, `qwen4exp_gpu_gate`).

## Limits

Not certified: sampled serving (greedy is the instrument), MTP and spec execution, vision, long
context (over 2,051 tokens, where the indexer prunes on real geometry), batching, multi-GPU, the
tokenizer and template serving surface, and any performance or context claim. Two-card expert
parallelism was designed and judged unable to reach its target
([`research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md`](../../research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md)).

Nothing here transfers to a sibling family by analogy.

## Deeper sources

- Lane: [`research/qwen4exp-bringup-20260829/`](../../research/qwen4exp-bringup-20260829/).
- Pack: `crates/memra-gguf/src/model_packs/qwen4_exp/`.
