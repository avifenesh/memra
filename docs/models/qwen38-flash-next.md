# Qwen3.8-Flash-Next

Support state: **bring-up only** — not yet NativeReference, because the native plan does
not execute: the `qwen4_exp` model pack is inspection/census only (`support: None`), and
the evidence below lives on the hand-written gate path. There is no serving surface, and
production admission still requires the NativeQualified gate set.

## What is gated

- Source: `Qwen/Qwen3.8-Flash-Next@de4b8e4d` — 48 text layers (36 GDN : 12 QSA at interval
  4), 512-expert softmax-top-10 MoE plus a sigmoid-gated shared expert every layer, a 51B
  n-gram/PLE block, a 1-layer QSA MTP head, vocab 248,320, 262,144-token native window
  (1M YaRN). Geometry was read from checkpoint headers, never by analogy to a sibling
  (`research/qwen4exp-bringup-20260829/ARCH.md`).
- Artifact: `Avifenesh/Qwen3.8-Flash-Next-NVFP4` — an experts-only NVFP4 mint (9 shards,
  174 GB) with the BF16 MTP head grafted back in; the upstream modeling code carries no
  MTP module, so a PTQ pass without the graft ships no draft head at all.
- Real-checkpoint gate
  (`research/qwen4exp-bringup-20260829/REAL-CHECKPOINT-GATE.md`): the 360 GB BF16 and the
  NVFP4 artifact both run final-logits argmax 10/10 against independent goldens, and the
  64-token greedy chains match the HF bf16 chains 64/64 on prompt 0 (both arms) and 64/64
  on prompt 2 (NVFP4), with the remaining divergences late.
- Engine path: hand-written `crates/memra-engine/src/qwen4exp_gpu.rs` (gate binaries
  `qwen4exp_real_gate`, `qwen4exp_gpu_gate`).

## What this does NOT certify

Sampled serving behavior (greedy is the instrument), MTP/spec execution, vision, long
context (>2051-token indexer pruning on real geometry), batching, multi-GPU, the
tokenizer/template serving surface, or any perf/context product claim.

Nothing above transfers to any sibling family by analogy.
