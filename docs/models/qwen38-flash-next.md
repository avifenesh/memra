# Qwen3.8-Flash-Next

Support state: **NativeReference** (bring-up). The eager engine path is gated at real
checkpoint scale; the ModelPlan loader path is not wired, there is no serving surface, and
production admission still requires the NativeQualified gate set.

## What is gated

- Source: `Qwen/Qwen3.8-Flash-Next@de4b8e4d` — 48 text layers (36 GDN : 12 QSA at interval
  4), 512-expert softmax-top-10 MoE plus a sigmoid-gated shared expert every layer, a 51B
  n-gram/PLE block, a 1-layer QSA MTP head, vocab 248,320, 262,144-token native window
  (1M YaRN). Geometry was read from checkpoint headers, never by analogy to a sibling
  (`research/qwen4exp-bringup-20260829/ARCH.md`).
- Artifact: experts-only NVFP4 mint, BF16 MTP grafted byte-exact
  (`tiyuvta/Qwen3.8-Flash-Next-NVFP4` on Hugging Face). The upstream modeling code carries
  no MTP module, so any PTQ pass without the graft ships no draft head at all.
- Real-checkpoint gate
  (`research/qwen4exp-bringup-20260829/REAL-CHECKPOINT-GATE.md`): the 360 GB BF16 and the
  NVFP4 artifact both run final-logits argmax 10/10 against independent goldens, and the
  NVFP4 arm matches the BF16 greedy chain 64/64.
- Engine path: hand-written `crates/memra-engine/src/qwen4exp_gpu.rs` (gate binaries
  `qwen4exp_real_gate`, `qwen4exp_gpu_gate`). The `qwen4_exp` model pack is
  inspection/census only — no native-plan execution.

## What this does NOT certify

Sampled serving behavior (greedy is the instrument), MTP/spec execution, vision, long
context (>2051-token indexer pruning on real geometry), batching, multi-GPU, the
tokenizer/template serving surface, or any perf/context product claim.

Nothing above transfers to any sibling family by analogy.
