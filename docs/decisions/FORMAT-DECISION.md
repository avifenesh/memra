# FORMAT-DECISION.md — what memra optimizes for (updated 2026-08-15)

**User's criterion (verbatim intent):** the decision is NOT what we already built and NOT the
current daily models — models change daily, the RIG doesn't. Optimize for the fastest possible
inference on THIS rig (sm_120a, 24GB, RTX 5090 Laptop, this CPU), for ANY model that may become
the strongest local candidate — including a future spilled/REAP'd MoE running CPU+GPU. Multi-month
rework is acceptable; this is a long-running inference project, not a weekend build.

## Decision: official safetensors semantics in; RIG-NATIVE layouts inside

Neither GGUF blocks nor HF/modelopt safetensors layouts are the optimization target. The target is
a **memra-internal weight layout chosen per weight-class by what the sm_120a datapath wants**, with
BOTH file formats repacked into it at load (one-time cost, irrelevant at serve time).

For new upstream models, the preferred semantic source is the official safetensors checkpoint as
a complete model family: tensor shards, `config.json`, tokenizer and chat template, quantization
metadata, and any auxiliary MTP/draft safetensors. This preference is about fidelity, not file
extension. The loader must inventory the complete set, preserve each tensor's declared class and
scale program, refuse duplicate or missing names, and fail closed rather than substitute another
activation, weight, KV, or compute format.

Why not GGUF-first: GGUF block layouts were designed for llama's dp4a/MMQ kernels. The FASTEST
sm_120a path for 4-bit is the mxf4nvf4 block-scale tensor-core mma (762 TFLOPS), and its operand
layout is NOT GGUF's 36-byte block_nvfp4. External implementations are research controls for that
operand contract; the shipped repack, swizzle, and CUDA kernels are Memra-owned. NVIDIA's native
NVFP4 releases (modelopt, faster than community GGUF quants of the same models) land in ST form
first; a GGUF-internal engine converts them THROUGH an unnecessary layout hop.

Why safetensors is not the internal layout: safetensors is a safe tensor container, not a complete
model program or a runtime layout. modelopt packing is closer to what the tensor core wants for
PREFILL, but decode (matvec, bandwidth-bound) and CPU-side spilled experts (AVX/AMX dot on the host
CPU) each have their own optimal layouts. The config, tokenizer/template, and quantization metadata
remain part of the semantic source. No file format is the compute answer; the RIG's three
datapaths are:

| datapath | optimal internal layout | today's state |
|---|---|---|
| prefill GEMM (tensor-core mxf4nvf4 / int8 mma) | operand-order + swizzled scale factors exactly as `mma.sync`/TMA consume them (modelopt-shaped for FP4) | GGUF blocks + per-load repack for Memra-owned tensor-core kernels; MMQ reads GGUF raw (llama-shaped, at parity) |
| decode matvec (HBM-bound) | coalesced per-warp block reads — current GGUF-raw layout is ALREADY at llama parity (42% DRAM, SOL) | GGUF raw — keep until a measured better layout exists |
| CPU-spilled experts (future MoE) | host-page-aligned, mmap-direct, dtype the CPU dots fast (int8/bf16 rows) | not built |

## Execution order (does NOT invalidate current work)

1. **Preserve current GGUF behavior and gates.** Existing optimized artifacts do not regress, and
   GGUF remains a portable self-contained import/export surface.
2. **Start new model onboarding from the official safetensors model family.** Freeze the source
   revision and build an exhaustive manifest across indexed shards and auxiliary files. Missing,
   duplicate, unclassified, or unsupported tensors stop the load.
3. **Loader unification (already 80% built):** `GpuTensor` is the abstraction; safetensors→repack
   at load already exists (modelopt NVFP4 → engine blocks, "no kernel change"). Formalize: every
   loader produces INTERNAL layout, tagged per weight-class; GGUF-raw is one internal layout among
   several, not the definition.
4. **Per-class migration only WITH measured wins** (the standing empirical rule): when a prefill
   FP4 kernel on modelopt-shaped operands beats the MMQ-on-GGUF floor interleaved, that class
   flips its internal layout and the GGUF loader gains the (already-written) repack for it. Decode
   stays GGUF-raw until a layout beats 42% DRAM.
5. **Spill tier (MoE arc)** designs against ST shards mmap'd directly (no conversion on the cold
   path), host-side layout chosen by CPU dot benchmarks — this is where ST-native pays off first.
6. **Dual support is the end state.** Safetensors is preferred for upstream semantic ingestion;
   GGUF remains a portable import and distribution format. The engine's identity is the
   rig-native layout, not either container.

## What this changes about ongoing optimization

- Tune-data records gain a `layout` field going forward (GGUF-raw vs repacked variants) so the
  autotuner learns layout as a dimension.
- New kernels are written against an OPERAND SPEC (what bytes in what order), not "the GGUF
  block" — provenance comments say which container maps to it and how.
- The FP4 prefill rebuild (PREFILL-GEMM-REBUILD.md) targets the tensor-core-native operand layout
  FIRST and treats the GGUF repack as the import step, not the other way around.
