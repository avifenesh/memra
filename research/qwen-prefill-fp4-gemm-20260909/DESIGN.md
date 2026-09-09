# SM120 FP4 trunk GEMM: pre-integration checkpoint

Base: memra 182819614874be818ff8bef0cfaf13cb3b8051e1. Chunk 1024; native SM120a warp MMA, unchanged NVFP4 weight bytes. No runtime integration at this checkpoint.

The first cell compares the current W4A8 C ABI against the existing K64 FP4 C ABI and a Memra-owned fused activation quantizer on the same binary. Include quantization in measured latency. Report raw interleaved samples, M/latency input tokens/s, effective operations/s and max absolute and max-normalized output deviation. Synthetic operands establish throughput and layout only, never model quality.

The candidate quantizer computes per-token amax and per-16 UE4M3 scales, packs with native round-to-nearest E2M1 conversion, and can fuse SiLU(gate)*up before quantization. The GEMM continues to consume the existing packed scale/operand layout. This is a new numerical program, including a different micro-scale choice from the existing five-candidate search.

## Prior arms deliberately skipped

- `research/prefill-gemm-20260806/VERDICT.md`: deleting the complete scale fold gained 3.17% end to end, not the predicted 44.7%. Do not rebuild the integer scale-fold chain.
- Same verdict and `research/prefill-ilp-20260806/VERDICT.md`: K16 accumulates into RZ already; extra independent accumulators do not repair a dependency that is absent. Do not repeat occupancy-only or staging-only experiments on that instruction.
- `research/w4a8-prefill-20260806/VERDICT.md`: plain f8f6f4 has half the issue rate of the block-scale form. It is not the FP4 K64 path.
- K64 mxf4nvf4 accepts E2M1 x E2M1, not q8. The q8-activation proposal cannot use this instruction directly. An FP8/K32 alternative is a separate program already studied in the W4A8 verdict.
- Prior FP4 precision refusals are not throughput refusals. The owner's 2026-09-09 quality and same-program consistency contract explicitly reopens this program. The failed top-channel correction is not repeated (`research/fp4-act-scoping-20260806/BRIEF.md`: 4/10 widened corpus).
- SM100 tcgen05 restrictions do not apply to SM120 warp MMA. Guard this arm by exact architecture when integrated.

## Integration gates, after owner checkpoint

Default OFF `MEMRA_PREFILL_FP4_TC`, decide-by 2026-09-23. OFF remains current W4A8; ON selects the qualified prefill program with rollback by unsetting the flag. Exact spec/plain K=1..8 on one binary under ON, zero argmax flips against OFF on roster prompts, four-turn greedy restore and cold/restored boundary identity, per-layer activation/logit deviation diagnostics, cheapest existing paired Qwen quality harness at the same seed, vendor-default sampled dspark-acc, interleaved cold 8k/32k/131k three boots/arm with nonce identity, c1 decode and eight-turn cache twin. Production configuration is derived from the private companion requalification JSON. No release or deployment.

Restore suffix and full-prime chunk geometries must use one deterministic numerical program. Batch-wide top-channel selection is excluded because it depends on which rows are grouped. The new quantizer is row-local.

Sources: NVIDIA PTX ISA, warp-level matrix instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/); SGLang v0.5.19 source and the private prior lane RESEARCH.md. No third-party implementation is added or linked.
