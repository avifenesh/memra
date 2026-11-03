# Subsystem write-ups

One write-up per slice of the repo, written by that slice's annotation agent from the code (citations are file:line at the map commit 5ba8a3b11a). They describe hot paths per family and route, device-specific parts, effect kinds and where the impact set is wide. The narrowing claims they mention were checked separately: `annotations.json` carries each region's verdict (`verified`, `verified-family-only` or `refuted`), and only the verified form shapes `map.json`.

| slice | covers |
|---|---|
| [S01](S01.md) | engine lib.rs, first half (Engine core, module/kernel resolution, launch helpers) |
| [S02](S02.md) | engine lib.rs, second half |
| [S03](S03.md) | hybrid_forward.rs, first half |
| [S04](S04.md) | hybrid_forward.rs, second half + hybrid_forward/ |
| [S05](S05.md) | hybrid loader, model, decode, decode_batch, prime graph, parallel |
| [S06](S06.md) | DeepSeek-V4 GPU model (dsv4_gpu.rs) |
| [S07](S07.md) | DeepSeek-V4 rest: engine dsv4_* + FFI, gguf dsv4 forward/decode, server dsv4 route |
| [S08](S08.md) | qwen4exp GPU model and siblings |
| [S09](S09.md) | multi-card: tp, tp_ar, tp_transport, glm5_tp*, pp |
| [S10](S10.md) | speculative core: spec.rs, spec/, eagle |
| [S11](S11.md) | spec routes: dflash, glm_spec, gemma_spec, glm5 decode graph, mimo mtp |
| [S12](S12.md) | memory tiers: tier_transfer, moe_cache, banked_residency, spill, cpu experts, pinned host + memra-tier |
| [S13](S13.md) | engine FFI shims, vision, mimo (non-MTP), kda, and every other engine file |
| [S14](S14.md) | CUDA matvec/GEMM: qmatvec, qmatvec_gemm, mmq_*, fp8/f16 prefill, headers |
| [S15](S15.md) | CUDA attention and core kernels: flash_attn, mla_attn, fa3, fattn_vendor, kernels.cu |
| [S16](S16.md) | CUDA model-specific and support kernels: dsv4, hybrid, kda, moe, router, spec_sample, tp_ar, mimo, tier_receipt |
| [S17](S17.md) | server worker.rs, first half |
| [S18](S18.md) | server worker.rs, second half |
| [S19](S19.md) | server lib.rs and the rest of memra-server |
| [S20](S20.md) | memra-gguf (plan, tensor contract, packs, manifest, config, source) + memra-cli |
| [S21](S21.md) | host crates: reference, tokenizer, sampling, kv, lanes, runtime, validate, net-guard, probe |
| [S22](S22.md) | build and gates: build.rs files, battery binaries (kernel_check, run_gen, run_spec), release battery tools, validation_plan, fast-gate, #1030 tools |
