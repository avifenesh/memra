# Engine-basics sweep: what the other stacks know that our NVFP4 TP2 lane should steal

Owner directive (2026-08-20): "research other inference engines and papers to have the basics
of the engine, then we can improve after we took the best of all." Scope: exactly the three
subsystems this lane is building — TP decode transport on PCIe P2P, NVFP4 MoE execution on
SM120, and the stream/event discipline around both. Sources verified 2026-08-20.

## 1. Decode-transport: PCIe one-shot allreduce beats everything at our message sizes

Source: local-inference-lab/rtx6kpro `optimization/pcie-oneshot-allreduce.md` (Luke Alonso's
SGLang kernel, commit d39236ae), measured on 4x RTX PRO 6000 (our exact card class, PCIe,
no NVLink), Qwen3.5-397B-A17B-NVFP4 + MTP.

- TP decode allreduces are 16-256 KB/message. NCCL ring overhead dominates there.
- Design: every GPU writes its shard directly into peer-visible buffers on all peers
  (cudaMemcpyPeer-free: direct P2P stores from the kernel), ONE system-scope CUDA barrier,
  then each GPU reduces locally. Double-buffered to kill the end barrier. Optional fused
  allreduce+RMSNorm epilogue. Auto-crossover benchmark at boot picks custom-vs-NCCL threshold
  (measured crossover 512 KB; 120 KB is the conservative 4-GPU setting).
- Measured latency: 6.1-11.8 µs for 1-64 KB vs NCCL 13.2-71.2 µs (1.9-6.0x). End-to-end
  single-user decode +11.3% (67.3 -> 74.9 tok/s). Same-NUMA only; cross-socket 8-GPU it LOSES
  to NCCL (system-scope barrier over IF too expensive).
- **Adopt for memra rung 2**: our per-layer TP2 combine payloads are exactly this class
  (down-partial reduce 16 KB, QKV/O gathers similar). Replace the host-bounce (2x PCIe + host
  add per op, ~100s of µs) with: rank kernels write partials into pre-registered peer-visible
  buffers -> system-scope release/acquire flag -> root reduce kernel. We already run NCCL-free;
  the one-shot design is the native end-state of our own bulk-P2P direction, minus the
  host round-trips. Double-buffer per layer. Fold the post-layer RMSNorm into the reduce
  epilogue as their fused variant proves out.

## 2. NVFP4 MoE kernels on SM120: the dequant class WINS at decode; native FP4 is a minefield

Source: flashinfer-ai/flashinfer issue #2723 (Mar 2026, 4x RTX PRO 6000 SM120, Qwen3.5-397B
NVFP4 — 512 experts + 1 shared, 10/token: our model's architecture cousin), plus vLLM PR
#40082 (b12x fused_moe for SM120), CUTLASS #2820/#2800.

Their measured single-user decode ladder (TP2+PP2 over 4 cards):

| backend | class | tok/s |
|---|---|---|
| Marlin | W4A16 dequant-in-kernel | **46-49** |
| FlashInfer CUTLASS + CUDA 13.0 `compute_120f` | native FP4 TMA grouped GEMM | 39.0 |
| FlashInfer CUTLASS + CUDA 12.8 `compute_120a` | native FP4, TMA tactics fail -> fallback | 14.6 |
| vLLM native CUTLASS grouped block-scaled FP4 | — | **garbage output** |

Findings that bind us:
- **At decode (bs=1, weight-bound), the dequant/dp4a class beats native FP4 tensor cores.**
  memra's `qmatvec_nvfp4_dp4a` + q8_1 activations is exactly this class — our kernel choice is
  validated by the best community result, not just our own history.
- CUTLASS grouped block-scaled FP4 GEMM is silently WRONG on SM120 (`compute_120`/`120a`
  templates) — the exact fluent-garbage failure class our laws exist for. Never adopt it
  blind; any native-FP4 prefill arm must be CUDA 13.x + `compute_120f` (13.0+ only), which is
  what finally made TMA WS grouped tactics work (14.6 -> 39 tok/s).
- SM120 is NOT SM100: different capability family, `a`/`f` suffix semantics, SM100 cubins
  crash. Any vendor kernel gated on `is_device_capability_family(100)` needs real SM120
  qualification, not a check-patch.
- Env quirks on this card class under vLLM: `NCCL_CUMEM_ENABLE=0`, spawn workers. (Their
  stack; noted in case we A/B against vLLM on-box.)

**Adopt for memra rung 1**: keep dp4a class for decode; move from per-expert loop to the
pointer-array batched launch (`qmatvec_nvfp4_batched_raw` exists) so one launch covers the
~10 routed experts per layer per rank; keep activations device-resident across
gate/up->act->down within a layer. Native-FP4 GEMM (`compute_120f`) is a PREFILL-only
candidate, later rung, behind our own raw-bit gates.

## 3. Roofline honesty: what >100 tok/s means against the field

- Our pair: 2x1.79 TB/s; Step-3.7-Flash active bytes/token ~= 3.7 GB NVFP4 experts +
  ~8.8 GB BF16 (attention + shared + dense) ~= 12.5 GB -> ~286 tok/s roofline.
- Best community single-user on the CLASS (bigger model, 4 cards, TP2+PP2): 46-49 tok/s
  (~15% of their roofline). vLLM on OUR exact model/cards: 94.6 tok/s (~33% of roofline).
- So >100 pre-MTP = beating vLLM's stack by ~6%+ at the same roofline fraction the best
  stacks reach. The levers that get us there vs their stack: no python/host round-trips
  (memra native), one-shot P2P transport (their own +11% lever), fused dp4a batched experts,
  and BF16 attention already rank-local from the FP8 arc. MTP rides on top (+26-45% at
  concurrency per vLLM's own GB200 NVFP4+MTP data; our frspec history says more at bs=1).

## 4. Stream/event discipline (the flake class we just hit)

- vLLM custom allreduce and Luke's one-shot both rely on EXPLICIT system-scope barriers or
  event ordering between producer stream and peer reader. memra's pp.rs does this right
  (evented TX/RX per boundary, publishing-stream copies); the step TP native collectives
  skipped producer fences in four places (fixed this lane, commit pending validation) —
  `bf16_column_parallel_resident_device_shards` broadcast, `gather_native_column_shards_device`,
  `step_bf16_row_parallel_resident_native` input scatter + partial collect.
- Rule to enshrine: **every cross-stream peer read fences its producer** (event preferred,
  stream-sync acceptable at decode sizes), and keepalives cover frees, not ordering.
- vLLM issue: custom allreduce HANGS on PCIe multi-NUMA Blackwell (their P2P/shm assumptions);
  our boxes are single-NUMA pairs — keep it that way for TP pairs, and keep the topology
  check in preflight (NODE/PIX, same NUMA) as a serving requirement.

## Sources
- rtx6kpro pcie-oneshot-allreduce.md (fetched 2026-08-20; benchmarks + design)
- flashinfer#2723 SM120 NVFP4 MoE debug journey (fetched 2026-08-20; full patch list + ladder)
- vLLM PR #40082 (b12x fused_moe SM120), CUTLASS #2820/#2800, vLLM forums Blackwell PCIe
  custom-allreduce hang thread, docs.vllm.ai moe_kernel_features
- Millstone AI 2x RTX PRO 6000 Step-3.7-Flash NVFP4 benchmark (earlier today, floor doc)
