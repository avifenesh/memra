# qwen4_exp decode PROFILE-0 — untuned eager NVFP4 arm, before any optimization (2026-08-29)

Perf lane phase 1 deliverable. Box: cloud-eval frankfurt, 2× RTX PRO 6000 Blackwell 96 GB
(sm_120a); single-card arm (GPU0), NVFP4 mint `~/data/q48fn-nvfp4`, memra
qwen4exp-bringup-20260829 @ a70f8a1ec8 (profiler instrumentation commit; forward math
identical to the REAL-CHECKPOINT-GATE run @ 0c57fc75ea). Greedy is the instrument here
(self-fed argmax, real goldens prompt, bounded steps) — never a serving claim.

## Baseline (unprofiled, warm)

```
binary: target/release/qwen4exp_real_gate  sha256=3e73cd75e72dfa95fff0d457ea9c342055c1d159e35698526f6ba335f88b3f1e
invocation: ./target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/perf0 \
    --label nvfp4 --goldens ~/realgate/dump --decode-timing 80 --profile 64
# decode_timing  steps=80  mean_ms=78.5  median_ms=78.5  p90_ms=78.6  tok_per_s_untuned=12.74
```

**78.5 ms/token = 12.74 tok/s** warm (self-fed greedy decode after the T=10 goldens
prefill; first post-prefill step excluded). p90 78.6 — the step time is flat, so this is a
launch/issue-bound program, not a variance story. Owner target ~90 tok/s ⇒ 11.1 ms/token,
i.e. a **7.1× gap**. Receipts: `profile0-nvfp4.tsv`, `run-profile0-nvfp4.log`.

## Per-section wall profile (64 warm steps, T_kv 94→158)

Method: `--profile 64` enables `qwen4exp_gpu::prof`, which synchronizes the stream at every
section boundary so a section's wall covers everything it queued. The sync boundaries
inflate the step total (115.6 ms profiled vs 78.5 ms unprofiled, 108.2 ms attributed), so
**shares are the signal and the unprofiled 78.5 ms is the absolute.** ms/token below is the
profiled attribution, not a claim about the unprofiled step.

| section | calls/token | ms/token | % attributed |
|---|---|---|---|
| **moe.dequant** | 480 | **31.11** | **28.8** |
| **hyper.read** | 96 | **30.01** | **27.7** |
| **moe.expert_gemms** | 480 | **17.21** | **15.9** |
| moe.idx_gather | 480 | 7.96 | 7.4 |
| gdn.proj | 36 | 5.12 | 4.7 |
| gdn.norm_gate_out | 36 | 2.50 | 2.3 |
| moe.shared | 48 | 1.99 | 1.8 |
| gdn.conv_scan | 36 | 1.90 | 1.8 |
| lm_head | 1 | 1.67 | 1.5 |
| hyper.write | 96 | 1.65 | 1.5 |
| qsa.proj | 12 | 1.48 | 1.4 |
| moe.router | 48 | 1.46 | 1.3 |
| qsa.sdpa | 12 | 1.11 | 1.0 |
| qsa.gate_wo | 12 | 0.76 | 0.7 |
| moe.reduce | 48 | 0.66 | 0.6 |
| qsa.idx_host (host twin) | 12 | 0.47 | 0.4 |
| qsa.idx_proj | 12 | 0.38 | 0.3 |
| ple.key_gate | 1 | 0.23 | 0.2 |
| exit.mixer | 1 | 0.13 | 0.1 |
| ple.conv_write | 1 | 0.12 | 0.1 |
| qsa.mask_h2d | 12 | 0.11 | 0.1 |
| logits.dtoh | 1 | 0.09 | 0.1 |
| entry.embed | 1 | 0.04 | 0.0 |
| ple.host_ngram_gather | 1 | 0.02 | 0.0 |
| ple.h2d | 1 | 0.01 | 0.0 |

**MoE is 54.6% of the attributed token** (dequant + expert GEMMs + index/gather + router +
reduce + shared), and the hyper-connection gates are another 29.2% (read 27.7 + write 1.5).
Everything the bring-up lane flagged as scary is small: QSA attention 3.4% total, the host
indexer twin 0.4%, the whole PLE block (host n-gram hash + 102 GB-table gather + H2D +
device conv) 0.4%, lm_head 1.5%.

## Launch census (nsys, exactly 8 warm decode steps)

```
invocation: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda \
    -o nsys-decode8 ./target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/perf0 \
    --label nvfp4-nsys --goldens ~/realgate/dump --decode-timing 10 --profiler-window
window: cuProfilerStart at decode step 2 → cuProfilerStop after step 9 (8 steps)
```

| API / op | per token | note |
|---|---|---|
| **kernel launches total** | **15,308** | cuLaunchKernel 9,214 + cudaLaunchKernel 5,634 (cuBLAS) + cuLaunchKernelEx 460 |
| cuMemAllocAsync / cuMemFreeAsync | 11,366 each | every eager temporary is a pooled alloc |
| cuMemsetD8Async | 1,685 | `e.zeros` per temporary |
| cuMemcpyHtoDAsync | 1,568 | per-expert index/weight uploads dominate |
| cuMemcpyDtoHAsync | 65 | 48 router logits + 12 indexer proj + logits |
| cuStreamSynchronize | 65 | one host boundary per dtoh |

Per-token kernel instances confirming the MoE story: `dsv4_nvfp4_deq_bf16_kernel` **1,440**
(= 480 routed experts × 3 projections), `bf16_to_f32` **1,440**, `scale_f32` 2,402 (macro
folds + gate scales), cuBLAS `gemvx` variants ~4,300. GPU-side kernel time is dominated by
gemvx (21.8% + 9.5% + 8.3% + 3.2%), the NVFP4 dequant kernel (16.8%) and its `bf16_to_f32`
twin (11.8%).

**This is the 27B lesson, one order worse for this family**: the deficit is not one hotspot
but ~15.3k launch boundaries and ~11.4k pooled allocations per token, each 2-4 µs of issue
latency. 48 MoE layers × 10 routed experts × (1 dequant + 1 upcast + 1 macro-scale + 3
GEMVs + gather + scatter + 3 small H2D) is ~4.8k of them by itself.

## Reading, per attack

- **(a) per-expert dispatch is the big rock, as expected — but the dequant, not the math.**
  `moe.dequant` (31.1 ms, 28.8%) is pure overhead: 1,440 kernel launches per token to
  materialize f32 copies of weights that are already resident, plus 1,440 upcasts and the
  macro scale, so a routed expert's 640×2560 projection is read, expanded 8× in bytes, and
  thrown away every token. `moe.expert_gemms` (17.2 ms) is cuBLASLt GEMV on M=1 — the wrong
  instrument for a matvec. Both die with one grouped kernel over the as-stored bank.
- **(b) the host indexer twin is NOT a problem at these lengths** (0.47 ms/token, 0.4%) —
  and the structural reason is now a coded fast path: at T < 2051 every complete block is
  selected regardless of score. Device scoring stays deferred to the long-context lane.
- **(c) hyper gates are the second rock** (30.0 ms/token, 27.7%, 96 read-gate calls/token):
  per gate, 4 RMSNorms + 4 rank-320 down GEMVs + 4 up GEMVs + 4 sigmoids + 4 muls + 4 axpys
  + 4×4 inject GEMVs — ~50 launches × 96 = ~4.8k launches/token. Fusion candidates in
  measured order: the inject GEMVs (16 per gate producing 4 scalars), then the per-stream
  norm/gate/mul/axpy chain.
- **(d) lm_head and PLE/H2D are not the problem** (1.5% and 0.4%). lm_head is one f32 GEMV
  over 248320×2560 in 1.67 ms ≈ 1.5 TB/s effective — near the card's f32 read bound, so
  the win there is a quantized head, not a launch fix. Nothing runs in fp32-by-accident.

Next: attack (a) grouped NVFP4 expert matvec, then (c) gate fusion. Both are launch-count
attacks, which is what the census says this decode step is.
