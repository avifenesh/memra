# FP4 clears the local component-cost threshold

Issue [#439](https://github.com/avifenesh/memra/issues/439). On one RTX 5090 Laptop
GPU, calibrated FP4 activation quantization plus GEMM is **2.500x** the INT8
throughput for the 272-slot input group at 4096 rows. This clears the issue's
approximately 1.25x engineering-value test. The full-program quality negative
remains unchanged.

## Measured result

Ten rounds per arm and shape, five AB and five BA, 64 launches per timed sample.
The aggregate divides the sum of slot-weighted INT8 median times by the
corresponding FP4 sum. It is not an average of per-class speed ratios.

| Rows | Input-group ratio | AB only | BA only | Conservative observed range |
|---:|---:|---:|---:|---:|
| 512 | 2.361x | 2.351x | 2.368x | 1.942x to 2.666x |
| 2048 | 2.451x | 2.423x | 2.454x | 2.015x to 2.811x |
| 4096 (primary) | **2.500x** | 2.475x | 2.509x | 2.248x to 2.719x |

The observed range pairs each class's fastest INT8 sample with its slowest FP4
sample for the lower bound, and reverses that for the upper bound. It is a
conservative description of these samples, **not a statistical confidence
interval**. Both order subsets and the primary lower bound clear 1.25x.

At 4096 rows, the weighted component-time sums are 1747.923 ms for INT8 and
699.095 ms for FP4 across the input group. These are sums of representative
projection timings, not a measured model prefill. The all-400-slot component
ratio is 2.464x; that does not rescue the rejected full A4 quality program.

| Class | Slots | K x N | INT8 median ms | FP4 median ms | Ratio |
|---|---:|---|---:|---:|---:|
| attn_gate | 48 | 5120 x 6144 | 3.062873 | 1.321591 | 2.318x |
| attn_k | 16 | 5120 x 1024 | 0.681676 | 0.353205 | 1.930x |
| attn_output | 16 | 6144 x 5120 | 3.083475 | 1.308785 | 2.356x |
| attn_q | 16 | 5120 x 12288 | 6.039613 | 2.611145 | 2.313x |
| attn_qkv | 48 | 5120 x 10240 | 5.066115 | 2.008670 | 2.522x |
| attn_v | 16 | 5120 x 1024 | 0.674263 | 0.352480 | 1.913x |
| ffn_down | 64 | 17408 x 5120 | 9.367976 | 3.902167 | 2.401x |
| ffn_gate | 64 | 5120 x 17408 | 9.617083 | 3.764454 | 2.555x |
| ffn_up | 64 | 5120 x 17408 | 9.748588 | 3.832006 | 2.544x |
| ssm_out | 48 | 6144 x 5120 | 3.510131 | 1.488609 | 2.358x |

## What the cell covers

The complete GGUF hash matches the original study:
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
The census validates all 400 frozen scale records against actual rank-two NVFP4
trunk tensors, groups them into ten uniform shapes, and finds 272 input slots.
Their MAC share is 0.6827586207, matching the issue's rounded 68.3%.

Each class uses its earliest-layer real weight tensor, its frozen v3-amax
multiplier, and deterministic synthetic F32 inputs in [-2,2). The same resident
input and split-plane weight bytes feed both existing FFI entrypoints:
`memra_mmq_nvfp4_w4a8` and `memra_mmq_nvfp4_calibrated_prefill`.

CUDA-event timings include activation quantization and GEMM. Allocations,
uploads, repacking, output checks and file I/O are outside the timed interval.
Scratch uses the separate size helper for each layout. Both arms get 20 warm
launches before the ten measured rounds. Host wall timings are retained as well.

All 30 class/row cells passed: finite, nonzero, distinct arm outputs; full output
hashes stable before and after measurement; and invalid-scale rejection 2902.
The raw table contains all 600 samples. The GPU and CPU censuses match exactly.
These are execution and fixture controls, not model-quality or support gates.

## Hardware and scope

- RTX 5090 Laptop GPU, 24463 MiB, driver 595.91.07, CUDA 13.1.
- Observed telemetry: 64 to 89 C, SM clock 1515 to 2055 MHz, memory clock 14001 MHz, power 31.01 to 179.28 W. Sampling interval 250 ms, 358 samples.
- Absolute timings varied. At 4096 rows, per-arm class ranges span roughly 6% to 30% of their medians. The balanced order results and conservative aggregate bounds remain well above the threshold.
- One exclusive GPU cell, 1200-second limit, 8192 MiB free-VRAM admission. It finished successfully in about 90 seconds. The sampled memory-use maximum was 1405 MiB. The full model was never loaded on the GPU.
- Source base `2873dd4ca37faa15cd4261e7b398926cced30106`, plus the standalone benchmark and its bin registration. Benchmark-source, binary, model, scale-table and representative-weight hashes are retained.

This measures repeated resident projection operations with synthetic inputs.
Grouped projection reuse, cache effects across the whole model, attention,
nonlinear operations, KV work, TTFT and model quality are outside this cell.
It does not establish a 2.50x model speedup or a PRO 6000 result.

No kernel, library implementation, runtime dispatch, default, calibration,
qualification tolerance or support record changed. The initial benchmark build
caught use of a private allocation helper; the final code uses the public stream
allocator. Targeted release Clippy and build pass, and the real GPU cell passes.
The topic push is development work with no native release qualification.

## Reproduce and decide

Build `a4-component-bench` in a private target directory with
`MEMRA_CUDA_ARCH=120a`. Run the CPU census before taking a GPU lease:

```sh
a4-component-bench MODEL.gguf scales.tsv census --census
```

Under the rig's exclusive resource controller, run the component cell with the
same pinned model and scale table, then analyze the complete output:

```sh
MEMRA_RP=1 a4-component-bench MODEL.gguf scales.tsv run1
python3 analyze.py run1 --out summary.json
```

The component economics justify the next bounded question. [QUALITY-ABLATION.md](QUALITY-ABLATION.md)
prepares a maximum of 28 fresh-process passes across the existing four windows,
with exact token-position identity, baseline repeats, the full-program negative
control and four class ablations. It requires NLL, KL and top-one reporting.
That plan has not run. The owner must choose its quality decision rule and
approve execution. No reduced-set implementation, recalibration, default flip,
program closure or cross-hardware promotion is included here.
