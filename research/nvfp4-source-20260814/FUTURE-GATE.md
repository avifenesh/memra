# Correctness-first FlashInfer SM120 admission gate

This is the required order for any future FlashInfer NVFP4 promotion. A failure stops the
campaign. Timing begins only after every correctness stage is green.

## 0. Freeze identity

Record before execution:

- FlashInfer release and exact commit;
- PyTorch, CUDA runtime, NVCC, driver, and CUTLASS/CuTe-DSL versions;
- GPU name, compute capability, and `FLASHINFER_CUDA_ARCH_LIST`;
- JIT/cubin cache state and the compiled target (`12.0f` for the SM120 candidate);
- source model/artifact hashes and generation manifest;
- candidate backend, scale layout, scale dtype, and all environment overrides.

Do not patch the compile target during the scored run. A patched exploratory build is a
different arm and needs its own manifest.

## 1. Quantizer-only gate

Run `nvfp4_quantize` followed by an independent dequantizer. Do not call GEMM.

Required historical regression matrix:

- `K=2048`, paired-context `N=8192`;
- `M={1,8,32,128,256,384,512,1024,1152,1280,1536,2048,3072,4096}`;
- `global_scale` supplied as BF16, FP16, and FP32;
- CUDA and CuTe-DSL quantizers;
- `SfLayout.layout_128x4` and `SfLayout.layout_linear`;
- shuffle off, then every layout/shuffle form the intended runtime will use.

Required real-shape extension:

- every `(K,N)` projection shape in the candidate model set;
- every decode/routed-row `M` from 1 through 256, without power-of-two sampling;
- every configured prefill/chunk M and observed routed-expert row count.

Pass criteria:

1. For a nonzero source tensor, an entire scale tensor may not be zero.
2. A zero block scale is allowed only when the corresponding source block is all zero.
3. Packed values, scale bytes, and dequantized output from a BF16/FP16 scale must be
   byte-identical to the control that promotes that same scalar value to FP32 before the
   call. This directly gates the #3398 dtype-reinterpretation class.
4. Dequantized values must be finite and non-collapsed.
5. Relative L2 error and median magnitude ratio must be frozen before the run against the
   independent CuTe-DSL/CPU reference. The CUDA result may not be worse than that reference
   envelope. Cosine alone is forbidden.

Persist one row per shape/dtype/layout/backend, including scale-zero fraction, relative L2,
median magnitude ratio, and hashes of packed values and scales.

## 2. Artifact-generation gate

For every candidate checkpoint tensor:

1. Start from the pinned BF16 source, never from a previously quantized artifact.
2. Record the generator commit/config and source tensor hash.
3. Dequantize the produced packed tensor without invoking the runtime GEMM.
4. Apply the same block-scale and numeric checks as stage 1.
5. Hash the final artifact and store the validation report beside its manifest.

An artifact made by a different tool is a separate arm. It is neither condemned nor
cleared by FlashInfer #3398.

## 3. GEMM-only gate

Generate A and B with the trusted stage-1 control, then hold their packed bytes and scales
fixed. Feed those exact inputs to:

- `mm_fp4(backend="b12x")`;
- `mm_fp4(backend="cutlass")`.

Compare each result with the BF16 matmul oracle and with the other backend. Require finite,
non-collapsed output and the predeclared NVFP4 error envelope. Store output hashes and full
error metrics. This stage must not call the candidate quantizer, so a failure belongs to
GEMM/dispatch rather than artifact generation.

## 4. Combined kernel gate

Run candidate quantization plus candidate GEMM over the full stage-1 shape matrix. Both
backends must pass. Repeat with PDL enabled and disabled wherever the API exposes it.

The combined result must agree with the composition of the separately accepted quantizer
and GEMM within the frozen tolerance. A new zero or magnitude shift is a hard failure.

## 5. Model and serving gates

Only after stages 1-4 pass:

- `kernel-check` against the existing oracle;
- `run-gen` same-prompt argmax gate;
- `run-spec` K=1..8 self-consistency;
- solo versus batched serving shapes;
- decode versus prefill/chunk transitions;
- graph versus eager and spec-verify versus plain where reachable.

No request may cross between different numerical programs unless the transition is proven
bit-identical by a serving-shape gate.

## 6. Performance admission

Benchmark only after the correctness report is sealed. Measure the current default and the
candidate in the same thermal/session regime on both the local RTX 5090 and an RTX PRO 6000
target. A one-rig result can set at most a one-rig policy.

Until all prior stages pass, the campaign verdict is **NO-GO** and the current runtime
default remains unchanged.
