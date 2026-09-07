# Dense FP8 to BF16 packed-conversion R5 preflight, 2026-09-07

## Verdict

The native packed E4M3x2 -> BF16x2 path is not compile-available in the
current CUDA 13.1 `sm_120a` toolchain. The standalone preflight contains the
exact PTX candidate and an independent packed bit oracle, but no R5 kernel was
built or run.

The corrected source form was:

```ptx
cvt.rn.bf16x2.e4m3x2 d, a;
```

with a 16-bit packed E4M3x2 source register and a 32-bit packed BF16x2
destination. `ptxas` rejects it for the engine target with:

```text
error: Unexpected instruction types specified for 'cvt'
ptxas fatal: Ptx assembly aborted due to errors
```

The first trial with `.satfinite` was also rejected as an illegal modifier,
then the corrected no-`.satfinite` form reached the type rejection. A second
preflight targeting `sm_120f` with the same CUDA 13.1 assembler produced the
same type rejection, so this is not a safe path to price on the current build
toolchain.

## Source and basis

`tools/dsv4-dense-tc-gate-r5.cu` is a new standalone preflight. Its `--basis`
path includes signed, zero, normal, and NaN-code E4M3 pairs and independently
constructs expected BF16 words, including the exact 16-bit E8M0 exponent
adjustment and zero handling. It was not executed because the compile gate
failed. R4 remains frozen and is the concrete alternative: vectorized `uint2`
K=128 staging, shared BF16 LUT, exact exponent adjustment, and four-warps per
64-row block.

```text
source d02c632c7f966f196bf8a86dcc0f9f9a726d74a26e41c167a4b29c9fba33942e
binary none (ptxas compile rejection)
```

## Compile command

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a \
  -o target/dsv4-dense-tc-gate-r5 tools/dsv4-dense-tc-gate-r5.cu
```

## Current primary NVIDIA references

- [PTX ISA 9.3 conversion syntax and target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
  documents `cvt.rn{.satfinite}{.relu}.bf16x2.e4m3x2` as a family-specific
  conversion introduced in PTX ISA 9.2, with support listed for `sm_120f`.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
  remains the source for the target architecture and memory rules; no power
  or clock attribution is made here.
