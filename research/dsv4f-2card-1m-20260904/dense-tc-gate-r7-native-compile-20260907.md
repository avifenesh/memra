# Dense FP8 to BF16 native full projection R7, compile-only, 2026-09-07

> SUPERSEDED semantic note: R7's normalizer also treated `mag==0` as `+0`.
> Current `dsv4_e4m3` preserves negative zero and only maps `mag==0x7f` NaNs
> to `+0`; use R9 for the corrected engine contract.

## Candidate

R7 keeps R4's four-warp/64-row, K=128 staging geometry, BF16 input contract,
E8M0 exact exponent-field transform, and K/MMA accumulation order. It replaces
only the shared BF16 LUT decode:

1. load eight resident FP8 codes per lane with the same vectorized `uint2`
   traffic;
2. normalize E4M3 `mag==0` and `mag==0x7f` bytes to zero, matching the engine
   contract proven by R6;
3. issue four native `cvt.rn.bf16x2.e4m3x2` conversions per lane;
4. apply the exact shared E8M0 exponent to both returned BF16 halves;
5. feed the unchanged shared A/B tiles and eight K=16 BF16 f32-accumulate MMA
   steps per K=128 tile.

SASS contains `F2FP.BF16.E4M3.UNPACK_B` and `HMMA.16816.F32.BF16`. Resource
   usage is 38 registers/thread and 21,504 bytes static shared memory.

## Artifacts

CUDA 13.3.73 compiled both cubin targets:

```sh
target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120f,code=sm_120f -cubin \
  -o target/dsv4-dense-tc-gate-r7-sm120f.cubin tools/dsv4-dense-tc-gate-r7.cu

target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a -cubin \
  -o target/dsv4-dense-tc-gate-r7-sm120a.cubin tools/dsv4-dense-tc-gate-r7.cu

/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda -o target/dsv4-dense-tc-gate-r7-driver \
  tools/dsv4-dense-tc-gate-r7-driver.cu
```

```text
target/dsv4-dense-tc-gate-r7-sm120f.cubin
  sha256 023f750f74dbf9aadb02268a13593fc20a5e0c9b7b7d8ae6932bdb7360299af1

target/dsv4-dense-tc-gate-r7-sm120a.cubin
  sha256 6b8a81e34c68d7a288954501d1a0d5e8ae62518594bf9d9915dddeecd37640bc

target/dsv4-dense-tc-gate-r7-driver
  sha256 f2d0e2ced0e575e7ef38a13fc7a876710d472cdd276c5d64ebd591192b861ecd
```

Source hashes:

```text
tools/dsv4-dense-tc-gate-r7.cu
  f4a49c19e8fde27c146c504398e83068a75beba182d2bd142b06c5e2b6a8bdbc
tools/dsv4-dense-tc-gate-r7-driver.cu
  2a75c921a214d137d275c78f83e69aecfa2c6a725475cbf3909c904caa91912d
```

The launcher is compiled with CUDA 13.1 and the actual current engine TU. It
uses the current `memra_dsv4_gemv_fp8_m` as control and loads the R7 cubin with
the CUDA driver API. It does not use CUDA 13.3 cudart or PTX JIT.

## Gate design

Before scored work, each case warms current and candidate once. Every scored
arm flushes a buffer sized to 2x the queried L2 before its event start. The
matched ABBA order is current/candidate/candidate/current for three cycles on
the full shape. Every arm checks finite per-row output, bit-repeatability,
numeric abs/relative bands, and a 16-float post-output guard. `--basis` runs
the same checks for rows 1, 64, 65, and 129 with K=128 and K=256, exercising
multi-K accumulation and padded-row tails.

Stable cubin ABI:

```text
r7_dense_native(const uint8_t* codes, const int8_t* scale_exp,
                const uint16_t* x, float* y, int rows, int k, int sc_cols)
```

The driver validates parameter offsets/sizes with `cuFuncGetParamInfo` before
launch. This receipt is compile/SASS/ABI-only; no cubin or launcher execution
was performed in this lane.
