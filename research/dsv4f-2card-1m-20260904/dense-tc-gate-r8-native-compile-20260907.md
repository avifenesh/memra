# Dense FP8 original-F32-order native decode R8, compile-only, 2026-09-07

> SUPERSEDED semantic note: R8's normalizer treated `mag==0` as `+0`.
> Current `dsv4_e4m3` preserves negative zero and only maps `mag==0x7f` NaNs
> to `+0`; R9 is the corrected engine-contract candidate. R8 F32-order and
> cancellation provenance remains attributable but its engine edge semantics
> are superseded.

## Candidate

R8 returns to the measured original `dsv4_gemv_fp8_m_kernel<1>` program:

- one output row per 128-thread block;
- each thread owns the same 8-value chunks and serial product/add order;
- the same 128-leaf shared reduction;
- the same FP32 scale and BF16-input semantics.

The only changed operation is FP8 decode. Each pair of normalized E4M3 bytes
uses native `cvt.rn.bf16x2.e4m3x2`, expands the exact BF16 words to f32, then
performs the original scale multiply and input product/add sequence. This is
intended as a bit-identity class, not an MMA reduction-order fork.

The double-stride path explicitly decodes both ranges first, then accumulates
all `i0` products followed by all `i1` products, matching the current kernel;
the previous alternating R8 ordering is not retained.

## Artifacts

CUDA 13.3.73 cubins:

```sh
target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120f,code=sm_120f -cubin \
  -o target/dsv4-dense-tc-gate-r8-sm120f.cubin tools/dsv4-dense-tc-gate-r8.cu

target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a -cubin \
  -o target/dsv4-dense-tc-gate-r8-sm120a.cubin tools/dsv4-dense-tc-gate-r8.cu

/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda -o target/dsv4-dense-tc-gate-r8-driver \
  tools/dsv4-dense-tc-gate-r8-driver.cu
```

```text
target/dsv4-dense-tc-gate-r8-sm120f.cubin
  sha256 288fbdd0f57fc0db9407695a11cc9708eeb7265466c05a29f5fee100d0846f3a

target/dsv4-dense-tc-gate-r8-sm120a.cubin
  sha256 fee70dd525597e26548b59c594584398d0d72f556f74e665872ba8bd117f79b7

target/dsv4-dense-tc-gate-r8-driver
  sha256 630f4b54d717fcc8ef02f87e4a7ad4c25a4fc230358604007dfbf2480eb27ee8
```

SASS contains `F2FP.BF16.E4M3.UNPACK_B`, FP32 `FMUL`/`FADD`, and the original
reduction structure. Resource usage is 36 registers/thread and 1,536 bytes
shared memory.

Source hashes:

```text
tools/dsv4-dense-tc-gate-r8.cu
  fad8ea82e2ec7a0bbe3e64a63b986c2343ea7f5e39dff810bfa93fd57d4fffce
tools/dsv4-dense-tc-gate-r8-driver.cu
  a113e3077e59e1e4415faf893c586a8d09d78b8a0ce5a7a5895dfc628ebfb645
```

## Driver gate

The CUDA 13.1 launcher includes the actual current
`memra_dsv4_gemv_fp8_m` control, loads the R8 cubin through the driver API, and
checks the 7-argument ABI before launching:

```text
r8_gemv_fp8_native(const uint8_t*, const float*, int,
                   const uint16_t*, float*, int, int)
offsets/sizes: 0/8, 8/8, 16/4, 24/8, 32/8, 40/4, 44/4
```

For every case, both arms warm first. Each scored launch flushes 2x queried L2
before the event start. Full shape uses three current/candidate/candidate/
current ABBA cycles. Every output row is finite-checked, candidate output is
compared byte-for-byte against current control, repeated arms are bit-checked,
and a 16-float post-output guard is checked after every arm.

`--basis` runs rows 1, 64, 65, and 129 at K=128, K=256, K=2048, K=4096,
and K=8192, plus an edge case containing E4M3 `+0/-0/NaN`, signed-zero
BF16 inputs, and scale exponents -4/+4. K>=2048 cases include a
cancellation-sensitive anchor with per-thread products `[2^30,1x7]` followed
by `[-2^30,1x7]`; the CPU preflight explicitly computes the shipped
all-i0-then-all-i1, old pair-interleaved, and individual-element-interleaved
F32 schedules. The expected control schedule gives 7 per thread, pair
interleaving gives 13, and individual-element interleaving gives 14. This
catches the prior R8 ordering defect before any full-shape timing.

No cubin or launcher was run on a GPU in this lane. R7 and R6 sources and
artifacts remain preserved.
