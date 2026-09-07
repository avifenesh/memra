# GU half2 phase-clock instrumentation, compile-only, 2026-09-07

## Scope

`tools/dsv4-gu-phase-clock-gate.cu` includes the current
`moe_f16_grouped.cu` and duplicates the exact
`moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT,false,true>` arithmetic/body order.
It adds only lane-0 `clock64()` stamps and a global per-CTA/per-warp trace.
The geometry fixture is the existing bank=128, hidden=4096,
intermediate=2048, active groups 1/3/4/6 setup.

The six reported phase accumulators are:

```text
prefix_lut, activation_copy_wait, weight_decode_store, mma, barriers, epilogue
```

`activation_copy_wait` includes both the `cp.async`/A-copy issue and every
`cp.async.wait_group` interval; wait time is not hidden outside the phase
receipt.

Each phase is a per-CTA/per-warp clock sample. The tool prints min/max/mean
over samples and a small per-CTA/warp sample, but never sums overlapping warp
times as a wall-time saving. The host compares the instrumented H output against
the current GU half2 launcher bit-for-bit, checks finite live rows and guards,
warms both arms, flushes 2x L2 before every scored launch, and runs three
current/instrumented/instrumented/current ABBA cycles for each active count.

## Compile receipt

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda \
  -o target/dsv4-gu-phase-clock-gate tools/dsv4-gu-phase-clock-gate.cu
```

```text
binary sha256 5d893d3d1a0750510498f869801fcb4f89141e5b8b01b2dfef5cf10f2cf56ed3
source sha256 22f3c524bc75571e3c0f4041d7bc1876e8a73f22902150a87fb8ec152ea494ba
runner research/dsv4f-devpair-20260905/run-gu-phase-clock.sh sha256 6fbafec9893f9bdf6b5e059edfa04efd0597f65b89e5fa3cfda1dc446175a871
instrumented resource: 150 registers/thread, 26128 bytes shared
current <108,false,true>: 150 registers/thread, 26128 bytes shared
```

No GPU execution, production/library edit, or Cargo change was made. The
instrumented timing includes clock sampling and trace writes; the receipt is
for identifying phase ownership and instrumentation overhead, not a speed
claim.

## Clock interpretation

CUDA documents `clock64()` as a per-multiprocessor counter incrementing each
cycle, and explicitly warns that sampled per-thread elapsed clocks include
time-slicing and are not the actual instruction cycles spent by the thread.
See the [CUDA 13.2 Programming Guide clock64 section](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf).
