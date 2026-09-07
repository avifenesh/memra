# Dense FP8 to BF16 MMA gate r4, 2026-09-07

## Mechanism

R3 was numerically valid after the x4.trans orientation fix but lost the
matched cold-flush cell by about 4.7x on both GPUs. R4 is a separate
standalone source and leaves the reproducible R3 source unchanged.

The candidate remains FP8 codes/scales resident and uses the same BF16 input
and `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` reduction-order
class. It changes the measured mechanism in four ways:

- Four warps cover 64 output rows per block, reducing the one-warp launch
  shape while retaining one independent 16-row MMA tile per warp.
- Each warp stages one 16x128 weight tile using 8-byte `uint2` code loads,
  instead of scalar code loads for every K=16 stage.
- A shared 256-entry BF16-bit LUT removes per-code FP8 f32 decode and
  conversion. The exact E8M0 scale exponent is applied to the BF16 exponent
  field, with a host proof that every resulting BF16 word equals the R3
  `f32 -> __float2bfloat16` shadow.
- A block barrier protects each K=128 stage; eight K=16 MMA steps consume the
  staged tile without another global decode or barrier.
- The accumulator is allocated once per warp before the outer K=128 loop and
  stored once after all tiles. Stage barriers remain on both sides of each
  tile so shared A/B storage can be reused without losing earlier K tiles.

The R4 kernel uses 38 registers/thread and 22,016 bytes static shared memory
(`cuobjdump --dump-resource-usage`), with 128 threads/block. This is a static
compile receipt only, not an occupancy or speed claim.

## Gate

The binary retains the actual current `memra_dsv4_gemv_fp8_m` control by
including the frozen R3 standalone source, and measures one warmup per arm plus
three cold-flush ABBA cycles (six scored runs per arm). Every scored row is
checked for finite output, candidate deltas use both absolute and relative
thresholds, and each arm is required to be bit-repeatable. The terminal
receipt is explicit `PASS` only after all checks.

`--basis` is R4-specific, not the inherited R3 probe. It runs K=128 and
K=256 through the R4 launcher for rows 1, 64, 65, and 129. The data sets make
only the first K=128 tile nonzero, so the independent oracle and explicit
first-tile anchor fail closed if a last-tile-only bug reappears.

## Compile

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas \
  -o target/dsv4-dense-tc-gate-r4 tools/dsv4-dense-tc-gate-r4.cu
```

Compile result: PASS on the CUDA 13.1 toolchain for `sm_120a`. The included
engine TU emits its existing `shd` linkage warning twice; no new warning was
introduced by the R4 source. The initial stage was compile-only. Subsequent
paired GPU execution is recorded below; no production integration was performed.

```text
binary 1ea312f336a552b10553a7bbdddbee7fbef1244d720693af29330d524072e827
source ea1e6bc12d560c09e73dff3db23897a40ca1e7d4b4be99807438579c0ed8351d
```

## Paired GPU result

R4 passes its actual K128/K256 basis, padded-row cases, full-shape numerical
band and memcheck on both cards. Max absolute/relative deltas remain
0.0341796875 / 0.000980377197, with the unchanged acceptance bar. It is still
not a bit-identical replacement and has no model-quality qualification.

Three-ABBA cold-cache current/candidate mean times are 43.563 / 84.517 us
on GPU 0 and 43.259 / 85.008 us on GPU 1. Verdict: reject integration of R4;
it is 1.94-1.97 times slower than the actual engine. Its improvement over
the older prototype is not an engine speedup. Raw controller namespaces:
`dense-tc-{basis,memcheck,rate}-20260907-r4`, all finished with status zero.

Current primary NVIDIA references used for the design:

- <https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf>
  (global coalescing, shared-memory/register resource limits, and asynchronous
  copy alignment requirements).
- <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>
  (16-byte alignment and shared/global staging constraints).
