# GU BDB schedule candidate, compile-only, 2026-09-07

## Raw phase finding

The completed two-card R3 receipt is at
`/home/avifenesh/projects/wt-dsv4f-ops-20260905/research/dsv4f-devpair-20260905/receipts/ondemand/gu-phase-clock-20260907-r3/`.
Across both cards, the corrected phase instrument shows weight decode/store as
the largest per-warp phase, roughly 147k--202k clock samples for active 1--6,
versus roughly 64k--96k MMA. Activation copy/wait is about 5.9k and barriers
896. Instrumentation overhead stayed about 1.01--1.05x, with exact H and guard
checks passing.

The highest-value bounded schedule seam is therefore B double buffering: stage
both gate and up B tiles into alternate shared buffers before the A wait/gate
MMA, then read gate from `B[cur]` and up from `B[cur^1]`. This preserves the
ascending gate/up MMA and epilogue order while moving up decode/store into the
existing A-wait window. It is a schedule candidate only; no speed is claimed
until the locked exact-H/rate cell.

## Candidate

`tools/dsv4-gu-bdb-schedule-gate.cu` includes the current production helpers
and exact geometry fixture. `r9_moe_kq_sktail_gu_bdb<108,false>` retains the
current A staging, `kq_fetch`, packed ModelOpt store, gate/up MMA arithmetic,
barriers, and epilogue. Only B storage/schedule changes: two B buffers and both
projection stores/fetches before the A wait and gate MMA.

The compile-only host symbol is `r9_gu_bdb_launch_compile_only`; no GPU launch
is made in this lane.

## Compile receipt

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda \
  -o target/dsv4-gu-bdb-schedule-gate tools/dsv4-gu-bdb-schedule-gate.cu
```

```text
binary sha256 20278363cb1178c18f831478d9d0bde22ab0b3f4f0ff46d192d2356ba4f5b759
source sha256 724a6960d298a2c0c4cefd5bc6ffa8ea2bae5a689bcfc9a6629b63c17db76baa
current resource 150 registers, 26128 shared bytes
BDB resource 132 registers, 35344 shared bytes
```

The BDB increase is 9,216 shared bytes, exactly one additional 64x72 BF16 B
tile; occupancy/rate remains a runtime question. No library, production, or
Cargo source changed and no GPU run was performed.

`clock64()` interpretation follows NVIDIA's [CUDA Programming
Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf):
it is a per-multiprocessor counter and sampled thread elapsed clocks include
time slicing. Phase samples are therefore not additive wall time.
