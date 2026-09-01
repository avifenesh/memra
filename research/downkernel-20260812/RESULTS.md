# Step B=1 `moe_down8` occupancy/tile study

Date: 2026-08-12

Branch: `lane/cx-downkernel`

Local branch base: `79c3c0b27`

Box1 measurement base: `f0ab104e7` (the touched runtime files were identical
between this commit and the local branch base before the arm)

Rig: box1, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SM/card),
stock clocks, serialized by `/tmp/memra-gpu.lock`

## Verdict

**GO for orchestrator promotion.** The Step-only top-8 slot-parallel arm is
bit-identical to the established serial kernel, removes the measured one-warp
CTA block-slot ceiling, and reduces the exact 40-launch semantic mix by
**8.178%** in an N=8/arm ABBA window.

This is a kernel-level result, not a new end-to-end Step or perf-board number.
No merge, tag, push, format, Nsys capture, or board operation belongs to this
lane. Promotion should retain the normal full-release validation boundary.

## Frozen baseline

The ncuspike Step B=1 decode anatomy records 40
`moe_down8_fma_dev_q8_rows_g` launches/token, 0.9045 ms/token, 9.62% of token
wall, 588.2 GB/s (36.83% card BW), 44.91% achieved occupancy, and 0.91
waves/SM. See the frozen [ncuspike result](../ncuspike-20260811/RESULTS.md).

The exact production shape is:

- IQ4_XS down weights, `in_f=1280`, `out_f=4096`, `row_bytes=680`;
- top-8 of 288 experts;
- `grid=(4096,1,1)`, `block=(32,1,1)`; and
- one output-row warp that executes all eight expert slots serially, including
  the slot-ordered `__fmaf_rn` routing-weight chain.

## Named mechanism

Fresh metric-scoped NCU 2026.1 on the exact shape reports:

| baseline launch property | value |
|---|---:|
| registers/thread | 44 (zero spills) |
| static shared memory | 0 B |
| block limit: SM hardware cap | 24 blocks/SM |
| block limit: registers | 40 blocks/SM |
| block limit: shared memory | 32 blocks/SM |
| block limit: warps | 48 blocks/SM |
| theoretical active warps | 24/SM |
| theoretical occupancy | 50.00% |
| achieved active warps | 21.51/SM |
| achieved occupancy | 44.82% |
| waves/SM | 0.91 |
| long-scoreboard stalls | 18.99 warps/issue |

The limiter is therefore **the hardware resident-block cap applied to a
one-warp CTA**, not register pressure or shared memory. Each SM can host only 24
of these one-warp blocks even though the device supports 48 resident warps.
The 4,096-row grid supplies about 21.8 blocks/SM before scheduling effects, in
line with NCU's 21.51 achieved active warps. The kernel has too few independent
warps to hide its long-scoreboard weight-load latency.

This interpretation follows NVIDIA's current definitions: the NCU Occupancy
section attributes the ceiling to block, register, shared-memory, warp, and
barrier limits, while `launch__waves_per_multiprocessor` measures grid coverage
relative to a resident wave and warns about partial-wave tails. References:
[Nsight Compute 13.3 Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [CUDA 13.2 Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).

Raw receipt: [baseline NCU details](raw/box1/baseline-ncu/details.txt),
[CSV](raw/box1/baseline-ncu/raw.csv), and
[ptxas build log](raw/box1/baseline-build.log). The binary NCU report was hashed
and removed; its hash remains in
[report.sha256](raw/box1/baseline-ncu/report.sha256).

## Arm 1: parallel slots, serial chain replay

`moe_down8_fma_dev_q8_rows_w8` launches `block=(32,8,1)`. Warp `j` computes
slot `j`'s expert dot for one output row, writes the reduced scalar to an
eight-float shared tile, and warp 0 lane 0 replays the original slot-order FMA
chain.

The following remain unchanged:

- dtype, IQ4_XS bytes, row layout, pointer-table lookup, and selected experts;
- each slot's `g = lane; g += 32` assignment and accumulation order;
- `expert_dot_g_v` arithmetic and the 32-lane reduction tree;
- routing-weight order and explicit `__fmaf_rn` chain; and
- output indexing and bytes.

Production dispatch selects the arm only for the frozen Step B=1 shape:
`t==1`, `in_f==1280`, `out_f==4096`, `n_used==8`, and IQ4_XS. Every other
shape keeps `moe_down8_fma_dev_q8_rows_g`.

The candidate compiles with 48 registers/thread, one barrier, 32 B static
shared memory, and zero spills. Raw receipt:
[candidate build log](raw/box1/candidate-build.log).

## Bit-identity gate

The lane harness fills every one of the 4,096 output rows with varied,
deterministic IQ4_XS bytes and varied top-8 Q8 activations/scales. It runs both
kernels against the same device buffers and dumps all f32 output bytes before
any timing.

```text
baseline  3c509060d071f171e0bd54ac9d0e29f98411fa5d4367b47d1b54480a0e2eccce
candidate 3c509060d071f171e0bd54ac9d0e29f98411fa5d4367b47d1b54480a0e2eccce
mismatches=0; cmp exit=0
```

Raw receipts: [comparison](raw/box1/exactness/comparison.txt),
[gate log](raw/box1/exactness/check.log), and the two retained 16 KiB dumps.

## N=8 ABBA semantic timing

The scored harness allocates 40 physically distinct top-8 expert banks, one
for each unclamped Step MoE layer that uses this exact symbol across the two
serial PP stages (the final two clamped layers use their separate path). A
synthetic token therefore walks 40 launches and 891,289,600 logical weight
bytes, larger than L2. Each sample repeats that token sweep 128 times. The
schedule is `ABBA` repeated four times, for N=8 per arm.

| arm | N | median ms/token-equivalent | median logical weight GB/s | per-launch µs |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.8591155 | 1,037.450 | 21.4779 |
| candidate | 8 | 0.7888570 | 1,129.849 | 19.7214 |

Candidate delta: **-8.178% time**, **+8.906% throughput**, and
0.0702585 ms/token-equivalent saved in this isolated 40-launch slice. Every
candidate sample is below every baseline sample; this is not a sub-noise result.

The card ran at stock clocks. The complete process thermal trace spans 26–42 C;
after the initial P-state ramp the trace shows P0 at 2,385–2,392 MHz. NCU is not
used as a timing authority because replay perturbs duration.

Raw receipts: [ABBA log](raw/box1/timing/abba.log),
[thermal trace](raw/box1/timing/thermal.csv), and
[host snapshots](raw/box1/timing/host-before.log).

## Counter movement

The paired standalone NCU passes use the same exact-shape harness and
cache-control policy:

| metric | baseline | candidate |
|---|---:|---:|
| block | `(32,1,1)` | `(32,8,1)` |
| registers/thread | 44 | 48 |
| static shared memory | 0 B | 32 B |
| theoretical active warps/SM | 24 | 40 |
| theoretical occupancy | 50.00% | 83.33% |
| achieved active warps/SM | 21.51 | 35.42 |
| achieved occupancy | 44.82% | 73.79% |
| waves/SM | 0.91 | 4.36 |
| long-scoreboard stalls | 18.99 | 16.64 |
| DRAM throughput | 688.54 GB/s | 1.02 TB/s |
| card-BW SOL | 43.19% | 64.30% |
| replay duration | 32.42 µs | 21.79 µs |

The arm raises the independent resident work exactly where predicted; the
block ceiling is replaced by a 48-register limit of five 8-warp CTAs/SM (40
warps, 83.33% theoretical occupancy). Raw candidate receipt:
[details](raw/box1/candidate-ncu/details.txt) and
[CSV](raw/box1/candidate-ncu/raw.csv).

An additional metric-scoped NCU run around the actual PP-2 production
`run-gen` binary observed the new symbol at the live decode launch
`(4096,1,1)x(32,8,1)`. It reports 74.85% achieved occupancy, 4.36 waves/SM,
916.89 GB/s, and 24.35 µs under replay. This is dispatch/mechanism evidence,
not a replacement for the ABBA timing. Raw receipt:
[production NCU details](raw/box1/production-ncu/details.txt) and
[run log](raw/box1/production-ncu/ncu-run.log).

## Correctness battery

The production Rust/CUDA dispatch was rebuilt from the isolated box1 worktree
and passed the required gates on the real Step artifacts:

- `kernel-check`: `ALL GREEN (88 cells, 21 skipped)`;
- `run-gen`: prefill/decode argmax MATCH and batched-prime/tokenwise argmax
  MATCH; and
- `run-spec`: self-consistency PASS for every K=1..8.

Raw receipts: [production build](raw/box1/production-build.log),
[kernel-check](raw/box1/gates/kernel-check.log),
[run-gen](raw/box1/gates/run-gen.log), and
[run-spec](raw/box1/gates/run-spec.log).
