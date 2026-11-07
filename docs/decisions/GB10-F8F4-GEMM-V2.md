# GB10 F8F4 prefill GEMM v2: per-call weight image, pipelined tile (2026-11-06)

**Status:** decided on one device class (unified-memory 120a family, measured on two DGX Spark
units), for F8F4 GEMMs of at least 2,048 rows over whole 256-value k groups. Discrete cards keep
the v1 tile.

## Question

At 131,051 prompt tokens the trunk GEMMs were 80.3 s of 181.1 s TTFT at 79 TFLOPS
(`research/gb10-prefill-20261106/round2-profile-128k/`). The v1 F8F4 tile folds NVFP4 weights to
e4m3 inside every tile at one 8-warp CTA per SM (187-200 registers, 94.7 KB smem), so the MMA
phase waits behind that ALU work, and three exact loader edits spilled or stayed flat (lever
ledger rows 3-6). Does the structural rework in `GEMM-REWORK-DESIGN.md` (dequantize once, a
pipelined mainloop without ALU staging, an L2-ordered raster) beat it end to end with the same
numeric program?

## Decision

`memra_mmq_nvfp4_f8f4_v2_preq` runs where the F8F4 route runs on GB10 and the GEMM has at least
2,048 rows (`F8F4_V2_MIN_ROWS`) on whole 256-value k groups:

- an image pre-pass folds the NVFP4 weight once per call, with the v1 loader's expressions, into a
  pooled scratch (e4m3 bytes row-major, per-32 scales k-major and lane-permuted, 1.125 B per
  weight; 100 MB for the 5120 x 17408 projection);
- the F8F4 quantizers (plain and the fused silu twin) write the same bytes and scales in a v2
  layout (row-major values plus a scale plane);
- one 16-warp CTA per SM computes a 256 x 128 tile (warp tile 32 x 64, 64 accumulators,
  128 registers) from a 3-stage cp.async ring of K = 64 slices with XOR-swizzled rows, ldmatrix
  for A and B, the same block-scale MMA at the ue8m0 identity over a fresh accumulator and the
  same `sum += (dA * dB) * C` per k32 block in ascending k;
- the raster walks 4 row tiles over every token tile before the next group.

`MEMRA_MMQ_F8F4_V2=0` restores v1; unset follows the device.

## Evidence

Exactness: kernel-check `MMQ-F8F4-V2` compares v2 against v1 on every f32 bit at every NVFP4
GEMM shape of Qwen3.8-27B (in 5,120 / 6,144 / 10,240 / 17,408, out 1,024 to 17,408) and of the
9B the nvfp4-gemm section loads, at T = 16 / 130 / 1,024 / 4,096, on the GGUF and the split-plane
weight layouts, plus `MMQ-F8F4-V2-SILU` for the fused quantizer: 0 mismatching outputs. The
`run-gen` logits are byte-identical and the served answers identical in every boot.

Kernel (`gemm-v2-probe/`, 8,192 rows, synthetic weights, both units): v1 73-77 TFLOPS; v2
106.8-116.3 TFLOPS GEMM-only and 101.7-110.2 with the image pass, against a measured ceiling of
132 TFLOPS for this epilogue at 16 warps per SM (`ceilings/ceil_probe2.txt`). `run-gen` pp3855:
4.167 -> 3.673 s (-11.8%), unit 2.

Served A/B, Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft, cells P V V P with one fresh
boot per cell and all three lengths per boot, unit 1, 8,192-row prime chunks
([`research/gb10-prefill-20261106/gemm-v2-served-unit1/`](../../research/gb10-prefill-20261106/gemm-v2-served-unit1/)):

| prompt tokens | base (round 2, `1ed40895ec`) | F8F4 v2 (`b6d10d41c7`) | change |
|---:|---|---|---:|
| 32,784 | 36.199, 36.238 s (median 36.218) | 31.376, 31.360 s (median 31.368) | -13.39% (-4.9 s) |
| 131,051 | 181.306, 181.239 s (median 181.273) | 161.785, 161.862 s (median 161.823) | -10.73% (-19.4 s) |
| 259,951 | 457.527, 458.321 s (median 457.924) | 418.932, 419.036 s (median 418.984) | -8.50% (-38.9 s) |

The answer text is identical in all four boots at every length; the MemAvailable floor at 259,951
tokens is 65.2-65.3 GiB in every cell.

At 4,096-row prime chunks, the GB10 default once #1073 drops 8,192 for peer fairness, both arms
with `MEMRA_PRIME_CHUNK=4096`, unit 1
([`gemm-v2-served-4k-unit1/`](../../research/gb10-prefill-20261106/gemm-v2-served-4k-unit1/)):

| prompt tokens | base `1ed40895ec` | F8F4 v2 `b6d10d41c7` | change |
|---:|---|---|---:|
| 32,784 | 36.286, 36.214 s (median 36.250) | 32.516, 32.562 s (median 32.539) | -10.24% (-3.7 s) |
| 131,051 | 182.609, 182.540 s (median 182.575) | 168.396, 167.852 s (median 168.124) | -7.91% (-14.5 s) |
| 259,951 | 461.140, 461.507 s (median 461.323) | 433.077, 433.143 s (median 433.110) | -6.12% (-28.2 s) |

Unit 2 ([`gemm-v2-served-4k-unit2/`](../../research/gb10-prefill-20261106/gemm-v2-served-4k-unit2/)):

| prompt tokens | base `1ed40895ec` | F8F4 v2 `b6d10d41c7` | change |
|---:|---|---|---:|
| 32,784 | 36.250, 36.340 s (median 36.295) | 32.593, 32.595 s (median 32.594) | -10.20% (-3.7 s) |
| 131,051 | 182.231, 182.374 s (median 182.303) | 167.747, 167.351 s (median 167.549) | -8.09% (-14.8 s) |
| 259,951 | 459.923, 461.138 s (median 460.530) | 432.283, 432.582 s (median 432.433) | -6.10% (-28.1 s) |

Identical answers in all eight boots; MemAvailable floor 66.2-66.8 GiB in every cell. The battery
on the timed binary (`gemm-v2-gates-unit1/`): kernel-check ALL GREEN in h2/db/t3 modes, argmax-margin
flips=0 bad=0, run-spec K=1..8 PASS, the `MEMRA_MMQ_F8F4_V2=0` rollback returns the same logits.

On the final commits, main `c2e04719b6` (with #1073-#1076) against main plus this change
`c23cd3a158`, unit 2
([`gemm-v2-served-final-unit2/`](../../research/gb10-prefill-20261106/gemm-v2-served-final-unit2/)):

| prompt tokens | main `c2e04719b6` | main + v2 `c23cd3a158` | change |
|---:|---|---|---:|
| 32,784 | 36.137, 36.245 s (median 36.191) | 32.525, 32.437 s (median 32.481) | -10.25% (-3.7 s) |
| 131,051 | 181.768, 182.177 s (median 181.972) | 167.487, 167.682 s (median 167.584) | -7.91% (-14.4 s) |
| 259,951 | 460.503, 459.085 s (median 459.794) | 431.770, 431.174 s (median 431.472) | -6.16% (-28.3 s) |

## Rejected

- Two 8-warp 128 x 128 CTAs per SM (the design's first target): reached (128 registers, 36 KB
  smem) and exact, but slower than the single 16-warp 256 x 128 CTA at 5 of the 7 swept shapes by
  0.5-9% (4-9% on the two FFN shapes, about 70% of trunk FLOPs) and faster by 1-3% only at
  5120 x 1024 and 5120 x 6144; the 256-row tile halves the activation re-reads from L2. One
  config is kept rather than a per-shape table.
- Resident images built at load: 0 s pre-pass but 28.9 GB of device memory; the 256K
  MemAvailable floor would fall from about 66 to about 38 GiB, and 256K context is the minimum.
- Reading the activation from the v1 `block_e4m3_mmq` layout: 45% excess L2 sectors, L2-bound at
  92 TFLOPS.
- 80 B padded smem rows: conflict-free for ldmatrix but 15-way conflicted for the cp.async
  writes; replaced by the XOR swizzle.
- K = 32 rings of 3-7 stages: 88-97 TFLOPS, one barrier per k32 step.
- 128 x 256 tiles: tie with 256 x 128 within noise; one config is kept.
- Packed f32x2 epilogue: ptxas lowers `fma.rn.f32x2` / `mul.rn.f32x2` to scalar FFMA/FMUL on
  sm_121a; no change at any occupancy.
- v2 below 2,048 rows: the image pass costs more than the GEMM saves (crossover 1,024-2,048 rows).
