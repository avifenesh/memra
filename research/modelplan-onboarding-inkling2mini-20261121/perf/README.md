# Inkling 2 Mini: decode and prefill performance of the GPU route (lane P, 2026-11-21)

Stacked on the GPU route of memra#1397 (`gpu-route/`). Card: RTX 5090 Laptop GPU, 24 GB, sm_120a, driver
615.71.09. Roofline at full clocks (`ceiling/FULL-CLOCKS.md` of the lane): DRAM read 865 GB/s, L2 3.3-3.4 TB/s,
BF16 MMA 116 TF, FP4 block-scaled 910 TF, FP8 block-scaled 432-443 TF. Floor of a decode token: about 5.55 GB
of weights read (BF16 trunk over 48 slots, 8 of 128 experts a slot, the 0.93 GB head), 6.4 ms or about
155 tok/s at c=1.

Rule of every step: the tuned route is bit-identical to the #1397 route on both numeric classes, the
declared act-quant class (the default) and the float bring-up class. Each step is checked three ways:

- `ink2-gpu-gate` (also in `kernel-check`): every kernel against its f64 reference operation, and each
  rewritten kernel bit-equal to the form it replaces at the route's shapes.
- Per-slot digests on the checkpoint: `ink2-gpu --score` folds the bits of every slot's two residual taps
  and every logit row into FNV-1a digests. The base and the new binary run the same tape: p1 (57 positions)
  one token a call, which is the decode path, and p2 (1531 positions) at the default chunk of 512, which is
  the prefill path plus 31 decode steps. Equal digests on all 48 slots and the logits mean equal bits.
- 32 greedy tokens from the p1 prompt in every timed take, equal across both binaries.

Timing: under a gpulock lease, `clampcheck.sh` first, `nvidia-smi` at 250 ms through the takes, a take is
void if any loaded sample (over 60 W) reads P2 or memory 11,001 MHz. Base and new interleave in both orders
(base-new, new-base, base-new), three takes each, 32 tokens of decode from the p1 prompt at c=1. The scripts:
`step.sh` (gate, digests, timed takes), `summ.py` (takes, telemetry voids, medians), `prof.sh` (nsys).

## Trajectory (c=1, declared class, 32 tokens from p1, median of 3 takes)

| Step | Change | tok/s | ms a token | launches a token | of the 155 tok/s floor | Receipt |
|---|---|---|---|---|---|---|
| base | #1397 at 0c88a56ca9 plus the digest runner | 5.38 | 185.9 | 1637 | 3.5% | `s1/summary.json` |
| 1 | BF16 decode GEMV at the read bandwidth, projections batched; controller and depth mix split over blocks | 53.0 | 18.9 | 1381 | 34% | `s1/` |
| 2 | expert GEMM reads input row pair / top_k: the MoE input is quantized once a token, not once a routed pair | 53.06 (vs 53.03, flat) | 18.85 | 1381 | 34% | `s2/` |
| rebase | onto #1397's review fixes (atomic forward, three-pass attention): base 5.36, steps 1+2 | 51.21 | 19.55 | 1381 | 33% | `s3/` arm1 |
| 3 | expert GEMM: ring issued before the activation stage, one warp a decode block | 51.79 | 19.33 | 1381 | 33% | `s3/` arm3 |
| rebase 2 | onto d789cb6f1c (exact SiTU, review fixes): base 5.34, steps 1-3 | 50.47 | 19.86 | 1381 | 33% | `s4-s7/` arm1 |
| 4 | decode attention split over blocks | 51.80 | 19.29 | 1381 | 33% | `s4-s7/` arm2 |
| 5 | CUDA graph of the decode step | 51.07 (p50 52.3) | 19.11 | 179 | 34% | `s4-s7/` arm3 |
| 6+7 | prefill GEMM and expert prefill groups (decode unchanged) | 51.06 | 19.08 | 179 | 34% | `s4-s7/` arm4 |

## Step 1: GEMV, controller and depth mix

The base profile (nsys, 8 decode tokens and the 26-token prefill, declared class) put 51% of GPU time in
`ink2g_gemv_kernel` and 29% in `ink2g_controller_kernel` (594 us a call, 1 block a row), then the tile GEMM of
the prefill (10%), the expert GEMMs (3.6%) and the mix (2.4%).

- `ink2g_gemv_kernel` gave one thread per output and walked its weight row alone: 32 rows a warp load, so
  each load instruction touched 32 rows. The new `ink2g_gemv_bf16_kernel` keeps the program (per 16-input group
  a fused multiply-add chain from zero, the group partials added in order) and moves the work: a lane forms one
  group's partial from 32 contiguous bytes, a warp reads 1 KB of one row a load pair, one thread per row adds
  the partials after. A decode step's q, r, sink, gate, k, v and dynamic-conv projections run as one launch of
  up to 8 jobs.
- The controller and the mix ran one 256-thread block per row, so a decode step ran each on one SM in f64
  (GeForce Blackwell runs f64 at 1/64 of f32). The split forms run the same per-thread orders and the same block
  reductions in one block per (rank output, row) and per (state, query, row), with the second phase one thread
  per channel. The #1397 kernels stay as `_v0` entry points, the gate's oracle.

Bit identity (`s1/`): gate ALL OK, 21 cells (`ink2g-gemv-jobs` on 5 shape sets, `ink2g-mix` and
`ink2g-controller` bit-differ 0 against `_v0`, the controller also in place). Digests equal on all 48 slots
and the logits for declared p1 one token a call, declared p2, float p1 one token a call and float p2
(`s1/id-*.json`, `digest`). Greedy ids equal across all six takes.

Timed (`s1/summary.json`): base 5.378 / 5.379 / 5.379 tok/s, new 53.02 / 53.01 / 53.00 tok/s; no void take
(no loaded sample at P2 or 11,001 MHz). Host load 12.9 at the start.

## Rebase onto the #1397 review fixes (7a24a2a50f)

Lane H's review fixes changed the base: the conv histories are double-buffered so a failed call commits
nothing (atomic forward), and attention recomputes its f64 scores in three passes so its shared memory does not
grow with the context. The new base (`base2`: 7a24a2a50f plus the digest runner) decodes at 5.363 tok/s and steps 1
and 2 on it (`s2r`) at 51.21 tok/s, against 53.0 before the rebase: the three score passes cost about 0.7 ms a
token at this context. Steps 1 and 2 re-verified on the new base: digests equal on all slots and the logits,
both classes (`s3/id-*-base.json` against `s3/id-*-arm1.json`).

## Step 3: expert GEMM issue order and one warp a decode block

Two changes to the act-quant expert GEMM, neither touching an output's MMA sequence: (a) each warp issues its
weight ring's first stages before the block stages the activations (the weight stream never waited on them);
(b) a one-row decode group runs one warp a block (lane D measured 1 beating 2 on 15 of 16 decode rows).
Digests equal at each arm, gate ALL OK (`s3/`). Timed at c=1 against step 2: (a) 51.21 -> 51.49 tok/s (+0.5%),
(a)+(b) 51.79 tok/s (+1.1%). Small: the decode expert GEMM is already at 48-81% of the read roofline (lane D's
full-clock measurement), and its cost is spread over 80 short kernels a token.

## Rebase onto the #1397 review round (d789cb6f1c, with #1396 at e6f9dc4e40)

The new base carries the exact SiTU (f64 tables), so the declared class's bits moved relative to the
first base; every step below is re-verified on the new base (`base3` = d789cb6f1c plus the digest runner).
Base3 decodes at 5.343 tok/s; steps 1 to 3 on it at 50.47 tok/s (`s4-s7/` arm1).

## Steps 4 to 7 (`s4-s7/`: base3, then arm1 = steps 1-3, arm2 = 4, arm3 = 5, arm4 = 6+7, arm5 and arm6 = rejected)

Digests equal at every link of the chain (base3 -> 1-3 -> 4 -> 5 -> 6+7 -> the two rejected arms), both
classes, p1 one token a call and p2 at chunk 512; gate ALL OK (28 cells) on the last arm; greedy ids equal in all
21 timed takes; no void take. Timed at c=1, 32 tokens from p1, three takes each in rotating order:

| Step | Change | tok/s (median) | p50 ms a token | launches a token |
|---|---|---|---|---|
| 1-3 on base3 | | 50.47 | 19.86 | 1381 |
| 4 | decode attention split over blocks (scores of 256 keys a block into f64 scratch, one block a head for the max, the denominator and the oldest-first value sum); the position read from the device | 51.80 | 19.29 | 1381 |
| 5 | the decode step as a CUDA graph, one per conv-history parity, captured after one eager step | 51.07 (mean over 31 steps, with one eager step and the capture) | 19.11 | 179 (1 a replayed step) |
| 6+7 | prefill only: BF16 tile GEMM 128 x 128 x 8 x 8 a thread; expert prefill groups of 32 (W4A4) or 16 (W8A8) rows | 51.06 | 19.08 | 179 |

Step 5: graph replay equals eager bit for bit (`graph-check/`: logits digests of every step and the ids equal for
declared p1 32 tokens, float p1 32 tokens and declared p2 64 tokens; 29, 29 and 61 replayed steps). The gain is
small (p50 -0.9%) because the step is GPU-bound: 1381 kernels in 19 ms hide their launches.

Steps 6 and 7, prefill (`pf-s6s7/`, measured on the first base, the same kernels): tokens/s at 512 / 2048 / 8192:
before 272 / 150 / 97, step 6 291 / 158 / 101, steps 6+7 335 / 171 / 107 (median of 3, rotating order, no void take).

Rejected, deleted (receipts kept in `s4-s7/` arm5 and arm6, digests equal, slower):
- the controller with the RMS factors and normalized rows formed once a row (one more kernel): p50 19.08 -> 19.24 ms.
- f32 and bf16 widened to f64 by moving the fields instead of the conversion instruction (bits equal): p50 19.24 ->
  19.99 ms. The f64 conversions are not what bounds these kernels.
- a prefill form of the split attention, rows in groups under a 64 MiB score scratch: 4x faster attention at 2048
  (`prof-n11/pf2048-grouped-split_cuda_gpu_kern_sum.csv`: 4.1 s of attention against 13.8 s for the tiled kernel)
  and bit-equal to the tiled kernel in the gate cells, but its digests on p2 differed from step 7 on 7 (declared) and 21
  (float) slots. Not shipped; the cause is open.

## Where a decode token goes now (`prof-n11/dec_cuda_gpu_kern_sum.csv`, graph replay, 16 tokens; per token)

| Kernels | ms a token | bytes or work | against the roofline |
|---|---|---|---|
| BF16 GEMV (trunk projections, router, unembed) | 4.7 | about 3.5 GB | 745 GB/s, 86% of 865 |
| W4A4 expert GEMM (40 slots x 2) | 3.7 | 1.43 GB | 385 GB/s, 45% |
| W8A8 expert GEMM (8 slots x 2) | about 0.6 | 0.51 GB | about 85% |
| controller, depth mix, norms, convs, router, head epilogues | about 8 | under 0.1 GB | f64-issue bound: one to nine blocks, the f64 pipe at 1/64 of f32 (ncu: 44-64% f64 pipe busy on the active SMs, 1-7% of the GPU) |
| attention (short context) | 1.9 | small | f64 scores and value sums |

At 4096 tokens of context the decode step is 54.5 ms (18.1 tok/s, `ctx4k-n11/`, measured on the rejected arm 6,
3.9% slower): the full-attention slots' f64 scores and value chains grow with the context. Prefill at 2048 spends
80% of its time in the tiled attention kernel (`prof-n11/pf2048_cuda_gpu_kern_sum.csv`).

## Open

- The f64 program is the ceiling of this route: about 10 of 19 ms a token are f64 reductions and epilogues the #1397
  program defines (norms, mixes, controllers, convs, router, attention), on a card whose f64 rate is 1/64 of f32.
  Bit-identity to that program allows spreading the work over more SMs, not removing it. An f32 program for these
  operations is a new numeric program: it needs the coordinator's call and a per-layer KL receipt.
- Long-context decode and prefill: attention. The grouped prefill split is 4x faster but differs on p2; find the
  cause, then ship it.
- W4A4 expert GEMM at 45% of the read roofline in decode.
- The prefill BF16 GEMM reaches about 12% of the f32 FMA peak at 26 rows (128-row tiles); prefill at the MMA peak needs
  bf16 activations, a new numeric program.

## Step 2: one quantized MoE input a token

The declared program quantizes the bf16 MoE input of a token once; the route held one copy a routed pair
(8 bit-identical copies) because the act-quant GEMM read and wrote one row index. `memra_ink2_expert_gemm_x`
takes `xdiv`: a group row r reads act row r / xdiv and writes out row r. The gate/up GEMM runs with xdiv = top_k
on one quantized row a token; the down GEMM keeps xdiv = 1 on the per-pair SiTU rows. The `ink2g-moe-declared-*`
cells now run this path and stay bit-equal to `memra_reference::inkling2_actquant::expert_block_*` (2304/2304).

Digests equal to step 1 on all slots and the logits, both classes (`s2/id-*.json`). Timed against step 1:
52.99 / 53.03 / 53.03 vs 53.06 / 53.02 / 53.07 tok/s, flat: a decode step's 8 quantizer blocks ran in parallel
before, so one block saves bytes, not time. Kept because it is the declared program's shape and removes 7 of 8
writes and reads of the input; the prefill quantizer runs rows / top_k as many blocks.

## Where a decode token goes after step 1 (`prof-s1/kern_sum.csv`, 8 decode tokens and the 26-token prefill)

| Kernel | per decode token (approx.) | what bounds it |
|---|---|---|
| `ink2_expert_gemm_kernel` (W4A4, 40 slots x 2) | 5.2 ms | 35.8 MB a slot, 41 us at 865 GB/s against 130 us measured: 32% of the read roofline |
| `ink2g_ctrl_down_kernel` + `ink2g_ctrl_up_kernel` (26 message slots x 4) | 4.7 ms | f64 chains on 64 blocks; 1.2 MB of weights a controller |
| `ink2g_gemv_bf16_kernel` | 4.0 ms | the trunk and the head, about 3.5 GB: near the read roofline |
| `ink2g_mix_*` | 1.5 ms | f64, small |
| `ink2g_attn_kernel` (short context) | 1.4 ms | f64 scores and the value sum, one block per head |
| `ink2g_conv_kernel` | 1.3 ms | one thread a channel, f64, 4 to 9 blocks |
| `ink2g_norm_kernel`, `ink2g_route_kernel`, `ink2g_heads_kernel`, the rest | 2 ms | one block a row, f64 |

## Step 8: the NVFP4 decode GEMV, multi-row calls, expert groups on the device (2026-11-21)

- The NVFP4 GEMV read e2m1 codes through a local-memory table and the e4m3 scales from global memory in
  the fold. The e2m1 decode is now exact bit arithmetic and the scales sit in shared memory, and the
  decode projections of a step run as one jobs launch with each warp's rows in flight together.
  nvfp4-rig c=1 (32 tokens, p1, three interleaved takes, no voids): 11.2 tok/s (p50 88.7 ms, merged
  #1401) to 47.3 (p50 20.1 ms, bitwise e2m1 and smem scales; s8/gi-rig) to 54.0 tok/s (p50 17.7 ms, jobs
  launch; s8/gl-rig, base 44.2 tok/s on the same window). Greedy ids equal across all takes.
- Calls of 2..8 rows (spec verify, batched decode) go through a GEMV that reads each weight once for
  up to four rows, and calls of up to 256 routed pairs group their experts on the device (no host round
  trip; empty records are skipped).
- Bit identity: the kernel cells (ink2g-gemv-rows, m 2, 3, 5, 8, BF16 and NVFP4) differ from the #1397
  tile kernel in 0 outputs; slot and logit digests equal on all 48 slots for shipped, float and
  nvfp4-rig at chunk 1, 4 and 512 against the previous binary (s8/gl-id), and at chunk 4 all-f64 against
  the merged route (s8/gk-rows). Gate ALL OK, 36 cells.

## Steps 10-11: attention split and arms, multi-row GEMV, branch-free e2m1 (2026-11-22)

All bit-identical to the step before under its program (gate ALL OK; digests equal on all 48 slots and the
logits at chunk 1, 2, 3, 4, 5, 6, 8 and 512, both classes; receipts s10-attention-rows, s11-rows-e2m1).
The precision program changed once, by ruling (docs/decisions/INKLING2-PRECISION-ARMS.md, round 4).

| Change | Measurement (nvfp4-rig unless named; c=1 3 interleaved takes, no voids) |
|---|---|
| attention fold split into top / exp / out | 4k decode 22.7 -> 38.1 tok/s; 32k decode 7.1 -> 17.6 (single runs) |
| adopted program (conv, router f32c; attention f32t) | 32k prefill 62.9 -> 109.6 tok/s, 32k decode 17.6 -> 26.8 (single runs) |
| multi-row GEMV rewrite (x transposed, 32 rows a block, rows per warp by size, next-row prefetch) | shipped 4-row call GEMV 18.3 -> 10.1 ms, 8 rows 36.4 -> 19.9 ms; rig 8 rows 40.7 -> 22.8 ms |
| branch-free e2m1 decode (byte-permute table, fp16 widen) | 5864 -> 2720 SASS instructions in the NVFP4 jobs GEMV; rig c=1 59.1 -> 67.5 tok/s |
| rejected: one pass of 5..8 rows (4 warps a block) | slower than two passes of 4: shipped 8 rows 19.9 -> 27.5 ms; deleted |

Final table on the adopted program (binary a18 for the long rows, a20 for c=1 rig; tok/s):

| | c=1 decode | 4k decode | 32k decode | prefill 512 / 2048 / 8192 / 32768 |
|---|---|---|---|---|
| nvfp4-rig | 67.5 (p50 14.2 ms) | 42.7 | 27.3 | 679 / 387 / 241 / 110 |
| shipped | 59.9 (p50 16.2 ms) | 44.1 | 26.1 | 703 / 399 / 246 / 98 |
| float | 15.9 | | | |

The decode floor of nvfp4-rig is 3.58 ms a token (3.10 GB at 865 GB/s, about 279 tok/s): c=1 runs at 25% of it.

## Step 12: grouped-query f32t attention (2026-11-22)

One block a (row, KV head, tile) serves its 3 query heads: the K and V rows are read 16 bytes at a time and
decoded once, every per-head chain and reduction keeps its order (bit-identical: gate ALL OK; digests equal
on shipped and nvfp4-rig at chunk 1, 4, 512 and on p3 8k). Three interleaved takes, no voids:

| | before | after |
|---|---|---|
| nvfp4-rig c=1 decode | 68.7 tok/s (p50 14.0 ms) | 71.3 (13.5 ms) |
| nvfp4-rig 4k decode | 47.9 (p50 20.2 ms) | 65.9 (14.6 ms) |
| shipped 4k decode | 45.2 (21.4 ms) | 57.5 (16.8 ms) |
| nvfp4-rig 4k prefill | 316 | 753 |
| shipped 4k prefill | 326 | 783 |

At 4k the attention is 1.32 ms a token (was 6.31); nvfp4-rig is now ahead of shipped at 4k. Receipts: s12-breakdown.

## Step 13: joint launches for paired small kernels (2026-11-22)

Bit-identical (gate ALL OK; digests equal on shipped and nvfp4-rig at chunk 1, 4, 512 and on p3 8k):
the K and V short convolutions in one launch, the K and V epilogues into the e4m3 cache in one launch, the
Q and R clips in one launch (the R job's narrower heads run in the Q block; the extra threads add exact
zeros), and the up to three attention-input norms of a slot in one launch. nvfp4-rig c=1 71.3 -> 74.0
tok/s (p50 13.43 -> 12.93 ms), 3 interleaved takes; launches a token 1693 -> 1493.

Measured and rejected (deleted): the depth mix's dot and output in one launch (the last block of a pair
forms all outputs; 1.53 -> 2.22 ms a token, slower); the expert GEMM streaming two 16-row tiles a warp
through one ring (4.24 -> 5.33 ms a token, slower).
