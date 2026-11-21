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
