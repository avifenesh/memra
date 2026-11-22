# Per-slot persistent kernel: design (lane P, 2026-11-22)

Goal: the latency-bound part of an nvfp4-rig c=1 token (about 6 ms of 13 ms; 1493 launches a token,
31 a slot) toward under 1 ms and under 300 launches, with every stage's per-output program unchanged
(the same bits as the route today: identity on chunk 1 / 4 / 512, p3, both classes, and lane H's
batch_gate).

## What a slot runs today (c=1, af6259f73d+, launches and us)

norm_n (1, 5) -> BF16 jobs (1, 6) -> NVFP4 jobs (1, 32) -> heads_qr (1, 2) -> conv2 (1, 3) ->
heads_e4m3_2 (1, 1.5) -> attn_t4g (1, 11) -> t4 merge (1, 2.6) -> gate (1, 2) -> NVFP4 jobs o (1, 14) ->
conv (1, 2.8) -> mix_dot (1, 9) -> mix_out (1, 2.8) -> norm (1, 5.4) -> BF16 router (1, 6.5) ->
route (1, 2.6) -> to_bf16 (1, 0.7) -> groups (1, 0.6) -> actq (1, 3.6) -> expert w13 (1, 65) ->
actq (1, 5.5) -> expert w2 (1, 21) -> sum_bf16 (1, 0.9) -> conv (1, 2.9) -> mix_dot / mix_out ->
(controller slots: ctrl_down, ctrl_up x3)

Each launch costs its own execution (most are one block or a few) plus the graph-node issue latency.
The chain is fully dependent.

## Shape

One cooperative launch per slot (graph node), blockDim 256, one block per SM x 2 (164 blocks). The
stages run in order; between dependent stages a grid barrier (cooperative groups grid.sync) or, where
the consumer is one block, a flag the single producer block sets (no grid-wide wait).

- Stage functions: each existing kernel body becomes a __device__ function of (virtual block index,
  thread index) with the same mapping it has now. A stage with B virtual blocks of 256 threads runs them
  as for (vb = blockIdx.x; vb < B; vb += gridDim.x). A stage of 32-thread blocks (the expert GEMM, wpb 1)
  runs 8 virtual blocks a real block, one a warp; its __syncthreads becomes __syncwarp (one warp is the
  whole virtual block, so the order of every operation is unchanged).
- Shared memory: the largest stage's (the NVFP4 GEMV, about 32 KB) sized once; each stage carves its
  own layout from it.
- Single-block chains (norm -> small BF16 GEMV of grid 1, route -> to_bf16 -> groups -> actq) run in
  one block back to back with __syncthreads between them, and the other blocks continue to the next
  wide stage or wait on its flag.
- Barrier count a slot: about 10 grid-wide (after norm, projections, conv/epilogue, attention, merge+gate,
  output projection, mix, expert w13, expert w2, sum) instead of 31 launches.

## Bits

Every stage keeps its thread mapping, its reduction trees and its per-output chains; only where a
virtual block runs and how the stages are separated changes. Identity is checked per stage first (each
stage's device function against its kernel in the gate), then end to end.

## Risks and measurements

- A grid barrier costs about 1-2 us on this card; 10 a slot is 0.5-1 ms a token: the target is reachable
  only if most single-block stages chain inside one block without barriers.
- Occupancy: one persistent block per SM at 256 threads limits the bandwidth stages (the GEMVs and the
  expert GEMM want 8-24 warps an SM). Option: keep the two expert GEMMs and the NVFP4 GEMVs as their own
  graph nodes and fuse only the latency-bound stages between them (about 22 of the 31 launches).
- Order of work: (1) the attention block of a slot (norm .. gate) as one kernel, measured; (2) the FFN
  head (norm .. actq); (3) the tails (sum, conv, mix).
