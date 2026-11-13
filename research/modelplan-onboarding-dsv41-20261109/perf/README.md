# DeepSeek-V4.1-Flash onboarding: kernel speed on one card, bytes unchanged (2026-11-09)

Verdict: the V4.1 layer kernels of `crates/memra-engine/cu/dsv41_gpu.cu` run a decode step in
0.62 / 0.60 / 0.50 ms per layer (from 3.99 / 3.39 / 2.97 ms at the lane's start) and a 4,096-token
prefill in 34.1 / 32.1 / 31.0 ms per layer (from 57.5 / 55.6 / 53.6 ms), on layers 2, 20 and 21 of the
real checkpoint on one RTX PRO 6000. Every change is a schedule change: `dsv41-gate kernels` proves each dispatched
kernel byte-identical to the reference schedule on the fixtures, the layer gate's saved tensors are
byte-identical between the binaries before and after, the N=2 partition gate still writes the
one-card bytes, and every timed run prints output hashes that match across binaries.

Since `29aeeee39f` (2026-11-10, the sections from "One row on one card" on): the one-row layer
on one card is 0.55 / 0.53 / 0.44 ms (`9ea7ec0b7f`; 0.62 / 0.60 / 0.50 at `29aeeee39f`), tensor parallelism at one row is
measured and is not bound by weight bytes, and the N-rank step has no host read and no thread
wake a layer.

This file covers the second half of the lane (base `aec3b0b401`): what moved, what did not, and
what the profiler said. The first half (the decode-shaped kernels, the pipelined GEMM, the device
MoE plan) is in the commit messages of `6e8bf3b41f` .. `aec3b0b401` with its A/B summaries under
`receipts/`.

Hardware and conditions: one RTX PRO 6000 Blackwell Max-Q (sm_120a, 96 GB, 188 SMs, 250 W cap,
clocks untouched), CUDA 13.4, the box's other two cards busy with another job at 280 W each for
every run here (co-tenant power is in each telemetry file). Timed runs are interleaved in one
lease per comparison (A B B A A B, three runs per arm, or the arms in order, reversed, in order),
250 ms telemetry next to them. A run times layers 2, 20 and 21 after prefilling the trunk below
them: the prefill of the layer (median of 3 reps) and 8 decode steps from the end of the prompt
(median of steps 2..8). Layer 2 is a ratio-2 compressor source, layer 20 a ratio-1 source with the
candidate pool and the indexer, layer 21 a Reuse layer.

## Result

| | lane start `26f82ae0aa` | base `aec3b0b401` | now | now / base |
|---|---|---|---|---|
| decode, ms per layer (layers 2, 20, 21; position 4,096) | 3.993, 3.392, 2.966 | 0.694, 0.682, 0.568 | 0.619, 0.597, 0.503 | 0.891, 0.875, 0.886 |
| decode step through layers 20 + 21, one wait, ms | - | 1.244 | 1.090 | 0.876 |
| prefill 4,096 tokens, ms per layer | 57.46, 55.56, 53.64 | 43.04, 40.62, 39.24 | 34.14, 32.13, 31.02 | 0.793, 0.791, 0.791 |
| prefill 4,096 tokens, tokens/s per layer | 71.3k, 73.7k, 76.4k | 95.2k, 100.8k, 104.4k | 120.0k, 127.5k, 132.0k | |
| prefill 32,768 tokens, ms per layer | 500.6, 541.1, 435.9 | 365.5, 377.5, 308.8 | 280.4, 286.1, 240.0 | 0.767, 0.758, 0.777 |
| prefill 32,768 tokens, tokens/s per layer | 65.5k, 60.6k, 75.2k | 89.7k, 86.8k, 106.1k | 116.9k, 114.5k, 136.6k | |
| prefill 131,072 tokens, ms per layer (GPU busy, profiled) | - | 2,075.8, 2,625.0, 1,493.9 | 1,558.9, 2,117.3, 1,154.7 | 0.751, 0.807, 0.773 |
| prefill 131,072 tokens, tokens/s per layer | - | 63.1k, 49.9k, 87.7k | 84.1k, 61.9k, 113.5k | |
| decode at position 131,072, ms per layer (GPU busy, profiled) | - | 0.869, 1.200, 0.693 | 0.770, 1.057, 0.631 | 0.886, 0.881, 0.911 |

Base and now are the arm medians of the interleaved runs (`receipts/multi-f34` at 4,096: three runs
per arm, ranges do not overlap, spread at most 1.2%; `receipts/multi-f34g/pfx2` at 32,768, spread at
most 1.6%). The lane-start column is the A arm of `receipts/ab-6e8bf3b41f-4k` and, at 32,768, the
single profile run of `receipts/base`. Card at 102 W median, 72 C median over the 4,096 runs and
127 W, 80 C (86 C peak) over the 32,768 runs, SM clock 2,272 MHz median; output hashes identical in
every run of both arms. The 131,072 rows are the means of two nsys profiles per arm, interleaved
A B B A (`receipts/profab-f34-128k`), not timed runs: a profile adds the tracer's cost to both arms.

The model has 40 layers. Three layers do not make a model number; as a scale, the mean of the three
times 40, without the head, Engram and the host:

| | base | now |
|---|---|---|
| decode step | 25.9 ms | 22.9 ms |
| prefill 4,096 tokens | 1.64 s (2,499 tokens/s) | 1.30 s (3,158 tokens/s) |
| prefill 32,768 tokens | 14.0 s (2,337 tokens/s) | 10.8 s (3,047 tokens/s) |
| prefill 131,072 tokens | 82.6 s (1,587 tokens/s) | 64.4 s (2,035 tokens/s) |

## What changed, per op

`prof_ab_summary.py` over interleaved nsys profiles (A B B A, NVTX range per op) gives each op's
GPU time in both binaries; an op only one change touches moves, the rest show the noise.

Base `aec3b0b401` (A) against this head (B), 4,096 tokens, layer 20 (`receipts/profab-f34-4k`, all six ranges in its `summary.md`):

### L20/prefill: GPU busy 42.668 -> 33.589 ms, wall 42.775 -> 71.220 ms (A n=2, B n=2 profiles)

| op | A runs ms | B runs ms | A mean | B mean | B / A |
|---|---|---|---|---|---|
| experts | 21.753, 21.985 | 16.483, 16.482 | 21.869 | 16.482 | 0.75 |
| sparse_attn | 4.973, 5.090 | 4.313, 4.307 | 5.032 | 4.310 | 0.86 |
| wo_a | 2.659, 2.683 | 2.499, 2.505 | 2.671 | 2.502 | 0.94 |
| q_b | 1.972, 1.987 | 1.720, 1.696 | 1.979 | 1.708 | 0.86 |
| shared_expert | 1.938, 1.961 | 1.658, 1.659 | 1.949 | 1.659 | 0.85 |
| wo_b | 1.870, 1.891 | 1.608, 1.610 | 1.880 | 1.609 | 0.86 |
| hc_ffn_pre | 1.710, 1.728 | 1.186, 1.190 | 1.719 | 1.188 | 0.69 |
| hc_attn_pre | 1.161, 1.172 | 0.897, 0.896 | 1.167 | 0.896 | 0.77 |
| router | 0.809, 0.820 | 0.733, 0.730 | 0.814 | 0.731 | 0.90 |
| attn_keys/index_score | 0.713, 0.728 | 0.398, 0.395 | 0.721 | 0.397 | 0.55 |
| attn_keys/index_q | 0.638, 0.651 | 0.602, 0.605 | 0.645 | 0.603 | 0.94 |
| hc_post | 0.490, 0.500 | 0.238, 0.238 | 0.495 | 0.238 | 0.48 |
| attn_keys/qr | 0.425, 0.430 | 0.373, 0.372 | 0.427 | 0.372 | 0.87 |
| moe_plan | 0.410, 0.411 | 0.017, 0.017 | 0.410 | 0.017 | 0.04 |
| attn_keys/kv_window | 0.257, 0.259 | 0.231, 0.232 | 0.258 | 0.231 | 0.90 |
| attn_keys/compressor | 0.229, 0.232 | 0.214, 0.211 | 0.230 | 0.213 | 0.92 |
| combine | 0.203, 0.204 | 0.204, 0.206 | 0.204 | 0.205 | 1.01 |
| attn_keys/topk | 0.095, 0.098 | 0.101, 0.103 | 0.096 | 0.102 | 1.06 |
| attn_keys/index_keys | 0.038, 0.039 | 0.049, 0.050 | 0.039 | 0.049 | 1.28 |
| attn_keys/kv_append | 0.035, 0.035 | 0.038, 0.038 | 0.035 | 0.038 | 1.09 |
| attn_keys/candidates | 0.019, 0.020 | 0.028, 0.028 | 0.020 | 0.028 | 1.42 |
| attn_idx_rows | 0.009, 0.009 | 0.009, 0.010 | 0.009 | 0.010 | 1.04 |

### L20/decode: GPU busy 0.658 -> 0.573 ms, wall 0.694 -> 0.612 ms (A n=2, B n=2 profiles)

| op | A runs ms | B runs ms | A mean | B mean | B / A |
|---|---|---|---|---|---|
| experts | 0.157, 0.157 | 0.158, 0.157 | 0.157 | 0.157 | 1.00 |
| wo_b | 0.065, 0.065 | 0.044, 0.044 | 0.065 | 0.044 | 0.68 |
| wo_a | 0.047, 0.047 | 0.049, 0.048 | 0.047 | 0.048 | 1.02 |
| sparse_attn | 0.046, 0.047 | 0.036, 0.036 | 0.047 | 0.036 | 0.76 |
| shared_expert | 0.044, 0.044 | 0.045, 0.045 | 0.044 | 0.045 | 1.02 |
| attn_keys/index_q | 0.043, 0.042 | 0.034, 0.034 | 0.043 | 0.034 | 0.79 |
| hc_ffn_pre | 0.036, 0.036 | 0.021, 0.021 | 0.036 | 0.021 | 0.58 |
| router | 0.034, 0.034 | 0.029, 0.029 | 0.034 | 0.029 | 0.83 |
| attn_keys/compressor | 0.032, 0.032 | 0.023, 0.023 | 0.032 | 0.023 | 0.73 |
| q_b | 0.031, 0.032 | 0.032, 0.032 | 0.031 | 0.032 | 1.01 |
| hc_attn_pre | 0.028, 0.028 | 0.017, 0.017 | 0.028 | 0.017 | 0.60 |
| attn_keys/topk | 0.014, 0.014 | 0.013, 0.013 | 0.014 | 0.013 | 0.89 |
| attn_keys/kv_window | 0.013, 0.013 | 0.014, 0.014 | 0.013 | 0.014 | 1.10 |
| moe_plan | 0.012, 0.012 | 0.012, 0.012 | 0.012 | 0.012 | 0.98 |
| attn_keys/qr | 0.012, 0.012 | 0.013, 0.013 | 0.012 | 0.013 | 1.09 |
| attn_keys/index_keys | 0.010, 0.010 | 0.011, 0.011 | 0.010 | 0.011 | 1.03 |
| attn_keys/kv_append | 0.009, 0.009 | 0.009, 0.009 | 0.009 | 0.009 | 1.03 |
| hc_post | 0.007, 0.008 | 0.003, 0.003 | 0.008 | 0.003 | 0.34 |
| attn_keys/index_score | 0.006, 0.006 | 0.003, 0.003 | 0.006 | 0.003 | 0.52 |
| combine | 0.004, 0.004 | 0.004, 0.004 | 0.004 | 0.004 | 1.02 |
| attn_keys/candidates | 0.003, 0.003 | 0.004, 0.004 | 0.003 | 0.004 | 1.26 |

## The changes

Each is a schedule change of one kernel. The numbers are the op's GPU time before and after in the
per-op tables above unless a line says otherwise.

1. **The pipelined block-scaled GEMM no longer touches local memory.** The scale codes of the stage
   two iterations ahead waited in `ring[i & 1]`: an array indexed with a runtime value, which the
   compiler keeps in a thread's local memory. With the block's shared memory at its largest split
   the SM's L1 is at its smallest, and ncu showed 96% of those local loads going to L2, one memory
   round trip per thread and stage (`receipts/look-s5`). The codes now wait in a register.
   `sass_audit.py` lists LDL / STL per kernel from `cuobjdump -sass`; the pipelined kernels are at 0.
2. **The fused-epilogue test moved from every lane to the thread that stores a scale.** A K block of
   a tile takes the one-fma epilogue when every live row code is in [64, 180] and every live column
   code in [82, 178]; the storing thread checks its code and each warp leaves a flag byte per
   stage. The per-lane min/max over 12 scales per K block is gone. The interval is a subset of the
   old condition, so the fused form runs only where it was already proven to round identically.
3. **A stage of the ring holds 3 K blocks on the 64-row tile and 4 on the 128-row tile, two stages
   in flight.** The per-stage work (copies, scales, barrier, the fp4 expansion) is spread over more
   MMAs, and a 64-row block stays under half an SM's 102 KB of shared memory, so two blocks share
   an SM (ncu on the first two-block form, `receipts/look-s5`: occupancy 16.5% to 32.4%).
4. **mHC.** `hc_post` and `hc_pre` give a thread 8 neighbouring columns (one 16-byte load per copy,
   all in flight, one store per output copy); the partial pass lands its x values through cp.async;
   a one-token pass runs `v41_hc_mixes_part1_kernel`, 4 chains per block with the token's x and
   each chain's fn in shared memory at once (one memory round trip for the pass).
5. **The indexer score keeps a warp's 16 keys in registers** as MMA fragments for 8 query rows and
   double-buffers the Q rows through cp.async, the next live row in flight while the current one
   runs: one barrier per row, 18 KB a block, five blocks per SM.
6. **Fragments through `ldmatrix`, four tiles per load,** in the bf16 tile GEMM, the sparse
   attention (Q, K, P; V two n8 tiles per transposed load) and the indexer score: ncu had the
   load/store unit as the busiest pipe of the sparse attention.
7. **The sparse attention stages the row's indices once,** so a key block's gather waits on one
   memory round trip, not on its index load and then its rows.
8. **Small-M GEMMs.** The bf16 kernel lands its rows through cp.async, the whole K slice at once
   when the launch is at most 256 warps (a few-column output is bound by one fetch after another);
   the block-scaled kernel loads 10 K blocks ahead when a warp's share of K is longer than 8 blocks
   (20 blocks: two rounds, not three) and loads its tile entry together with the tile count.
9. **Small kernels.** The router loads its logits and bias together and keeps a lane's values in
   registers (`v41_route_kernel<12>` up to 384 experts); the MoE placement is one thread per token
   over a per-expert token bitmap (`moe_plan` 0.41 to 0.017 ms at 4,096 tokens); RMSNorm keeps a
   thread's values and weights in registers for both passes; the fp32-FMA tile GEMM stages a K tile
   with one 16-byte load per operand.
10. **The top-k emit has two forms.** Many rows at once (prefill): the keys are read strided, each
    warp ballots its keys above and equal to the threshold into a table, a prefix over the equal
    ballots ranks the ties and a prefix over the taken ballots gives the output slots; six barriers a
    row whatever its length, where the first form ran four per 256 keys. A launch of at most 4 rows
    (decode): each thread owns a contiguous run of the row and two block scans place it. The
    contiguous form alone (the first commit of this half, `a29d1db474`) was 2x slower than the base
    on 131,072-token prefills (`receipts/profab-f24-128k`: top-k 126 to 257 ms, candidates 100 to
    179 ms on layer 20); at 32,768 tokens the table form takes
    4.3 ms where the contiguous form took 10.1 and the base 5.2 on layer 20, and 2.2 against 4.5 ms
    for the candidates (`receipts/profab-f30-32k`, `receipts/prof-aec3b0b401`). At 4,096 tokens
    the tables cost 0.006 ms (top-k) and 0.008 ms (candidates) a layer over the base.

## What did not work

Kept here so nobody runs them again. Each has its raw runs under `receipts/`.

| tried | result | receipts |
|---|---|---|
| Small-M block-scaled GEMM staged through a per-warp cp.async ring (fragments from shared memory), scale codes staged by a load-then-store loop | decode +6..11% per layer: experts 0.157 to 0.250 ms, `q_b` 0.031 to 0.047; `wo_b` 0.065 to 0.036 | `multi-f5a`, `look-s5` |
| The same with the whole K range of a warp in flight when two blocks fit an SM, scale codes loaded into registers first | experts 1.14x, `q_b` 1.20x, shared expert 1.09x slower in the layer; `wo_b` (K 8,192, 640 blocks) 0.56x. Net flat on one card; dropped. A K-keyed dispatch for `wo_b` alone is left (see below) | `multi-f14`, `profab-f14` |
| bf16 tile GEMM on a two-stage cp.async ring | `wo_a` prefill 1.10x slower (30 KB a block halves the blocks per SM); the plain staging stays | `profab-f9` |
| 64-row tiles for every pipelined GEMM (two blocks per SM everywhere) | dense shapes 1.01..1.07x slower, experts at 512 rows 1.11..1.15x slower; 128-row tiles stay above 128 rows | `gemm-f15x` |
| Sparse attention with 8 warps per block (8 keys and 64 output dims per warp, the fp32 P sums re-added in the 4-warp order) | byte-identical, prefill 1.10x and decode 1.08x slower; the 4-warp layout stays | `multi-f20`, `profab-f20` |
| mHC partial pass with each chain's fn chunk in shared memory instead of registers (no spills) | flat at prefill; the register form stays | `multi-f18` |
| The router's selection unrolled to 32 experts per lane | 15.1 us against 13.0 us; unrolled to the 12 a 384-expert router needs instead | `profab-f9`, `profab-f20` |
| The largest shared-memory split for the 61..86 KB pipelined blocks | nothing to gain: an SM has 102 KB, one such block fits either way | `ncu-524c0e693f`, `look-s5` |
| 16 warps per block splitting K in the small-M block-scaled GEMM (512 threads) | decode 1.09..1.10x slower per layer | `multi-f28` (arm `f25x`) |
| The small-M block-scaled GEMM with every load unconditional and what a lane lacks selected away afterwards (no per-lane branches) | decode 1.12..1.17x slower with it; the same arm without it is within 2.5% of its base. A lane that loads nothing is cheaper than a branch | `multi-f28` (arm `f28`), `multi-f30` |
| The indexer score with a block's Q rows serving 8 key tiles (71 KB, one block per SM) | `index_score` 1.39..1.47x slower at 32,768 tokens; the one-tile form with five blocks per SM stays | `profab-f30-32k` |
| A tighter sparse-attention gather loop (addresses by shifts) | 0.99 at prefill, 1.00 at decode; dropped | `profab-f30-32k` |
| The table top-k emit for a single row | 0.040 to 0.062 ms on a 32,768-key row; a launch of at most 4 rows keeps the contiguous form | `profab-f30-32k` |
| Fragment loads hoisted ahead of the MMA chain in the small-M bf16 GEMM, and the next chunk loaded ahead of the fma chain in the narrow fp32-FMA kernel | flat (20.3 to 20.8 us, 21.9 to 21.6 us): both kernels already sit on their chains | `multi-f33a`, `profab-f33a-4k` |
| Row addresses hoisted out of the small-M block-scaled GEMM's load loop | flat on every shape and in the layer: the compiler had hoisted them | `multi-f33b` |
- The query projection on the card's second stream under the keys side (one token): 0.583 /
  0.562 / 0.468 -> 0.581 / 0.557 / 0.468 ms a layer (`receipts/multi-x1`, arms B and C). A fork
  and a join between two streams cost 7 to 10 us; the keys side of a plain layer is 14 us.
- The routed-expert tile kernel at three blocks an SM with 8 K blocks loaded ahead instead of 10:
  decode 1.02 to 1.03x slower (`receipts/multi-x6`). The blocks are not waiting for SM slots.

## Arms

The timed arms are binaries of the working tree at successive points; a receipt directory's
`arms.txt` or `bins.sha256` names them. Each later arm is the one before it plus or minus the
listed changes.

| arm | content |
|---|---|
| `aec3b0b401` | the base commit |
| `s5` | staged small-M GEMMs (block-scaled and bf16), router bias loaded ahead, top-k emit scans, indexer staging through cp.async, sparse-attention indices staged once, fused-epilogue flags, 3-stage ring on the 64-row tile, `hc_post` in 16-byte chunks |
| `f9` | `s5` without the staged small-M block-scaled GEMM; plus the pending scale codes in a register, mHC x staging through cp.async, a one-token mHC pass with 80 fn loads in registers over two 13-warp blocks, the bf16 tile GEMM on a cp.async ring, the fp32-FMA tile staging, the router unrolled to 32 |
| `f14` | `f9` plus the staged small-M block-scaled GEMM with the whole range in flight, the one-token mHC pass over three 9-warp blocks (it spills), `hc_pre` in chunks, the register RMSNorm, the top-k emit's key masks |
| `f15x` | 64-row tiles for every pipelined GEMM (compare its pipelined rows only) |
| `f17` | `f9` without the bf16 tile ring; plus `v41_hc_mixes_part1_kernel`, `hc_pre` in chunks, the register RMSNorm, `v41_route_kernel<12>`, the top-k emit's key masks, the bitmap MoE placement |
| `f18` | `f17` plus the mHC fn chunks in shared memory and the indexer with its keys in registers |
| `f20` | `f18` plus the 8-warp sparse attention and the two-stage ring of 3 / 4 K blocks |
| `f21x` | `f20` plus the 10-block load-ahead of the small-M block-scaled GEMM and the tile entry loaded with the count |
| `f24` | `a29d1db474`: `f21x` without the 8-warp sparse attention and the shared-memory fn chunks, with the ldmatrix fragment loads and the load-ahead keyed on K |
| `f25x` | `f24` with 16 warps per block in the small-M block-scaled GEMM |
| `f28` | `f24` plus the small-M block-scaled GEMM's unconditional loads, the tighter sparse-attention gather and the table top-k emit |
| `f30` | `f28` without the unconditional loads; plus the indexer with 8 key tiles per block |
| `f33a` | `f24` plus the table top-k emit above 4 rows, the hoisted fragment loads of the small-M bf16 GEMM and the look-ahead load of the narrow fp32-FMA kernel |
| `f33b` | `f33a` plus the hoisted row addresses of the small-M block-scaled GEMM |
| `f34` | this head: `f24` plus the table top-k emit above 4 rows |

## What the profiler said

- An SM of this card has 102.4 KB of shared memory and a block at most 99 KB. A kernel's blocks
  per SM follow from that: 49 KB for two, 33 KB for three.
- A plain load-then-store staging loop runs one memory round trip after another; cp.async, or
  loads into registers first, puts them in flight together. This was the whole gain in the mHC
  kernels, the top-k emit and the sparse attention's index stage.
- Local memory is global memory. `sass_audit.py` finds it statically: any LDL / STL in a hot kernel
  is an array indexed with a runtime value or a spilled register.
- Per-lane fragment loads from global memory are not slow by themselves: the small-M block-scaled
  GEMM that loads 4 bytes per lane straight into registers beat both staged versions on every
  shape but one. A lane that skips a load behind a branch is cheaper than a lane that loads and
  discards.
- Two decode kernels sit on a dependency chain and nothing around it helps: the small-M bf16 GEMM
  runs K / 16 accumulating MMAs one after another (320 for K = 5,120: 20 us, about 62 ns each), the
  narrow fp32-FMA kernel K fmas (5,120: 22 us). Both chains are the reference's order.
- A kernel alone on the card and the same kernel with every SM full want different schedules: a
  thread reading its own contiguous run is the fastest top-k emit for one row and the slowest for
  8,192 rows at once.

## One row on one card (2026-11-10)

A decode layer reads about 307 MB of weights a token, 186 us at the card's copy rate, and took
482 us (short context) to 573 us (position 4,096, layer 20). Five kernels fill the card (the
routed-expert tiles, wq_b, wo_a, wo_b, the shared expert); the rest are a few blocks each on a
dependency chain (the router's fp32 chain, the plan, mHC, the norms, the window KV) and leave it
empty. Per launch, one rank, short context (`receipts/nrank-m6-n1`):

| kernel (grid) | us | what bounds it |
|---|---|---|
| expert tiles, w1 / w3 (288 x 8 blocks), w2 (640 x 8) | 51.2, 50.3 | see below |
| wo_a, bf16 (32 blocks) | 46.2 | 67 MB at 1,456 GB/s |
| wo_b (640 blocks, K = 8,192) | 43.2 | four load rounds a warp; 980 GB/s |
| wq_b (4,096 blocks, K = 1,280) | 28.5 | 42 MB at 1,470 GB/s |
| router, fp32 chain (96 blocks) | 22.7 | 5,120 dependent FMAs |
| shared expert w1 / w3 (288 blocks), w2 (640) | 12.6, 12.4 | two load rounds a warp, one wave of blocks |
| wq_a (160), wkv (64) | 9.2, 6.2 | one wave of blocks |
| `hc_mixes_part1`, `hc_finish`, RMSNorm, route, plan scan | 6.7, 5.0, 3 to 5, 6.9, 5.3 | one block on a chain |

What moved the layer, each interleaved against the head before it, ms a layer at position 4,096 on
layers 2, 20, 21 (three runs an arm, spreads under 1%, hashes identical):

| change | before | after | ratio | receipts |
|---|---|---|---|---|
| the shared expert on the card's second stream, under the router chain and the plan (`13d16037b4`) | 0.615, 0.593, 0.500 | 0.592, 0.568, 0.475 | 0.961, 0.959, 0.950 | `multi-f1` (0.947, 0.951, 0.937 on quiet cores, `multi-x1`) |
| expert tiles with the MMA's operands exchanged, up to 8 rows (`3c83d8c1a6`) | 0.591, 0.568, 0.476 | 0.563, 0.535, 0.446 | 0.953, 0.942, 0.937 | `multi-m3` |
| the one-part tile instance without the loop over parts of K (`9ea7ec0b7f`) | 0.563, 0.539, 0.446 | 0.552, 0.529, 0.437 | 0.981, 0.983, 0.980 | `multi-m5` |
| fp32 FMA small-M for 5..8 rows in one launch (`74e1e68de1`), the keyed attention entry (`9500e9960a`) | 0.550, 0.528, 0.436 | 0.557, 0.530, 0.437 | flat | `multi-m7`, `multi-k1` |

Three-layer mean times 40, without the head: 20.2 ms a step (22.9 at `29aeeee39f`). `1a2bdbef7d`
(launchers, mHC rows, Engram staging) is gated (`gates-a1`) and not yet timed.

## Calls of 2 to 8 rows (a drafted block's verification, a batch of sequences)

- The tile list scales with its tiles, not worse. One projection launch, one row a tile
  (`receipts/gemm-k2`): 6 / 12 / 29 / 48 expert tiles take 43.9 / 78.6 / 237.5 / 386.5 us for w1 or
  w3 and 37.8 / 68.0 / 202.9 / 337.0 for w2: 21 us a distinct expert at 6 tiles, 23 at 29 and 48
  (15 tiles of two rows: 19.6). In an 8-token forward of layer 21 the experts are 706 us of 1,144
  (`receipts/rows-k1`); a block lives about 20 us for 5 us of MMAs, waiting on four load rounds
  whose K blocks it holds in registers.
- mHC mixes: the one-token kernel ran only for exactly one row; 2 to 8 rows took the 16-token pass,
  53 us against 6.7, twice a layer. `1a2bdbef7d` gives the one-token kernel a block column a token
  up to 8 rows (`schedule/hc_mixes/rows8` in the kernel gate: its rows equal the 16-token pass's).
- The fp32 FMA linears (router gate on every layer, the head once a call) took at most 4 rows a
  launch, so 5 to 8 rows ran them twice: a fixed step of about 2 ms on a 9-layer call (the batching
  lane's rows table: 10.48 -> 12.91 ms from 4 to 5 rows). `74e1e68de1`: one launch (8 rows on a
  256-thread block for the head; K landed in two halves for the router), and the step is gone
  (10.28 -> 11.50).
- The Engram staging buffers were copied whole to the device on every call. They are sized for the
  largest prefill chunk so far: 50.3 MB after one of 8,192 tokens, 0.88 ms of every decode step per
  Engram layer (`receipts/nrank-k1-n1-9l`, 128,745 tokens of context), 23 MB after a 3,745-token
  prompt. `1a2bdbef7d` copies the call's rows.
- Launchers: a kernel's shared-memory limit is one setting a function a context. The launchers set
  it to the call's size before each launch, so two threads on one card with different sizes could
  launch above the limit the other had set (CUDA error 1 or 701 in a pipeline whose stages share a
  card). `1a2bdbef7d` sets each limit and carveout once a device; the kernel gate's
  `stress/sparse_attn_two_threads_two_widths` holds it.
- `9500e9960a` adds `memra_v41_sparse_attn_keys`: a query row's window rows and compressed rows
  named by a table entry, so a decode step reads the ring and the compressed KV where they live
  (the copy into one key buffer is 134 MB and 123 us a layer at position 131,072 on a ratio-1
  layer) and a batch reads each sequence's own.

## One row against vLLM, by family (us a layer)

vLLM nightly on a 9-layer copy of the checkpoint, one card, one graph replay a step (the baseline
lane's nsys profile, kernel time a step / 9) against this route at about 4k context on a plain
layer (`receipts/prof1-m1-4k`, the per-launch table above):

| family | vLLM | here | note |
|---|---|---|---|
| dense MXFP8 linears | 132 (5.3 launches at 24.7) | 138 (wq_a 9, wq_b 29, wkv 6, wo_b 43, shared expert 38, `act_quant` 13) | level |
| MoE: grouped GEMM and helpers | 107 + 19 | 128 + 22 | the tiles read at 0.9 TB/s where theirs is at the bus |
| bf16 GEMM: wo_a and the router | 47 | 46 to 50 + 23 to 29 | wo_a is read-bound; the router is this route's fp32 FMA chain |
| mHC with its norms | 27 | 40 | four launches a half-layer against two fused |
| sparse attention | 19 | 37 at 640 keys | the same schedule; four blocks walk ten key tiles in turn |
| norms, quant, RoPE, KV insert | about 28 | about 28 | level |
| Engram staging copy | 0 | 47 (0.42 ms a call at 3.7k) | removed by `1a2bdbef7d` |
| head, ms a step | 0.94 + 0.13 | 0.92 | 1.32 GB of bf16 at 1.44 TB/s: read-bound |
| layer | about 380 | about 467 + 47 | |

The router: its 23 to 29 us are 5,120 dependent fmas an output, the reference schedule's order. A
faster router needs a different reduction order, which is not a schedule change: it would be the
first item of this route to need a distribution-level gate against the vendor floor (router ids
and weights over a corpus) instead of the byte gate. vLLM runs the router as a bf16 GEMM launch;
that is the precedent. Not opened.

The second stream: a branch that depends on nothing the main stream is about to compute starts
behind a fork event and is joined before its first reader. The shared expert reads only the FFN
input, so it runs while the router's chain and the plan leave the card empty. It saves 32 us of
the shared expert's 45: the router's chain takes 42 us with the shared expert on the card against
29 alone, and a fork and a join cost 7 to 10 us.

The expert tiles. Three probes say what does not bound them: three blocks an SM (1.02 to 1.03x
slower, `multi-x6`), half the weight-load instructions (paired lanes and a warp shuffle, 1.01x
slower, `multi-t2`), half the FP4 conversion instructions (the weights given to the tensor core as
e2m1 through `mma kind::f8f6f4`, exact by the kernel gate, 1% faster, `multi-m4`, not kept: it
ties the kernel to one architecture's MMA kinds). Two say what does. The e4m3 MMA tile is 16 x 8;
the small-M body put 8 weight rows on its 8-wide side and the activation rows, at most 8, on its
16-wide side, so half of each MMA was zeros. With the operands exchanged (16 weight rows on the
16-wide side) the six experts of a token go from 163 to 128 us (`prof1-m1-4k`): the MMAs halved
and 35 us went. The tensor core gives the same bits with the operands exchanged (kernel gate,
125/125). What is left is about 35 us of MMAs (414K a token-row: one per K block per 16 weight
rows, run one after another by a tile's tensor unit at about 62 ns) and about 90 us that scale
with the weight rows read, 113 MB at about 1,250 GB/s, near what wq_b and wo_a reach. So a
token-row of experts costs about 0.10 ms a layer (0.13 before) and is close to a read bound plus
an MMA bound; rows that share an expert share both (a tile of r rows is one expert's read).

The dense FP8 GEMMs keep the 8-column body. With 16 rows of fragments a warp has registers for 5
K blocks a load round instead of 10, and a kernel that is one wave of blocks is bound by its
rounds: wo_b 44 -> 68 us, the shared expert 45 -> 59 with the exchanged form.

## N ranks, one token: where a decode step goes

`dsv41-tpbench` decode on four ranks, 8 layers resident, `MEMRA_TP_TRANSPORT=peer-pull`, under nsys;
`decode_window.py` and `thread_gaps.py` read the trace. Every rank is on the one card these runs
had, so the four streams share its SMs and its memory bus: the host numbers are the code path's,
the kernel times are contended and are not what four cards give.

Build `8c8c5544b9` (`receipts/nrank-m7-n4-pp`), per layer:

| | us per layer |
|---|---|
| wall | 965 (one rank on the card: 644, stream busy 93%) |
| a rank's stream busy | 445 (46%) |
| a rank's stream idle before a kernel that was already launched (an event wait, or the card busy with another rank) | 260 |
| a rank's stream idle waiting for the host's launch | 44 |
| a rank's issue thread inside non-blocking driver calls | 303: 48 launches, 44 `cuMemAllocAsync`, 44 `cuMemFreeAsync`, 11.5 peer copies, 18 event waits, 6 records |
| a rank's issue thread between two sections of the layer | 435: 8.5 gaps of 10 to 200 us; the longest is the routing readback, 150 us on the calling thread |

The layer is 19 fork-join sections (one per group of per-rank work, three per gather) and one
blocking host read (rank 0's expert offsets, which size the expert gather). One card with one rank
has none of this: its stream is busy 93% of the step and a launch waits 0.3 us.

Two changes on the host, no kernel and no order on a stream touched (`16ed52eeb3`), then the host
read itself (`1559672e22`):

- The rank threads are the group's own: rank 0 is issued by the calling thread, the others by
  threads that spin on an atomic for their next section (200 us, then they park). rayon's
  `install` + `par_iter` woke sleeping workers at every section.
- `Ops::tmp` and `Ops::buf` take a freed buffer of the same size from the stream's free list (up
  to 1 MiB a buffer, 64 MiB a stream) and a dropped buffer goes back to it: the 88 allocation calls
  a layer are 2. A buffer is reused only on its own stream, so the device sees the order a
  stream-ordered free and allocation give.
- A forward of up to 4 tokens keeps the MoE plan on the device. Every rank runs its tile lists
  into its own region of a buffer of N regions, the regions are gathered whole (61 KiB a rank at
  one token), and `v41_moe_owner_kernel` offsets each entry of the combine's order table to the
  region of the rank that owns the pair's expert. The combine adds the same rows in the same
  order. The host's check on the offsets is a sticky device flag read with the logits.

What that does to the host with four ranks on the one card (`receipts/nrank-n1-n4-pp`,
`nrank-n2-n4-pp`; both traces ran with a build on the same cores, so read the counts, not the
times): driver calls a layer a rank 202 -> 138 (allocation calls 88 -> 2), gaps of 10 to 200 us
between two sections of a layer 8.5 -> 0.5 a layer, blocking device-to-host reads 12 -> 8 a step
(the 8 are the logits and the flag). The wall on the one card barely moves, 7.752 -> 7.651 ms a
step (`receipts/nrankab-n1`, three runs before a quiet window cut it): four streams on one card
are bound by the card. With a build compiling on the rank threads' cores the spinning threads are
preempted and the step is slower than with parked workers (8.18 -> 8.50 -> 10.15 ms,
`receipts/nrankab-n2`): the rank threads need their cores. The separate-card runs are in the next
section.

Gates on the final tree: `dsv41-gate kernels` 123/123; `layer` 161/161 with its 111 saved tensors
byte-identical to `aec3b0b401`'s; `tp` at N=2 358/358 and at N=4 704/704, each on peer-pull and
on the default transport; `decode` 240/240, 240/240 and 60/60 on its three captures
(`receipts/gates-n3`); generated ids identical across every timed run of every arm.

## Tensor, expert and pipeline parallelism at one row

The same decode on two cards, one rank a card (20 layers resident, peer-pull), and on four ranks,
two a card, against one rank on one card. Timed runs are interleaved (A B B A A B, three rounds of
64 steps a run); the per-layer rows are from nsys traces (`decode_window.py`), the trunk layer
without the head.

| | one rank, one card | two ranks, a card each | four ranks, two a card |
|---|---|---|---|
| step, ms (20 layers + head), `8c8c5544b9` | - | 9.986 | 12.251 |
| step, ms, `1559672e22` + the top-k change | - | 9.802 (0.982) | 12.071 (0.985) |
| trunk layer, us (median, traced) | 482 | 432 | 523 |
| a rank's kernels busy, us a layer | 487 | 349..360 | 345..357 |
| copies | 7 | 13 | 20 |
| waiting for the host's launch | 2 | 2 (34 at `8c8c5544b9`) | 62 (146) |
| waiting for a peer before a gather copy | - | 46..56 | 74..96 |
| a launched kernel waiting (an event, or the card's other rank) | 17 | 36 | 90..107 |
| routed-expert tiles: mean rank / busiest / idlest, us a layer | 152 | 82 / 105 / 60 | 51 / 86 / 18 |

(`receipts/nrankab-sep-n2`, `nrankab-sep-n4`, `nrank-sep-*`, `nrank-m6-n1`; the second card runs
at a 280 W cap, the first at 250 W.)

Two ranks on two cards run a layer in 0.90 of one card's time, where half the bytes would be
0.50; four ranks do not get a rank's kernels below two ranks' (345..357 us against 349..360). The
step at one row is not bound by weight bytes. A layer reads about 307 MB of weights a token
(routed experts 106, wo_a 67, wq_b 42, wo_b 42, the shared expert 35, wq_a and wkv 9, the router
4), 186 us at the card's 1,650 GB/s copy rate, and one rank takes 482 us for them:

- kernels that fill the card: wq_b reads at 1,500 GB/s and wo_a at 1,456 (at the bound), wo_b at
  980, the shared expert's three launches at about 900, the routed-expert tiles at 743;
- kernels that are a few blocks on a dependency chain and leave the card empty: the router's fp32
  chain (28 us), the mHC pair (27), RMSNorm (16), the plan and the route (19), `act_quant` (12),
  sparse attention (11), SwiGLU, RoPE, the combine. These are replicated on every rank and do not
  shrink at any N: about 190 us a rank at N = 2 with the shared expert, wq_a and wkv, which are
  also loaded whole on every rank;
- what splitting halves is the fill kernels only: dense block GEMMs 139 -> 98 us a rank (five of
  their seven launches are replicated), tiles 152 -> 82, wo_a 49 -> 38; and three gathers a layer
  give most of that back (46..56 us of waiting, 36 of event waits, 13 of copies).

The host changes of this branch remove the host from a separate-card step (34 -> 2 us a layer of
launch wait) and are worth 2% there; they are worth more where ranks share a card's context.

Expert layout. The route today gives each rank whole experts (96 a rank at N = 4). With top-6
routing the busiest rank's tiles take 1.28x the mean at N = 2 and 1.68x at N = 4, and every rank
waits for it at the expert gather. A split of every expert across the ranks like a TP MLP (w1 and
w3 by hidden rows, a gather of the hidden rows, w2 by output rows; every step is row-wise, so the
bytes hold) balances the ranks, lets the shared expert split the same way and needs no host read
of the plan at any row count, for one more gather a layer. From the numbers above it is worth 20
to 40 us of a 400 to 520 us layer at one row. It is not built: at one row the expert bytes are not
the bound. They are the only term that grows with rows (the drafting lane measured +0.13 ms a
layer a row, the six experts' 106 MB), so the split pays for verify rows and batched sequences,
not for one stream.

Pipeline parallelism, from the one-rank number: a request's step through four cards in series is
one card's layer time, 0.48 ms x 40 = 19.3 ms plus three hand-offs of the stream (22.9 ms from the
layer bench at position 4,096), with no collectives and four requests in flight, one a stage.
Four ranks today take about 29 ms a step (the 4-card lane's number) and two ranks a card 0.52 to
0.61 ms a layer here. For this model with these kernels the pipeline is at least as fast for one
request and scales with concurrency without batching; tensor parallelism wins only if the
replicated, latency-bound part of the layer shrinks.

## Four ranks on four cards: the experts split by rows, gathers inside kernels (2026-11-10)

The first runs with a rank a card on four cards (two with a 250 W limit, two with 280 W; the
limits are the box's and were not changed), all 40 layers resident (71.3 GiB a rank), one lease,
`receipts/four`. Every timed run reads the cards' slowdown counters before and after: none moved.

Arms, each on top of the one before:

| arm | commit | what |
|---|---|---|
| A | `dfb6d457d7` | experts by owner: a rank holds 96 whole experts and the whole shared expert; peer-pull gathers |
| B | `173d61cf6f` | the experts split by rows: a rank holds 1/4 of every expert (hidden rows of w1 and w3, output rows of w2), routed and shared; no host read in the MoE |
| C | `940e7e6289` | B with `MEMRA_TP_TRANSPORT=device-push`: a gather of at most 256 KiB a rank is two kernels a rank (push, pull), no event and no copy call |
| D | `5d74b1db2a` | C with w1 + w3 + SwiGLU of the routed pairs in one launch, the shared expert's hidden slice on the second stream, the plan of a few rows in one kernel |

Byte gates at four ranks on four cards (`dsv41-gate tp`, the capture's prefill and two decode steps
against the one-card route): B 704 checks, 0 failed on peer-pull and on the default transport; C and
D 706 checks, 0 failed on device-push. The two extra checks are the small-gather stress (20,000
gathers of five region shapes with no host wait between them, every consumer checking every
producer's region: 0 words differ) and its control, which has to fail to count: with the sequence
word stored before the data, 2,000 gathers give 30.6 million differing words.

One-token decode, ms a step (arms in order, reversed, in order; 3 rounds of 48 steps a run; the
generated ids are identical across every run of every arm):

| arm | run medians | step | us a layer, all in | against A |
|---|---|---|---|---|
| A | 17.486, 17.545, 17.575 | 17.545 | 438.6 | 1.000 |
| B | 15.677, 15.665, 15.691 | 15.677 | 391.9 | 0.894 |
| C | 12.996, 13.012, 13.003 | 13.003 | 325.1 | 0.741 |
| D | 12.351, 12.363, 12.345 | 12.351 | 308.8 | 0.704 |

A block of R rows (a forward of the first R rows of a prompt through the 40 layers, then the head
for the last row and its read-back; median of 16, ms). The time is to the first token: the bench's
`forward_s` is the host's issue time and is not a result.

| rows | A | B | C | D |
|---|---|---|---|---|
| 1 | 16.97 | 15.29 | 12.64 | 12.02 |
| 2 | 22.11 | 17.77 | 15.28 | 13.87 |
| 4 | 33.74 | 20.12 | 17.55 | 16.05 |
| 8 | 33.48 | 26.22 | 22.88 | 20.79 |

D adds 1.25 ms a row from 1 to 8 rows.

### Where D's layer goes

From the traced run of D (`receipts/four/trace-D`: 12.76 ms a step under the profiler), per rank
per layer. Inside a layer the stream is busy all the time (276 us median, 278 us of kernels): the
kernels run one after another and nothing waits on a peer.

| family | us a layer a rank (launches) |
|---|---|
| dense MXFP8 linears: wq_a, wkv, wq_b, wo_b, the shared expert's three | 62.7 (7.25, 8.6 each) |
| routed experts: w1 + w3 + SwiGLU 34.0, w2 12.8 | 46.8 (2) |
| `wo_a` | 38.1 (29.5 a launch) |
| mHC: mixes 11.9 + 9.0, post 3.9, collapse 2.2 | 27.0 (8) |
| the router's fp32 chain | 26.6 (23 a launch) |
| gathers: push 16.2, pull 14.7 | 30.9 (4 gathers, 7.7 each) |
| RMSNorm 15.7 (3.8 a launch), `act_quant` 11.9 (10.25 launches) | 27.6 |
| sparse attention 11.2, RoPE 2.9, key quantization 1.6, index 1.9 | 17.6 |
| MoE plan 6.3, route 5.9, combine 3.6, shared SwiGLU 1.3 | 17.1 |
| the head, a step | 220 us a rank (5.5 a layer) |
| outside the 40 layers, a step | 1.7 ms: the head, the logits read back, the bench's scoring of 129,280 logits, the embedding upload |

A gather cost 33 us at two ranks through events and copy calls (20 us waiting on the peer's event);
inside kernels it is 7.7 us (push 4.0, pull 3.7) and the wait for a peer is 1.3 us a layer.

The host: a rank's issue thread makes 52 launches a layer at 2.4 us and spends 79 us between calls,
204 us of issue a layer against 276 of card time. Below about 200 us a layer the issue thread is
the bound, so past the next cuts a rank's step has to become one CUDA graph replay.

### One row after the first pass: what paid and what did not

Second and third pass of the same block (`receipts/four2`, `receipts/four3`; every arm gated at
four ranks on device-push and on peer-pull before it was timed; no slowdown flag in 46 runs).

| change on top of D | step, ms | against D |
|---|---|---|
| D (`5d74b1db2a`) | 12.349, 12.360 (two passes) | 1.000 |
| the mHC collapse and its RMSNorm in one launch | 13.327 | 1.079 |
| that plus device copies of at most 1 MiB as kernels | 13.384 | 1.084 |
| the mHC mixes of a few rows on the second stream, under their sublayer | 11.624 | 0.940 |

The fused collapse + RMSNorm kernel was a regression of 25 us a layer at every row count (blocks
of 1, 2, 4, 8 rows: +1.0 ms each): one thread did 80 single-element strided loads where the two
kernels load 16-byte chunks; its launch took 18 us against 5.5 for the two. The copy kernels moved
3.9 us a layer from copy calls into kernels and bought nothing. Both are removed. The mixes under
the sublayer are worth 18 us a layer: nothing between a sublayer's mixes and their first reader
(the post-mix after the sublayer) needs them, so the two launches a sublayer run on the second
stream beside the attention and the MoE. The best exact one-row step on four cards is 11.62 ms
(290.6 us a layer all in), from 17.55.

### Prefill on four cards: transfers between the cards

A prefill on four separate cards was slower than on one: 4,096 tokens in 9.19 s (A) and 7.58 s (B),
32,768 in 73.8 s and 59.9 s, where one card takes 32.5 ms a layer. The trace of B
(`receipts/four/pre-trace`) has it in one row: per rank per layer the peer copies take 114 to 172 ms
for 0.214 GB, 1.2 to 1.9 GB/s, and every kernel together 12.9 ms (expert tiles 5.08, dense 1.89,
mHC mixes 1.55, `act_quant` 1.37, sparse attention 1.06). The unsplit route's own capture
(`receipts/prefill-n4`) shows the same: 138 ms of peer copies a layer for 0.267 GB.

What it is not: the stride. Packing a region of columns into a contiguous block on the producer
and copying that changed nothing (7.25 s against 7.59; `receipts/four2`). What the copy records say
(`copy_rate.py`, `copy_pairs.py` over the traces): the same copy has a best time at the link's rate
and a median 25 times slower. 10 MB: 0.19 ms best (54 GB/s), 4.6 ms median. 84 MB: 1.49 ms best,
42.6 median. 264 MB: 4.7 ms best (56 GB/s), 139.6 median. A copy that runs alone gets 50 to 56 GB/s
on almost every pair of cards; with four in flight the median is 1.2 to 1.9 GB/s, in any pattern:
pulling from the peers in ring order (each card the source of one copy and the destination of one
at a time) left the rate of a copy where it was.

| 4,096 / 32,768 tokens, time to first token, s | |
|---|---|
| experts by owner, peer copies (A) | 9.19 / 73.8 |
| split by rows, peer copies (B) | 7.58 / 59.9 |
| + contiguous blocks | 7.25 / 57.5 |
| + padding skipped in the tile body | 7.21 / 57.5 |
| + the mHC sums split across ranks (one more gather a sublayer) | 7.47 / 58.4 |
| + hidden columns gathered as codes, ring order (peer copies) | 4.25 / 33.1 |
| the same build, large gathers as kernel stores into the peers (device-push) | 2.55 / 20.0 |

The last two rows are one build on its two transports. Gathering the split MoE's hidden columns
as e4m3 codes and scales (each rank quantizes its own columns; the 32-wide blocks lie inside a
rank's columns) took the gathered bytes from 0.214 to 0.164 GB a rank a layer, which is the copy
path's gain. Under device-push a gather over 256 KiB makes no copy call: the producer's kernel
stores its region into its slot of every peer's inbox and publishes an event, the consumers unpack.
That is 2.85 times the copy path as it was and still 63.7 ms a layer where the kernels are 13: the
stores land at a few GB/s a card with twelve flows at once, the same order as the copies. One
transfer at a time at the single-flow rate would move a layer's 0.66 GB (all four ranks) in 12 ms.
The mechanism of the collapse is not known yet; the four cards sit on four root buses.

### Two cards after the block: what a transfer costs

`dsv41-bench p2p` on two of the cards (`receipts/p2p-01`; GB/s a flow, best equals median to 0.1):

| | 8 MiB | 64 MiB | 256 MiB |
|---|---|---|---|
| copy engine, one flow | 54.2 | 55.9 | 56.1 |
| copy engine, both directions at once | 47.6 | 49.4 | 49.6 |
| kernel storing into the peer, one flow | 7.6 | 7.7 | 7.7 |
| kernel storing into the peer, both directions at once | 7.5 | 7.5 | 7.5 |

Between blocks that live across the copies, two flows do not collapse; the kernel store is the
same 7.6 GB/s a source whatever else runs. The same two cards on a prefill (two ranks, 20 layers,
`receipts/two`, time to first token at 4,096 / 32,768 tokens):

| | |
|---|---|
| plain strided peer copies (`5d74b1db2a`) | 27.5 s / stopped |
| packed peer copies, both ranks at once | 0.72 / 22.6 s |
| packed peer copies, one transfer at a time | 0.75 / 5.29 s |
| kernel stores, both at once | 0.96 / 7.02 s |
| kernel stores, one at a time | 1.23 / 9.34 s |

So the collapse needs both: blocks allocated for the gather (the large ones of a 32,768-token
prefill, 264 MB) and transfers at once. One-token decode on the pair, 20 layers: 9.304 ms a step
unsplit, 7.621 at `5d74b1db2a`, 7.339 with the mHC mixes under their sublayer.

Byte gates of this branch's head on one card (`receipts/gates-pu2` for `5d74b1db2a`): kernels 138
checks, layer 161 with the 111 saved tensors equal to the reference set, tp at 2 and 4 ranks on
peer-pull, the default transport and device-push (the small-gather stress on a shared card: 0
words differ, its control caught), decode on three captures: 0 failed. The control of the launcher
fix (`receipts/red1`): the previous head fails `stress/sparse_attn_two_threads_two_widths` with a
CUDA error, as it has to.

### Kept blocks, the link, and what stays open

Keeping the gather's packed block and landing block per rank across gathers (`f308e48578`)
removes the collapse on the pair (`receipts/two-pu12`, time to first token at 4,096 / 32,768
tokens): blocks allocated a gather 0.72 / 22.6 s, kept blocks 0.73 / 4.98 s; one transfer at a
time on top of kept blocks 0.73 / 5.23 s, so the switch was removed again, and with it the kernel
stores for large gathers (7.6 GB/s a source against the copy engine's 50). It has not run at four
ranks on four cards.

Why the per-gather blocks collapsed is not known. Two things it is not. Not the first use of new
memory: the probe's line for blocks allocated for each copy runs at the kept blocks' 54 to 56
GB/s (`receipts/p2p-01-pu12`). Not the link: an idle card's link sits at generation 1 and comes
up within two seconds of load, but with each card's generation sampled every 50 ms through a
prefill (`receipts/link`) both links are at generation 5 while the per-gather arm's copies crawl.
That arm is also erratic: the same build gave 0.72 s and 14.85 s for the same 4,096-token prefill
an hour apart, where the kept-block arm gave 0.73 and 0.77 s. Every lease's telemetry carries the
link generation and width since then.

### The host inside a prefill: the Engram tables

With the copies out of the way a one-card prefill showed the card waiting for the host 33 ms a
layer (`receipts/prefill-n1-pu11`), and the host's CUDA calls of the whole forward add up to 20 ms.
A sampler (`receipts/hostprof-fin1`: the bench under gdb, stopped 30 times by a signal) put every
sample of the forward in `Ctx::engram`, most in `memcpy`. Each Engram layer (1 and 14) gathers
about 24 rows a token from a table of 384 million 256-byte rows, 98 GB, memory-mapped from the
checkpoint. Host-only probe, 98,304 random rows (a 4,096-token prompt):

| how the row's page is held | 1 thread | 8 threads |
|---|---|---|
| file mapping, page not yet touched by the process | 6,779 ns a row, 666 ms | 924 ns, 90.8 ms |
| file mapping, page mapped, 4 KiB | 35.9 ns, 3.5 ms | 11 to 14 ns |
| anonymous memory, 4 KiB, touched | 31.9 ns | 13.3 ns |
| anonymous memory, 2 MiB transparent huge pages | 22.0 ns | 17.6 ns |

It is the first touch of the page, 190 times the row. `madvise(MADV_POPULATE_READ)` on the mapping
populates 38 GB/s, 2.6 s a table, with no memory of its own (the pages are the page cache's), so
`fb0ac3627c` maps each table when its layer is loaded, once a process, and splits a large gather
over threads. One card, 8 layers, a first-seen prompt (`receipts/prefillab-en2-n1`): 4,096 tokens
0.654 / 0.540 -> 0.302 / 0.302 s, 32,768 tokens 3.703 / 3.327 -> 2.277 / 2.286 s; a repeated
prompt was 0.32 / 2.27 s before and after. The one-row decode step is unchanged (4.915 ms on one
card both ways, first round included): its two dozen rows a table were not paying the fault in
this bench. The bench's cells record the process's page faults, so pages the cache drops under
pressure show as a count. Two ranks on two cards, 20 layers (both Engram layers), first-seen
prompts (`receipts/two-en2`): 4,096 tokens 0.728 / 0.710 -> 0.481 / 0.477 s, 32,768 tokens
4.748 / 4.660 -> 3.363 / 3.364 s; the one-row step 7.278 -> 7.287 ms. The 2.6 s a table is with
the checkpoint in the page cache; after something dropped the cache the first load reads the
table from disk.

The same pair's traced one-row step (7.33 ms): the main stream is busy 92% of it, so the cards
bound the step. The 31 us a layer it idles are the local copy calls of the window ring and the
compressed keys into the attention's key buffer (10 us, 2.4 calls a layer at 4.3 us each), the
join of the second stream (3 us), and at the step's end 120 us between the logits read-back and
the next step's first call (the bench scoring 129,280 logits, the runner's step setup).

### One-card candidates measured in the same slot

| change | commit | measured |
|---|---|---|
| launcher fix, mHC rows kernel, Engram staging (`1a2bdbef7d`) | landed earlier | flat at one row: 0.552 -> 0.555, 0.530 -> 0.532, 0.438 -> 0.439 ms a layer |
| staged FP4 expert tiles | `eaa42ee764` | one row 0.555 -> 0.533, 0.530 -> 0.513, 0.437 -> 0.419 ms a layer; 48 tiles 386.5 -> 304.1 us (w1/w3), 337.0 -> 304.2 (w2) |
| fp32 head, 16-byte loads | `72b2cf30c3` | the head at 1 / 4 / 5 / 8 rows 0.900 -> 0.868, 1.235 -> 1.072, 1.414 -> 1.200, 1.796 -> 1.450 ms |
| one-row top-k and candidate blocks over a long row (entries, not wired) | `1b7f65695d` | top-k 0.058 / 0.209 / 0.403 -> 0.048 / 0.049 / 0.049 ms at 32,768 / 131,072 / 262,144 keys; candidates 0.035 / 0.106 / 0.197 -> 0.078 / 0.105 / 0.105 ms: the long top-k above 32,768 keys, the long candidates above 131,072 |
| dense small-M body with K in parts | dropped | one row flat; `wo_b` in an 8-row forward 0.056 -> 0.052 ms, 4 us a layer: not worth a second form of the body |

## A layer's second half as one graph, the fused launch on the staged body, the transport default (2026-11-10)

### What a launch, a graph launch and a node update cost

`dsv41-bench graph` issues a chain of RMSNorm launches of one 5,120-wide row three ways on every
rank at once: one by one, as one captured graph, and it rewrites every node's parameters
(`receipts/gprobe-g1`, medians of 15).

| | one rank | four ranks on one card |
|---|---|---|
| a launch issued one by one, host | 1.6 us | 2.0 us a rank |
| the chain done, a launch | 3.6 to 3.9 us | 4.3 to 4.7 us |
| one graph launch, host (16 / 64 / 256 nodes) | 1.4 / 1.9 / 3.8 us | 5.6 / 6.4 / 7.6 us for the four graphs |
| the chain done as a graph, a node | 3.5 to 3.7 us | 3.7 to 4.1 us |
| a node parameter update (`cuGraphExecKernelNodeSetParams`) | 0.08 to 0.09 us | |

A graph launch costs the host what one or two launches cost, whatever it holds, and a parameter
update is 25 times cheaper than a launch. So a captured step can take whatever still varies from
step to step (an input buffer's address, a key count, a position) by rewriting the nodes that hold
it, and nothing has to stay outside the graph for cost.

### The graph door

`MEMRA_DSV41_DECODE_GRAPH=1` (`dsv41_gpu::graph`). `Group::capture` runs a closure with every
rank's stream recording: the launches become that rank's graph, the second stream's work a branch
of it, and nothing executes. Three things make a recorded launch valid later. Its buffers stay
where they were: under a capture every temporary comes from a free list of the capture's own,
allocated outside the stream's order (a stream-ordered allocation inside a capture would become a
node), and kept with the graph. Its inputs are followed: the capture finds every node argument
that holds an address inside one of its input buffers and a replay whose inputs sit elsewhere
rewrites them first. And nothing in it waits for the host: a capture that recorded anything but
kernels and zero fills is refused. The small gathers needed one change for this: a gather's
sequence number was a host counter passed to the push and pull kernels; it is now the rank's own
count of gathers, kept in its device memory, so a recorded gather is the gather of whatever step
replays it.

The first two replays of a span run it both ways, launch by launch on the call's inputs and as the
graph on copies of them (addresses the capture never saw), and compare what each wrote byte for
byte.

Captured so far: the second half of a layer for calls of at most 8 rows (the post-mix of the
attention output, the FFN's mixes, norm, router, plan, expert tiles, shared expert, the MoE's
gathers, combine, post-mix), about 30 of a one-row layer's 52 launches. The attention half takes
positions as launch arguments; it joins when the step's host data sits in one block on the card
(the batching lane's step block) and the rest is retuned by node updates.

Two cards, 20 layers, device-push, three arms forward / reversed / forward, ids identical across
every run (`receipts/pair-g3`): A is the lane head before this section, B adds the staged fused
launch below, C is B with the door on.

| | A | B staged fused | C + the graph half |
|---|---|---|---|
| one-row decode, ms a step | 7.302 | 7.183 | 7.033 |
| a block of 1 row | 7.05 | 6.92 | 6.76 |
| 2 rows | 8.36 | 8.08 | 7.89 |
| 4 rows | 9.84 | 9.26 | 9.17 |
| 8 rows | 13.98 | 12.94 | 12.99 |

The replay is 2.1% of a one-row step (7.5 us a layer a rank) and level at 8 rows: as expected
small, because the stream was already busy the whole layer. What it changes is the floor: the
host's 204 us a layer no longer bounds a layer whose kernels get shorter. Of 2,900 layer calls
2,860 were replayed (the others are the first sight and the recording call), 100 after a retune.
On one card (9 layers, `receipts/nrankab-g4-n1`, `-n4`): one rank 4.876 -> 4.863 ms, four ranks
sharing the card 6.667 -> 6.630; there every replay is a retune (1,287 of 1,287), which costs
nothing measurable.

Gate: under the door `dsv41-gate tp` repeats every decode step four times untraced and compares
each repeat whole (stream, pre-mix, every state buffer) with the single-card block: one, two and
four ranks on one card 480 / 985 / 1,955 checks, two ranks on two cards 985, 0 failed
(`receipts/ggate-g6`, `receipts/pairgates-g6`).

### The fused w1 + w3 + SwiGLU launch on the staged body

The staged small-M tile kernel (`eaa42ee764`) sits inside `memra_v41_gemm_f8_tiles`. One card runs
w1, w3 and w2 through that entry; two or more ranks run the hidden rows through the fused launch,
which still had the 16-column register body in each half. So the pair kept the old read pattern:
in the batching lane's per-kernel table of an 8-sequence call on two cards the fused launch was
5.1 of about 18 summed ms and did not move when the staged tiles were merged. Each half now runs
the staged body (a part of K in shared memory at once, both projections' weight rows and the
tile's activation rows once for the two). Columns B against A above: 1.04 ms off an 8-row call
(7.4%), 0.58 at 4 rows, 0.28 at 2, 0.12 at one row.

### The push limit and the default transport

`dsv41-bench p2p` times a gather by the size of a rank's region both ways under device-push: 64
gathers back to back, one wait at the end (`receipts/gather-g3`, medians of 9, three runs within
0.5 us).

| a rank's region | in kernels, us a gather | kept blocks and the copy engine |
|---|---|---|
| 1 KiB | 4.4 | 14.5 |
| 16 KiB | 7.1 | 20 |
| 64 KiB | 11.3 | 22.2 |
| 128 KiB | 16.7 | 23.2 |
| 256 KiB | 27.4 | 25.5 |
| 512 KiB | 48.9 | 35 |
| 1 MiB | 92 | 47.8 |

A push is one block of a kernel storing the region into every peer: 4.3 us plus 86 us a MiB. The
copy-engine gather pays launches, events and copy calls first (14 us) and then moves bytes at the
link's rate. They cross at 227 KiB on two cards, so the old limit of 256 KiB a region was close.
Since the kernel stores the region once a peer, the limit is now on what a rank stores: region
times peers at most 224 KiB. Every gather of a call of up to 8 rows stays inside kernels at two
and at four ranks.

With no transport named, two to eight ranks now take device-push (small gathers in kernels, large
ones through kept blocks). It led peer-pull at every width of a decode call in the batching lane's
quiet-pair A/B (3.65 against 3.92 ms at one sequence, 7.99 against 8.18 at eight), and the build's
default, host-canonical, had no measurement behind it for this route. Cards that grant no peer
access fall back to host-canonical with a note.

### Prefill expert tiles: once a token, and 64 against 128 rows

The routed experts' input was gathered into pair order (six copies of every token row) and
quantized there. It is now quantized once a token and each tile reads its pairs' tokens through
the plan's pair-to-token list (`memra_v41_gemm_f8_tiles_rows`), a sixth of that quantization and
the same codes. The first form read the list in front of every stage's copies and was flat to 1%
slower on one card (`receipts/prefillab-g5-tiles`: 340.6 -> 345.3 ms at 4,096 tokens); each
thread now reads its rows' offsets once a tile.

| time to first token, best / median ms | pairs gathered | once a token |
|---|---|---|
| one card, 9 layers, 4,096 tokens | 347.1 / 363.3 | 343.4 / 357.4 |
| one card, 9 layers, 32,768 tokens | 2,616.6 / 2,649.9 | 2,586.6 / 2,607.1 |
| two cards, 20 layers, 4,096 tokens | 466.2 / 487.2 | 459.0 / 471.1 |
| two cards, 20 layers, 32,768 tokens | 3,274.0 / 3,316.9 | 3,211.9 / 3,231.4 |

(`receipts/prefillab-g6-tiles`, `receipts/prefillab-g6-tiles-pair`, A B B A, 4 values an arm.) The
quantization of the pairs is not split across ranks, so its share grows with the rank count.

Tile rows, forced both ways in the same run on one card: 128-row tiles at 4,096 rows (mean 64
rows an expert, the rule picks 64) 351.2 against 345.3 ms, 0.8% slower even with the padding
skipped; 64-row tiles at 32,768 rows (the rule picks 128) 2,695.9 against 2,591.2 ms, 3.5%
slower. The rule stands.

### The keys side under wq_b

After wq_a and its norm the rest of the keys side (the window KV into the ring, the compressor, the
index keys, the index query and selection, the appends) reads only x and qr. On calls of at most 8
rows it now runs on the second stream under wq_b and its RoPE, and the heads join it before their
first read of the keys (`6daaffa1ef`). Two cards, 20 layers (`receipts/pair-m2`): one-row decode
7.179 -> 7.148 ms. Small, and far under the 17 us a layer the kernel table suggested: wq_b is a
wide launch and a branch under it gets little of the card, where the shared expert under the
router's narrow chain got most of its time back.

### Calls of 9 to 64 rows

Above 8 rows every family left its small-row body for the 64-row pipelined one. A call of N rows
on two cards, 20 layers (`receipts/wide-m2`): 12.7 ms at 8 rows, 35.3 at 12, 38.5 at 16, 44.5 at
32, 50.3 at 64. So the cost is one step of 22.7 ms between 8 and 12 rows, and two calls of 8 beat
one of 16. In a traced 16-row call the pipelined expert tiles are 38% of the kernel time, the
pipelined dense body 21%, the router's gate through the 64 x 64 fp32 tile kernel 18%, bf16 7%, the
mHC pass split across the ranks 5%.

Now a call of 9 rows and more keeps the small-row bodies up to a limit a family, issued once a
group of rows inside the call: 8-row expert tiles (an expert with more rows gets several), the
fp32 FMA and bf16 small-row kernels once a group of 8, the one-token mHC kernel a row, and for the
block-scaled dense GEMM a 16-row form of the 8-column body: the tensor-core tile's 16-wide side
held 8 activation rows and 8 rows of zeros, and the second 8 rows now ride on the same MMAs, so a
weight is read once for twice the rows.

Where each family's groups stop paying, from sweeps with one family's limit raised at a time
against all at 0 (ms a call, the family's column minus "none"; `receipts/widesweep-w3-n2` two
ranks on two cards, 20 layers; `receipts/widesweep-w3-n1` one rank, 9 layers):

| two ranks, rows | none | dense | bf16 | fp32 FMA | mHC | tiles |
|---|---|---|---|---|---|---|
| 12 | 35.4 | -5.5 | -0.9 | -6.0 | -1.5 | -4.1 |
| 16 | 38.4 | -3.9 | -0.9 | -5.7 | -1.6 | -5.1 |
| 24 | 42.3 | -1.9 | -0.1 | -5.0 | -1.4 | -5.4 |
| 32 | 44.4 | +0.2 | +0.8 | -4.2 | -1.1 | -5.1 |
| 48 | 47.9 | +4.3 | +2.3 | -2.9 | -0.8 | -3.5 |
| 64 | 50.4 | +8.3 | +4.1 | -1.5 | -0.2 | -2.0 |
| 96 | 54.5 | +17.0 | +7.7 | +2.0 | +0.7 | +1.0 |

On one rank a dense launch is twice as wide and its groups pay only to 12 rows (level at 16,
+1.6 ms at 24); bf16 to 16, fp32 FMA to 48, mHC to 32, tiles to 48. The limits are therefore per
family and rank count (`wide_rows_for`): at this point one rank 16 / 16 / 48 / 32 / 48, two ranks
and more 24 / 16 / 64 / 48 / 64 (dense, bf16, fp32 FMA, mHC, tiles; the next section moves the
dense one); four ranks take the two-rank limits until they are measured.

| a call of N rows, ms | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| two cards, 20 layers, before | 12.7 | 35.4 | 38.4 | 42.3 | 44.4 | 47.9 | 50.4 |
| two cards, each family at its limit | 12.7 | 17.7 | 21.3 | 28.6 | 34.0 | 40.7 | 46.7 |
| one card, 9 layers, before | 9.8 | | 22.7 | | 26.1 | | |
| one card, the defaults (`receipts/prefillab-w4-wide-n1`) | 9.8 | | 15.0 | | 21.1 | | |

One call of 32 rows on two cards is 34.0 ms where four calls of 8 are 50.3. What is left in a
32-row call on one card (`receipts/wide-w4-n1`, kernel time a layer): the 8-row expert tiles 54%
(they read each routed expert's weights once, which is the bytes a batch of distinct routes has to
read), the pipelined dense body 20%, the fp32 FMA groups 11%, bf16 7%.

Gate: a row's bytes do not depend on the group it runs in, and the gate checks it on the same
call: the prompt's first 12, 16 and 32 rows as one call with every limit at 0 (the pipelined
bodies) and with every family through its groups, output against output, on the single card and on
each rank, on layers 1, 2, 14, 20 and 24. (A first form compared a short call with the same rows
of the full prefill; that holds on a layer without compressed keys and not on a kv source, where
the selection width follows the call's key count and the attention sums a row's keys in another
grouping.)

### The dense body for 7 to 32 rows, the router's gate in one launch

A traced 32-row call on two ranks after the section above (`receipts/wide-w4-n2`, 1,606 us of
kernels a rank a layer): the pipelined dense tile 28%, the fused hidden tiles 25%, the w2 tiles
13%, the router's gate 9% as four launches of 8 rows, bf16 8%.

What bounds the pipelined dense body there is its shape. Its launches by grid in that trace: wq_a
(10 blocks, K = 5,120) 68 us, wkv (4 blocks, K = 5,120) 66 us, the shared expert's w1 and w3 (9
blocks) 66 us each, wq_b (128 blocks, K = 1,280) 21 us. Time follows K and nothing else: a block
lands 3 K blocks a stage, a stage is about 1.25 us of round trip, K = 5,120 is 54 stages, and most
of the card is idle. A rank's dense weights are 55 us a layer at the copy rate. The one-pass
small-row form has the opposite problem: it keeps all of K's terms in shared memory, one block an
SM at 16 rows.

`v41_gemv_f8_rows_kernel` is the 8-column body with K in parts: a part of K is split over the
block's eight warps, a warp loads the weight fragments of its K blocks once and runs them against
each set of 16 activation rows, the part's scaled terms go to shared memory, the (row, column)
threads add them in K order onto the sum they carry. A block holds one part's terms (24 KB, four
blocks an SM) and a weight is read once for all the rows. It replaces the 16-row form above. The
router's gate (any narrow fp32 FMA output) at 9 to 32 rows is one launch: the narrow kernel's
block is 4 columns by up to 32 chains, with K landed in four parts. Past the dense limit a call of
up to 32 rows takes a 32-row tile of the pipelined body.

Two ranks, 20 layers, the family alone against the pipelined path, ms a call
(`receipts/widesweep-w7-n2`):

| rows | none | dense in parts | (the 16-row form) | the gate in one launch | tiles | mHC | bf16 |
|---|---|---|---|---|---|---|---|
| 12 | 33.7 | -4.1 | -3.8 | -6.5 | -4.2 | -1.6 | -0.9 |
| 16 | 36.9 | -3.9 | -2.3 | -6.6 | -5.3 | -1.7 | -0.9 |
| 24 | 40.7 | -3.3 | -0.2 | -6.5 | -5.6 | -1.5 | -0.3 |
| 32 | 42.8 | -2.5 | +1.8 | -6.4 | -5.3 | -1.3 | +0.6 |
| 48 | 47.9 | -1.4 | +4.3 | -2.9 | -3.7 | -0.8 | +2.3 |
| 64 | 50.5 | -0.3 | +8.2 | -1.5 | -2.1 | -0.4 | +3.9 |
| 96 | 54.6 | +4.3 | | +1.9 | +0.7 | +0.6 | +7.6 |

Limits now (`wide_rows_for`; dense, bf16, fp32 FMA, mHC, tiles): one rank 24 / 16 / 48 / 32 / 48,
two ranks and more 48 / 16 / 64 / 48 / 64. A call on two cards with the defaults
(`receipts/wide-w10-n2`, best of 16): 8 rows 12.6 ms, 16 rows 19.0, 24 rows 24.0, 32 rows 27.6, 48
rows 39.3, 64 rows 46.9; a 32-row call was 44.4 ms before this work and four calls of 8 are 50.5.
Its kernels are now 1,325 us a rank a layer: the fused hidden tiles 30% and the w2 tiles 16% (the
routed experts' weights, read once), dense 19% (252 us, from 451), bf16 10%, the gate 4%.

The body also beats the one-pass form at 7 and 8 rows, and loses below
(`receipts/prefillab-w9-parts8-n2`, `-parts5-n2`, `-parts5-n1`, the body forced down, A B B A):
two ranks 1 row +1.3%, 4 rows +2.0%, 5 and 6 rows level, 7 rows -1.4%, 8 rows 12.77 -> 12.48 ms;
one rank 7 rows -1.4%, 8 rows 9.90 -> 9.53 ms. Dense calls of 7 rows and more take it.

### The second stream above 8 rows, the free list's cap

The three second-stream overlaps stopped at 8 rows with the small-row bodies. One limit raised at
a time, a call of N rows, ratio to the committed tree (`receipts/prefillab-w11-caps-n2` two ranks,
20 layers; `receipts/prefillab-w11-caps-n1` one rank, 9 layers):

| | two ranks: 16 rows | 32 | 64 | one rank: 16 rows | 32 |
|---|---|---|---|---|---|
| the shared expert under the router's chain | 0.961 | 0.957 | 0.934 | 0.976 | 0.942 |
| the mHC mixes under their sublayer | 0.978 | 0.983 | 1.003 | 0.990 | 0.992 |
| the keys side under wq_b | | | | 1.008 | 0.992 |
| the free list at 8 MiB a buffer, 256 MiB in all | 1.001 | 1.000 | 1.003 | 0.994 | 0.998 |
| shared expert and mixes together | 0.937 | 0.948 | 0.934 | | |

The shared expert goes to the second stream up to 64 rows and the mixes up to 32
(`SIDE_LIMITS`); the keys side stays at 8 (level on one rank; on two ranks only the batched step
takes it). A 32-row call's rows are 1.3 and 2.6 MB against the free list's 1 MiB cap, and raising
the cap changes nothing: the stream-ordered device pool is not what a wide call waits for. Two
cards, 20 layers, with both limits: 16 rows 18.87 -> 17.64 ms, 32 rows 27.45 -> 25.97, 64 rows
46.59 -> 43.23.

### A 16-row form of the bf16 small-row kernel

The small-row bf16 kernel carries 8 rows on the tile's 16-wide side. Its 16-row form (`R16`) holds 16,
so a warp's chain over K carries twice the rows on the same MMAs, and a call of 9 to 16 rows is one
launch of it instead of two launches of 8 rows. Alone, against the tree before (second stream held
at 8 rows; A B C D D C B A, medians of 10):

| a call of N rows, ms | 12 | 16 | 24 | 32 |
|---|---|---|---|---|
| two ranks, 20 layers, before | 16.58 | 18.91 | 24.09 | 27.39 |
| the 16-row form | 16.03 | 18.32 | 24.02 | 27.27 |
| + groups of 16 up to 32 rows | 16.30 | 18.61 | 23.55 | 26.86 |
| the 16-row form and the second stream's limits | 15.61 | 18.08 | 23.43 | 26.57 |
| one rank, 9 layers, before | 12.19 | 14.00 | 17.12 | 19.40 |
| the 16-row form | 11.85 | 13.98 | 17.13 | 19.42 |
| + groups of 16 up to 32 rows | 12.04 | 14.15 | 17.54 | 20.20 |

(`receipts/prefillab-w14-bf16-n2`, `-n1`.) The limit stays 16 at both rank counts: groups of 16 up
to 32 rows gain 2% at 24 and 32 rows on two ranks, lose 1.7% at 12 and 16, and lose at every width
on one rank. The form and the second stream's limits add: in a run of the tree before, the limits
alone and both (`receipts/prefillab-w14-side-n2`), the ratios to the tree before are 0.958 / 0.967 /
0.969 / 0.977 for the limits alone and 0.928 / 0.942 / 0.965 / 0.973 for both at 12 / 16 / 24 / 32
rows.

The gate's wide cases (calls of 12, 16 and 32 rows, every family's limit at 0 against 32) ran the
second stream in both passes, so a branch on the second stream above 8 rows had no reference in
stream order. The reference pass now runs with the three second-stream limits at 0
(`receipts/gates-w14`, `ggate-w14`, `pairgates-w14`: 0 failed).

### A timed cell and the card's power limit

The same build's 16-row call measured 13.6 ms in a rotation of 8 / 16 / 32 rows and 14.0 to 14.5 in
12 / 16 / 24 / 32. It is not the widths. One width alone, 48 calls back to back
(`receipts/longrounds-w14`): 13.08 to 13.12 ms for the first 150 ms, 14.5 at 250 ms, 14.7 to 14.8
from 400 ms on. Every width alone and in sets of two and three for 6 rounds (`receipts/rot-w14`):
13.2 to 13.7 in every set; only the four-width set, the one busy for longer than 0.3 s, shows 14.7
to 14.8. Traces of the rotation and of each width alone (`receipts/rottrace-w14`): every kernel is 1
to 4% slower in the rotation, small and large alike, the host's launch call costs the same, and no
call allocates. A uniform factor on every kernel is the clock.

`nvidia-smi` at 20 ms under a run of 3.3 s (read only; its values refresh about twice a second;
`receipts/clockprobe-w14-*`):

| | first 100 ms | from 1 s on | ratio | settled clock, draw |
|---|---|---|---|---|
| one card (card 0), 9 layers, 32 rows | 18.03 ms | 19.98 | 1.108 | 2,272 -> 1,747 MHz at 250.0 W, 69 C |
| one card (card 1), 9 layers, 32 rows | 17.99 | 19.30 | 1.072 | 2,272 -> 1,807 MHz at 250.0 W, 48 C |
| one card, 9 layers, one-row decode | 4.66 | 5.05 | 1.084 | 2,280 -> 1,867 MHz at 250.0 W |
| two cards, 20 layers, one-row decode | 7.13 | 7.27 | 1.019 | 2,280 and 2,227 MHz at 243 and 214 W: no step |
| two cards, 20 layers, 8 rows | 12.41 | 13.62 | 1.097 | 1,905 and 2,107 MHz at 250.0 W |
| two cards, 20 layers, 32 rows | 25.86 | 28.24 | 1.092 | 1,845 and 2,032 MHz at 250.0 W |

A card under continuous load reaches its power limit after 0.15 to 1.5 s and its clock settles 8 to
23% lower; a call then costs 7 to 13% more, the same factor at every width. The settled clock
depends on the card's temperature (the warmer card settles 60 MHz lower alone and 190 MHz lower in
the pair), and a pair runs at the pace of its slower card. A one-row decode on two cards stays
under the limit (each card waits for the other's gathers) and keeps the fast clock; on four cards
no card passed 214 W in the last four-card pass, so the one-row figures (7.0 to 7.3 ms on two
cards, 11.62 on four) are steady-state figures. The wide-call figures in the sections above are
not: their cells are bursts of 0.1 to 0.4 s from an idle card. Ratios inside one interleaved run
hold (each arm crosses the step at the same place); the absolute ms are low against a served step
by the factor in the table. In steady state on two cards a call of 8 rows is 13.62 ms and of 32
rows 28.24.

From here a reported figure is the steady one: the median over calls that start at least 0.5 s
after the run's first call (`steady.py`), with the burst figure beside it. Raising the free list's
caps does not change any of this (`receipts/prefillab-w11-pool-r4-n1`: level to 2% slower in the
long rotation).

### The graph door with several sequences, and a whole layer as one graph

A bug behind the door, fixed in `bafaf69aa5`: a replayed half-layer handed back the capture's own
pre-mix buffer, which the span's next replay rewrites. One sequence a forward never saw it. A
`Runner::forward` of several sequences' one-token calls replays a layer's span once a sequence,
so every sequence but the last read another's pre-mix: 7 of 8 sequences generated other tokens.
The replayed half now returns a copy. The gate that was missing is `dsv41-tpbench --step-gate`:
every decode cell's steps with the door off, then on from fresh sequences fed the same rows,
every step's stream and pre-mix on every rank and its ids compared, as one call a sequence and as
one batched call. Before the fix 8 sequences as one call a sequence fail 36 of 37 checks
(`receipts/step-r1`); after it 1 and 8 sequences in both forms pass on one and two ranks
(`receipts/step-x2`, `step-x2-n2`). The door is off by default, so no default-path number moved.

A whole layer of a batched step as one graph (branch only, not landed). The batched step keeps
its host data in the rank's step block, so a layer's launches name positions and key tables by
address and the whole layer can be recorded: inputs followed by address (stream, pre-mix, the
layer's tables and positions, the selection rows a reader layer takes), and the three launches a
sequence of an index layer that hold what grows with it (key address and count, position,
candidate pool) found through the stream's capture state right after each launch and rewritten
before every replay. Step gate on one rank: 1 and 8 rows x 12 steps, 0 of 37 failed, 99 replays.
One rank, 9 layers, steady state, 745 steps an arm (`receipts/step-s1`):

| a step, ms | one row | 8 sequences |
|---|---|---|
| one call a sequence, host scoring of the logits | 4.912 | 40.47 |
| the same, the second half of a layer replayed | 4.875 | (the bug above) |
| the batched step, device argmax | 4.843 | 11.375 |
| the batched step, the whole layer replayed | 4.800 | 11.253 |

The whole layer replayed is 0.9% of a one-row step and 1.1% at 8 rows on one rank, where the
card and not the issue thread bounds a layer. It stays on its branch until a four-rank layer is
close to the issue thread's 204 us; the door is decided then (on by default with its gates, or
its module deleted). The download and host scan of the logits are 0.07 ms a step.

### Where a four-rank step's wall goes, by layer

`step_wall.py` cuts a traced decode step at every layer's first mix launch, so the layers'
windows add up to the step's wall. Last night's four-card trace (`four2/trace-B`, 13.30 ms a step
under the profiler): a typical layer 300 us, 40 of them 12.01 ms, the rest 1.29 ms: the last
layer's window with the head 450 us above a typical layer (the head's fp32 GEMV 231 us a card,
the rest the read-back and the host), the two Engram layers 150 us each (an FP8 GEMV of 110 us),
the kv and index source layers 2, 8, 14 and 20 85 to 100 us each (the fp32 chain 45, bf16 22,
quantization and the selection), the index layers 24 to 36 42 us each.

### A gather as one launch, and what a removed launch is worth

A small gather was a push and a pull launch back to back on a rank's stream; they are one kernel
with a barrier between the halves (`c9e06330eb`, exact). Two ranks on two cards, 20 layers, steady
state, A B B A A B, 1,227 steps an arm (`receipts/nrankab-f2-pair`): a one-row step 7.167 ->
7.092 ms, every run after below every run before (7.165 / 7.167 / 7.168 against 7.100 / 7.098 /
7.044); blocks of 4 and 8 rows 9.15 -> 9.11 and 12.71 -> 12.68 (`receipts/prefillab-f2-rows-pair`).
On one card, where the ranks share the card, it is level (`receipts/nrankab-f1-n4`, `-n2`).

That is 0.9 us a gather where the trace showed 7.7 us for the two kernels. A small kernel's traced
duration is mostly its own work and its wait, not a floor that a merged launch removes. The class
"launch-latency-bound small kernels" in the accounting above was priced from traced durations (45
to 55 us a layer); by this calibration it is worth 2 to 4% of a one-row step, not 16 to 19%.

### The fp32 FMA calls at verify widths

A bf16 weight asked for an fp32 output runs as FMA chains in K order. Three callers a step: the
router gate (every layer), the ratio-2 compressor's kv and gate rows, the head's logits. A batched
step of 1 row against 6 rows, one rank, 9 layers, traced (`receipts/steptrace-x2`; the prompts'
prefill is in the totals):

| a step, ms | 1 row | 6 rows |
|---|---|---|
| the head (`v41_gemv_ffma`) | 0.94 | 1.56 |
| the router and the compressor (`v41_gemv_ffma_narrow`, `_parts`) | 0.43 | 0.70 |

The head at one row is read-bound (1.32 GB once). For 5 to 8 rows it takes the 8-row form: the
same weights read once, six chains a column, and it becomes chain-bound. The narrow kernel stages
its 4 weight rows and the call's rows whole in a block; (4 + rows) x 5,128 x 2 bytes passes the
99 KB limit at 6 rows, so 6 to 8 rows take K in two parts: 34 -> 43 us a launch. On four ranks
the head splits by ranks and the router does not. Both belong with the order-bound chains (the
router gate, `wo_a`, the head, the compressor's pair).

### The page cache under a timed cell

A cell that starts after the page cache was dropped reads its Engram tables from the disk: the
first tp gate of a rehearsal took 105 s instead of 18, 74 s of it one 98 GB table mapped by one
thread at 1.3 GB/s, and the first prefill cell showed 981 to 2,730 major page faults where a warm
one shows none (`receipts/night-pair`). The kernel log names every drop
(`drop_caches`). Since then: the table is mapped by eight threads; every timed run records the
last drops before and after it and a run with a drop inside is flagged; the prefill tables carry
each cell's page faults; and a pass starts with a read of the checkpoint into the cache
(`perf_warm.sh`: 475 GiB in 29 to 55 s when 107 to 274 GiB of it come from the disk).

### The bf16 family in the K-segment order

`wo_a` and the other bf16 linears summed each output over K as one chain of K / 16 dependent MMAs,
so a few-column call was bound by the chain's latency. The family now sums K in the K-segment
order: K is cut into 8 contiguous segments, each segment's sum is one fp32 chain in K order from
zero, and the 8 sums meet in one fixed tree in index order, ((s0 + s1) + (s2 + s3)) + ((s4 + s5) +
(s6 + s7)), rounded once. Both kernels (the tile kernel above 16 rows, the small-row kernel up to
16) use it at every row and rank count, so a row's bytes do not depend on the path that computed it
(the kernel gate checks the tile kernel's first 8 rows against the small-row kernel's bytes). It
moves `wo_a`, the ratio-1 compressor, the index keys and the index head weights; the router, the
head and the ratio-2 compressor's fp32 pair keep their FMA chains.

The order changes bytes, so it was judged against the vendor, not against the old route
(`receipts/ordertapes-k1`, `tools order_gate.py`): the 18 decode-lane tapes (13 chat, 4 long, the
257k needle; 714 steps) teacher-forced through the whole trunk, top-32 KL against the vendor's own
run, d = KL(vendor || new) - KL(vendor || old) with a bootstrap 95% interval (10,000 resamples). The
pass line: d's interval contains zero or lies below it.

| group | steps | KL old | KL new | KL of the vendor's second program | d [95%] | top-1 old / new | verdict |
|---|---|---|---|---|---|---|---|
| chat | 517 | 1.631e-02 | 1.509e-02 | 1.432e-02 | -1.22e-03 [-5.55e-03, +2.70e-03] | 97.3% / 96.9% | pass |
| long | 194 | 3.679e-02 | 2.402e-02 | 3.051e-02 | -1.28e-02 [-3.05e-02, +2.24e-03] | 93.8% / 95.9% | pass |
| needle 257k | 3 | 1.618e-04 | 1.062e-04 | 3.643e-05 | -5.56e-05 [-1.74e-04, +6.95e-06] | 100% / 100% | pass |
| all | 714 | 2.181e-02 | 1.745e-02 | 1.866e-02 | -4.35e-03 [-1.01e-02, +6.43e-04] | 96.4% / 96.6% | pass |

The route log (`dsv41-run --routes`) shows the same six experts on 42% (chat), 49% (long) and 37%
(needle) of the 23.5 M routed rows, from layer 2 on: drift between two valid realizations through
the quantization steps, not flips at equal inputs (tokens whose six experts equal the vendor's on the
needle capture's 3,761, new / old / the vendor's second program: layer 1 3,717 / 3,717 / 3,692, 2
3,647 / 3,645 / 3,526, 14 3,691 / 3,692 / 3,601, 20 3,617 / 3,716 / 3,534, 21 3,731 / 3,731 / 3,728,
24 3,725 / 3,723 / 3,556; layer 20, the ratio-1 kv source, is the one that moves, and stays above
the vendor's second program). CheckpointParity on the pack's probe passes
in its band (max abs 2.91 against 2.14 before, band 6.4; the vendor's argmax;
`receipts/ckpt-k3/verify-capital`). Greedy ids on the 13 chat cases (`receipts/ckpt-k3`, the decode
gate): every case follows the vendor's tokens to the same step as the old route, 8 to the end and
gsm 5, gsm_think 3, story 19, hebrew_para 23, code_pal 19, each divergence a vendor near-tie; the
run stopped at 57 steps with story's and hebrew_para's tails unfinished, past both divergences.
The new reference set of the layer gate's 111 tensors was written on a GB10 build of this tree:
a GB10 (121a) build of the old tree writes the 120a reference set byte for byte (111 of 111;
`receipts/bytecheck-k3-gb10`).

The first form (k1) kept the 8 accumulators in one warp, so a stage held one k16 step of each
segment: a weight row was read as 32-byte pieces 1 KB apart. It was 13% slower at one row on one
card and 7% on the pair; the trace put all of it on the small-row kernel (56.6 -> 139.1 us a layer
on one rank, 40.9 -> 67.5 on the pair; `receipts/steptrace-{o2,k1}t{1,2}`). The landed form gives
each segment its own warp: a block is 8 warps over one 8-column group, warp w streams segment w
contiguously as the one-chain kernel did, and the 8 sums meet in shared memory. Its bytes are k1's
(all 111 tensors the layer gate saves, `receipts/eq-k1-k3`).

| steady ms, A B B A against a449d3183c | one-row decode | 4-row block | 8-row block |
|---|---|---|---|
| one rank, 9 layers (`nrankab-k3-n1`) | 5.012 -> 4.965 (0.991) | | |
| the pair, 20 layers (`nrankab-k3-pair`, `prefillab-k3-rows-pair`) | 7.119 -> 6.856 (0.963) | 9.870 -> 9.492 (0.962) | 13.905 -> 13.561 (0.975) |
| four cards, 40 layers (`k3four`) | 10.968 -> 10.004 (0.912) | 14.216 -> 12.935 (0.910) | 18.546 -> 17.302 (0.933) |

Receipt names: `k1` is the first form, `k2` an attempt on it (the accumulator index fixed at compile
time, the tile's fold by segment: no gain), `k3` the landed form, `o2` the base (a449d3183c); the
first form's byte gates are in `gates-kseg-k1` (`gates-k1` is an earlier build of that name).

Prefill on one rank (`prefillab-k3-prefill-n1`): 4,096 tokens level (median 0.990), 32,768 tokens
1.019 (best 1.030). The tile kernel's three saved tree partials take it from 64 to 195 registers a
thread, so fewer blocks fit a multiprocessor; that cost is open (a narrower tile keeps every
element's program).

### The fp32 FMA family: the K-segment order moved the outputs; the chain streamed instead

The router's gate (every layer), the ratio-2 compressor's kv and gate (the kv-source layers 2, 8 and
14) and the head take an fp32 output of a bf16 weight: each output is one fma chain over K = 5,120 in K
order. At one row the narrow outputs (384 and 512 wide) are bound by that chain: 36 us a call on an RTX
5090, against 8.3 us for 5,120 dependent fmas.

r1 (not landed) summed these outputs in the bf16 family's K-segment order. On four cards it bought
10.050 -> 9.419 ms at one row, 13.017 -> 12.467 at 4, 17.293 -> 16.651 at 8, and failed the order gate:
chat d +3.93e-03 [+8.4e-04, +7.09e-03] by tape, top-1 96.9 -> 95.9% (the vendor's second program
97.5%); all 18 tapes +5.25e-03 [+2.26e-03, +7.82e-03], top-1 96.6 -> 95.7%. Against two other
realizations, k3 and the CUTLASS build (which passed), r1 sat furthest from the vendor on 7 of the 9
tapes of 20 steps or more (`receipts/r1-dig/og-r1/per-tape-r1-cutlass-k3.md`; 7 or more of 9 at 1 in 3
a tape: p 0.008).

At equal inputs the order is not further from the vendor: on the kernel gate's vendor fixtures r1's
router weights sit closer (exact 70 -> 100 of 270, rel_l2 2.0e-7 -> 1.2e-7) and so does the ratio-2
compressor's latent (rel_l2 8.5e-5 -> 1.3e-6). The bisect used a research build that takes the order
per call site (`receipts/r1-dig/r1x-site-switch.patch`; never landed):

| arm | layer-gate tensors moved (of 111) | attention output rel_l2 vs the vendor, layer 2 / 14 | chat tapes: d [by-tape 95%], top-1 |
|---|---|---|---|
| none (the switch off everywhere) | 0: k3's set byte for byte | 8.65e-3 / 9.60e-3 | (k3: 96.9%) |
| router only | 13: the MoE outputs' last bits; no metric moves in 4 digits | 8.65e-3 / 9.60e-3 | +4.59e-03 [+9.74e-04, +7.48e-03], 96.5% |
| compressor only | 32: layers 2 and 14 from the latent on | 4.43e-3 / 1.36e-2 | +3.90e-09 [-5.26e-09, +2.65e-08], 96.9%: 511 of 517 steps' top-32 byte-equal to k3's |
| both (r1) | 41 | 4.43e-3 / 1.36e-2 | +3.93e-03 [+8.44e-04, +7.09e-03], 95.9% |

The router alone carries the shift: its chat KL equals r1's to the last digit on 6 of the 13 tapes and
its d is r1's within the interval, while the compressor alone moves only the two needle tapes' 6 steps
of the 517 (it does move the 3,761-token layer gate's attention: on the chat prompts its last bits
rarely reach the logits). The route log is not a factor: r1's tapes ran with it, the arms without,
and they agree. A router order change moves only the last bits of each layer's MoE output at equal
inputs, and through 40 layers of fp8 and fp4 quantization that is a different trajectory; the
K-segment one lands further from the vendor's. It is not that the one chain is the vendor's order:
on an RTX 5090 (torch 2.14 with CUDA 13.0, TF32 off as the vendor's oracle runs) the vendor's fp32
`F.linear` matches neither order bit for bit at 1 to 4,096 rows (the one chain 3.5 to 5.6% of outputs,
the K-segment order 11.6 to 16.9%), and the K-segment order sits about 5x closer in max error
(`receipts/r1-dig/bench/cublas-order-5090.txt`). So the order gate, not closeness at the
source, decides: a changed router order is a new trajectory, and this one measured worse.

What landed instead keeps every output's one chain, so the bytes are the current route's, and takes
the speed from the kernel. `v41_gemv_ffma_ring_kernel` (a narrow output at 1 to 8 rows): a block is one
warp that owns 4 columns; K streams through a 4-stage cp.async ring, so the first stage is the only
wait; the chain reads its next 8 values of each operand one step ahead, and a 32-bit word's two bf16
values are widened by a shift and a mask (3.4 instructions an fma instead of about 5, under the fma's
4-cycle latency). The compressor's kv and gate run in one launch (`memra_v41_gemm_ffma2`). Measured and
dropped: a second warp that widened each stage to fp32 for the chain's warp (22 us at one row).

| RTX 5090, K = 5,120, us a call | current (one chain) | r1 (K-segment order) | ring (one chain) |
|---|---|---|---|
| router, 1 row (kernel, graph-timed, weights from memory) | 36.0 | 11.3 | 12.8 |
| router, 4 rows | 39.0 | 13.2 | 15.8 |
| router, 8 rows | 49.1 | 29.6 | 15.8 |
| compressor (512 wide), 1 row | 37.4 | 12.4 | 13.7 |
| router 1 / 4 / 5 / 8 rows, the engine's own timer (`dsv41-bench gemm`, mean of 2 runs) | 52.0 / 55.4 / 55.6 / 67.5 | 18.1 / 21.0 / 24.5 / 37.7 | 22.5 / 24.3 / 24.4 / 24.5 |
| compressor's pair, 1 / 4 / 8 rows, two calls -> one (engine timer) | | | 35.1 -> 21.4 / 38.0 -> 24.8 / 38.8 -> 39.7 |

GB10 (121a), the same kernels: router one row 42.3 / 12.7 / 15.6 us with the weights in L2, 55.5 /
25.5 / 23.4 from memory (`receipts/r1-dig/bench`). Bytes: the ring's outputs equal the current route's
on every bench shape (1 to 8 rows, both widths, the pair); the engine's router outputs hash the same
at 1, 4, 5 and 8 rows; the kernel gate passes 177 of 177 on the 5090 (ring = parts = groups of 8 = tile;
the pair = two calls; the vendor fixtures' router and compressor lines unchanged from k3's). On GB10
(121a, `receipts/r1-dig/gates-r2-gb10`): kernels 177 of 177, the layer gate 161 of 161 with its 111 saved
tensors equal to k3's set (the hash list's sha256 46164854b745e1d8, k3's), and the order gate's 13 chat
tapes byte for byte k3's (78 of 78 files: d = 0 on the chat group); tp at 2 and 4 ranks of the card
(peer-pull) 478 / 0 and 914 / 0, decode 240 / 0, 240 / 0 and 60 / 0 (`receipts/r1-dig/gates-r2-gb10`).
Above 8 rows the narrow kernel with K in parts stays: r1's segment kernel was slower there (16 rows
71 against 55 us).

PRO 6000 (2026-11-13, `receipts/r1-dig/pro6000`: the run summaries, and the raw runs in `r2-bytes/` and
`r2-pair23/` with gzipped telemetry): the layer gate 161 of
161 on both builds with all 111 saved tensors equal to kpc's; every `dsv41-bench gemm` output hash
equal to kpc's (head and router at 1, 4, 5 and 8 rows, the pair against two single calls); tp on
the pair 480 / 0 (default) and 478 / 0 (peer-pull). The engine timer, kpc (runs 1 and 4) against
r2 (runs 2 and 3), us a call:

| PRO 6000, K = 5,120 | 1 row | 4 rows | 5 rows | 8 rows |
|---|---|---|---|---|
| router, kpc | 24.3 / 24.3 | 26.1 / 26.1 | 26.5 / 26.5 | 33.6 / 33.6 |
| router, r2 | 20.0 / 19.8 | 21.6 / 21.3 | 22.0 / 21.6 | 21.9 / 21.6 |
| compressor pair, two calls -> one (r2) | 35.5 -> 20.9, 35.3 -> 20.6 | 38.7 -> 23.0, 38.5 -> 22.7 | | 40.1 -> 25.4, 39.9 -> 25.1 |

On the RTX 5090 Laptop the pair at 8 rows is slower as one launch: 38.1 -> 39.2 and 39.4 -> 40.2 us in the two
runs (`receipts/r1-dig/gemm-5090`); at 1 and 4 rows it is faster, and the PRO 6000 is faster at every width.

The pair's step (cards 2 and 3, 20 layers, kpc A against r2 B, A B B A B A A B, 4 runs an arm,
steady calls past 0.5 s of load): one-row decode 6.670 -> 6.505 ms (-2.5%; 919 and 912 calls; by run,
the median of each run's steady calls, kpc 6.664 [6.663, 6.679] against r2 6.499 [6.492, 6.514] over 4 runs an arm,
the ranges apart; `r2-pair23/per-run.md`). The 4-row block (8.845 -> 8.824 ms) and the 8-row block
(12.362 -> 12.344 ms) are level (36 calls an arm each). No run flagged for a slowdown, no page-cache
drop inside a run.

At r2's pushed head (`1008d7cac5`, on the PRO 6000 pair, 2026-11-13, `receipts/r2-gates-pro6000-eve`):
the kernel gate 182 / 0 on one card, its partial-block lines included (the ring at 2 rows, n 30
against the first 2 rows of a 24-row call on the narrow kernel; the pair's two outputs against the
24-row calls on each weight). The graph door (`MEMRA_DSV41_DECODE_GRAPH=1 dsv41-gate tp`, the needle
capture, layers 1, 2, 14, 20, 21 and 24, two decode steps, two ranks on the pair): default 1,105 / 0
and device-push 1,105 / 0 (6 spans captured, 42 replays, 12 checked); peer-pull 478 / 0, where the
door captures no span (`spans=0`), so that run is the eager peer-pull gate. No run flagged for a
slowdown, no page-cache drop inside a run.

The ring at one and two stages (2026-11-14, one GB10, 121a, `receipts/r2-ring-k256-k512-gb10`, the
#1283 review's nit): the ring has 4 stages of 256 elements and its prologue lands 3, so at K 256 and
512 a call holds fewer stages than the prologue lands, and both the prologue's guard and the loop's
refill guard skip. The kernel gate now runs the ring there beside the (2, 30, 5120) shape: widths 30
(the last 4-column block partial) and 32 (whole blocks), 2 rows and 8 (every lane of the warp), each
against the first rows of a 24-row call on the narrow kernel with K in parts (a form that is not the
ring), and the pair against the 24-row calls on each weight. 202 of 202 built from the change's commit
(`6b1617fa46`) against 184 of 184 on main at `df1d23a764` (`base-main/`; the change's base, main
`bbc6f83aea`, has the same gate and kernel sources): the 18 new checks pass, every output byte equal,
and the existing entries keep their names.

### A four-card prefill's copies, and the copy call that was slow

On four cards a 4,096-token prefill took 4.0 s (vLLM's: 0.24 s). The night pass's trace
(`night/pre-trace`, not kept) shows where: per rank and layer 170.5 MB arrive in 21.6 peer copies,
82 of the layer's 102 ms (678 of 769 ms at 32,768 tokens), each copy running at 2.0 GB/s. By
tensor, into a rank a layer: the `wo_a` columns 50.3 MB, the experts' hidden columns as FP8 codes
and scales 51.1 MB, the experts' and shared expert's output columns 31.5 MB, the `wo_b` output
columns 31.5 MB, a small per-row gather 4.9 MB, the index selection 1.3 MB. The bytes are within
1.35 to 2.7x of what a ring all-reduce of the layer's output moves a rank; the rate was 15x off.

A standalone probe on the four cards with no model (`p2p_probe.py`, `receipts/p2p-probe`; an
all-gather of S bytes a rank, GB/s a rank received, median of 15):

| method | 10 MiB | 16 MiB | 64 MiB |
|---|---|---|---|
| one copy alone (its own rate) | 54.1 | 54.5 | 55.1 |
| the 12 copies one at a time | 13.4 | 13.5 | 13.7 |
| each consumer pulls its 3 on its stream, 4 at once | 30.9 | 31.1 | 31.1 |
| the 12 copies at once | 26.6 | 26.1 | 26.2 |
| kernel stores into the peers, 4 at once | 2.1 | 2.1 | 2.1 |
| NCCL all-gather (P2P/CUMEM, 4 channels) | 36.7 | 37.3 | 38.0 |

So the pattern was not the cause. The engine's own gather probe on the same cards, nothing else
running, gave 2.3 GB/s a rank at 1 MiB: the cause was in its copy path. Its factors one at a time
(`p2p_probe2.py`, `receipts/p2p-probe2`; consumers pull in ring order as the engine does):

| GB/s a rank | 1 MiB | 16 MiB |
|---|---|---|
| cudaMalloc blocks, the peer copy call, pull | 27.4 | 49.7 |
| cudaMalloc blocks, a copy by device address, pull | 27.5 | 50.0 |
| pool blocks, the peer copy call, pull | 27.6 | 49.8 |
| pool blocks, a copy by device address, PUSH (the producer's stream) | 32.6 | 50.9 |
| pool blocks, a copy by device address, pull (the engine) | 2.9 | 2.67 |
| pool blocks allocated for each gather, by device address, pull | 2.9 | 2.47 |

Only one combination is slow: a stream-ordered pool block read by a consumer through a copy by
plain device addresses, which is what every large gather did. The fix keeps the stream, the ring
order and the events and changes the call: every copy between two ranks' cards is
`cuMemcpyPeerAsync` naming both contexts (`copy_peer` in `tp.rs`: the packed gather, both
peer-pull branches, and the bench's copy probe). Pure movement, so the bytes are unchanged.
Where it reaches: under the default transport (device-push) every gather of a decode call of up
to 8 rows stays inside kernels at two and at four ranks (the push limit above), so a decode step
of up to 8 rows never calls `copy_peer` and runs the build's decode as before. The large gathers
of a prefill (and of wider calls) go through it, as does every gather under
`MEMRA_TP_TRANSPORT=peer-pull`. So the pair's decode A B B A has nothing to measure; the prefill
is measured below. The byte gates below cover the pair (the copy crossing cards), one GB10 card
(four ranks, no copy crossing cards) and four cards. The four-card byte gate (`dsv41-gate tp
--devices 0,1,2,3`, the needle capture, layers 1, 2, 14, 20, 21 and 24, two decode steps;
2026-11-13, the cards held 15:36:05 to 15:37:35, `receipts/gate4-kpc-seam`) passes on two builds:
`seam-kpc` (`4eb63a1052`, the branch as pushed then) and `seam-kpcR` (kpc on the route's review fix
`5e71a5929b`, whose `tp.rs` is the pushed route's), default 916 / 0 and peer-pull 914 / 0 each. In
those runs the push-gather stress finds 0 words differing over 20,000 gathers, and its word-first
control is caught.

Measured after the fix (2026-11-12 and 13; the cards at their set limits, 250 W on cards 0 and 1,
280 W on 2 and 3; no run flagged for a slowdown, no page-cache drop inside a run):

- The pair (cards 2 and 3, `receipts/pairgates-kpc-23`): the tp gates pass on every transport,
  default 480 checks, peer-pull 478, host-canonical 478, the graph door 1,105, 0 failed, with the
  push-gather stress (20,000 gathers, 0 words differ) and its control.
- The copy probe on the pair (`receipts/p2p-probe3-pair23`, median of the probe's runs): at 16 MiB a
  rank 51.4 to 52.3 GB/s for every block kind through the peer call (and cudaMalloc blocks by device
  address), each direction alone 54.3 to 55.0; the old call, a pool block read by device address,
  runs at 46.3 at 16 MiB and 30.25 at 1 MiB on the pair (36.9 through the peer call at 1 MiB), and a
  block allocated for each gather at 25.2. The 2.5 to 2.9 GB/s of the old call is the four-card
  figure (`receipts/p2p-probe2`), where it was slow.
- The gather crossover (`receipts/gather-kpc-pair23`): 1,024 KiB a rank in 8 rows on two ranks
  takes 46.5 us through kept blocks and 92.6 us inside kernels, so the push limit (inside kernels
  up to 224 KiB a rank) stays.
- Four cards, 40 layers, k3 (A) against this build (B), first-token time of a prefill
  (`receipts/prefillab-kpc-four`, `receipts/prefillab-kpc-four-131k`, the block's tables in
  `prefillab-kpc-four/block/`): A B B A twice at 4,096 and 32,768 tokens, so 4 runs (processes) an
  arm with three rounds each, n 12 an arm; A B B A once at 131,072, 2 runs an arm with two rounds
  each, n 4 an arm. The telemetry is gzipped (`lease.telemetry.csv.gz`).

| tokens | n a arm | k3 median [min, max] ms | kpc median [min, max] ms | kpc / k3 |
|---|---|---|---|---|
| 4,096 | 12 | 4,006.4 [3,960.3, 4,021.4] | 642.4 [633.6, 652.6] | 0.160 (6.24x) |
| 32,768 | 12 | 32,247.4 [31,251.8, 32,707.4] | 4,553.6 [4,514.5, 4,588.1] | 0.141 (7.08x) |
| 131,072 | 4 | 126,661.5 [125,276.4, 127,722.1] | 19,479.2 [19,257.4, 19,692.4] | 0.154 (6.50x) |

The ranges are apart at every length and the token after the prompt is the same on both arms
(48, 646, 851). The bytes moved are the same (288 gathers, 3,456 copies and 27.3 GB a prefill at
4,096 tokens); the host's issue time, which waits on the copies, falls from 3.20 to 0.52 s there.
The 53-tape order gate puts this build in k3's byte family (d = 0, the section below).

### The order gate on 53 tapes: the four speed changes of 2026-11-13

The order gate above ran on the decode lane's 18 tapes. Its interval was wide, so 35 more vendor
tapes (`ext35`: chat and prose, English and Hebrew, 3,011 steps) joined it, and the route and the
four speed changes after it ran all 53 on one RTX PRO 6000 each (`dsv41-run --cases`,
`MEMRA_FULL_PREC=1`, as `ordertapes-k1` ran; `receipts/ext35-gate/`, the by-tape `order_gate.py` on
a CPU box, the current route `cur` = k3).

The builds fall into two byte families on every tape, the gate's files (top ids, top values, lse,
argmax) and the full fp32 rows alike (`receipts/ext35-gate/wrx/families.txt`):

- A: k3 (this route), kpc (the peer copy call), emb (a prefill's host work at load) and r2 (the fp32
  FMA family's narrow chain streamed) write k3's bytes, so d = 0 exactly on all 53 tapes;
- B: the CUTLASS GEMMs and their small-row fix (cfix) write one set of bytes between them, a new
  one, judged against the vendor:

| tapes, family B against k3 | steps | KL cur | KL new | d [by-tape 95%] | top-1 cur / new / vendor's second program | verdict |
|---|---|---|---|---|---|---|
| the 18 decode-lane tapes | 714 | 1.745e-02 | 1.913e-02 | +1.68e-03 [-8.39e-04, +5.21e-03] | 96.6% / 96.1% / 97.2% | pass |
| ext35 (35 tapes) | 3,011 | 1.967e-02 | 1.870e-02 | -9.76e-04 [-4.70e-03, +2.34e-03] | 97.4% / 97.7% / 97.8% | pass |
| all 53 | 3,725 | 1.925e-02 | 1.878e-02 | -4.68e-04 [-3.54e-03, +2.34e-03] | 97.2% / 97.4% / 97.7% | pass |

The 257k needle alone (one tape, 3 steps) has d +8.90e-04 above zero, read by d alone as before.
The interval did not narrow (width 6.05e-03 on 18 tapes, 5.88e-03 on 53): the prose tapes spread
more by tape (sd of the per-tape d 9.8e-03 against 4.0e-03 on the 13 chat tapes). Card time 34.9
minutes on two cards; no thermal stop.

### A prefill's host work, moved to the load

The same trace's host side (`receipts/prefill-host`: `prefill_host.py`, `window_hostgaps.py`,
`alloc_slow.py` over its sqlite export; four ranks, 40 layers) holds three serial items that the
2 GB/s copies hid behind queued work and that a prefill pays once its copies run at 50 GB/s:

| item | 4,096 tokens | 32,768 tokens |
|---|---|---|
| the prompt's embedding rows, gathered on the host and uploaded from pageable memory to every rank | 41.9 MB a rank at 9 GB/s, 4.4 to 4.7 ms | 335.5 MB a rank at 9.5 GB/s, 35 ms |
| Engram's pinned staging, allocated at a forward's first Engram layer when it is too small | 29 to 44 ms and 3 to 16 ms a rank thread (`cuMemHostAlloc`) | 51 to 91 ms (grown) |
| the device pool growing past its high-water mark, one allocation at a time | 6 to 14 ms for each of three allocations a rank thread | 18 and 11 ms |

The host's own issue cost is far below the GPU's: a 4,096-token prefill issues 3,058 launches,
1,728 event waits and 1,042 pool allocations a rank (medians of microseconds; the long totals in
a trace are a full queue's back-pressure), about 75 ms a rank against about 0.6 s of GPU work once
the copies are fast. So what stays on the time to the first token is the serial part before the
first launch and these items.

The 4,096-token window of `prefill_window.py` ended at the next prompt's first layer, so it
counted the 32,768-token prompt's upload (36 ms of the "host to device" row) and its start-up (79
of the 88 ms of host-late idle) as the 4,096 one's. Inside the 4,096 prompt's own layers the host
left a stream idle 9.4 to 11 ms in all.

Each item now happens at the load:

- The embedding table on the card. Each card keeps one copy of `embed.weight` (bf16
  `[129,280][5,120]`, 1.32 GB), read by the card's ranks, from the load (`Runner::load_embed`, with the head; a runner that never loads its residency
  takes it at its first forward). A forward checks the call's ids on the host, uploads them (16 KB
  at 4,096 tokens) and `v41_embed_rows` writes each token's row into its hc stream slots and the
  one-hot pre-mix: the bytes of the host gather, its upload and the hc copies (`dsv41-gate
  kernels`, `schedule/embed_rows`). `forward_batch`'s step block and the draft head keep the host
  rows (one to a few rows a call, already in one upload).
- Engram's staging at construction. `Runner::new` pins each rank's staging for a whole slice of
  `chunk` rows (at most `max_pos`), the size `dsv41-tpbench` reports as the run's pinned memory; a
  forward grows it only past that.
- The pool's reserve at the load. `load_resident` allocates the runs' reserve on each device (the
  longest prefill's workspace and the sequences' state, what it already checks the cards for) and
  frees it. The pool's release threshold is pinned at the maximum (`Engine::new`), so the block
  stays mapped and the runs' allocations of any size come out of it. On GB10 (`pool_probe.py`,
  `receipts/emb-gates-gb10/pool-probe.txt`; ten allocations of a 32,768-token prefill's sizes,
  4.19 GiB):

| pool | allocation calls, host ms (sum, largest) | pool reserved after |
|---|---|---|
| empty | 160.2, 53.9 | 4.19 GiB |
| after one 7.34 GiB block was allocated and freed (264 ms, once) | 0.0, 0.01 | 7.34 GiB |
| a second pass | 0.0, 0.00 | 7.34 GiB |

What stays serial is traced with `dsv41-tpbench --nvtx` (`nsys -t cuda,nvtx`): a range a prefill
cell and, inside a forward, its hashing (`v41 hash`) and its embedding (`v41 embed`), with every
op's range. `prefill_serial.py` splits each prompt's time from its cell's start to its first launch
(the bookkeeping before the hashing, the hashing, the embedding, the rest) with the CUDA calls in
that span, and lists over the cell what the load should have removed; with the ranges,
`prefill_window.py` ends a prompt's window with its own cell.

Checked on GB10 (121a, `receipts/emb-gates-gb10`; GB10 writes the 120a bytes): fmt; the kernel
gate 151 checks, 0 failed, the two new ones byte-equal; the layer gate 161 checks, 0 failed, its
111 saved tensors equal to k3's; teacher-forced runs through `Runner::forward` (`dsv41-run --cases`: capital, code_pal and a 3,761-token prompt over the first 4 layers, Engram layer 1 inside) write every file byte-equal to the transport fix alone's on one rank and on two ranks of the card (18 of 18 each), one rank's equal to two ranks'; the tp gate at 2 and 4 ranks 480 and 916 checks, 0 failed. `dsv41-tpbench` does not fit on GB10 with the checkpoint in its page cache (1.3 to 2.2 GiB free against a 5.2 GiB reserve, both builds), so the pool's reserve at the load runs only in the four-card window. On the 120a target: the release build, fmt and
clippy `--all-targets -D warnings`.

Measured on the pair (2026-11-12, cards 2 and 3, 20 layers, `receipts/prefillab-emb-pair23`): the
transport fix alone (A, kpc) against this build on top of it (B), A B B A, three prompts a run in
the order 4,096, 32,768, 131,072 and back, first-token time:

| tokens | every round, median [min, max] ms (n 6 an arm) | rounds 1 and 2, median [min, max] (n 4) | B / A, every round / rounds 1-2 |
|---|---|---|---|
| 4,096 | 487.4 [471.9, 569.4] -> 480.7 [452.1, 485.6] | 487.4 [485.8, 491.5] -> 482.7 [479.7, 485.6] | 0.986 / 0.990 |
| 32,768 | 3,298.3 [3,287.6, 3,531.3] -> 3,226.0 [3,175.1, 3,248.0] | 3,298.3 [3,295.3, 3,299.0] -> 3,235.7 [3,222.2, 3,248.0] | 0.978 / 0.981 |
| 131,072 | 13,682.4 [13,564.3, 13,731.1] -> 13,348.1 [13,224.3, 13,481.3] | 13,686.0 [13,564.3, 13,731.1] -> 13,400.9 [13,310.8, 13,481.3] | 0.976 / 0.979 |

The first round of the first run is where the load's reserve shows most (A 569.4 ms at 4,096 tokens,
the pool growing; B's first round is its fastest); from the second round on the ranges are apart
at every length. No run flagged for a slowdown, no page-cache drop inside a run. The traced prefill
(`receipts/pre-trace-emb-pair23`, `serial.md`): the host's serial part before a prompt's first
launch is 0.76 ms at 4,096 tokens and 6.20 ms at 32,768, nearly all of it Engram's hashing (0.69
and 6.14 ms); no pinned allocation and no pool allocation of 1 ms or more inside a cell. The
53-tape order gate puts this build in k3's byte family (d = 0).

That first A/B had 2 runs an arm with a cold first A run, measured on kpc plus this change; at 4,096
tokens its second-round ranges touched. The A/B on this change's own base, at the pushed heads
(2026-11-13, cards 2 and 3, 20 layers, `receipts/prefillab-emb-route-pair23`, `per-run.md` there):
the route `7450e9f065` (A) against emb `21bb0500b3` on it (B), A B B A in three blocks, 6 runs an arm,
3 rounds a run, first-token time:

| tokens | A, each run's median (n 6): median [min, max] ms | B, same | best B / A | median B / A (every round, n 18) |
|---|---|---|---|---|
| 4,096 | 490.2 [488.0, 492.1] | 483.8 [480.6, 484.4] | 0.987 | 0.989 |
| 32,768 | 3,317.3 [3,304.8, 3,327.5] | 3,243.1 [3,226.2, 3,246.6] | 0.989 | 0.978 |
| 131,072 | 13,753.8 [13,545.5, 13,783.4] | 13,397.4 [13,313.6, 13,415.3] | 0.983 | 0.974 |

The per-run ranges are apart at every length: -1.3% at 4,096 tokens, -2.2% at 32,768 and -2.6% at
131,072 (the per-run medians' medians). No run flagged for a slowdown, no page-cache drop inside a run.
Owed: the four-card A B B A (40 layers).

The memory the table costs: one copy a card, 129,280 x 5,120 bf16 = 1.32 GB (1.23 GiB). On the pair
the peak a rank rose 1.23 to 1.26 GiB (71.08 -> 72.34 GiB at 4,096 tokens, 83.39 -> 84.62 at
131,072), room the server's memory book no longer has for requests. Ranks on one card share the
copy (the review's fix; before it, four ranks on one GB10 kept 5.3 GB). The pool warm-up at the
load is a speed step only: when its block cannot be had it logs and the load goes on; a sync error
still stops the load. The shared copy checked on GB10 (`receipts/emb-share-gb10`): `dsv41-run
--cases` (capital, code_pal, the 3,761-token needle; 4 layers, Engram layer 1 inside) on two ranks
of one card, which now share the table, writes the one-rank run's 12 value files byte for byte,
and both equal the build before the fix (a table a rank). The build's source is commit `076fa74ed3`
(`source.txt` there: the commit, and the blob ids of the engine sources the run reached).

`dsv41-tpbench` did not run on GB10 for either build (`emb-gates-gb10/gates-b.txt`, `tpbench rc 1`:
with the checkpoint in the page cache the card had about 2.2 GiB free for the 5.2 GiB reserve, does
not fit), so the changed load path has no GB10 run. The serve smoke on the pair (2026-11-13, emb
`21bb0500b3`, `receipts/serve-smoke-emb-pair`): `memra-server` on two ranks, a rank a card, device-push,
`MEMRA_CTX=8192`, up in 5.6 s with the table and the pool warm-up in the load; one `decode_c1`
request (2,048 prompt tokens, 512 out) answered, rc 0, TTFT 0.11 s. The run set
`MEMRA_DSV41_TRUNK_LAYERS=9` (9 of the 40 trunk layers: a rehearsal of the load and the serving loop,
whose answers are not the checkpoint's), so its 283 tokens a second is not a model figure.

## What exact bytes cost at one row on four ranks, and what each relaxation would buy

Every change in this lane keeps the bytes of the one-card route. At one row on four ranks three
items are then fixed by the order of a sum, not by bytes read or by launches. None is opened here;
the numbers are for the decision.

| relaxation | what is exact today | what it costs a layer | relaxed form | would buy | what changes |
|---|---|---|---|---|---|
| The router as a tensor-core GEMM | 384 scores, each a chain of 5,120 fused multiply-adds in fp32 in K order (`v41_gemv_ffma_narrow_kernel`), on every rank, ahead of the plan and the experts | 28 us at any rank count (3.9 MB of weights: 3 us of reads) | bf16 MMAs with an fp32 accumulator, the dense GEMMs' form | about 20 us | the scores' last bits; a token whose sixth and seventh scores are closer than that picks another expert |
| `wo_a` on tensor cores | 8 groups of 1,024 outputs, each over 4,096 bf16 weights with fp32 accumulation, since 2026-11-11 in the K-segment order (8 chains of 512 and a fixed tree; see the bf16 family above), before it one chain in K order (`v41_gemv_bf16_kernel`); split by groups | was 49 us on one rank, 38 on two, about 30 on four, where its bytes (17 MB) are 13 us; the K-segment order took 24 us a layer off a four-card one-row step | the same MMA form | landed (the K-segment order) | the last bits of the attention output (judged by the order gate) |
| Sums of partial outputs in place of two gathers | `wo_b` and the experts' w2 are split by output rows, so each needs all of its input: the `wo_a` columns and the hidden columns are gathered first, then the outputs (4 gathers a layer) | 2 of the 4 collectives | `wo_b` and w2 split by K (a rank holds the columns of its own `wo_a` groups and hidden columns), each rank's partial output summed across ranks (2 reductions a layer, as the baselines do) | 2 collectives a layer at the measured cost of one | the K sum is N partial sums added instead of one chain: last bits of every attention and MoE output |

The first two are one-row items (at more rows both GEMMs fill the card and the chain is not the
bound). The third holds at every row count and also halves the gathered bytes of a prefill.

## Batched decode across sequences: design, not started

Where it plugs in: `decode.rs` `Runner::forward` already takes several calls and walks the trunk
layer by layer, but runs `Group::block_forward_owned` once per call inside each layer. The batch
form runs one block forward per layer for all one-token calls: B rows, row b = call b.

What takes a batch dimension as is (row-wise kernels; all already run with rows > 1 at prefill,
and the kernel gate already proves one-token instance == row of the multi-row instance):
- mHC: `hc_mixes` (<16,16> partial pass == part1 per token, in the kernel gate), `hc_finish`,
  `hc_pre8`, `hc_post8`, `rmsnorm32`.
- every dense GEMM with its `act_quant`: wq_a, wq_b, wkv, index_q, wo_a (bf16 grouped), wo_b, shared
  expert, router (`gemv_ffma_narrow` is per row) and `route`. Rows 1..4 use the small-M kernels, more
  rows the tile/pipelined kernels: same bytes per row (kernel gate).
- MoE: `moe_plan` over B rows (pairs grouped by expert: the prefill machinery), `experts_tiles`,
  `swiglu`, `moe_combine` per row. At N ranks the staged gather of 1559672e22 covers up to 4 tokens
  (256 KiB bound); for more rows raise the bound (48 pairs = 491 KiB a rank at B = 8) or keep the
  exact gather with its host read.
- the collectives (`allgather` moves row blocks) and the head (`linear_bf16` over B rows, one
  readback).

Where CSA2 per-sequence state forces per-sequence work:
1. Positions. RoPE kernels take one start position and assume consecutive rows: need a per-row
   position array (small kernel change), also for `attn_idx` (window/visible bounds per row).
2. Window ring write (128 x 1,024 B a layer a sequence, slot = pos % 128): B small copies, or one
   scatter kernel over a pointer table.
3. Compressor open group (ratio 2/4 layers emit one compressed key every ratio tokens): sequences
   are at different phases; per-sequence launches of two tiny kernels, or one kernel with a per-row
   "emit" mask and per-row destination index.
4. Index keys / candidates / top-k: per-sequence key count n_c and key base. `index_score` (per row:
   Q row x that sequence's K), `candidates`, `topk` only run on index layers (0.12 launches a layer
   on average): per-sequence launches are acceptable (B x ~25 us on those layers), or pad to the
   longest sequence with -inf and pass a per-row K base.
5. Sparse attention reads window ring + compressed KV + selected rows of ONE sequence through an
   index list. Batched: all sequences' rings and compressed KV of a kv-source group in one arena per
   rank (a slab per sequence), and `attn_idx` writes arena-relative indices = slab offset + index.
   `sparse_attn` then runs B rows unchanged (it already takes one index list per row).
6. Engram: per-sequence n-gram hashes on the host, the gathered rows are row-wise.

Byte gate: batched == solo. `dsv41-gate decode --batch B`: B sequences of different lengths
(straddling the 128 window, the ratio-2 and ratio-4 emit phases and an index refresh), each stepped
solo with the per-layer traces saved (attn_out, moe_out, selection, logits), then stepped as one
batch; every sequence's every layer compared byte for byte, plus `dsv41-tpbench` generated-ids
hashes at conc B against conc 1. The argument is the same as for a schedule change: no element's
operation sequence changes, rows never mix.

Estimate for conc 8 from measured kernel times (one card, position 4,096, layer 20,
the per-op table above): a decode layer is 0.573 ms of kernels, 0.157 of it the routed experts
(6 experts, 106 MB of weights a token) and 0.416 everything else. Everything else is latency- or
single-read-bound and row-insensitive (wo_b 44 us a row at decode against 0.39 us a row at 4,096
rows; router 29 against 0.18): about +10% at 8 rows. The experts are bandwidth-bound and scale with
distinct experts: 8 tokens touch up to 48 of the 384 (45 if routing were uniform), so about 7x the
weight traffic: 1.1 ms.
- one card: 0.416 x 1.1 + 1.1 = 1.56 ms a layer, 62 ms a step of 8 tokens = 128 tok/s aggregate
  (one stream today: 22.9 ms a step = 44 tok/s; 8 unbatched streams: the same 44).
- four ranks: the row-insensitive part is today's N-rank layer (0.72 ms measured on two cards, the
  target after the host work about 0.35 to 0.45), x 1.1; the experts split by rank, 11 a rank on
  average and the busiest rank near 1.5x that: about 0.3 to 0.4 ms. Layer 0.8 to 1.2 ms,
  step 32 to 48 ms for 8 tokens: 170 to 250 tok/s aggregate, without drafting.
- with the keys side left per sequence (items 2 to 5 as B launches each): +8 x (about 60 us of
  kernels and 12 launches) a layer, about +0.5 ms: 110 to 150 tok/s. So the arena (item 5) and the
  per-row position kernels (item 1) are part of the first version, not a follow-up.

## What is left, with the expected gain

| what | where it stands | expected |
|---|---|---|
| Four ranks, one row: launches | 11.62 ms a step on four cards at the last four-card pass (290.6 us a layer all in; 17.55 unsplit), 52 launches a rank a layer and 204 us of issue a layer. The second half of a layer now replays as one graph a rank (about 30 of the 52 launches): 2.1% of a one-row step on two cards, measured; four cards not yet | the whole step as one graph a rank on the batched call (positions, tables and Engram rows in one block on the card, key counts retuned by node updates at 0.08 us each): the host out of the layer, and forks free, so the wkv chain, the compressor and the index query can run beside the wq chain (about 25 us a layer on the critical path) |
| The push limit | measured on two cards: in kernels against kept blocks cross at 227 KiB a region; the limit is 224 KiB of stores a rank (region times peers); device-push is the default transport at two to eight ranks | the four-card crossover from the same probe (5 seconds inside a four-card pass) |
| Prefill on N ranks: gathers | kept blocks measured on a pair only (0.73 / 4.98 s at 4,096 / 32,768 tokens, 20 layers); four ranks last measured with per-gather blocks | at the single-flow rate a layer's 0.66 GB of gathers (four ranks) is about 12 ms; the attention's wo_a columns gathered as codes like the hidden ones take another quarter of the bytes |
| Prefill: expert tiles | 4.83 ms a rank a layer at 4,096 tokens on four ranks at the last four-card pass. Since then the input is quantized once a token (2 to 3% of a pair's prefill, 1 to 1.5% of one card's); 128-row tiles at 4,096 rows measured 0.8% slower than 64-row ones, so the tile-row rule stands | the read of each expert's weights once a tile is what is left: a tile that serves two experts' rows, or the weights staged once for an expert's tiles |
| Expert tiles through shared memory | landed for the one-card tiles (`eaa42ee764`) and for the fused w1 + w3 + SwiGLU launch of the split path: 1.04 ms off an 8-row call on two cards | the same body for w2 at 9 to 32 rows, where a batch of more than 8 sequences falls back to several calls of 8 |
| The head at 5 to 8 rows | 2.1 ms against 0.9 at one row: every thread of the 256-thread block runs a chain and the block is bound by its instructions; 16-byte loads of the staged values are built, not gated | about 1.2 ms |
| Dense FP8 at 7 to 32 rows | the 8-column body with K in parts: 2.2 to 4.1% of an 8-row call, 2.5 to 4.1 ms of a 12 to 32-row call on two cards; dense is still 252 us a rank a layer at 32 rows where the weights are 55 us of reads | fewer, larger parts (the terms of a part in registers instead of shared memory), and the same body for bf16 (`wo_a`: 127 us a rank a layer at 32 rows through the 64 x 64 tile kernel) |
| `act_quant` | 10 launches a layer, 1.2 us each in a stream (0.6 ms a 9-layer call when every launch is synchronized, which is the profiler's floor, not the step's) | folding it into each GEMM block would repeat the quantization in every block; into the producers (the norm before wq_a / wkv and the FFN, SwiGLU before w2) it saves 7 launches, about 10 us a layer |
| N-rank decode: batching across sequences | each call runs alone through each block, so 8 streams give one stream's tokens/s | the design above: 170 to 250 tokens/s aggregate at 8 streams on four ranks, from measured kernel times |
| N-rank decode: CUDA graph of a rank's step | the capture, replay, retune and self-check module is in (`dsv41_gpu::graph`) with the second half of a layer as its first span; the batched call takes its host data from one block a rank | the attention half and then the whole step: one upload, one graph launch and the index layers' node updates a rank a step |
| Top-k over long rows | rows up to 133,632 keys take the ballot tables since the last commit of this branch (two blocks an SM): layer 20 at 131,072 tokens 253 -> 167 ms (`receipts/prof1-n4-128k`), against 125 at `aec3b0b401`; a decode step's one row of 131,072 keys takes the scan form in one block, 225 us, and the candidate selection 106 | prefill: the 40 ms above `aec3b0b401` on ratio-1 index layers; decode at long context: a select over several blocks, about 0.25 ms a step on an index layer at 131,072 |
| Decode at long context: the attention's key buffer | every step copies the window ring and the whole compressed KV into one buffer for the sparse attention: 134 MB and 123 us at position 131,072 on a ratio-1 layer, 9 us at 4,096 | a persistent buffer per sequence and kv group with the ring's 128 rows in front: the copy's time, on every compressed-attention layer |
| Decode experts (3 launches, 113 MB of MXFP4 a token-row) | 0.128 ms a token-row after the exchanged-operand tiles: about 35 us of MMAs and 90 us of reads at 1,250 GB/s | 0.02 ms a row if the reads reach the copy rate; the MMA term needs a tile that holds more weight rows per MMA than 16; across cards the expert split by rows divides both terms by the ranks (built: `173d61cf6f`) |
| `wo_b` at decode through the staged ring | measured 0.065 to 0.037 ms with the staged kernel, 0.044 now with the deeper load-ahead | 0.007 ms a layer on one card with a K-keyed dispatch; unmeasured at 1,280 columns per rank |
| Sparse attention | one 86 KB block per SM, 4 warps, load/store unit the busiest pipe; 13% of a 4,096-token prefill, 0.036 ms of a decode layer | a key tile of 32 keys (two blocks per SM) or the 8-warp layout on top of the ldmatrix loads: unmeasured |
| Decode launch count | 58 kernels a layer, 35 of them under 8 us, each about one memory round trip plus the launch | fusing the `attn_keys` chain and the mHC pair: 0.05..0.08 ms a layer; needs the Rust side |
| Pipelined GEMM instruction count | about 30 instructions per MMA against 9.5 for the MMA, its 8 fp32 ops and the fragment loads | 10..20% of the expert and dense GEMMs |
| `act_quant` one thread per 32-block with 16-byte loads | 4% of a 32,768-token prefill, 450 GB/s | 2% of prefill |
| CUDA graphs | measured: a graph launch costs the host 1.4 to 3.8 us for 16 to 256 nodes, a node update 0.08 us, a launch 1.6 to 2.0 us; one card is not launch-bound (the door moves a one-rank step 0.3%) | see the N-rank row |
| The head and Engram | not in these runs (`dsv41-bench layer` times trunk layers) | the head reads 1.3 GB of bf16 per token: 0.8 ms at the copy roofline on one card, a quarter of that per rank |

## Files

- `ab_summary.py`: an interleaved layer A/B (run medians, arm medians, spread, range overlap, output
  hashes, telemetry); `--arms X,Y` compares two arms of a multi-arm run.
- `gemm_ab_summary.py`: the same for `dsv41-bench gemm` (the layer's GEMM shapes in isolation).
- `prof_attr.py`, `prof_compare.py`, `prof_ab_summary.py`: per-op GPU time from nsys profiles, two
  reports side by side, and the interleaved per-op A/B.
- `sass_audit.py`: instruction and local-memory census per kernel of a CUDA object.
- `decode_window.py`, `thread_gaps.py`: where a decode step's wall goes in an nsys trace of an
  N-rank run: each stream's busy and idle time (host-late or device-late), the driver calls a layer,
  each host thread's time inside and between them. `nrank_ab_summary.py`: an interleaved
  `dsv41-tpbench` decode A/B.
- `prefill_window.py`: the same for a prefill: per stream, kernels by name and copies by kind and
  size with their rate, and the idle time before copies and kernels. `prefill_ab_summary.py`: an
  interleaved `dsv41-tpbench` prefill A/B on the time to the first token (the bench's `forward_s` is
  the host's issue time).
- `copy_rate.py`, `copy_pairs.py`: the copies of a trace by size (best and median duration and
  rate) and by card pair, with how many are in flight at once.
- `step_wall.py`: a traced decode step's whole wall by layer index: each layer's median window,
  and what the heavy ones (the head, the Engram layers, the index layers) add to a typical layer
  by kernel and host call.
- `p2p_probe.py`, `p2p_probe2.py`: transfer probes on N cards with no model (torch's NCCL, a small
  CUDA extension): an all-gather by copy pattern, kernel stores and NCCL; the engine's copy factors
  one at a time (pool or cudaMalloc blocks, the peer call or device addresses, pull or push) and one
  copy alone a card pair. `copy_sizes.py`: a trace's peer copies by size into one card.
- `prefill_host.py`, `window_hostgaps.py`, `alloc_slow.py`, `memkind_enum.py`, `big_htod_all.py`: a
  prefill trace's host side: its host-to-device copies by source memory, the host-late idle of each
  stream by where it falls (the prompt's layers or the next prompt's start), the pool allocations
  of 0.5 ms and more, the export's own memory-kind numbering (not CUPTI's: 0 is pageable), and
  every large upload of the run. `pool_probe.py`: whether a block reserved in a device's pool and
  freed serves later allocations of other sizes without the pool growing (ctypes, no model).
  `prefill_serial.py`: a `dsv41-tpbench --nvtx` trace's serial part of each prompt before its first
  launch, by the forward's ranges.
- `steady.py`: the calls of `dsv41-tpbench` runs by how long the card has been busy (the first
  0.15 s, to 0.5 s, from 0.5 s on): the steady figure a served step pays next to the burst one.
  `trace_rot.py`: kernel time by name of a mixed-width trace against the sum of its widths traced
  alone.
- `order_gate.py`: the distribution-level gate of a reduction-order change: a build's teacher-forced
  tapes against the current route's and the vendor's floor (top-32 KL, the paired bootstrap interval
  of the difference per tape group), and the route log's router table (same six experts, flips,
  weight and MoE-output differences).
- `receipts/`: one directory per comparison (`multi-*`: timed arms with their kernel gates;
  `profab-*`: per-op profiles; `gemm-*`: GEMM shapes; `look-*`, `ncul-*`, `ncu-*`: ncu reports;
  `base`, `check-*`, `ab-*`, `prof-*`: the first half of the lane; `nrank-*`: N-rank decode traces;
  `nrankab-*`: N-rank decode A/Bs; `gates-*`: every byte gate on one build; `four`, `four2`,
  `four3`: the four-card phases, one lease each: gates, decode A/B, traces, rows, prefill; `two`,
  `two-pu12`, `link`: the same on a pair; `p2p-*`: the transfer probe; `hostprof-*`: host stack
  samples of a prefill; `prefillab-*`: prefill A/Bs; `prefill-n4`: the unsplit route's prefill trace
  on four cards; `red1`: the control of the launcher fix; `gprobe-*`: launch, graph launch and node
  update costs; `gather-*`: a gather by size, in kernels against kept blocks; `ggate-*`,
  `pairgates-*`: the tp gates under the graph door and on a pair; `pair-g3`: the three-arm pair
  pass of the staged fused launch and the graph half; `night-pair`: the rehearsal of a four-card
  pass on a pair; `wide-*`, `widesweep-*`: calls of 8 to 128 rows and the per-family limits of
  the small-row bodies; `ordertapes-*`, `ordergate-base`: the order gate's tables (the tapes'
  tensors and route logs stay on the box); `eq-*`: two builds' saved layer tensors compared byte
  for byte; `steptrace-*`: kernels by name of a traced step; `ckpt-*`: CheckpointParity and the
  greedy decode gate of a build; `bytecheck-*`: a GB10 build's saved layer tensors against the 120a
  set, and its one-card gates; `k3four`: an arm on the four cards after a night pass; `prefill-host`: the host side of the night pass's prefill trace; `emb-gates-gb10`: the gates of the prefill's host work moved to the load). nsys and ncu report files are
  not kept; their exports and summaries are. The 250 ms telemetry files are gzipped; the box's
  working directory in the logs reads `$LANE`.
