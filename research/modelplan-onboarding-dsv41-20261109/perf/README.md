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

## What exact bytes cost at one row on four ranks, and what each relaxation would buy

Every change in this lane keeps the bytes of the one-card route. At one row on four ranks three
items are then fixed by the order of a sum, not by bytes read or by launches. None is opened here;
the numbers are for the decision.

| relaxation | what is exact today | what it costs a layer | relaxed form | would buy | what changes |
|---|---|---|---|---|---|
| The router as a tensor-core GEMM | 384 scores, each a chain of 5,120 fused multiply-adds in fp32 in K order (`v41_gemv_ffma_narrow_kernel`), on every rank, ahead of the plan and the experts | 28 us at any rank count (3.9 MB of weights: 3 us of reads) | bf16 MMAs with an fp32 accumulator, the dense GEMMs' form | about 20 us | the scores' last bits; a token whose sixth and seventh scores are closer than that picks another expert |
| `wo_a` on tensor cores | 8 groups of 1,024 outputs, each a chain over 4,096 bf16 weights with fp32 accumulation in K order (`v41_gemv_bf16_kernel`); split by groups | 49 us on one rank, 38 on two, about 30 on four, where its bytes (17 MB) are 13 us | the same MMA form | about 15 us at four ranks | the last bits of the attention output |
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
| Expert tiles through shared memory | built, not gated (`v41_gemv_f4s_tiles_kernel` in the working tree): a part of K landed by cp.async, nothing in flight in registers, four blocks an SM | the experts toward the read rate: 706 -> about 400 us a layer at 8 distinct-routed rows, 128 -> about 80 at one |
| The head at 5 to 8 rows | 2.1 ms against 0.9 at one row: every thread of the 256-thread block runs a chain and the block is bound by its instructions; 16-byte loads of the staged values are built, not gated | about 1.2 ms |
| Dense FP8 at 8 rows | +0.76 ms on a 9-layer call (the batching lane's per-kernel table); wo_b's terms pass two blocks an SM at 7 and 8 rows | K in parts for the 8-column body, as the tiles have |
| `act_quant` | 10 launches a layer, 1.2 us each in a stream (0.6 ms a 9-layer call when every launch is synchronized, which is the profiler's floor, not the step's) | folding it into each GEMM block would repeat the quantization in every block; into the producers (the norm before wq_a / wkv and the FFN, SwiGLU before w2) it saves 7 launches, about 10 us a layer |
| N-rank decode: batching across sequences | each call runs alone through each block, so 8 streams give one stream's tokens/s | the design above: 170 to 250 tokens/s aggregate at 8 streams on four ranks, from measured kernel times |
| N-rank decode: launches | 48 kernels, 11.5 peer copies, 18 event waits and 6 records a layer a rank; a launch costs the issue thread 2.1 us alone on the card and 2.7 to 3.6 us with four ranks on one context | fusing `act_quant` into its GEMM (11 launches), the mHC pair and `hc_pre` + RMSNorm (4), the plan's three zero-fills: about 20 fewer launches a layer a rank, 0.05 to 0.07 ms of issue time and 0.02 to 0.03 ms of kernel floor |
| N-rank decode: CUDA graph of a rank's step | not built: buffers are now stable (the free list) and the MoE has no host read at decode, which were its two blockers; positions, key counts and the ring slot have to move from kernel arguments to a small device block | the issue cost of a step (about 0.2 to 0.3 ms a layer a rank) down to one launch a step; only worth it when the cards, not the host, bound the step |
| Top-k over long rows | rows up to 133,632 keys take the ballot tables since the last commit of this branch (two blocks an SM): layer 20 at 131,072 tokens 253 -> 167 ms (`receipts/prof1-n4-128k`), against 125 at `aec3b0b401`; a decode step's one row of 131,072 keys takes the scan form in one block, 225 us, and the candidate selection 106 | prefill: the 40 ms above `aec3b0b401` on ratio-1 index layers; decode at long context: a select over several blocks, about 0.25 ms a step on an index layer at 131,072 |
| Decode at long context: the attention's key buffer | every step copies the window ring and the whole compressed KV into one buffer for the sparse attention: 134 MB and 123 us at position 131,072 on a ratio-1 layer, 9 us at 4,096 | a persistent buffer per sequence and kv group with the ring's 128 rows in front: the copy's time, on every compressed-attention layer |
| Decode experts (3 launches, 113 MB of MXFP4 a token-row) | 0.128 ms a token-row after the exchanged-operand tiles: about 35 us of MMAs and 90 us of reads at 1,250 GB/s | 0.02 ms a row if the reads reach the copy rate; the MMA term needs a tile that holds more weight rows per MMA than 16; across cards an expert split by rows divides both terms by the ranks (not built: loader, plan and one more gather) |
| `wo_b` at decode through the staged ring | measured 0.065 to 0.037 ms with the staged kernel, 0.044 now with the deeper load-ahead | 0.007 ms a layer on one card with a K-keyed dispatch; unmeasured at 1,280 columns per rank |
| Sparse attention | one 86 KB block per SM, 4 warps, load/store unit the busiest pipe; 13% of a 4,096-token prefill, 0.036 ms of a decode layer | a key tile of 32 keys (two blocks per SM) or the 8-warp layout on top of the ldmatrix loads: unmeasured |
| Decode launch count | 58 kernels a layer, 35 of them under 8 us, each about one memory round trip plus the launch | fusing the `attn_keys` chain and the mHC pair: 0.05..0.08 ms a layer; needs the Rust side |
| Pipelined GEMM instruction count | about 30 instructions per MMA against 9.5 for the MMA, its 8 fp32 ops and the fragment loads | 10..20% of the expert and dense GEMMs |
| `act_quant` one thread per 32-block with 16-byte loads | 4% of a 32,768-token prefill, 450 GB/s | 2% of prefill |
| CUDA graphs | not measured: the host issues a layer's 58 launches in 0.13 ms while the card is busy 0.57 ms, so one card is not launch-bound | nothing on one card; revisit when a rank's layer is under 0.15 ms |
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
- `receipts/`: one directory per comparison (`multi-*`: timed arms with their kernel gates;
  `profab-*`: per-op profiles; `gemm-*`: GEMM shapes; `look-*`, `ncul-*`, `ncu-*`: ncu reports;
  `base`, `check-*`, `ab-*`, `prof-*`: the first half of the lane; `nrank-*`: N-rank decode traces;
  `nrankab-*`: N-rank decode A/Bs; `gates-*`: every byte gate on one build; `four`, `four2`,
  `four3`: the four-card phases, one lease each: gates, decode A/B, traces, rows, prefill; `two`:
  the same on a pair; `p2p-*`: the transfer probe; `prefill-n4`: the unsplit route's prefill trace
  on four cards; `red1`: the control of the launcher fix). nsys and ncu report files are
  not kept; their exports and summaries are. The 250 ms telemetry files are gzipped; the box's
  working directory in the logs reads `$LANE`.
