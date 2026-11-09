# DeepSeek-V4.1-Flash onboarding: kernel speed on one card, bytes unchanged (2026-11-09)

Verdict: the V4.1 layer kernels of `crates/memra-engine/cu/dsv41_gpu.cu` run a decode step in
0.62 / 0.60 / 0.50 ms per layer (from 3.99 / 3.39 / 2.97 ms at the lane's start) and a 4,096-token
prefill in 34.1 / 32.1 / 31.0 ms per layer (from 57.5 / 55.6 / 53.6 ms), on layers 2, 20 and 21 of the
real checkpoint on one RTX PRO 6000. Every change is a schedule change: `dsv41-gate kernels` proves each dispatched
kernel byte-identical to the reference schedule on the fixtures, the layer gate's saved tensors are
byte-identical between the binaries before and after, the N=2 partition gate still writes the
one-card bytes, and every timed run prints output hashes that match across binaries.

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

Base and now are the arm medians of the interleaved runs (`receipts/multi-f34` at 4,096: three runs
per arm, ranges do not overlap, spread at most 1.2%; `receipts/multi-f34g/pfx2` at 32,768, spread at
most 1.6%). The lane-start column is the A arm of `receipts/ab-6e8bf3b41f-4k` and, at 32,768, the
single profile run of `receipts/base`. Card at 102 W median, 72 C median over the 4,096 runs and
127 W, 80 C (86 C peak) over the 32,768 runs, SM clock 2,272 MHz median; output hashes identical in
every run of both arms.

The model has 40 layers. Three layers do not make a model number; as a scale, the mean of the three
times 40, without the head, Engram and the host:

| | base | now |
|---|---|---|
| decode step | 25.9 ms | 22.9 ms |
| prefill 4,096 tokens | 1.64 s (2,499 tokens/s) | 1.30 s (3,158 tokens/s) |
| prefill 32,768 tokens | 14.0 s (2,337 tokens/s) | 10.8 s (3,047 tokens/s) |

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

## What is left, with the expected gain

| what | where it stands | expected |
|---|---|---|
| Decode experts (3 launches, 106 MB of MXFP4 per token) | 0.157 ms, 670 GB/s against a 1,650 GB/s copy roofline; latency-bound per block (tile header, two load rounds, the ordered sum) | up to 0.06 ms a layer on one card; little per rank at EP4 (1..2 experts a rank) |
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
- `receipts/`: one directory per comparison (`multi-*`: timed arms with their kernel gates;
  `profab-*`: per-op profiles; `gemm-*`: GEMM shapes; `look-*`, `ncul-*`, `ncu-*`: ncu reports;
  `base`, `check-*`, `ab-*`, `prof-*`: the first half of the lane). nsys and ncu report files are
  not kept; their exports and summaries are. The 250 ms telemetry files are gzipped; the box's
  working directory in the logs reads `$LANE`.
