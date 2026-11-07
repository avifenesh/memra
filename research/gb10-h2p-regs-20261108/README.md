# GB10 prime attention: giving h2p its registers back (2026-11-08) - negative

Device: DGX Spark (GB10, compute capability 12.1, 48 SMs, unified LPDDR5x, driver 580.178.04),
SM clock 2,171 MHz at the end of the timed rounds, CUDA 13.0 on the device. Subject: the shipped
`fa_prefill_qw_h2p` (main `7781d4ed66`, `crates/memra-engine/cu/flash_attn.cu` sha256 `e759f4ac`), the
Qwen3.8-27B prime attention on GB10 (24 query / 4 KV heads x 256, causal, f32 Q, BF16 K/V
workspace, bf16 `mma.sync m16n8k16` with f32 accumulation). Served 131k attention was 40.5 s of
about 127 s TTFT (`research/gb10-prefill-next-20261107/`).

**Verdict:** neither register lever is faster. Both are exact: 0 mismatching floats against h2p
at every depth, on random data, real model inputs and ragged shapes. Both lose at depth 131,072.
Freeing 16 Q registers through shared memory gives 78.77 against 85.00 TFLOPS (-7.3%, 7.9% more
kernel time); the warp-pair split gives 65.06 (-23.5%, 30.7% more time). A newer ptxas, tried on the
side because the profile's NOPs pointed at it, gives 77.32 against 85.14 (-9.2%). h2p stays as
shipped and nothing reaches the engine. The probe sources, raw logs and ncu digests are here.

## The premise (from the h2p profile, `research/gb10-prefill-attn-20261107/`, and ncu at 131k)

h2p runs at 85.0 TFLOPS at depth 131,072, 0.83 of the 102 TFLOPS ceiling of the bf16 `mma.sync`
forms it issues. ncu: tensor pipe 78.9% active, issue slots 37.5%, no eligible warp on 62.5% of
cycles, 2.0 warps per scheduler at 255 registers, stalls `wait` 45.5%, `selected` 18.7%,
`math_pipe_throttle` 16.5%. NOPs were 21% of samples and 42% of `wait` stalls. The reading was:
at 255 registers ptxas cannot keep enough independent chains live to cover HMMA latency, and two
warps per scheduler cannot hide it either. Two exact levers would give registers back:

- **(a)** keep Q k-tiles in shared memory (the h2 probe `qh12` had measured +4.6% this way on h2's
  barriers);
- **(b)** split each 16-row tile over a warp pair, about 168 registers per warp, so a CTA holds 12
  warps (3 per scheduler instead of 2).

Both had to keep h2p's per-row program: the same 16-deep QK chains, the same softmax expressions,
the same bf16 P rounding, the same fold of a fresh two-MMA PV chain into O (`fma(O, alpha, C)`; the
shipped SASS executes exactly 130 FFMA per warp-tile: 128 O folds and 2 row sums) and the same
per-lane row-sum order.

## The probe (`probe/`)

`probe3.cu` includes main's `flash_attn.cu` verbatim (the shipped h2p and its helpers) plus the
candidates in `attn3.cuh` and `attn3b.cuh`, runs T = 4,096 rows at depths 4,096 to 131,072 (one
warmup, median of 3 per cell), byte-compares every candidate's O with h2p, and reports TFLOPS on
the causal work and the share of 102. `harness/probe_rounds.sh` runs three rounds with the
candidate order reversed in round 2; `harness/probe_table.py` takes the median over rounds. With
`PROBE_CUBIN`, a kernel named `x<name>` runs that entry symbol from a cubin through the driver API
(the same PTX assembled by another ptxas). Built with nvcc 13.0 on the unit
(`nvcc -O3 -gencode arch=compute_121a,code=sm_121a probe3.cu -o probe3 -lcuda`).

### The shared-memory budget forces a smaller ring

h2p's three K+V stages take 98,368 of the 101,376 B a CTA may use, so no Q k-tile fits beside
them (one k-tile for 8 warps is 4 KB). Both levers therefore need an 80 KB ring. Two were built:

- **K3V2** (`attn3.cuh`): three K planes, two V planes, per-plane full/empty mbarriers. V(t + 1)
  lands in V(t - 1)'s slot, so PV waits on a second barrier in mid-tile.
- **K2V3** (`attn3b.cuh`): two K planes, three V planes and one full barrier per tile. A thread
  issues V(t) at the end of tile t - 2 and K(t) at the top of tile t - 1, and its
  `cp.async.mbarrier.arrive.noinc` after K(t) fires once all its earlier copies landed, V(t)
  included. So a warp waits once per tile, at the top, as in h2p, and both copy points sit where
  h2p's producer sits.

Under ptxas 13.0, every timed K3V2 arm spills inside the tile loop (12 to 19 local loads per
tile in r32e, qa12e, pw0e and pw1q15e: ptxas reloads 11 to 16 O accumulators around the mid-tile V
wait), while K2V3 cuts them to 0 to 5 (rb, qb12 to qb14, pb0, pb0q15; the skewed pair splits keep
7 to 8) (`ptxas/registers.txt`, `ptxas/sass-loop-census.txt`). K2V3 is the base of both final
levers.

### Arms

| arm | change | registers | stack | loop LDL/STL | loop NOPs |
|---|---|---:|---:|---:|---:|
| h2p | shipped | 255 | 8 | 1 / 0 | 70 |
| rb | K2V3 ring, all of Q in registers (control for a) | 255 | 24 | 5 / 3 | 75 |
| qb12 | (a): rb + Q k-tiles 12..15 as per-lane fragments in shared memory (one LDS.128 per k-tile) | 255 | 0 | 0 / 0 | 74 |
| qb13 | Q k-tiles 13..15 | 255 | 0 | 0 / 0 | 79 |
| qb14 | Q k-tiles 14..15 | 255 | 16 | 3 / 0 | 67 |
| pb0 | (b): 12 warps, 2 heads x 3 pairs x 16 rows (BQ 48), K2V3, no skew | 168 | 16 | 3 / 3 | 38 |
| pb0q15 | pb0 + Q k-tile 15 in shared memory | 168 | 8 | 2 / 2 | 32 |
| pb1 | pb0 + groups of 4 warps released one QK apart | 168 | 32 | 8 / 8 | 41 |
| pb1q15 | pb1 + Q k-tile 15 in shared memory | 168 | 32 | 7 / 7 | 43 |
| pb2 | pb0 + group 1 after group 0's QK(0), group 2 after its softmax(0) | 168 | 32 | 8 / 8 | 39 |
| r32e | K3V2 ring, all of Q in registers, refills at the tile's end | 255 | 48 | 12 / 11 | 72 |
| qa12e | (a) on K3V2 | 255 | 80 | 19 / 15 | 76 |
| pw0e | (b) on K3V2, no skew | 168 | 48 | 12 / 12 | 37 |
| pw1q15e | (b) on K3V2, skewed, Q k-tile 15 in shared memory | 168 | 56 | 14 / 14 | 44 |

Registers, stack and the hot loop's static counts are ptxas 13.0's (the hot loop is the backward
branch whose body holds the most HMMAs; per tile and warp). The pair split (b): warp 2p computes S
for keys 0..15 of the 32-key tile with h2p's full-d chains (h2p's `kg = 0` iteration), warp 2p + 1
keys 16..31 (`kg = 16`). They exchange the per-row max through shared memory (max is exact and
order-free), then P as bf16 A fragments and the first warp's per-lane partial row sum; the
second warp continues that sum in h2p's order (`g = 0, 1` then `g = 2, 3`), applies the 4-lane
reduce and keeps l, handing it over at the end. Each warp runs PV for its 128 O columns with h2p's
chains. Two named-barrier syncs per pair per tile; the producer spreads the 2,048 16-byte copies of
a tile over all 384 threads (rows warp + 12j).

## Results (`probe-runs/probe-r1.log`, `probe-runs/probe-x134.log`; unit b23d)

TFLOPS, median over 3 rounds (each a median of 3), and change against h2p in the same rounds; every
cell 0 mismatching and 0 non-finite floats:

| arm | 4,096 | 16,384 | 32,768 | 65,536 | 131,072 | share of 102 at 131k | 131k ms (3 rounds) |
|---|---:|---:|---:|---:|---:|---:|---|
| **h2p** | **62.43** | **78.45** | **82.44** | **84.03** | **85.00** | **0.833** | 152.571, 152.805, 152.889 |
| h2 (main's previous) | 52.29 | 63.75 | 66.10 | 67.30 | 67.70 | 0.664 | 191.346, 191.834, 191.854 |
| rb | 60.26 (-3.5%) | 75.25 (-4.1%) | 78.74 (-4.5%) | 80.59 (-4.1%) | 81.19 (-4.5%) | 0.796 | 159.115, 159.975, 160.214 |
| qb12 | 57.89 (-7.3%) | 73.01 (-6.9%) | 76.00 (-7.8%) | 77.84 (-7.4%) | 78.77 (-7.3%) | 0.772 | 164.522, 164.885, 164.921 |
| qb13 | 62.22 (-0.3%) | 76.41 (-2.6%) | 77.52 (-6.0%) | 77.92 (-7.3%) | 78.27 (-7.9%) | 0.767 | 165.629, 165.943, 166.100 |
| qb14 | 60.36 (-3.3%) | 75.65 (-3.6%) | 79.05 (-4.1%) | 81.17 (-3.4%) | 81.84 (-3.7%) | 0.802 | 158.637, 158.703, 159.194 |
| pb0 | 43.54 (-30.3%) | 61.28 (-21.9%) | 63.44 (-23.0%) | 64.14 (-23.7%) | 64.80 (-23.8%) | 0.635 | 200.032, 200.441, 200.466 |
| pb0q15 | 43.66 (-30.1%) | 60.99 (-22.3%) | 63.42 (-23.1%) | 64.69 (-23.0%) | 65.06 (-23.5%) | 0.638 | 198.915, 199.642, 199.738 |
| pb1 | 41.48 (-33.6%) | 56.54 (-27.9%) | 58.91 (-28.5%) | 60.23 (-28.3%) | 60.23 (-29.1%) | 0.590 | 215.568, 215.644, 215.944 |
| pb1q15 | 42.86 (-31.3%) | 58.87 (-25.0%) | 61.58 (-25.3%) | 62.45 (-25.7%) | 62.91 (-26.0%) | 0.617 | 205.711, 206.461, 207.166 |
| pb2 | 41.69 (-33.2%) | 58.12 (-25.9%) | 59.51 (-27.8%) | 60.66 (-27.8%) | 61.00 (-28.2%) | 0.598 | 212.245, 212.906, 213.161 |
| r32e | 56.44 (-9.6%) | 68.15 (-13.1%) | 71.11 (-13.7%) | 72.31 (-13.9%) | 72.99 (-14.1%) | 0.716 | 177.936, 177.943, 178.502 |
| qa12e | 56.43 (-9.6%) | 67.87 (-13.5%) | 70.73 (-14.2%) | 71.94 (-14.4%) | 72.43 (-14.8%) | 0.710 | 178.511, 179.307, 179.905 |
| pw0e | 44.27 (-29.1%) | 54.33 (-30.7%) | 56.61 (-31.3%) | 57.58 (-31.5%) | 58.42 (-31.3%) | 0.573 | 222.225, 222.306, 223.047 |
| pw1q15e | 43.18 (-30.8%) | 52.58 (-33.0%) | 55.35 (-32.9%) | 55.54 (-33.9%) | 56.25 (-33.8%) | 0.551 | 230.572, 230.892, 232.597 |

Every arm separates completely from h2p at 131k: h2p's slowest round (152.889 ms) is faster than
any candidate's fastest.

- **(a) frees registers and gains nothing.** The 80 KB ring alone (rb) costs 4.5%. Against that
  control, Q k-tiles 14..15 in shared memory (8 registers) give +0.8%, 13..15 (12) -3.6% and 12..15
  (16) -3.0%, with every spill gone: there is no monotone register effect, and the Q loads put
  shared-memory latency into the head of the QK chains. ptxas already interleaves both 16-key halves
  of h2p's QK (four live accumulators on one Q fragment load), so the "wider interleave" the freed
  registers were meant to buy is the schedule h2p already has; an explicit four-accumulator QK loop
  (`r32w`, `qa12w` in `attn3.cuh`) compiles to the same loop (1,037 against 1,038 and 1,002 against
  1,003 instructions).
- **(b) loses about a quarter.** Twelve warps at 168 registers do the same MMAs with far more
  instructions per MMA: the hot loop is 785 to 829 instructions for 64 HMMAs against h2p's 969 for
  128, so a unit of work costs 1.6x the instructions; the CTA covers 96 rows instead of 128, so the
  producer issues a third more copies per FLOP; two pair barriers per tile lock each pair's softmax
  together; and the 1,032-CTA grid (BQ 48) is 21.5 waves on 48 SMs against h2p's 16.0. The skewed
  variants lose more than the unskewed ones.
- The K3V2 ring is worse than K2V3 in every pairing (its mid-tile V wait and the spills around it
  cost 7.5 to 9.6 points: r32e against rb, qa12e against qb12, pw0e against pb0).

## The ptxas side arm: the NOPs are not the bottleneck (`probe-runs/probe-x134.log`)

ptxas 13.0 puts a NOP between back-to-back HMMAs: h2p's loop holds 70 NOPs, and they are where the
profile's 1.735e9 executed NOPs come from (70 per warp-tile, 41.6% of `wait` samples). The same PTX
assembled by ptxas 13.1 or 13.4 (here, `ptxas -arch=sm_121a -O3`) keeps 255 registers and the 8 B
stack and reschedules the loop: 909 instructions against 969 under 13.4, 3 NOPs, and 41 `UIADD3`,
whose predicated-off no-op form `@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ` holds 12 of the 14 top
`wait` instructions of the 13.4 build (`ncu/cb33-xh2p-131072-wait.txt`). So 13.4 still pads, more
sparsely, with another no-op. Only the 13.4 build was timed; 13.1 builds the same 3-NOP loop.
Driver 580.178 loads the 13.4 cubin, and its O equals h2p's bit for bit (scheduling only):

| arm | 4,096 | 16,384 | 32,768 | 65,536 | 131,072 | 131k ms (3 rounds) |
|---|---:|---:|---:|---:|---:|---|
| h2p (ptxas 13.0) | 62.47 | 78.40 | 82.18 | 83.81 | 85.14 | 152.489, 152.552, 152.968 |
| xh2p (ptxas 13.4, same PTX) | 60.66 (-2.9%) | 73.08 (-6.8%) | 75.52 (-8.1%) | 76.71 (-8.5%) | 77.32 (-9.2%) | 167.889, 167.981, 168.002 |
| xrb | 60.32 (-3.4%) | 75.03 (-4.3%) | 77.92 (-5.2%) | 80.09 (-4.4%) | 81.04 (-4.8%) | 160.021, 160.268, 160.313 |
| xqb12 | 57.51 (-7.9%) | 69.48 (-11.4%) | 72.28 (-12.0%) | 73.57 (-12.2%) | 74.19 (-12.9%) | 175.044, 175.060, 175.082 |
| xpb0q15 | 43.51 (-30.4%) | 60.46 (-22.9%) | 63.09 (-23.2%) | 64.40 (-23.2%) | 64.70 (-24.0%) | 200.448, 200.755, 200.777 |

On GB10, ptxas 13.0's HMMA, NOP, HMMA pacing is 9.2% faster for h2p than 13.4's sparser schedule.
The `wait` stalls on those NOPs are pacing, not lost work, and an upgraded GB10 toolkit would need
its own per-kernel A/B before it could become the build's default.

## ncu at depth 131,072 (`ncu/`)

`ncu --set full` on one launch per arm (`harness/ncu_cand.sh`; `ncu_sum.py` and `sass_wait.py`
digests). h2p, rb and qb12 ran on unit b23d (SM 2.127 GHz under ncu), then h2p again, qb14, pb0q15,
pb0 and xh2p on unit cb33 (2.144 GHz). The two h2p runs agree within 0.1 point on every share
below (b23d: 157.6 ms, tensor 78.9%, issue 37.5%, 2.005 / 0.456 warps, `wait` 45.5%). The h2p
column is cb33's, so compare durations within a unit: rb and qb12 against 157.6 ms.

| | h2p | rb | qb12 | qb14 | pb0q15 | pb0 | xh2p |
|---|---:|---:|---:|---:|---:|---:|---:|
| registers / threads per CTA | 255 / 256 | 255 / 256 | 255 / 256 | 255 / 256 | 168 / 384 | 168 / 384 | 255 / 256 |
| duration under ncu (ms) | 156.9 | 165.5 | 170.2 | 162.8 | 204.3 | 205.9 | 173.0 |
| tensor pipe active | 78.9% | 75.8% | 73.3% | 76.3% | 62.6% | 62.0% | 71.8% |
| issue slots active | 37.5% | 37.5% | 34.7% | 36.2% | 45.9% | 46.5% | 31.9% |
| warps active / eligible per scheduler | 2.02 / 0.455 | 2.00 / 0.469 | 2.00 / 0.435 | 2.00 / 0.458 | 2.97 / 0.685 | 2.99 / 0.701 | 1.99 / 0.425 |
| instructions executed | 24.10e9 | 25.26e9 | 24.14e9 | 24.12e9 | 37.90e9 | 38.23e9 | 22.61e9 |
| local loads (warp instructions) | 24.8e6 | 123.9e6 | 0 | 74.3e6 | 99.9e6 | 149.9e6 | 24.8e6 |
| stall `wait` | 45.5% | 41.5% | 43.1% | 41.5% | 35.3% | 34.1% | 34.2% |
| stall `math_pipe_throttle` | 16.5% | 17.5% | 22.5% | 21.6% | 15.8% | 17.0% | 33.9% |
| stall `selected` | 18.7% | 18.8% | 17.3% | 18.0% | 15.5% | 15.4% | 15.9% |
| stall `long_scoreboard` / `barrier` | 2.8% / - | 4.4% / - | 3.6% / - | 3.2% / - | 7.8% / 3.4% | 8.1% / 3.1% | 2.3% / - |
| NOP share of samples / of `wait` | 20.8% / 41.6% | 20.9% / 45.4% | 20.0% / 42.4% | 19.0% / 41.0% | 9.8% / 24.3% | 11.8% / 30.0% | 0 (`UIADD3` no-ops 7.9% / 20.7%) |

- Freeing Q registers (qb12, qb14) removes the spill traffic (qb12: 0 local loads) and leaves the
  instruction count where h2p's is, but `math_pipe_throttle` rises from 16.5% to 22% and the tensor
  pipe falls 3 to 6 points: the warps reach their HMMAs no more evenly, they just queue at the pipe
  in a different place.
- The pair split does what it was built for, and it does not help: 3 warps per scheduler with 0.69
  eligible instead of 0.46, issue 46% instead of 37.5%. But it executes 57% more instructions for
  the same 3.17e9 HMMAs (3.20e9 with the zero rows of the partial last pair), and the tensor pipe
  falls to 62%.
- With ptxas 13.4's sparser padding (xh2p), 1.5e9 fewer instructions issue, `math_pipe_throttle` doubles to
  33.9% and the tensor pipe falls to 71.8%. The NOP between two HMMAs paces one warp's HMMA issue
  and so leaves issue slots to the other warp of the scheduler; the `wait` samples on those NOPs
  were that pacing, not idle tensor cycles.

## Exactness: random data, ragged shapes, real model inputs

- Random data (the rounds above): every arm, 0 mismatching floats at all five depths.
- Ragged shapes (`probe-runs/edge-e1.log`, unit cb33, `harness/edge.sh`): T in {128, 130, 176,
  192, 1,000, 1,024, 3,567, 4,033, 4,075, 4,096} with T_kv from T to 262,144 (tails that are not
  multiples of 32, 48 or 64, fresh primes, partial last CTAs, the served 131k and 260k final-chunk
  shapes): 24 cells x 11 candidates (rb, qb12, qb13, qb14, pb0, pb0q15, pb1, pb2 and the ptxas 13.4
  builds of h2p, qb12 and pb0q15), all equal to h2p bit for bit, none non-finite.
- Real inputs (`probe-runs/real-r1.log`, `harness/real_probe.sh`): the research binary
  `6c739dc807` (`research/gb10-h2p-regs-dump`, `probe/research-dump-hook.patch`: an
  `ATTN3_DUMP_DIR` hook after the h2-family launch in `fa_prefill_view_ws`, never landed) served one
  cold 131,051-token prompt (`harness/dump_run.sh`, the served harness of
  `research/gb10-prefill-next-20261107/`) and wrote the f32 Q and the BF16 K/V workspace of the
  first and the last full-attention layer (layers 3 and 63) at depths 16,384, 32,768, 65,536 and
  131,008 (T = 4,096 / 4,096 / 4,096 / 4,032 rows) and of the 43-row final chunk at 131,051. On all
  ten, every candidate equals h2p bit for bit (120 cells; the input sha256 prefixes are in the
  log). Their speed matches the random rounds: at depth 131,008 h2p 83.57 / 83.38 TFLOPS (layers
  3 / 63), rb 80.47 / 80.08, qb12 78.00 / 77.43, qb14 80.64 / 80.69, pb0q15 67.26 / 66.82, xh2p
  75.98 / 75.77. (The served prompt's last 43 rows run outside the h2 route, which takes T >= 128;
  the probe times them only as a shape.) The dumps (3.7 GB) were deleted at close; the hook commit
  and `dump_run.sh` regenerate them in about ten minutes.

## What this rules out, and what is left

- More registers per warp is not h2p's limit: with 16 freed and no spill, ptxas does not schedule
  the chains better. Lever (a) needs the ring to shrink, and the smaller ring costs more than the
  registers give.
- More warps per scheduler through a pair split is not worth its instruction overhead at this tile
  shape.
- ptxas 13.4's schedule (3 NOPs and predicated-off `UIADD3` no-ops in place of 13.0's 70 NOPs)
  makes h2p slower.
- Not tried here: a longer key tile (BK 64) that halves the per-tile softmax and barrier overhead
  per MMA (it changes the 32-key online-softmax order, so it is a different float program), and
  split-KV (also a different program; `research/gb10-prefill-attn-20261107/`).
