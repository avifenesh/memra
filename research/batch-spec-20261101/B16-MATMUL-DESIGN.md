# Design: an exact 16-row decode matmul at the 8-row tier's bandwidth (GB10)

Status: design, not built (2026-11-04). Target: the 16-row tick's matmuls from 91.5 to about 72 ms at depth 12288, the 8-row tier's time for the same 15.7 GB of weights. Measured per-part numbers: `GB10-PLAIN-DECODE.md`, section "Combined opt-in build and where the 8- and 16-row ticks go now", and `gb10-combo/step/`.

## Why it matters

The served cell plateaus at c8 because the 16-row tick costs more than the speculation it buys. Speculation at 8 sessions pays only while the 16-row tick is under (1 + acceptance) x the 8-row tick, about 1.49x. Today, fp32: 173.3 vs 168 ms; bf16 state: 162.0 vs 159.6. At 16 rows the tick is 91.5 ms of matmuls against 71.7 at 8 rows. A 16-row matmul at the 8-row time puts the 16-row tick near 148 ms fp32 and 137 ms bf16 (after the memset fix), so spec at c8 and plain at c16 both gain:

| cell (depth 12288) | today | with the target matmul |
|---|---:|---:|
| 16-row tick, fp32 | 173.3 ms (92.3 tok/s) | about 148 ms (108 tok/s) |
| spec at c8, acceptance 0.49 | at break-even, gate serves plain | about 8 x 1.49 / 0.148 = 80 tok/s predicted |

## Where the 19.8 ms goes

nsys at 16 rows (`gb10-combo/step/d12288-b16-kern.csv`) against 8 rows (`d12288-b8-kern.csv`), ms per step:

| kernel | 8 rows | 16 rows | delta |
|---|---:|---:|---:|
| `mmvq_bN_rpsc` singles (256 launches, mostly the FFN tensors) | 52.8 | 67.3 | +14.5 |
| group4 door (64 launches) | 13.9 | 17.7 | +3.8 |
| Q5_K output head (1 launch) | 5.0 | 6.5 | +1.5 |

The singles carry three quarters of the gap. They are the FFN tensors: `ffn_gate`/`ffn_up` 5120 -> 17408, `ffn_down` 17408 -> 5120, 50.1 MB each in NVFP4. At 226 GB/s one costs 222 us. The 16-row launch runs about 264 us, the 8-row one about 206 us.

The weight bytes from DRAM are identical at 8 and 16 rows. What doubles is the activation traffic into the SMs. A block is 4 warps x 2 weight rows = 8 rows (`MEMRA_MMVQ_ROWS 4`, rpsc `WROWS 2`), and every block reads all MCOLS activation rows of the projection:

| tensor | blocks | activation bytes per block at 16 rows | L2 -> SM activation traffic per launch, 8 rows / 16 rows |
|---|---:|---:|---:|
| ffn_up / ffn_gate (in 5120, out 17408) | 2176 | 16 x 5120 = 80 KB | 87 MB / 174 MB |
| ffn_down (in 17408, out 5120) | 640 | 16 x 17408 = 272 KB | 89 MB / 178 MB |

So at 16 rows each FFN launch moves about 3.5x its own DRAM weight bytes through L2 into the SMs. ncu (`gb10-ncu/`) agrees: the 16-row rpsc kernel is latency-bound on the L1TEX scoreboard (23.8 cycles per warp, issue slots 19% busy, 39% occupancy at 90 registers), not on DRAM (26% memory throughput).

## The exactness contract

Every design keeps the dp4a chain per (weight row r, column c): lane l walks g = l, l+32, ... in order; per g, `partial = ws0*S0; partial += ws1*S1; acc += adg*partial`; then the 32-lane xor butterfly, lane 0's value written. Integer sums S may come from any instruction, since integer sums are order-free. The float chain must replay in this order. Gate: kernel-check `RP BATCHED` and `FUSED4` bit-bad=0 at m=9..16, the bench output hash, and decode TOKFP equal.

## What has already failed, and what it rules out

| tried (16 rows) | result | what it rules out |
|---|---|---|
| next-quant register prefetch | slower | more registers per lane: occupancy is the binding constraint |
| cp.async activation staging into smem, 8 rows per block | slower | staging that does not cut the per-block re-read: same traffic, plus two block barriers per step |
| four weight rows per warp (16 columns) | slower | 64 accumulators per lane: occupancy again |
| column split 2 x 8 (2 rows) | bench -12%, step flat | splitting columns across blocks: each block still re-reads its activation half, and the FFN singles did not move (268 vs 264 us) |
| exact tensor-core form at 16 rows | 7.09 vs 6.50 ms per pass; step 149 vs 128 | the int8 mma: the integer part is not the cost |

The common thread: no variant reduced the L2 -> SM activation bytes per weight byte. Two levers were tried. One cut the re-read but paid in registers (four rows per warp). The others kept the 8-row block, which fixes the re-read factor.

## Proposed design: a wide block with a shared activation tile (design A)

Keep the rpsc lane program unchanged: same registers per lane, 2 rows per warp, scale rows prestaged. Change three things:

1. **Rows per block:** from 8 to 32. That is 16 warps x 2 rows, or 8 warps x 2 rows x 2 row-tiles walked in sequence by a persistent block.
2. **Activation source:** for each j-step (g = l + 32j), one producer warp copies the step's activation tile, 16 columns x 32 groups x 32 B = 16 KB, plus 16 x 32 `adg` floats, from global into smem. It uses `cp.async.bulk` (the bulk-copy engine, on sm_120/121) and signals on an mbarrier. The tile is double-buffered (32 KB) so the next step's copy is in flight during this step's math.
3. **Activation reads:** each consumer warp reads the tile from smem. Lane l reads its 32 B per column as two `int4` loads from a column-major tile: 32 lanes x 32 B = 1 KB contiguous per column, no bank conflicts in 4 phases. It waits on the mbarrier phase, so no block barrier.

L2 -> SM activation traffic falls by the rows-per-block factor: 32 instead of 8 is 4x less, 44 MB instead of 174 MB per FFN launch, below the 8-row tier's 87 MB. The per-lane chain is untouched: same g order, same ops, only the address the 32 activation bytes come from changes. So it is bit-identical by construction.

Resources per block (16 consumer warps + 1 producer):
- **smem:** 32 KB tile + 16 x 2 x 272 x 4 B prestaged scale rows (34.8 KB) = 67 KB, so one block per SM.
- **Registers:** about 90 x 544 threads = 49K of 64K.
- **Occupancy:** 17 warps per SM against about 18 today, so similar latency hiding with a quarter of the L2 traffic.
- **Tail:** ffn_up has 544 row-tiles over 48 SMs, 11.3 waves; the persistent form walks row-tiles to avoid the 0.7-wave tail.

Predicted: the singles go from 264 to about 206-222 us per FFN launch, bounded below by the 222 us DRAM time. That is -12 to -14.5 ms per 16-row step. The same tile scheme applies to the group4 door (-2 to -3.8 ms) and to the Q5_K head (-1 to -1.5 ms) as follow-ups.

## Alternatives kept on the list

- **B. Four rows per warp with an 8-column split across two warps of the same block.** 32 accumulators per lane, as today; 32 weight registers instead of 16. The activation re-read halves because a block covers 16 rows. It is cheaper to write than A, but the extra weight registers repeat the rpsc4 occupancy loss unless the compiler keeps the count near 100. One kernel template change; worth a bench round before A.
- **C. The tensor-core form with a shared activation tile.** At 16 rows the integer part is not the cost, so this matters only if A moves the bound to issue slots. Not first.

## Measure before building

1. **ncu on GB10:** `mmvq-b16-bench`, FFN shapes only, m=8 and m=16, sections MemoryWorkloadAnalysis plus `lts__t_sectors_srcunit_tex_op_read.sum` and `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`. The hypothesis holds if L2 -> SM read bytes per launch are about 3.5x the weight bytes at m=16 (about 1.7x at m=8), and L2 throughput is the top sub-unit. If it fails, if the L2 sectors equal the weight bytes because L1 already shares the tile across co-resident blocks, design A is wrong and the next suspect is the per-lane load issue rate (design B or C).
2. **Bench, the FFN shapes only:** the step's singles, not blocks 0-3 (lever 1's bench mixed shapes and misled us).

## Gates and adoption

kernel-check bit-identity at m=9..16; bench hash; TOKFP equal; decode tick at 16 rows depth 2048 and 12288, 3+3 in both orders; then the served cell c8/c16. Adopt per box at boot if it wins (the wave probe already times the 16-row arm), with a 5090 measurement before any 5090 default.
