# The F8F4 prime GEMM tile after v3: what blocks 140 TFLOPS, and what to try (round 4, lever 6)

## Where it stands (131k, the stack head's profile, `profile-r4-stack-131k/shapes.txt`)

`mul_mat_f8f4v3<256, 128, 8, 2, 64, 3, EPI>` (16 warps, one CTA per SM, 128 registers, k64 x 3
stages fed by TMA under mbarriers) is 55.07 s of the 117.8 s of kernels, 6.38 PFLOP at 115.8
TFLOPS (548 MAC/clk/SM at 48 SMs x 2.2 GHz). By launch shape:

| bucket (grid) | GEMMs | calls | s | ms/call | TFLOPS | waves on 48 SMs |
|---|---|---:|---:|---:|---:|---:|
| 4,352 | gate/up silu, 34,816 x 5,120 | 2,048 | 23.69 | 11.57 | 126 | 90.7 (99.6%) |
| 640 | down 5,120 x 17,408; ssm_out and wo 5,120 x 6,144 | 4,096 | 19.59 | 4.78 | 103 | 13.3 (95.2%) |
| 1,280 | the GDN wqkv 10,240 x 5,120 | 1,536 | 5.71 | 3.71 | 116 | 26.7 (98.8%) |
| 768 | the GDN gate z 6,144 x 5,120 | 1,536 | 3.39 | 2.21 | 117 | 16.0 |
| 1,536 | attention q + gate 12,288 x 5,120 | 512 | 2.25 | 4.39 | 117 | 32.0 |
| 128 | attention k / v 1,024 x 5,120 | 1,024 | 0.44 | 0.43 | 99 | 2.7 (88.9%) |

The weight image pre-pass (5.95 s) runs at the DRAM copy rate and is not in these figures.

## The program, and what it costs per MMA

For output row i, token j and k32 blocks b in ascending order, one fresh m16n8k32 block-scale MMA
per block and `s += (dA[i,b] * dB[j,b]) * C_b` (`f8f4v2_compute_stage`). Per warp and k32 step the
tile issues 16 QMMA, 64 FMUL (each `dA * dB` product is unique to one output and one block, so none
is shared), 64 FFMA, 2 + 4 `ldmatrix.x4` and 1 + 4 `LDS.128` for the scales. Per SM sub-partition
(4 warps) that is 1,030 tensor cycles against about 670 issue slots. The exact program's ceilings
(ledger rows 33-37, `GEMM-V3-DESIGN.md`):

| form | MAC/clk/SM | TFLOPS |
|---|---:|---:|
| QMMA alone, accumulating | 1,018 | 215 |
| the exact epilogue, register operands | 821 | 173 |
| the shipped loop over resident smem (no loads, no barriers) | 567-664 | 120-140 |
| v3 as shipped (all shapes) | 548 | 116 |

## What the register file blocks

- 16 warps x 128 registers is the whole 64K file. The exact epilogue needs one f32 accumulator per
  output per thread: a 256 x 128 tile over 512 threads is 64 accumulators, half the budget, before
  any fragment, scale or address.
- So the A/B fragments of step k+1 cannot be loaded while step k computes (24 more registers for a,
  b; 20 for the scales): the 664 -> 821 gap is that missing double buffer, and it is the largest
  piece above the feed.
- Fewer warps with more registers lose (rows 35-36: 12 warps at 168, -6.8..+2.3%; 8 at 235, -18%):
  with the math pipe throttling at 3.3 stalls per issued instruction, the loop needs 4 warps per
  scheduler to cover the QMMA-to-FFMA latency.
- A producer warpgroup leaves 120 registers per consumer and spills (-56%).
- sm_121a has no asynchronous MMA (no wgmma, no tcgen05 / TMEM), so accumulators cannot leave the
  register file the way Hopper or B200 tiles do.
- Smem: a k64 stage is 27 KB (A 16 KB, B 8 KB, scales 3 KB), so 3 stages (81 KB) is the deepest ring
  under the 99 KB per-CTA limit; k32 stages 6-7 deep lose 17-21% (row 37).

Nothing exact gets past 821. 140 TFLOPS (664) is the resident-smem loop's best shape: a tile that
reaches it needs the feed fully hidden. Past it, it needs fewer operand-staging stalls per MMA than
the shipped loop has.

## Candidates (all keep the per-element program, so all are bit-identical by construction)

1. **Finer-grained slot release.** v3 refills a slot once all 16 warps leave it, so the elected
   thread waits for the slowest warp before every refill (14% of stall samples sit on one mbarrier
   check, row 34). The A part of a slot is read only by the 2 warps of its 32-row group, the B part
   only by the 8 warps of its 64-column group. Separate empty barriers per A group (8) and B group
   (2), with the refill split into 10 boxes, let a group's data refill as soon as that group is done.
   Target: v3 -> the resident loop's rate on the feed share (548 -> 600-620).
2. **Persistent CTAs with cross-tile prefetch.** Each CTA loops over tiles (raster order kept) and
   starts the next tile's first stages before the current epilogue. That hides the per-tile ring
   fill and drain: K / 64 = 80 steps per tile (272 for down), 3 of them exposed per tile. It also
   overlaps the gate/up silu quantize epilogue. Target: +2-4% on the 640 and 4,352 grids.
3. **Wave balance for the 640-tile bucket** (19.6 s at 95.2% wave efficiency). Run the last wave as
   128-row tiles (each CTA keeps its own full k order; split-K and stream-K are out because they
   change the sum's order), so 640 tiles become 13 full waves plus a half-height one. Target: about
   1 s at 131k.
4. **Partial fragment double buffer.** Within the 128 registers, prefetch only `b` and `dbv` of the
   next `pp` (8 registers) ahead of the current `pp`'s 4 MMAs. The SASS decides whether ptxas already
   hoists it. Target: part of 664 -> 821; unknown until the SASS and ncu are read.

## Probes (cb33, one lease per step)

- P0: fresh ncu (`--set full`) of the shipped v3 entries at T 4,096 on the gate/up silu, down, wqkv
  and ssm_out shapes (the design lane's ncu was of v2): stall reasons per issued instruction, tensor
  pipe active %, issue slot use, and the SASS instruction mix per k32 step. Also mode 3 (resident
  smem) on v3's stage compute, to re-measure the feed ceiling under the CTA-local copy form.
- P1: candidate 1 in the probe (`tma_probe.cu` lineage), bit-compared to the shipped tile, timed in
  both in-process orders, 11 launches.
- P2: candidate 2, then 4, the same way; candidate 3 last, because it changes the launch shapes.
- Engine work starts only on an arm that beats the shipped v3 by at least 5% on its shapes (n >= 5,
  both orders). The served A/B and the gates follow the usual pattern.

## Expected TTFT

GEMM time scales with the prompt (55.07 s at 131,051 tokens, so 13.78 s at 32,784 and 109.2 s at
259,951). Saved seconds for a tile throughput gain across all shapes:

| v3 -> | 32k | 131k | 260k |
|---|---:|---:|---:|
| 125 TFLOPS (+8%) | 1.1 s (-4.9%) | 4.4 s (-3.7%) | 8.8 s (-2.8%) |
| 140 TFLOPS (+21%, the resident loop) | 2.4 s (-10.9%) | 9.6 s (-8.1%) | 19.1 s (-6.1%) |

(percentages against the stack head's served TTFT: 22.1 / 118.4 / 315.3 s.) Candidates 1 + 2 + 3
together are aimed at 125-130 TFLOPS. Reaching 140 also needs candidate 4, or another way to stage
operands that the probes have not found yet.

## h2p attention, for comparison

40.8 s at 131k at 83-85 TFLOPS against a 102 TFLOPS mma.sync ceiling, at 255 registers with two
warps per scheduler. Rows 47-49 found no exact schedule above h2p. At the ceiling the gain is about 7
s at 131k (up to 20 s at 260k, where attention grows faster). It needs the same kind of register
budget the GEMM lacks. The GEMM goes first: it is about half of the remaining time at 131k and has
two candidates (1, 2) that free no registers at all.

## Probe results so far (ledger rows 59-61, `gemm-v4/`)

- P0 (row 59): the shipped v3 under ncu. Tensor pipe 48-58% active, issue 39-49%. QMMA carries 45%
  of the stall samples (math-pipe throttle), FFMA 11%, the mbarrier waits (SYNCS) 11%. MIO and LG,
  v2's feed stalls, are now small. FMUL and FFMA are 65% of the executed instructions, four of each
  per QMMA, which is the exact program's fixed cost.
- P1 (row 60), candidate 2, persistent CTAs: bit-identical, but slower on every shape (-0.1..-6.7%).
  The tile loop's state spills 16-40 B in the mainloop, and the hardware's CTA scheduling balances
  better than a static walk. Rejected.
- P2 (row 61), mode 3 on v3: with no feed after the prologue, the GEMMs run 6-10% faster on gate/up
  and down, 2-5% on wqkv, flat on ssm_out. So a perfect feed is worth about 6% of the GEMM time
  (about 3.3 s at 131k, 6.5 s at 260k), and that is the ceiling for candidate 1. Everything above
  mode 3 is the stage compute under the 128-register cap, which no measured structure changes.
  140 TFLOPS is out of reach for the exact program on this tile.
- h2p sizing (row 62): mode 3 (no K/V feed) matches h2p within 0.1% above 64k (+1.3% at 32k), and
  ncu puts h2p's tensor pipe at 78.6% active against 79.7% in mode 3. The attention prime is
  compute-bound. Its idle fifth is the dependency chain of 2 warps per scheduler at 255 registers;
  no feed or wait lever exists there, and rows 47-49 found no exact schedule above it.
