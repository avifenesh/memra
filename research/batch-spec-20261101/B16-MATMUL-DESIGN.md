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

## The first hypothesis, tested and refuted: activation traffic

A block is 4 warps x 2 weight rows = 8 rows (`MEMRA_MMVQ_ROWS 4`, rpsc `WROWS 2`), and every block reads all MCOLS activation rows. On paper that doubles the L2 -> SM activation bytes at 16 rows: 87 -> 174 MB per ffn_up launch, about 3.5x the 50 MB of weights. The first draft of this design (a 32-row block sharing a cp.async.bulk activation tile in smem) rested on that.

ncu says otherwise (`gb10-b16-ncu/`, blk.0 tensors, `lts__t_sectors_srcunit_tex_op_read.sum`):

| launch | m=8: L2 -> L1 sectors, L1 hit | m=16: L2 -> L1 sectors, L1 hit | L2 throughput m8 / m16 |
|---|---|---|---|
| ffn_gate (2176 blocks) | 5.50 M, 78% | 4.91 M, 90% | 27% / 26% |
| ffn_up (2176 blocks) | 6.11 M, 76% | 5.41 M, 89% | 27% / 20% |
| ffn_down (640 blocks) | 2.13 M, 92% | 4.76 M, 90% | 13% / 8% |

Co-resident blocks already share the activation rows in L1: the L1 hit rate rises with m and the L2 sectors stay flat on the 17408-row tensors. Neither L2 (8-27%) nor L1 (16-34%) throughput is near a limit. The shared-tile design would cut traffic that is not the bottleneck, so it is dropped before building.

## What the 16-row kernel actually waits on

ncu with boost clocks, stall sampling and SASS attribution on the FFN launches (`gb10-b16-ncu/stalls.txt`):

| ffn_gate, one launch | m=8 (`b8_rpsc`) | m=16 (`b16_rpsc`) |
|---|---:|---:|
| duration | 321.6 us | 415.9 us |
| registers per thread, block limit | 72, 7 blocks | 90, 5 blocks |
| active warps per scheduler | 6.20 | 4.70 |
| executed instructions | 26.6 M | 40.7 M (1.53x) |
| warp cycles per issued instruction | 27.8 | 19.1 |
| of which long scoreboard (global load not yet back) | 22.7 | 14.3 |
| issue slots busy | 19.7% | 23.3% |

Time follows instructions x cycles per instruction / resident warps: 26.6 x 27.8 / 6.20 against 40.7 x 19.1 / 4.70 is a 1.39x ratio; the measured ratio is 1.29x. Stall samples sit on the first use of loaded registers: `IDP.4A` (dp4a) operands 51% and the `SHF` that consumes the loaded quant word 26%. So the 16-row kernel loses on two fronts at once:

1. It runs 1.53x the instructions per weight byte, because the column loop doubles while the weight decode and loads stay.
2. Its 90 registers (32 accumulators plus 16 decoded weight words plus the column's activations) drop residency from 6.2 to 4.7 warps per scheduler. So fewer warps hide each load's latency.

The column loop issues each column's two activation `int4` loads and its `adg` load right before that column's dp4a chain. The `if (c >= m) break` inside the unrolled loop keeps the compiler from hoisting later columns' loads. So a warp waits once per column on an L1 round trip, and on L2 for the 10-13% that miss.

## The exactness contract

Every design keeps the dp4a chain per (weight row r, column c): lane l walks g = l, l+32, ... in order; per g, `partial = ws0*S0; partial += ws1*S1; acc += adg*partial`; then the 32-lane xor butterfly, lane 0's value written. Integer sums S may come from any instruction, since integer sums are order-free. The float chain must replay in this order. Gate: kernel-check `RP BATCHED` and `FUSED4` bit-bad=0 at m=9..16, the bench output hash, and decode TOKFP equal.

## What has already failed, and what it rules out

| tried (16 rows) | result | what it rules out |
|---|---|---|
| next-quant register prefetch | slower | prefetch paid for with registers: residency is the binding constraint |
| cp.async activation staging into smem, 8 rows per block | slower | smem staging: same latency class as an L1 hit, plus two block barriers per step |
| four weight rows per warp (16 columns) | slower | 64 accumulators per lane: residency again |
| column split 2 x 8 (2 rows) | bench -12%, step flat | splitting columns across blocks: the FFN singles did not move (268 vs 264 us) |
| exact tensor-core form at 16 rows | 7.09 vs 6.50 ms per pass; step 149 vs 128 | the int8 mma alone: the integer part is not the cost |

## Proposed design: compile-time width, batched column loads, fixed residency (design B2)

Keep the rpsc lane program and its chain. Change only the schedule of loads within a g step and the register budget:

1. **Compile-time width.** Instantiate the 16-row kernel for the exact m=16 the served tick runs, with no `c >= m` test inside the column loop; m=9..15 keep today's kernel. The compiler can then schedule the 16 columns' loads freely.
2. **Column groups of 4.** Issue the 8 activation `int4` loads and 4 `adg` loads of columns c..c+3 together, then run their dp4a chains while the next group's loads are in flight. That is two groups in flight: 2 x (8 x 4 + 4) = 72 registers of activations in the naive form. Too many, so the staged values are the packed `int4` words consumed in place: 36 extra registers in the worst case and 20 if the `adg` loads ride one `float4`. The float chain per (r, c) is untouched.
3. **Hold residency.** `__launch_bounds__(128, 5)` keeps the 5-block (4.7 warps per scheduler) residency it has now, and the build fails if the schedule needs more. If it fits in 4 blocks only, the batching must pay for the lost warp; the bench decides.

Expected effect: the per-column L1 round trip leaves the critical path. With 4 columns per group the waits per g step fall from 16 to 4. At today's 14.3 long-scoreboard cycles per instruction that is the bulk of the stall, so the target is the 8-row kernel's 1.29x instruction-time ratio less the latency share. If cycles per issued instruction fall from 19 to about 12 at 4.7 warps, the FFN launch drops from 416 to about 280 us under ncu. That is the 8-row kernel's time, and about -14 ms per 16-row step.

Risk: the compiler may spill or reorder the groups. The SASS must show the grouped `LDG`s ahead of each group's `IDP` chain, and no local memory.

## Alternatives kept on the list

- **C. Fewer instructions per weight byte.** The m=16 kernel decodes each weight word once per g (shared across 16 columns), so the 1.53x is the dp4a and float epilogue, not the decode. The integer part could move to int8 `mma.sync` with the weights as A (16 rows) and the 16 activation columns as B (two n8 tiles), keeping the float epilogue. The earlier tensor-core form lost at 16 rows, but it was built for 32 rows and folded the butterfly through smem. Worth a second look only if B2 leaves the kernel issue-bound.
- **A (dropped).** The shared activation tile. Refuted above.

## Measure before building (done for A, to do for B2)

1. Done: ncu L2 -> L1 sectors and stall attribution (`gb10-b16-ncu/`). They rule out A and point at load latency per column plus residency.
2. Next: a one-kernel B2 prototype in `mmvq-b16-bench` on the FFN shapes, `MEMRA_BVAR` door, ncu stall attribution against `b16_rpsc`. It goes forward only if the FFN launch time drops at equal hash.

## Gates and adoption

kernel-check bit-identity at m=9..16; bench hash; TOKFP equal; decode tick at 16 rows depth 2048 and 12288, 3+3 in both orders; then the served cell c8/c16. Adopt per box at boot if it wins (the wave probe already times the 16-row arm), with a 5090 measurement before any 5090 default.

## Prototype result (2026-11-04): adopted in part

`research/b16-rpscx-proto` (584a8522de) built `nvfp4_mmvq_batched_rp_scx<MCOLS, WROWS, CG>` and forced it with `MEMRA_BVAR`. Every variant hashes identically to rpsc on the bench (31 tensors, blk.0-3), and kernel-check `RP BATCHED m=8/16` stays at bit-bad=0 (`gb10-b2/`).

Bench pass, GB10, 3 rounds (ms; `gb10-b2/sweep.txt`, `round2/sweep.txt`):

| 16 rows | ms per pass | | 8 rows | ms per pass |
|---|---:|---|---|---:|
| rpsc (today) | 6.38-6.62 | | rpsc (today) | 4.65-4.78 |
| width only (CG=1) | 6.26-6.62 | | width only (CG=1) | 4.65-4.75 |
| **CG=2** | **5.60-5.77** | | CG=2 | 4.63-4.75 |
| CG=2, `__launch_bounds__(128,5)` | 5.65-5.77 | | CG=4 | 4.20-4.21 |
| CG=4 | 7.01-7.09 | | CG=8 | 4.08-4.21 |
| CG=4, `__launch_bounds__(128,5)` | 7.40-7.48 | | | |
| CG=8 | 7.26-7.27 | | | |

ncu, FFN launches (`ncu-*.csv`, boost clocks, one launch each):

| variant | ffn_down (640 blocks) | ffn_gate / ffn_up (2176) | registers | warps per SM | local memory |
|---|---:|---:|---:|---:|---|
| 16 rows rpsc | 517 us | 399-406 us | 90 | 19.4 | 0 |
| 16 rows CG=2 | 500 us | 380-381 us | 96 | 19.1 | 0 |
| 16 rows CG=2 bounded | 483 us | 375-379 us | 96 | 19.1 | 0 |
| 16 rows CG=4 | 530 us | 437-444 us | 132 | 11.8 | 0 |
| 16 rows CG=4 bounded | 608 us | 487-489 us | 96 | 18.9 | 160 MB spill |
| 16 rows CG=8 | 520 us | 468-472 us | 171 | 7.7 | 0 |
| 8 rows rpsc | 437 us | 317-328 us | 72 | 25.6 | 0 |
| 8 rows CG=8 | 340 us | 300-303 us | 128 | 14.0 | 0 |

The FFN launch went from 403 to 380 us, short of the ~350 us goal. Grouping cuts long-scoreboard stalls only while residency holds. Past two columns per group, the extra registers cost more warps than the overlap buys back.

Decode step, one binary, `MEMRA_BVAR` unset (rpsc) vs forced, 3+3 in both orders (`gb10-b2/step/ab.txt`). TOKFP is identical in every run.

| cell | rpsc (ms/tick) | forced (ms/tick) | change |
|---|---|---|---:|
| 16 rows, depth 12288, CG=2 | 167.46, 167.21, 167.34 | 161.70, 161.90, 161.73 | **-3.4%** |
| 16 rows, depth 2048, CG=2 | 127.07 | 121.90 | **-4.1%** |
| 8 rows, depth 12288, CG=8 | 109.97, 109.90, 110.13 | 110.18, 110.27, 110.35 | +0.2% (flat) |
| 8 rows, depth 2048, CG=8 | 90.22 | 90.28 | flat |

nsys at 16 rows depth 12288: the singles go from 67.31 to 61.86 ms per step. The 16-row matmul falls from 91.5 to 86.1 ms, so 14.1 ms remain over the 8-row tier. At 8 rows the singles stay at 53.1 ms. That tier is already at the bandwidth ceiling in the step: the bench gain comes from the small tensors, which the step routes through group4.

Kept: `qmatvec_nvfp4_mmvq_b16_rpscx` (CG=2, unbounded) for full 16-row calls, in 88805bc1f1. Deleted (SHAs on the research branch): CG=1/4/8, the bounded forms and every 8-row twin.

What remains: 14 ms of the 16-row matmul, now with 96 registers at 19 warps per SM. Grouping alone cannot remove it. The next candidates both cut instructions per weight byte rather than hide latency:
- design C, the integer part on `mma.sync` with the float epilogue unchanged;
- 16-row coverage in the group4 door (17.7 ms at 16 rows against 13.9 at 8).
