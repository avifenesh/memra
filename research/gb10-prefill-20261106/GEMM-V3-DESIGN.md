# F8F4 prefill GEMM on GB10: a TMA-fed, warp-specialized tile (design lane, 2026-11-07)

Status: measured, not taken to the engine. The best exact arm is worth +7.6% on gate+up in a
standalone probe; the lane's bar for engine work was 10%. Receipts: `gemm-v3-probe-unit2/` (unit 2,
CUDA 13.0, SM clock 2.10-2.19 GHz under load). MAC/clk/SM figures use 48 SMs at 2,200 MHz, as the
round-3 map does; ledger rows 33-37.

## Starting point

At 131k the shipped v2 tile (`mul_mat_f8f4v2<256,128,8,2,64,3,1>`, one 16-warp CTA per SM, 128
registers) is 60.3 s of 124.9 s of kernels, 509-549 MAC/clk/SM by shape. The question was whether a
producer/consumer structure on TMA, or a register reallocation that admits fragment double
buffering, closes the gap to the exact program's ceiling.

## What the exact program allows (measured)

| form (16 warps per SM) | MAC/clk/SM | source |
|---|---:|---|
| QMMA accumulating into C (no software scale) | 1,018 | round-3 `probe_mma_f8f4.cu` chain |
| fresh C, `s += da * db * C` with the products hoisted out of the loop | 880 | round-3 `scaled` (SASS: all 16 FMULs before the loop) |
| fresh C, the exact epilogue with scales that change every k32 step, register operands | 821 | `src/coissue.cu` `tile` (16 QMMA + 64 FMUL + 64 FFMA per step) |
| the shipped tile's loop over resident smem: no global loads, no barriers (mode 3) | 567-664 | `stage1.log` |
| the shipped tile | 508-549 | `stage1.log`, every stage's REF lines |

The round-3 880 ceiling hoists the four `da * db` products; with non-hoistable scales the register
form reaches 821, so the 26 s of headroom estimated from 880 shrinks to about 22 s. Of the gap
between the shipped tile (~525) and 821, the feed (cp.async, the per-stage `__syncthreads`, L2
traffic) is ~525 -> 615 (mode 3), and staging the A/B fragments and scales from smem under the
128-register cap is 615 -> 821. `ceil_probe3.cu`'s 132 TFLOPS (628 MAC/clk/SM) was pessimistic: its
A operand is rewritten by a dependent IADD before every MMA.

ncu at the gate shape (`ncu/summary-gate.txt`): the shipped tile issues on 38.8% of cycles with the
tensor pipe 50.5% active; per issued instruction the stalls are math-pipe throttle 3.33, fixed
latency 1.25, barrier 1.03, MIO throttle 0.68, LG throttle 0.43. Mode 3 keeps the math-pipe throttle
(3.25) and drops barrier, MIO and LG (0.04, 0.10, 0.02); tensor 62.4%.

## What sm_121a offers (compile and runtime checks)

- TMA 2D tile loads (`cp.async.bulk.tensor.2d`, SASS `UTMALDG.2D`), bulk copies (`UBLKCP`), mbarrier
  with transaction counts (`SYNCS.*`), `elect.sync`: compile and run.
- `setmaxnreg`: compiles to `USETMAXREG.DEALLOC/TRY_ALLOC` and works at runtime (`nreg_rt.cu`), and
  ptxas allocates past the entry count after `.inc` (`nreg_alloc.cu`: 168 at entry, R0-R203 used
  after `.inc 232`, no spills).
- TMA multicast compiles with an advisory that the feature has "substantially reduced performance"
  on this target; not used.
- TMA `SWIZZLE_64B` on 64-byte rows puts 16-byte chunk c of row r at c ^ ((r >> 1) & 3), which is
  `f8f4v2_layout<.., 64>::swz`, so the ldmatrix addresses are unchanged (`SWIZZLE_32B` matches the
  32-byte rows of a k32 stage the same way).

The register file decides the design space: 16 warps x 128 registers is the whole 64K file. A
producer warpgroup leaves at most 120 per consumer thread (512 x 120 + 128 x 32), and the consumer
loop needs 128 (it spills a little even there). Twelve consumer warps fit at 160-168.

## Probe (`src/tma_probe.cu`, bit-identical to the shipped tile)

The shipped mainloop and epilogue verbatim; the four stage operands (A image, B activation, A and B
scale planes) arrive by TMA into the same swizzled smem; one full and one empty mbarrier per slot
replace `__syncthreads`. Every arm returned the shipped tile's output bit for bit on 5 shapes at
T 4,096 and on gate+up at T 1,024 and 130 (`bitdiff=0` on every line of `stage*.log`).

| arm (vs the shipped tile timed in the same process; `pairs-hint.txt`, `pairs-nohint.txt`) | gate+up | down | wqkv | ssm_out |
|---|---:|---:|---:|---:|
| elected thread issues stage kt+2 at the top of iteration kt | +0.3% | -2.0% | +3.9% | +0.9% |
| elected thread issues stage kt+3 at the end of iteration kt (issue-end), plain wait | +5.2% | +1.8% | +7.1% | +4.5% |
| issue-end, waits suspend on the barrier (n = 7) | **+7.6%** (5.0..10.0) | +1.9% | +7.8% | +5.4% |
| issue-end + scale loads in the shared window (n = 3) | +8.2% (5.5..10.6) | +4.1% | +6.9% | +5.7% |
| last warp out of a slot refills it (no empty wait) | +2.8% | -0.7% | +4.4% | +1.6% |
| issue after compute into the slot left one stage earlier (fixed or rotating issuer) | -1.2 / +2.1% | -1.9 / +2.4% | +3.1 / +3.0% | +1.6 / -9.9% |
| k32 stages, 6 or 7 deep | -25% | -20% | -17% | -19% |
| producer warpgroup + `setmaxnreg` (16 consumers at 120) | -56% (spills) | | | |
| 12 warps of 32 x 64 at 168 registers (192 x 128 tile), 3 / 4 stages | -6.8 / +2.3% | -9.3 / 0.0% | -6.8 / +1.9% | -5.0 / +1.6% |
| 8 warps at 235 registers (128 x 128 tile), 5 stages | -18% | -21% | -18% | -16% |

The issue-end arm removes the barrier, MIO and LG stalls (1.03, 0.68, 0.43 -> 0.01, 0.04, 0.10) but
the waits move to the mbarriers: the elected warp waits for the slowest warp before every refill,
and the rest sleep on the barriers (0.96 per issued instruction; 14% of stall samples sit on one
mbarrier check, `ncu/sass-stalls-v3-issue-end-gate.txt`). More registers per thread with fewer warps loses: this loop is latency-bound
and wants 4 warps per scheduler.

## Expected engine gain

Applying the issue-end arm's measured speedups to the 131k shape split (`round3` profile: gate+up
26.7 s, down 20.8 s, wqkv 6.2 s, the 6,144-, 12,288- and 1,024-row GEMMs 6.6 s) saves about 3.1 s
of the 60.3 s (3.6 s with the shared-window scale loads), about 2.4% of a 127.6 s cold 131k
prefill. A perfect feed (mode 3's rates) would save about 9.7 s; the remaining 615 -> 821 is the smem operand
staging under the 128-register cap, and no structure measured here keeps 16 warps and frees
registers.

## Verdict

Not taken to the engine: +7.6% on gate+up (n = 7, 5.0-10.0%) against the 10% bar. If the owner
wants the ~3 s, the build is the issue-end arm with suspended waits and shared-window scale loads,
host-side tensor maps per call (four `cuTensorMapEncodeTiled` per GEMM), the GB10 route and the
v2 kernel-check bit cells; nothing in it changes a value.
