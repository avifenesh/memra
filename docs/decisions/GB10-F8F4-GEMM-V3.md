# GB10 F8F4 prefill GEMM v3: the v2 tile fed by TMA under mbarriers (2026-11-08)

**Status:** decided on one device class (unified-memory 120a family), measured on DGX Spark
unit 1; the unit 2 confirmation follows as ledger row 43. The plain and residual-epilogue GEMMs that take the v2 tile run the v3 mainloop; the fused
gate/up silu launch keeps v2. `MEMRA_MMQ_F8F4_V3=0` is the rollback seam. Discrete cards keep v1
(v2 is off there).

## Question

The v2 tile (`docs/decisions/GB10-F8F4-GEMM-V2.md`) is 60.3 s of a cold 131k prefill's kernels at
508-549 MAC/clk/SM. Its loop over resident smem without loads or barriers reaches 567-664, and the
exact epilogue's register-operand ceiling is 821 (`research/gb10-prefill-20261106/GEMM-V3-DESIGN.md`,
ledger rows 33-37). ncu puts barrier, MIO and LG throttle at 1.03, 0.68 and 0.43 stalls per issued
instruction. Does a TMA feed under mbarriers recover part of the feed's share end to end, with the
same numeric program?

## Decision

`mul_mat_f8f4v3` replaces the v2 mainloop in `memra_mmq_nvfp4_f8f4_v2_preq` and
`_v2_preq_resid` (both take `v3`, from `MEMRA_MMQ_F8F4_V3`, on for GB10):

- the same 256 x 128 tile, 16 warps, K = 64 x 3 stages, 4-row-tile raster groups, image pre-pass and
  activation layout;
- per stage, thread 0 loads the image rows, activation rows and both scale planes with
  `cp.async.bulk.tensor.2d` into v2's swizzled smem (SWIZZLE_64B on 64-byte rows is
  `f8f4v2_layout<.., 64>::swz`); four tensor maps are encoded per call on the host;
- one full (expect_tx) and one empty (one arrival per warp) mbarrier per slot replace the per-stage
  `__syncthreads`; thread 0 refills a slot with stage kt + 3 once every warp has left it; waits carry a
  suspend hint;
- the scale planes load through shared-window addresses;
- the stage compute and the plain and residual epilogues are shared device functions
  (`f8f4v2_compute_stage`, `f8f4v2_epilogue`) holding v2's operation sequence, so every output is
  bit-identical; the v2 kernels' SASS is unchanged (per-function compare).

`memra_mmq_nvfp4_f8f4_v2_gateup_silu` keeps the v2 mainloop: under v3 its silu epilogue raises the
register pressure until the mainloop spills (CUDA 13.0: 28 / 32 B against none for the plain
epilogue), and it ran slower (below).

## Evidence

Served A/B on one GB10 (unit 1), Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft at 262,144
context, the final head (`4924aeddf2` on main `c347e782d2`), one binary, B = `MEMRA_MMQ_F8F4_V3=0`
(the v2 mainloop), L = the default, cells B L L B, one fresh boot per cell, one cold prompt per length
per boot ([`gemm-v3-engine-final-unit1/served/`](../../research/gb10-prefill-20261106/gemm-v3-engine-final-unit1/served/)):

| arm | 32,784 tokens | 131,051 tokens | 259,951 tokens |
|---|---|---|---|
| v2 mainloop | 24.437, 24.472 s | 127.771, 127.891 s | 333.900, 334.377 s |
| v3 (gate/up on v2) | 24.084, 24.085 s (-1.51%) | 126.524, 126.553 s (-1.01%) | 331.138, 331.194 s (-0.89%) |

The answer text is identical in all four boots, needles found, no request error. The gains (0.37 /
1.29 / 2.97 s) are at least six times the largest within-arm spread at each length (0.035 / 0.12 /
0.48 s). The second unit's confirmation is ledger row 43.

First campaign, every v2 entry on v3 including gate/up (unit 1, `3146763781` on main `e7b375532d`,
[`gemm-v3-engine-served-unit1/`](../../research/gb10-prefill-20261106/gemm-v3-engine-served-unit1/)):
24.44 -> 24.20 s at 32k (-0.99%), 127.99 -> 127.00 s at 131k (-0.78%), 333.96 -> 332.20 s at 260k
(-0.53%), identical answers; about a third of what the probe predicted. The kernel split of a served
cold 32k prefill under nsys, v3 off / on / off
([`gemm-v3-engine-nsys-unit1/`](../../research/gb10-prefill-20261106/gemm-v3-engine-nsys-unit1/)),
found the gap: the fused gate/up launch, the largest bucket, went 6.601 / 6.591 -> 6.675 ms per call
at its 2,176-tile grid (4,480 calls, 29.6 s, +1.2%) and 13.107 / 13.135 -> 13.329 ms at its 4,352-tile
grid (576 calls, +1.6%), while the 640-tile GEMMs went 2.869 / 2.853 -> 2.727 ms (-4.7%) and wqkv 4.030 / 4.070 ->
3.780 ms (-6.7%). A driver over the engine's own entries
([`gemm-v3-engine-epi-unit1/`](../../research/gb10-prefill-20261106/gemm-v3-engine-epi-unit1/),
both orders, image pass included) agreed: gate/up silu -1.2 / -0.2%, down +1.0 / +0.9%, down with the
residual epilogue +2.7 / +1.7%, wqkv +6.8 / +7.0%, ssm_out +5.1 / +5.3%. Re-deriving the tile origin
from `%ctaid.x` after the loop and per issued stage, to cut the gate/up spills, halved them but lost
on every shape (`runb*.log`: the issuing thread's per-stage divisions slow warp 0), so gate/up went
back to v2.

Exactness: kernel-check ALL GREEN (124 cells, 5 skipped) in the default, `MEMRA_PRIME_KV_H2=0` and
`MEMRA_PRIME_KV_T3=1` modes on unit 2 (`gemm-v3-engine-gates-unit2/`) and in the default mode on the
final head on unit 1 (`gemm-v3-engine-final-unit1/kernel-check/`), every `MMQ-F8F4-V2*` cell (plain at every NVFP4
shape of the 27B and the 9B, T 16 / 130 / 1,024 / 4,096, both weight layouts; row tails; silu; gate/up;
residual) run under both mainloops against the v1 tile: 0 mismatches. argmax-margin flips=0 bad=0
PASS; run-spec K=1..8 self-consistency PASS (acceptance 17/98 and 17/112, as before). On unit 1
(`gemm-v3-engine-gates-unit1/`): the 16k argmax-margin gate PASSES with v3 and with
`MEMRA_MMQ_F8F4_V3=0` (flips 3, budget 3, every flip margin-explained) and the two probe tables are
byte-identical (`a42e74b3f28cbd96`); run-gen pp3855 logits are identical with v3, `MEMRA_MMQ_F8F4_V3=0`
and `MEMRA_MMQ_F8F4_V2=0` (`f81e969042a7e155`), at 3.361 / 3.429 / 3.926 s.

## Rejected

- v3 for the fused gate/up silu launch: slower than v2 (above).
- From the design lane (ledger rows 34-37): issuing at the top of the iteration (+2.0% in the probe),
  the last warp out refilling the slot (+2.8%), issuing after compute into the older slot (-1.2 / +2.1%),
  a producer warpgroup with `setmaxnreg` (the 16 consumer warps keep only 120 registers and spill:
  -56%), 12 or 8 warps with more registers (-6.8..+2.3% / -18%), and k32 stages 6-7 deep (-17..-21%).
