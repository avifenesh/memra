# GB10 F8F4 prefill GEMM v3: the v2 tile fed by TMA under mbarriers (2026-11-08)

**Status:** decided on one device class (unified-memory 120a family), measured on both DGX Spark
units. Every GEMM that takes the v2 tile runs the v3 mainloop, the fused gate/up silu launch
included, with CTA-local TMA copies (the follow-up below; #1102 first shipped v3 for the plain and
residual launches only). `MEMRA_MMQ_F8F4_V3=0` is the rollback seam. Discrete cards keep v1 (v2 is
off there).

## Question

The v2 tile (`docs/decisions/GB10-F8F4-GEMM-V2.md`) is 60.3 s of a cold 131k prefill's kernels at
508-549 MAC/clk/SM. Its loop over resident smem without loads or barriers reaches 567-664, and the
exact epilogue's register-operand ceiling is 821 (`research/gb10-prefill-20261106/GEMM-V3-DESIGN.md`,
ledger rows 33-37). ncu puts barrier, MIO and LG throttle at 1.03, 0.68 and 0.43 stalls per issued
instruction. Does a TMA feed under mbarriers recover part of the feed's share end to end, with the
same numeric program?

## Decision

`mul_mat_f8f4v3` replaces the v2 mainloop in `memra_mmq_nvfp4_f8f4_v2_preq`,
`_v2_gateup_silu` and `_v2_preq_resid` (all take `v3`, from `MEMRA_MMQ_F8F4_V3`, on for GB10):

- the same 256 x 128 tile, 16 warps, K = 64 x 3 stages, 4-row-tile raster groups, image pre-pass and
  activation layout;
- per stage, thread 0 loads the image rows, activation rows and both scale planes with
  `cp.async.bulk.tensor.2d` (CTA-local destination) into v2's swizzled smem (SWIZZLE_64B on 64-byte rows is
  `f8f4v2_layout<.., 64>::swz`); four tensor maps are encoded per call on the host;
- one full (expect_tx) and one empty (one arrival per warp) mbarrier per slot replace the per-stage
  `__syncthreads`; thread 0 refills a slot with stage kt + 3 once every warp has left it; waits carry a
  suspend hint;
- the scale planes load through shared-window addresses;
- both mainloops run the same stage compute and epilogues (`f8f4v2_compute_stage`,
  `f8f4v2_epilogue`), so every output is bit-identical; a call whose tensor maps cannot be encoded
  runs v2; the map encoder is resolved through the runtime (`cudaGetDriverEntryPointByVersion`), so
  no engine binary links libcuda.

#1102 kept `memra_mmq_nvfp4_f8f4_v2_gateup_silu` on v2: with `.shared::cluster` copies, its silu
epilogue raised the register pressure until the mainloop spilled (CUDA 13.0: 28 / 32 B against none
for the plain epilogue), and it ran slower (below). The CTA-local copies reversed that (follow-up).

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
0.48 s). On unit 2 (ledger row 43, cells 1 and 6 against 2 and 5 of the campaign below) the same
binary gave 24.388 -> 24.050 s (-1.38%), 127.532 -> 126.141 s (-1.09%) and 333.484 -> 330.406 s
(-0.92%).

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

## Follow-up: CTA-local copies, gate/up on v3 (2026-11-08, ledger rows 44-45)

The independent review of #1102 traced the issuing thread's cost to the copy form. With a
`.shared::cluster` destination, the compiler wraps every copy in a CTA-id check and a call slow path
(16 local stores). The `.shared::cta` form compiles without either: the v3 plain kernel drops from
1,688 to 1,152 instructions with the same 128 FFMA, 192 FMUL and 32 QMMA (CUDA 13.4 SASS).

Per launch on unit 1, drivers over the engine entries with the image pass included, 4 runs per cell
([`gemm-v3-cta-probe-unit1/`](../../research/gb10-prefill-20261106/gemm-v3-cta-probe-unit1/)):

- The plain and residual launches gain another 1.7-7.1% over the landed v3. down gains the most:
  +6.2 / +7.1% at T 2,048 / 4,096.
- The fused gate/up silu launch on the CTA form runs 12.6 / 11.4% faster than on v2. The cluster form
  was 1.2 / 0.2% slower.

So the follow-up puts gate/up back on v3. Under CUDA 13.0 its v3 kernel spills 12 / 12 B. The cluster
form spilled 28 / 32 B, and v2's gate/up kernel spills 12 / 16 B. v2 now calls the same stage
compute and epilogues, so one copy of the numeric program serves both mainloops. A call whose
tensor maps cannot be encoded runs v2. kernel-check's forced v3 arm passes `v3 = 2`, which returns an
error instead of falling back, so that arm always tests v3.

Exactness on unit 1 ([`gemm-v3-cta-engine-unit1/`](../../research/gb10-prefill-20261106/gemm-v3-cta-engine-unit1/)):
- kernel-check is ALL GREEN, with GATEUP under both mainloops.
- pp3855 logits are identical with v3, `V3=0` and `V2=0`, and to #1102's (`f81e969042a7e155`).

On unit 2 (`gemm-v3-cta-served-unit2/argmax16k-v3/`), the 16k argmax-margin gate PASSES: flips 3,
budget 3, every flip margin-explained. Its probe table is byte-identical to #1102's
(`a42e74b3f28cbd96`).

Served on unit 2, one campaign, cells B L N N L B, one fresh boot per cell
([`gemm-v3-cta-served-unit2/`](../../research/gb10-prefill-20261106/gemm-v3-cta-served-unit2/)).
B and L are #1102's binary (B = `MEMRA_MMQ_F8F4_V3=0`), and N is the follow-up's binary, with the
same server code:

| arm | 32,784 tokens | 131,051 tokens | 259,951 tokens |
|---|---|---|---|
| v2 mainloop (B) | 24.405, 24.371 s | 127.080, 127.985 s | 333.763, 333.204 s |
| #1102 (L) | 24.054, 24.047 s | 126.137, 126.144 s | 330.659, 330.153 s |
| follow-up (N) | 23.054, 23.019 s (-4.22% vs L) | 121.990, 121.979 s (-3.29%) | 322.052, 322.270 s (-2.50%) |

Against the v2 mainloop, N is -5.54 / -4.35 / -3.40%. The answer text is identical in all six
boots, needles found, no request error. The gains over L (1.01 / 4.16 / 8.25 s) are at least 16 times
the largest within-arm spread of the L and N arms at each length (0.035 / 0.011 / 0.506 s).

## Follow-up: the half-height last wave (2026-11-09, ledger rows 64-65b)

A v3 launch with more tiles than SMs whose last wave is partial runs that wave on `rem` of the SMs.
When `2 x rem <= SMs`, the launch now runs its last `rem` tiles in the raster's order as `2 x rem`
128 x 128 tiles: the same template at `PBM = 256`, 4 x 2 warps of the full tile's 32 x 64 span, with
the image and its scale plane in 128-row boxes. Each tile keeps its own k order and runs the same
stage compute and epilogue, so every output is bit-identical. On GB10 the measured case is the 640-tile
grids at T 4,096 (down, its residual twin, ssm_out, wo: 13 full waves and 16 left over). The 1,280-tile wqkv
grid and gate/up do not split at T 4,096.

- Probe on unit 2, launch-interleaved (`research/gb10-prefill-20261106/gemm-v4/p4b-unit2/`): down
  +1.67..+1.77%, down with the residual epilogue +1.37..+1.47%, ssm_out +0.67..+1.46%; bytes identical.
- kernel-check `MMQ-F8F4-V2` and `MMQ-F8F4-V2-RESID` at the 640-tile shapes, and `MMQ-F8F4-V2-TAIL` at
  1,152 rows with a token count picked per SM count (its second half tiles lie wholly past the last row).
- Served cold TTFT against the commit before it, B T T B B T, 3 per arm: unit 2 22.066 -> 22.021 s
  (-0.20%) and 117.949 -> 117.704 s (-0.21%), both separated, 260k a tie; unit 1's medians -0.28% /
  -0.23% / -0.69%, overlapping through two reduced-clock cells. Identical answers.

`MEMRA_MMQ_F8F4_V3=0` turns the split off with the rest of v3.

## Rejected

- v3 for the fused gate/up silu launch with `.shared::cluster` copies: slower than v2 (above);
  superseded by the CTA-local copies.
- From the design lane (ledger rows 34-37): issuing at the top of the iteration (+2.0% in the probe),
  the last warp out refilling the slot (+2.8%), issuing after compute into the older slot (-1.2 / +2.1%),
  a producer warpgroup with `setmaxnreg` (the 16 consumer warps keep only 120 registers and spill:
  -56%), 12 or 8 warps with more registers (-6.8..+2.3% / -18%), and k32 stages 6-7 deep (-17..-21%).
