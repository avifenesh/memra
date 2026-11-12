# DSV4 matrix expert program as the loaded default, 2026-09-11

## Decision

DeepSeek-V4 loads the matrix whole-request expert program for prefill, one-row decode and
speculative verification (memra #461). It was the default-OFF door `MEMRA_DSV4_MOE_PROGRAM=matrix`
until the owner flipped it on 2026-09-11 and the door was deleted. The scalar reference executor
stays in the tree as the arm every class row compares against, reachable only through
`dsv4_doors::arm_reference_expert_program_for_gate`. The verdict and the doors deleted with the flip
are in the [FLAGS.md](../FLAGS.md) removed-doors ledger, entry "Removed doors, 2026-09-11 (the
matrix expert program is the default; split-K goes with the flip)".

## Why it is a different numeric class

No CUDA kernel was written for the dispatch, but it selects a different kernel family. The
reference walks each slot with scalar f32 FMA over 16- or 32-element FP4 groups on the packed
`experts_w`/`experts_sc` plane and reduces through a fixed 128-lane tree. The matrix program runs a
tensor-core MMA chain over the split-plane ModelOpt table with a persistent split-K tail, fuses gate
and up, and folds the routing weight and clamp inside the kernel. Both consume the same FP8-QAT
activation; the weight plane, the arithmetic unit, the K decomposition and the fusion point differ.
Bit-equality is unavailable by construction, so the matrix arm owes drift rows, not an exactness
gate. Program-tagged state refuses matrix/reference crossings. Kernels: `memra_moe_kq_*` and
`memra_dsv4_moe_fused_*` ([KERNELS.md](../KERNELS.md)).

## Rejected

- The scalar reference program as the loaded default: slower at every measured prefill length
  (below), and kept only as the class oracle.

## Measurement that settled it

- Class (`dsv4_moe_program_class_gate`): 0 of 448 rows bit-equal across seven domains, 20 top-1
  changes, KL mean 0.011106 and max 0.220805; sampler stream and refusal ordering unchanged.
- Drift at the canonical arm (matrix with EP off), as recorded in the FLAGS.md ledger: 4.464% top-1
  on the whole tape, 6.250% on the 320 rows that can flip, 0.000% on code, KL mean 0.0111 and max
  0.273; HC24 split dots S16, an accepted class, drifts 3.571% on the same tape.
- Accuracy control: 179/300 against 179/300, exact McNemar p = 1.000000.
- Gain at the canonical arm, prefill only, admission minimum held in both arms: +119.5% at 153
  tokens, +43.3% at 981, +28.2% at 3,686 (reference 135.7 and 135.5 tok/s against matrix 173.8 and
  173.8 at 3,686).

Receipts for the class rows: `research/dsv4f-2card-1m-20260904/matrix-request-program.md` and
`research/dsv4f-2card-1m-20260904/matrix-distribution.md`. The prose as it stood in the kernel
inventory is in [history/KERNELS.md](../history/KERNELS.md).
