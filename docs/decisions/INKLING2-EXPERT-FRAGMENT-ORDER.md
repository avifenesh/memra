# INKLING2-EXPERT-FRAGMENT-ORDER: the declared expert GEMM reads its weights in MMA-fragment order (2026-11-22)

Scope: the Inkling 2 Mini declared expert GEMM (`cu/inkling2_actq.cu`, `ink2_expert_gemm_kernel`), both classes that
run it (shipped, nvfp4-rig). The float class keeps the stored row-major layout (`gemm_bank` refuses a fragment-order
bank and the declared GEMM refuses a row-major one).

## Chosen

The expert banks are repacked once at load (`ink2_frag_values`, `ink2_frag_scales`): per 16-row tile and K step, the
512 bytes of the tile's A operand in lane order, each lane's fragment 16 contiguous bytes; the scales as
`[tile][word][16 rows]`. Each lane loads its fragment with one 16-byte `ld.global.nc.L1::no_allocate` straight into
registers, 16 K steps ahead for one-row groups (8 otherwise). The bytes, the MMA operands and the MMA order are
unchanged, so every output is bit-identical (ink2-actq-gate 48/48 against the reference chains; route digests equal
to the pins). A layout change, not a requantization.

Measured (research/modelplan-onboarding-inkling2mini-20261121/perf/s15-lanek/lk3): nvfp4-rig c=1 67.6 -> 78.6 tok/s,
shipped 54.2 -> 63.0, 4k 61.9 -> 72.5; the decode expert GEMM 60.8 -> 31.5 us a launch, 84% of the 865 GB/s read
roofline (was about 52%).

## Rejected

- The stored row-major layout through a private 16-stage `cp.async` ring per warp read back by `ldmatrix` (the
  kernel before): 8 one-warp blocks an SM by shared memory, MIO-throttle and long-scoreboard bound (lane P's ncu).
- The same ring at 8 stages (twice the blocks an SM, one wave): slower (lk1: 76.8 -> 74.8 tok/s); bytes in flight a
  warp, not occupancy, bound it.
