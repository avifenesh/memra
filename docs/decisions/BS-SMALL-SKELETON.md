# The shared small-batch skeleton of the block-scaled GEMMs, DeepSeek-V4.1 its first caller (2026-11-18)

## Decision

The small-batch machinery of the DeepSeek-V4.1 route's block-scaled GEMMs moves out of the route into two
header-only units that any route can instantiate:

- `crates/memra-engine/cu/bs_small.cuh`: the device MoE plan and its tile list (`bs::moe_plan_launch`, kernels
  `bs_moe_count_kernel<CH>`, `bs_moe_scan_kernel<NE>`, `bs_moe_place_kernel<CH>`, `bs_moe_plan_small_kernel<NP,
  NE>`), the grouped expert GEMM's small-row body on its two cp.async rings (`bs_small_moe_kernel<Ring, Math,
  Epi, CW, PK>` with `Ring` = `bs::Warp6`, the six-stage warp ring, or `bs::Block2`, memra#1324's block ring;
  `bs::SmallRowMoe`), and the dense small-row body (`bs_small_dense_kernel<Math, Epi, CW, MR>`,
  `bs::SmallRowDense`, `bs::dense_small_launch` on a caller's `bs::DenseLaunch`).
- `crates/memra-engine/cu/bs_cutlass.cuh`: the CUTLASS configuration template `bs::cut::Cfg<EAct, EW, AlW,
  TileN, Swap, Grouped>` over CUTLASS 4.8.1's sm_120 block-scaled kernels, the dense launcher `bs::cut::plain`,
  and the grouped (ptr-array) call with its device setup kernel (`bs_cutlass_setup_kernel<Cfg>`,
  `bs::cut::grouped`).

The first math policy is `bs::MxHw32<WF>`, the V4.1 program exactly; the first epilogue is `bs::EpiBf16`. The
`memra_v41_*` FFI entry points stay as the route's wrappers (contracts, return codes, the kernel gate's ring seam
and every speed rule: which calls take a small body, the ring and column choices, the dense geometry), so no Rust
caller changes. The extraction
is held to the same SASS, kernel for kernel, on sm_120a and sm_121a (below).

## What is shared and what stays with the route

Shared (the census of 2026-11-18 found these generic): the tile list and the device plan (they take `bm`, `topk`
and the rank's expert range, and know nothing of the model), both cp.async rings of the grouped body, the dense
small body, each instance's shared-memory size and limits (`Warp6` sets its limit once a device; every `Block2`
instance is asserted under the default 48 KB at the most tile rows, since a policy's operand bytes set the stage
size), the register-carry bound of the dense body's stage (`bs::dense_carries`), and the CUTLASS normal and
swapped forms. The grouped body reads the fixed expert table record `bs::ExpW` (three projections' codes and scale
streams, their second-level scales), not a policy type: a route whose experts have another table shape brings a
record change with it.

Per route: the math policy (operand formats, the scale stream's layout and meaning, the MMA, what follows a
scale word's K blocks and the end of the chain), the epilogue, the FFI contracts and return codes, and every speed
rule, each measured by its caller on its own shapes and cards. For the V4.1 route: which calls a small body takes
against the route's CUTLASS forms (`memra_v41_moe_small_fits`, `memra_v41_dense_small_fits`); the grouped ring by
card (`v41_moe_blk_card`: sm_120 with at least 188 SMs takes `Block2`, as memra#1324 measured on the RTX PRO
6000) and each ring's column rule (Block2 8 columns while the launch is one wave of four blocks an SM, else 16;
Warp6 16, 8 or 4 by warps an SM); the dense geometry (`v41_dense_geo`: an SM's columns in one block, the stage as
deep as about 48 KB a SM of weights in flight allows); the form rule of its CUTLASS calls (`pick`); the scale-row
kernel of its grouped calls (`sfrow_kernel`) and the seams its gates set (`memra_v41_moe_small_ring`). The
skeleton takes geometry as arguments (`bs::SmallRowMoe<Ring, Math, Epi, CW, PK>::launch`, `bs::DenseLaunch`) and
never picks it, so a new caller does not inherit V4.1's tuning: it measures its own and passes it.

## Why the math is a policy and not a scale parameter

The three programs the census found differ in where the scaling happens, not only in its granularity:
DeepSeek-V4.1 scales inside the tensor core (`kind::mxf8f6f4 block_scale scale_vec::1X`, ue8m0 codes per 32 on
both sides, the scale selectors picking each lane's code); Step-3.7's FP8 experts run the same instruction at the
identity scale and fold the 128-wide block's weight and activation scales into a second accumulator in software
(`cu/mmq_fp8_blk.cu`); DeepSeek-V4-Flash's `moe_kq` streams dequantize to f16 before an m16n8k16 MMA
(`cu/moe_kq_prims.cuh`). A granularity number cannot say which of these runs, and each route's bytes depend on
its own program (AGENTS.md: one numeric program per request; a format is not supported by substituting another
program). So the program is a type: an instantiation names its policy, its kernel symbol carries the policy
(`bs_small_moe_kernel<bs::Block2, bs::MxHw32<1>, bs::EpiBf16, 8, 16>`) and a trace shows which program ran. A
policy also cannot be combined with a scale stream it does not read, which a numeric parameter would allow.

Instantiations with the same template arguments in two units are one symbol: the plan kernels (no policy), a
policy two routes share, a shared CUTLASS configuration's setup kernel. Every unit that instantiates the same
`bs::` kernel must therefore compile it identically (the same header revision and the same nvcc flags, `-fmad`
included); otherwise which unit's copy runs is undefined. Today only the V4.1 units include the headers, and
`dsv41_gpu.cu` (`-fmad=false`) and `dsv41_cutlass.cu` instantiate disjoint kernels.

The weight format is part of the policy (`MxHw32<0>` e4m3, `MxHw32<1>` e2m1), not a separate template argument
of the bodies as first proposed: Step-3.7's experts are e4m3 under a different program, so a body that fixed the
experts' format would not fit them.

## The policy interface, and how the later policies fit

A policy gives the skeleton: `kWB`, `kWT` (a weight row's bytes a K block of 32, and where lane t's A register
starts), `kAB` (an activation row's bytes a K block), `wreg(p)` (the A register from staged bytes), `sf_off` (the
byte offset of a row's scale word of four K blocks), `wrow`, `arow` (which rows' scale words a lane reads, one
or two of each), `kAcc`, `zero`, `mma` (the four MMAs of a scale word), `block_end` and `finish`. The epilogue
gives `Out` and `conv`. The skeleton owns the staging, the loop over stages, the m16n8k32 fragment addressing and
the ascending K order.

- `IdFold128` (Step-3.7 FP8 experts; not in this change): `kWB` 32 and `kWT` 4 (the e4m3 fragments of
  `MxHw32<0>`); the scale word of a row and four K blocks is the 128-wide block's f32 scale, the same word a row
  the rings already stage (`sf_off` maps a weight row to its 128 x 128 block's scale and an activation row to its
  per-128 scale); `wrow` gives rows g and g + 8 and `arow` rows 2t and 2t + 1 (the scales of the lane's four
  accumulator elements); `mma` runs the identity-scale MMA into a block accumulator; `block_end` folds it into
  the chain's accumulator (`kAcc` 8) in `mmq_fp8_blk.cu`'s order and clears it. The fold must be written with
  explicit `__fmul_rn` / `__fmaf_rn` so the including TU's `-fmad` setting cannot move it.
- `F16Deq` (DeepSeek-V4-Flash `moe_kq` streams; not in this change): the scales live inside the weight blocks,
  the MMA is m16n8k16 on f16 with the weights dequantized into B registers. The staging and the loop over stages
  fit as they are; the fragment step does not, since the skeleton's fragment addressing is m16n8k32's. That
  policy will bring a fragment hook in place of the skeleton's A and B loads, and the change that adds it must
  show the `MxHw32` instantiations' SASS unchanged again.
- Epilogues `EpiF32` and `EpiF32Alpha`: `Out` = float; an alpha would come in as a kernel argument of its own
  instantiations, which leaves the existing instantiations' parameter lists alone.

## Byte identity: the method and the result

No GPU was available for this change. The proof is the compiled code: `cu/dsv41_gpu.cu` and
`cu/dsv41_cutlass.cu` with build.rs's flags (nvcc 13.1, `-O3 -std=c++17 --expt-relaxed-constexpr`, `-fmad=false`
for the first, the vendored CUTLASS includes for the second) for sm_120a and sm_121a, before (`801243f06e`, the
head of #1337; `17c6b4fd75` since the rebase below) and after; `cuobjdump -sass` and `-res-usage` of each object; each kernel's SASS body (every
instruction and encoding) and its resource line compared, kernel for kernel, after masking nvcc's
directory-derived internal-linkage prefixes (the kernel-object method of the V4.1 record's
`perf/receipts/cfix-pro6000/kcmp/`) and mapping each moved kernel's new name to its old one. Result: 104 of 104
kernels of `dsv41_gpu.cu` and 9 of 9 of `dsv41_cutlass.cu` equal on each arch (226 of 226; 79 + 79 unmoved, 25
+ 25 moved, 8 + 8 CUTLASS kernels unmoved and 1 + 1 setup kernel moved). The other build arches compile
`dsv41_gpu.cu` too (the CUTLASS unit is its stub there): 104 of 104 equal on each of sm_89, sm_90a and sm_100a.
Receipts: `research/bs-small-skeleton-20261118/`.

Rebased on 2026-11-19 onto `17c6b4fd75` (#1337 rebased onto main), the skeleton's plan carries memra#1315's row
map as main has it: `bs_row_bad` and the `rows_bad`, `row0`, `rows_cap` arguments on the count kernel, the
one-block plan kernel and `bs::moe_plan_launch`, with the argument check in the route's wrapper. Against the new
base the result is again 104 of 104 and 9 of 9 on each arch; the two plan kernels memra#1315 changed differ
between the old and the new base and between the old and the rebased head with the same digests, so the rebase
adds the row map and nothing else (`research/bs-small-skeleton-20261118/README.md`, rebase section).

What had to match for that, found by the comparison and kept in the header's comments:

- A thread index read inside a called (inlined) function is bounded by 1024, one read in the kernel by its
  launch bounds; the warp ring's `rw0` came out as an add instead of an or. Both ring bodies are therefore in the
  kernel itself (`if constexpr` on the ring), with their reads where the route's kernels had them.
- Expression forms stay those of the route's kernels: the output element is converted before its address is
  formed (`C[i] = Epi::conv(...)`, not a store helper taking the address first), fragment and scale-word
  addresses are written out as before rather than through address callbacks, and the lane's scale rows are plain
  ints rather than a struct (an aggregate copy reaches the optimizer later; the block ring's 8 x 16 shape moved
  two address computations across its loop, every other kernel did not).

## Rejected

- One scale-granularity parameter for every route (above).
- Ring bodies as called functions taking the thread index as an argument: the same SASS for 103 of 104
  kernels, but the block ring's 8 x 16 shape differed; the kernel-body form gives 104 of 104.
- Speed rules in the skeleton (a first version of this change kept the ring choice by card, the ring column rules
  and the dense in-flight target in the shared launchers; the independent review of #1343 flagged them): they
  were measured on this route's shapes and cards, so they moved to the route, and the skeleton takes geometry as
  arguments. The SASS comparison below was rerun after the move.

## Not shown by this change

The SASS comparison proves the device code of the two units unchanged; it does not run anything. The GPU battery
(`research/bs-small-skeleton-20261118/gpu-battery.sh`) ran on four RTX PRO 6000 on 2026-11-22, comparing the change
with memra#1337's head. The kernel gate gave 1,091 of 1,091 on each arm, with the check lines equal. The four-rank
tensor-parallel gates gave 1,056 / 1,054 / 1,054 on the three transports on each arm, with the layer hashes equal
(`research/bs-small-skeleton-20261118/receipts-l30/`). The pair battery on the board models was not run: no board
model reaches these kernels. The landing's sm_120a record runs the device battery on the landing commit.
