# The shared small-batch skeleton: device code before and after (2026-11-18)

Decision record: [BS-SMALL-SKELETON](../../docs/decisions/BS-SMALL-SKELETON.md). The change moves the V4.1
route's tile list and device MoE plan, both cp.async rings of the grouped small-row body, the dense small-row
body and the CUTLASS configuration into `cu/bs_small.cuh` and `cu/bs_cutlass.cuh`, with the route as the first
caller. No GPU was used. The evidence is the compiled device code.

## Method

- Arms: base `801243f06e` (memra#1337's head, the branch this change stacks on) and the change's tree.
- `nv_build.sh TAG`: `cu/dsv41_gpu.cu` and `cu/dsv41_cutlass.cu` with build.rs's flags (nvcc 13.1; `-O3
  -std=c++17 --expt-relaxed-constexpr`; `-fmad=false` for the first; `-DNDEBUG` and the vendored CUTLASS include
  dirs for the second), each for `sm_120a` and `sm_121a`, on a CPU build host. `arch_build.sh`: `cu/dsv41_gpu.cu`
  for `sm_89`, `sm_90a` and `sm_100a`, both arms (the CUTLASS unit is its stub there).
- `dump.sh TAG`: `cuobjdump -sass` and `-res-usage` of each object (cuobjdump 13.1).
- `sass_cmp.py BASE NEW`: splits each dump at `Function :`, demangles, masks nvcc's directory-derived
  internal-linkage prefixes (the kernel-object method of
  `research/modelplan-onboarding-dsv41-20261109/perf/receipts/cfix-pro6000/kcmp/`), maps each moved kernel's new
  name to its old one (table `RENAME`), and compares every kernel's SASS body (every instruction with its
  encoding) and its resource line (registers, stack, shared, constant bank) byte for byte. `--digests` writes the
  sha256 of each normalized body.
- While the extraction was being made exact: `quick.sh` (one unit, sm_120a, object and PTX), `ptxdiff.py` (one
  kernel's PTX with virtual registers masked) and `kdiff.py` (one kernel's SASS diff) located each difference.

## Result

| unit | arch | kernels | equal | unmoved | moved |
|---|---|---|---|---|---|
| `dsv41_gpu.cu` | sm_120a | 104 | 104 | 79 | 25 |
| `dsv41_gpu.cu` | sm_121a | 104 | 104 | 79 | 25 |
| `dsv41_cutlass.cu` | sm_120a | 9 | 9 | 8 | 1 |
| `dsv41_cutlass.cu` | sm_121a | 9 | 9 | 8 | 1 |
| `dsv41_gpu.cu` | sm_89 | 104 | 104 | 79 | 25 |
| `dsv41_gpu.cu` | sm_90a | 104 | 104 | 79 | 25 |
| `dsv41_gpu.cu` | sm_100a | 104 | 104 | 79 | 25 |

The 25 moved kernels of `dsv41_gpu.cu`: 16 dense small-row instances, 3 warp-ring and 2 block-ring instances, the
4 plan kernels. The moved CUTLASS kernel is the grouped call's setup kernel; the 8 CUTLASS GEMM kernels keep
their names (their types are CUTLASS's own). Reports: `sass-cmp-120a-121a.txt`, `sass-cmp-89-90a-100a.txt`;
per-kernel digests: `digests-120a-121a.tsv`, `digests-89-90a-100a.tsv`; object hashes:
`objects-120a-121a.sha256`, `objects-89-90a-100a.sha256` (the objects differ as files: symbol names); build exit
codes: `build-rc.txt`.

The objects build.rs itself compiled (the workspace clippy build of the committed tree, `MEMRA_CUDA_ARCH=120a`,
taken from memra-engine's build out dir) give the same result against the base: 104 of 104 and 9 of 9
(`sass-cmp-cargo-120a.txt`, `digests-cargo-120a.tsv`, `objects-cargo-120a.sha256`), so the script's flags are
build.rs's. The committed tree rebuilt with `nv_build.sh` after the last edits (comments and two unused
`using` declarations in `dsv41_cutlass.cu`) gives the same per-kernel digests: `sass-cmp-final-tree-120a-121a.txt`.

`sass-cmp-120a-121a-round1.txt` is the first comparison of the extraction, before its expression forms were
made exact: 18 of 104 kernels of `dsv41_gpu.cu` differed on each arch (all dense small-row instances but three,
every grouped-body instance), while every CUTLASS kernel and every plan kernel was equal. The comparison catches
a changed kernel; the decision record lists the three causes found and fixed.

## Not shown here

Nothing was run on a card. `gpu-battery.sh` is the GPU battery prepared for the merge, with the expected counts
(the kernel gate, the four-rank tensor-parallel gates, the full pair battery), to run when an RTX PRO 6000 box is
free.
