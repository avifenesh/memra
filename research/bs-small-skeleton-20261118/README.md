# The shared small-batch skeleton: device code before and after (2026-11-18)

Decision record: [BS-SMALL-SKELETON](../../docs/decisions/BS-SMALL-SKELETON.md). The change moves the V4.1
route's tile list and device MoE plan, both cp.async rings of the grouped small-row body, the dense small-row
body and the CUTLASS configuration into `cu/bs_small.cuh` and `cu/bs_cutlass.cuh`, with the route as the first
caller. No GPU was used. The evidence is the compiled device code.

## Method

- Arms: base `801243f06e` (memra#1337's head, the branch this change stacks on) and the change's tree. Since the
  rebase of 2026-11-19 the base is `17c6b4fd75`, memra#1337's head rebased onto main (section below).
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

After the review of #1343 moved the speed rules (ring choice, column rules, dense geometry) from the header into
the route, the tree was rebuilt and compared again on sm_120a and sm_121a: 104 of 104 and 9 of 9 on each arch,
with every kernel's digest equal to the earlier head's (`sass-cmp-review-120a-121a.txt`). The move is host code only.

`sass-cmp-120a-121a-round1.txt` is the first comparison of the extraction, before its expression forms were
made exact: 18 of 104 kernels of `dsv41_gpu.cu` differed on each arch (all dense small-row instances but three,
every grouped-body instance), while every CUTLASS kernel and every plan kernel was equal. The comparison catches
a changed kernel; the decision record lists the three causes found and fixed.

## Rebase onto memra#1337's rebased head (2026-11-19)

memra#1337 was rebased onto main, and this change with it: base `17c6b4fd75`. Main had gained memra#1315, the
row map (a router id that names no expert marks its row in the call's bitmap `rows_bad`, at bit `row0 + t`,
clamped to the map's last bit, so only the requests whose rows faulted fail), and memra#1327 (the split MoE's
gather rule, host code). In the device code memra#1315 changed only the plan: `v41_row_bad` and the
`rows_bad`, `row0`, `rows_cap` arguments of the count kernel and the one-block plan kernel, and their check in
`memra_v41_moe_plan`. The rebase carries them into the skeleton as main has them: `bs_row_bad` (the same body),
the same three arguments on `bs_moe_count_kernel<CH>` and `bs_moe_plan_small_kernel<NP, NE>` and on
`bs::moe_plan_launch`, the same call sites (the count kernel at its pair's token, the one-block plan at pair
`tid`'s row `tid / topk`), and the argument check stays in the route's wrapper. The scan and place kernels take
no row map, as on main. The host bindings, the tensor-parallel caller and the gate are main's, unchanged by this
change. The gate's plan checks already cover the row map on both forms: every plan checks that a clean selection
marks no row, and `schedule/moe_plan/out_of_range_id` (3 rows, the one-block form, slice at row 37 of its call)
and `schedule/moe_plan/out_of_range_id_rows700` (the three kernels) check exactly the faulted rows marked.

The comparison, rebuilt with `nv_build.sh` and compared with `sass_cmp.py` on sm_120a and sm_121a:

| pair | dsv41_gpu.cu (each arch) | dsv41_cutlass.cu (each arch) |
|---|---|---|
| new base `17c6b4fd75` vs the rebased head | 104 of 104 equal (79 unmoved, 25 moved) | 9 of 9 equal |
| old base `801243f06e` vs new base `17c6b4fd75` | 102 of 104; differ: `v41_moe_count_kernel`, `v41_moe_plan_small_kernel` | 9 of 9 |
| old head `e658b44cd8` vs the rebased head | 102 of 104; differ: the same two kernels | 9 of 9 |

`rowmap_chain.py` chains the three runs' digests kernel for kernel: the old pair's bodies and resource lines are
equal, the new pair's are equal, and the two kernels that changed between the bases changed between the heads
to the same bodies (count kernel 22 to 26 registers, both kernels 16 more bytes of parameters: the three
arguments). So the rebase adds exactly memra#1315's row map to the skeleton and nothing else. Reports:
`sass-cmp-rebase-120a-121a.txt`, `sass-cmp-rebase-bases-120a-121a.txt`, `sass-cmp-rebase-heads-120a-121a.txt`,
`rowmap-chain-120a-121a.txt`; digests `digests-rebase-120a-121a.tsv`; objects `objects-rebase-120a-121a.sha256`;
build exit codes `build-rc-rebase.txt`.

The battery's kernel-gate and tensor-parallel counts came from memra#1337's receipts, which predate main's
added checks, so `gpu-battery.sh` now takes each count from the base arm (or an `EXPECT_*` pin) and holds the
receipts' figures as a floor; the dry run gained three cases for it.

## Not shown here

Nothing was run on a card. `gpu-battery.sh` is the GPU battery prepared for the merge, with the expected counts
(the kernel gate, the four-rank tensor-parallel gates, the full pair battery), to run when an RTX PRO 6000 box is
free.

`gpu-battery-dryrun.sh CASE` runs the battery against fake binaries, a fake `nvidia-smi`, `cargo` and
`release-battery.sh` (no GPU): the passing pair, kgate and tp4 phases, and one case for each failure path (a
missing, partial or newline-less BOARD, an absent model, a lock that cannot be opened, a missing binary, a run-gen
MISMATCH, a FLIP-NEARTIE that passes, a failing release battery, kernel-gate lines that differ between the arms,
kernel-gate counts that differ between the arms, a count below the floor, a pinned count missed, a relative
OUT). Results: `gpu-battery-dryrun.txt`.
