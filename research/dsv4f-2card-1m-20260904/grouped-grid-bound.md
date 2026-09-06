# Bound sparse MoE persistent grids, 2026-09-06

Implemented, default OFF, hardware qualification pending. This targets a measured
operation, not a power-limit hypothesis or a claim that tensor cores are unused.
The sampled decode profile attributes 39.2% of kernel sum to the existing
`moe_kq_sktail_kernel<108>`, which already uses tensor-core MMA.

## Source finding and change

The device-prefix path launches the full resident grid even when only a few
expert tiles exist. On the profiled PRO cards, the deep-tail grid is 188 SMs x
three resident blocks = 564 CTAs. Every CTA computes the prefix/schedule before
discovering whether it owns work. The DSV4 route preparation already reads a
validated live-row count; no additional device read is needed to bound the grid.

For S live rows and E possible expert groups, let G=min(S,E). A safe bound on
the number of 32-row tiles is `G + floor((S-G)/32)`. There can be at most G
nonempty groups, and each extra tile needs another 32 rows. Multiply by
`ceil(out_features/64)` for a launch bound. This also bounds the 128-row form.
For example, three owned rows give at most 96 gate/up CTAs or 192 down CTAs,
instead of 564. This is a launch-count example, not a speedup prediction.

`memra_moe_kq_gemm_sk_grid` adds an explicit cap. Zero delegates to the unchanged
generic ABI. Positive caps are admitted only for split-plane ModelOpt weights
with device-owned prefixes; unsupported formats, host-count metadata, negative
caps and invalid dimensions refuse. All existing generic callers remain unchanged.
No GEMM kernel body, unpacked value, scale or ascending MMA accumulation chain
changes. A positive cap does not truncate work: the persistent loop continues
with `t += gridDim.x` until it exhausts the device-defined tile count, including
when cap=1. This is consistent with the current primary-source explanation of
[persistent grouped scheduling](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html).
No CUTLASS kernel or new runtime dependency is introduced.

`MEMRA_DSV4_GROUPED_GRID=full|bounded` defaults to full. It is checked before
artifact/CUDA load and cached for process lifetime. The explicitly gate-only
all-rank override is changed only after draining both stages. Successful
positive-cap submissions are counted; graph replays do not increment that host
counter. This remains separate from removing route/mirror host synchronizations
and does not make the complete MoE graphable by itself.

## Verification and queued cells

- CPU tests cover strict flag parsing, integer-partition bounds, real projection
  shapes, zero dimensions and overflow. Full engine suite: 411 passed, 11 ignored;
  server suite: 604 passed. Strict library/bin clippy and formatting pass.
- The compiled component extends the existing partition/global-bank oracle with
  six cap controls (including 0, 1 and the derived bound), independent guards,
  invalid-cap/host-metadata refusals and live-routing GEMM graph replays.
  It has 62 cases: prior tail/empty/partition shapes plus actual 4096x2048 and
  2048x4096 projection dimensions at row counts 1/6/32. Graphs use cap=1 to force
  repeated visits and transitions through empty partitions.
- Component SHA256:
  `d84c3a891cbd489239d46bf8a9ea97f783872c8ee787847e5212c75c71b9878d`.
  The build retains the pre-existing shared-symbol linkage warning in its raw log;
  it has not been suppressed or treated as a sanitizer pass.
- Full-model candidate SHA256:
  `4e99bc01ca06721a452b0435f5691c781caea12639bb32500eff4c708bd19b4f`.
  Its `grid-ab` protocol fixes radix sampling, matrix/EP, active C4 and the tiled
  indexer/scorer, then compares full/bounded grids across plain/DSpark with four
  warmups and six interleaved observations per combination/context. Contexts are
  256/8192, 256 sampled outputs, the same seed and frozen output hashes. Positive
  submission counters must engage only on the bounded arm. The summary checks
  all 56 rows, schedule, hashes, loops, process ownership and telemetry.

The target controller waited for the real near-1M cell to release the canonical
pair lock, then ran component, memcheck and synccheck separately on both GPUs.
Only passing components permitted the full-model timing/identity comparison.
The long prompt's terminal verdict was recorded independently, never relabeled
by the subsequent optimization gate.

## Final target decision, 2026-09-06

The target full-model comparison completed with six eligible rows per
arm/context and output identity on all 56 rows. The cap engaged on every
bounded row (64,803 plain projection calls and 32,508 DSpark projection calls
per row), but means were flat/no-go:

| context | arm | full | bounded |
| ---: | --- | ---: | ---: |
| 256 | plain | 27.1001 | 27.1614 |
| 256 | DSpark | 25.4919 | 25.5216 |
| 8192 | plain | 24.8942 | 24.9887 |
| 8192 | DSpark | 33.0654 | 32.9748 |

The `MEMRA_DSV4_GROUPED_GRID` door, cap FFI entry, gate override/counter and
bound tests were removed. The generic full-grid visitor remains. This file is
the historical receipt, not a pending promotion plan; `full` is now the only
code path.
