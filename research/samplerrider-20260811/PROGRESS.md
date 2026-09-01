# Sampler rider progress

Date: 2026-08-11
Branch: `lane/cx-samplerrider`
Rig: local RTX 5090, stock/released clocks

## Plan

1. Locate the decode-epilogue `argmax_token_device_col` per-row launch and record the exact row, dtype, and tie-break contract.
2. Replace only that loop with one batched CUDA launch spanning all batch rows, preserving byte-identical selected tokens and row-to-token mapping.
3. Run the frozen-model exactness gates: `decode-batch-gate` with zero differing bits, the b1fix one-hash matrix, `run-gen` argmax MATCH, and `run-spec` K=1..8 self-consistency PASS.
4. Capture baseline and candidate with N=5 interleaved A/B runs in one thermal window through `tools/perf-ab.sh`, under `nice`/`ionice`, recording clocks and any concurrent GPU activity.
5. Write `RESULTS.md` with a GO/NO-GO verdict, N, thermal regime, raw-log locations, and separate correctness and timing conclusions.

Part (b), overlapping detokenize/SSE emission with next-chunk issue, is explicitly excluded because the prior SOL study measured it flat.

## Status

- Implemented a two-pass batched argmax over contiguous decode rows. Each row retains the frozen
  256-block scan/reduction and smallest-token-id tie break; the launches now use `grid.y = row`.
- Added a separate lazy-grown batched-partials pool so CUDA-graph-captured scalar argmax pointers
  cannot be invalidated by a later B>1 allocation.
- Positive-temperature rows retain the existing Gumbel perturbation and scalar overwrite; the
  all-greedy scored path uses two launches total instead of two per row.
- Release build of `kernel-check`, `decode-batch-gate`, `run-gen`, and `run-spec` passed on sm_120a
  with CUDA 13.1. Receipt: `raw/build.log`.
- Local RTX 5090 Laptop GPU exactness is green on the pinned q9 NVFP4 GGUF:
  `decode-batch-gate` config B=1/2/4/8 and strict B=4 passed (greedy, sampled, and mixed-meta
  epilogues); `kernel-check` finished ALL GREEN (106 cells, 1 skipped); `run-gen` reported both
  argmax comparisons MATCH; and `run-spec` reported self-consistency PASS for K=1..8.
  Receipts: `raw/exactness/`.
- The requested `tools/perf-ab.sh` is absent from this branch, `origin/main`, all local worktrees,
  and local Git history. The closest current repository protocol is the alternating-order N=5
  live-server harness in `research/rowwalk-20260811/perf-box1.sh`; a lane-local 5090 adaptation
  will preserve its one-lock, raw-log, thermal-sampling, and baseline-binary controls.
