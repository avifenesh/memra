# Multi-row FA decode entry progress

## Lane contract

- Branch/worktree: `lane/cx-fadecoderow` / `/home/avifenesh/projects/wt-cx-fadecoderow`
- Initial HEAD: `ba3e70c9af455320dc661ab023e5c653539bc447`
- Target: replace the Step3.7 B>1 per-`(layer,row)` FA decode launches with one launch per layer over all rows.
- Exactness invariant: preserve each row's KV view, physical rows, SWA window, mask, kernel math, and within-row accumulation order bit-for-bit.
- Stop condition: if batching requires a cross-row reduction or otherwise changes numeric order, record a NO-GO instead of landing the arm.
- Promotion boundary: this lane may commit implementation and raw 5090 evidence, but must not merge, tag, push, move generated boards, or run `cargo fmt`.

## Status

- 2026-08-12: lane initialized; steering and `research/rowwalk-20260811/RESULTS.md` read. No implementation change has begun.
- 2026-08-12: fetched `origin/main`; branch point confirmed at `ba3e70c9af455320dc661ab023e5c653539bc447`.
- 2026-08-12: implemented an FA-v3 multi-view row-grid entry for the Step3.5 B>1 serving walk. The entry carries adjusted per-row K/V bases, `t_kv`, and split rung; B=1 and non-v3/ring/repeated-cache paths retain the old dispatch.
- 2026-08-12: added a focused hd128 B=4 view-offset/length bit-identity cell. `cargo check --release -p memra-engine --bin kernel-check` passed under CUDA 13.1 / auto-detected sm_120a.
- 2026-08-12: candidate release binaries built at source `3845bda8358a6fe5883095250d3d8e6df84fda2a`. The focused B=4 row-grid cell reported `bitdiff=0`, and full `kernel-check` completed `ALL GREEN (102 cells, 1 skipped)`.
- 2026-08-12: both requested 9B artifacts passed config B=1/2/4/8 and strict B=4 with zero differing bits. Both also passed the two `run-gen` argmax comparisons and naked `run-spec` K=1..8 self-consistency (8/8 plus aggregate PASS).

## Required evidence

- [x] Confirm branch point against current `origin/main` and record baseline/candidate provenance.
- [x] NVFP4 and Q8_0 `decode-batch-gate`: config B=1/2/4/8 plus strict B=4, zero differing bits.
- [x] `kernel-check` ALL GREEN.
- [x] `run-gen` argmax MATCH.
- [x] `run-spec` K=1..8 self-consistency PASS.
- [x] Interleaved N=5 run-gen-style A/B in separate target directories, with distinct source/binary hashes and one recorded thermal window.
- [x] Verify B=1 does not regress: median `132.9 -> 133.0 tok/s` (`+0.075%`, below noise).

## Final timing verdict

- 2026-08-12: valid local window `raw/perf-nvfp4-rerun2/` completed with all warmups and ten
  timed arms exit 0. Median deltas: B=2 `+0.000%`, B=4 `+0.114%`, B=8 `+0.131%`; paired deltas
  cross zero at every width. The +2–6% expected gain did not appear.
- The local 32-layer 9B artifact did not emit the unconditional first-B>1 Step35 dispatch marker,
  so it is also a non-target timing proxy rather than proof of the changed Step3.7 walk.
- Final lane verdict: **NO-GO**. See `RESULTS.md`.
