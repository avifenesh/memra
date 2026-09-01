# Dual-active PP-2 decode increment 1 progress

Date: 2026-08-11

Lane: `lane/cx-dualpp1`

Hardened rebase base: `1592253f` (contains the required sec7/reusepool tip `2ddb9bd2`)

The increment-0 history was rebased onto that base after the orchestrator invalidated the original
pre-hardening fork. No increment-0 or pre-rebase increment-1 verdict is carried forward.

## Scope and invariants

- Keep dual-active decode default OFF and require both `MEMRA_DUAL_PP=1` and
  `MEMRA_PP_OVERLAP=1`.
- Move the arbitrary-live-width split into the worker scheduling decision. Each scheduled tick
  must form two balanced waves, with `ceil(c/2)` requests in wave A and `floor(c/2)` in wave B,
  while respecting the exact-tier cap per wave.
- Preserve mixed lane/priority membership, original request order, and one final publication to
  the caller stream. A width-one tick stays on the serial path.
- Prove the alternating PP boundary slots remain collision-free on both box1 devices under a
  sustained mixed-width detached soak.

## Required evidence

- [x] Worker scheduler policy tests cover arbitrary even/odd live widths, exact-tier caps,
  c=1 fallback, and mixed lane/priority ordering.
- [x] Final-rebase local CPU compile/test receipts are green for the scheduler and metrics-scope
  implementation.
- [x] Final-source box1 release rebuild completes before any GPU battery.
- [x] Final-source `kernel-check`, strict decode-batch, `run-gen`, `run-spec` K=1..8,
  split-arm bit identity, and one-hash serving matrix are all green.
- [x] Detached cross-device soak runs under one `/tmp/memra-gpu.lock` hold, records N and the
  thermal regime, and preserves raw stdout/stderr plus exit receipts.
- [x] Frozen N=5 interleaved performance block reports the arbitrary-width curve and verifies the
  >=+15% c>=8 floor without changing the generated performance board.

## Intake

- Read the lane inbox, `/home/avifenesh/projects/bw24/CLAUDE.md`, increment-0 `RESULTS.md`, and
  the dual-active `DESIGN.md` before code inspection.
- The worktree was clean on the requested isolated branch at intake.
- `lanectl` was not on PATH; the inbox was cleared through
  `/home/avifenesh/projects/lanectl/lanectl inbox cx-dualpp1`.
- A later inbox gate identified the old `3d485a22` merge base as pre-hardening. The detached box1
  process was stopped during B=11, its partial logs were retained under
  `raw/box1-pre-rebase-invalidated/`, and the branch was rebased onto `main@126e6642`.
- No merge, tag, push, formatting run, or performance-board change is authorized.

## Implementation checkpoint

- The per-model scheduler policy now keeps the model's existing batch cap as a per-wave cap.
  Only the three-way conjunction of dual requested, overlap requested, and a live PP-2 batched
  path combines two caps into one scheduled tick.
- Each combined chunk carries its explicit balanced midpoint into the engine. The engine rejects
  a non-ceil/floor midpoint and validates the exactness cap against the larger individual wave,
  allowing the Step checkpoint's live c=16 tick to execute as 8+8 instead of two independent
  B=8 dual calls.
- Request order remains the already-stable lane-priority order. The focused odd-width test pins
  concatenation across a mixed-membership 3+2 boundary, and c=1 carries no dual midpoint.
- `/metrics.dual_pp` now records completed wave pairs, use counts for boundary slots 0 and 1,
  and rejected same-slot pairs. The soak can require both slots, equal use, and zero collisions
  without enabling timing events in the scored process.
- The complete `dual_pp` metrics block is gated by `MetricsScope::operator()`. A populated-snapshot
  regression proves both completion-domain and tenant credentials see no wave, slot, timing, or
  topology data while the operator scope retains the block.
- The PP exactness gate has a serial two-wave oracle for widths beyond one exact cap, so the
  final box1 replay can compare direct B=9..16 dual ticks against the same cap-bounded numeric
  class with zero differing bits.
- Post-rebase local receipts in `raw/local/implementation/`: scheduler policy tests 2/2, engine
  dual policy/timing tests 4/4, server metrics isolation tests 8/8, and focused target checks
  green. No GPU workload ran locally and `cargo fmt` was not run.

## Box1 harness checkpoint

- The detached parent takes `/tmp/memra-gpu.lock` once, verifies the staged source, rebuilds the
  release server and every required gate binary, then retains the same lock descriptor through
  correctness, soak, and performance.
- The final-source model battery expands the split replay to direct B=1..16. Widths 9..16 use a
  sequential ceil/floor serial oracle so both sides stay in the Step checkpoint's exact B<=8
  numeric class.
- The collision soak alternates ten fresh serial/dual processes. Boots 1/2 cover the complete
  c=1..17 one-hash matrix; the remaining boots run rotated mixed widths for N=101 points per arm.
  Every point mixes interactive, judge, and harvest membership and must match the frozen
  `21b8293f...445bb6de` completion per request. Each dual boot must report equal slot-0/slot-1
  use, positive overlap, and zero collisions.
- The frozen curve is N=5 interleaved serial/dual at every c=2..17, 512 tokens/request, with one
  continuous 250 ms thermal trace. Per-point metrics require both slots once per completed wave
  pair and zero collision deltas. The reducer reports, but does not hide, any c>=8 median below
  the +15% floor.
- All four Bash drivers pass `bash -n` and `shellcheck`; both embedded Python reducers and the
  mixed-lane QoS probe compile without writing bytecode.

## Pre-sec7 box1 checkpoint — invalidated for final review

- Final implementation and scored source: `365e1eb71c6b635872447d4d1af1aeac4d7c087f`.
  It is based on `main@126e6642`, which contains the required `afb9be7b` hardening merge.
- A fresh release server rebuild completed in 3m52s, followed by the release gate-binary rebuild
  in 12.46s; both exit receipts are zero. The detached driver was PPID 1 from lock acquisition and
  retained one `/tmp/memra-gpu.lock` descriptor through build, correctness, soak, and performance.
- Final-source correctness is green: `kernel-check` 86 cells (21 skipped), direct PP-2 B=1..16
  split and unsplit identity, strict decode-batch, both `run-gen` argmax comparisons, and
  `run-spec` K=1..8 self-consistency. The direct split replay compared 140,238,848 f32 logits with
  zero differing bits; all 15 dual B=2..16 cells advanced overlap liveness.
- The ten-boot alternating collision soak completed 101 points and 929 golden-matched requests per
  arm. Across the five dual boots it recorded 9,104 completed slot pairs, 9,104 uses of each slot,
  and zero same-slot collisions. Its c1..17 serial/dual matrix was one-hash green in all 34 cells.
- The soak thermal trace contains N=13,990 samples at 250 ms, 28--51 C and 180--2422 MHz, with no
  artificial cooldown. CUDA timing instrumentation remained disabled in scored processes.
- The frozen performance block completed 160/160 points and 1,520 request rows without errors:
  N=5 interleaved serial/dual at c2..17, 512 tokens/request. The weakest c>=8 median was c8 at
  +19.812% against the +15% floor; c16 measured +31.530% and c17 +27.776%.
- The performance thermal trace contains N=39,804 samples at 250 ms, 30--49 C and 180--2422 MHz,
  with no artificial cooldown. Its 80 dual points accumulated 40,956 slot pairs, equal per-slot
  use, and zero collisions.
- That source's increment verdict was **HOLD**, but the verdict is not carried across the sec7
  rebase. Its receipts are preserved under `raw/box1-pre-sec7-invalidated/` and summarized in
  `PRE_SEC7_RESULTS.md` for provenance only.

## Superseding sec7 gate

- A later orchestrator inbox update required the sec7-fixed main tip before merge review because
  the intervening constrained-decoding fail-closed latch did not rearm. The branch rebased cleanly
  onto then-current local `main@1592253f`, which contains the named `2ddb9bd2` sec7/reusepool tip.
- The rebased dual implementation through operator metrics scoping is `c51aec29`. Main's two
  `AbandonedWorkerLimit` mappings remain present, as does the sec7 rearm implementation.
- Final-rebase local receipts under `raw/local/sec7-rebase/` are green: scheduler policy 2/2,
  engine dual eligibility/timing 4/4, server metrics isolation 8/8, and focused target checks.
- The prior final evidence commit was quarantined rather than deleted. No result from source
  `365e1eb7` will be used as the final-source verdict.
- The replacement run used a fresh detached checkout, rebuilt release before the battery, and took
  `/tmp/memra-gpu.lock` as its first logged action. It reran the complete exactness, soak, and
  frozen performance blocks, making the final HOLD decision source-local.
- Dual mode remains default OFF behind `MEMRA_DUAL_PP=1` plus `MEMRA_PP_OVERLAP=1`. No merge, tag,
  push, formatting run, or performance-board change is authorized.

## Final sec7-source checkpoint

- Final implementation and scored source: `a8f24074331647baf79189f190a402054d1af314`, based on
  hardened `1592253f` and containing `2ddb9bd2`. The focused sec7 rearm test passes, and both
  `AbandonedWorkerLimit` mappings remain present.
- The detached `setsid` driver acquired `/tmp/memra-gpu.lock` at 2026-08-11T14:08:23Z before any
  build or GPU work. The fresh release server build completed in 226.07 seconds; gate binaries
  rebuilt in 12.00 seconds. Both build receipts are zero.
- Final-source exactness is green: `kernel-check` 86 cells (21 skipped), direct PP-2 B=1..16 split
  and unsplit bit identity, B=2..16 dual liveness, strict decode-batch, both `run-gen` argmax
  comparisons, and `run-spec` K=1..8 self-consistency. Every `.exit` receipt is zero.
- The alternating ten-boot soak completed 101 points and 929 golden-matched requests per arm. Its
  serial/dual c1..17 one-hash matrix passed all 34 cells. Five dual boots recorded 9,114 slot pairs,
  equal use of slots 0 and 1, and zero same-slot collisions.
- Soak thermal regime: N=14,126 samples at 250 ms, 28--51 C and 180--2422 MHz, with no artificial
  cooldown. Scored CUDA timing instrumentation remained disabled.
- The N=5 interleaved performance block completed 160/160 points and 1,520 request rows without
  errors. Every c>=8 median cleared +15%; the minimum was c8 at +20.753%. Its 80 dual points
  recorded 40,956 balanced slot pairs and zero collisions.
- Performance thermal regime: N=39,788 samples at 250 ms, 30--49 C and 180--2422 MHz, with no
  artificial cooldown. The imported final tree contains 1,076 files after adding the detached
  launch receipt; the raw-log failure-pattern scan is empty.
- Final verdict: **HOLD**, not promotion. The operator-only metrics boundary is documented in
  `docs/FLAGS.md`; default activation and the separate admission-residual gate remain out of scope.
