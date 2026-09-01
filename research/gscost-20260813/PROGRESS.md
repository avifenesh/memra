# GraphSession/B1FAST demotion cost — progress

Date: 2026-08-13
Branch: `lane/cx-gscost`
Pinned base: `904a5d5f32a1b9170bc8628f2392cb0287572dbe`
Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition

## Scope and invariants

- Measure one binary in two runtime configurations only:
  - `REPAIRED`: default environment (`MEMRA_SERVE_B1FAST` and `MEMRA_SERVE_GS` unset).
  - `EAGER`: `MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1`.
- Record the binary SHA-256 and verify it is identical for both arms.
- Score Q27 and Q35 at concurrency 1, 4, 16, and 40.
- Use N >= 5 independent launches per cell, alternate arm order by repetition, and keep both arms in one uninterrupted `/tmp/memra-gpu.lock` hold.
- Keep GPU 1 idle during scored work because both devices share the PIX path. Reject any
  substantive activity signal and any sustained or frequent one-MiB idle signature.
- Tee raw output before parsing; report medians, min..max spreads, and paired deltas. Treat a delta inside the observed spread as flat.
- Do not merge, tag, push, or edit the generated performance board.

## Checkpoints

- [x] Confirm clean dedicated worktree and exact base commit.
- [x] Inspect the current serving/benchmark harness and existing evidence conventions.
- [x] Verify box1 access, model paths, both GPUs idle, shared lock free, ports clear, and thermal/clock regime.
- [x] Build/stage one binary and seal its SHA-256.
- [x] Run the interleaved Q27/Q35 concurrency ladder under one lock.
- [x] Run mixed cache-hit cells at c=4 and the model knees if the thermal/lock window remains valid.
- [x] Run the pre-release correctness battery in the same lock hold if practical.
- [x] Confirm tenant-clean shutdown: lock released, GPUs at 0 MiB, ports clear.
- [x] Parse raw evidence and write `RESULTS.md`.
- [x] Commit the complete lane and stop.

## Live log

- 2026-08-13: Worktree verified clean on `lane/cx-gscost` at `904a5d5f3`; `origin/main` was at the same commit at initial inspection.
- 2026-08-13: Box1 preflight passed: both RTX PRO 6000 Server Edition cards reported 0 MiB / 0% at 26 C, topology is PIX, `/tmp/memra-gpu.lock` was free, scored ports were clear, CUDA is 13.2, and both requested model plus draft artifacts are present on local NVMe.
- 2026-08-13: Pinned the live-server measurement surface. The primary table uses aggregate sustained decode after first visible tokens; end-to-end output rate is retained so GraphSession capture/TTFT cannot be hidden. The optional money-shape block imports the byte-frozen cx-requal 4,860+60 mixed90 harness but runs one endpoint on GPU0 so GPU1 remains idle.
- 2026-08-13: Q35's eager B=1 fusion is ineligible by current source contract, so its EAGER arm measures GraphSession at c=1; the driver requires a non-scored graph-census capture for both models before accepting any scored cell. Q27's EAGER arm enables both requested doors.
- 2026-08-13: Added a fresh-build script, single-endpoint mixed-cache adapter, deterministic reducer, and one-lock box1 driver. Scored points use fresh servers, N=5, cell-level arm alternation (odd REPAIRED-first, even EAGER-first), continuous 250 ms GPU telemetry, and a fail-closed GPU1 activity guard.
- 2026-08-13: Staged the complete lane as a verified Git bundle into the new `/opt/scratch/nvme/cx-gscost/memra` checkout and built source `71682fdbdf4a3616b3ab1794bf4b75607d35dcfd` fresh with CUDA 13.2 / auto-detected sm_120a. `memra-server` SHA-256 is `d314dfc211918523d93e14454a56a323dfa8544c974254a0e7e236e822848846`; both arms will execute that one file. Post-build preflight again found both GPUs at 0 MiB, the shared lock free, and port 18468 clear.
- 2026-08-13: Attempt 1 stopped before scoring because its deliberately cold, cache-disabled activation request produced no graph census. Source audit found why: phase (a0) tests GraphSession eligibility before phase (b) prefill, while phase (c) emits token 1 in the same tick that cold prefill completes; the next tick fails `generated.is_empty()`. The current post-prefill GraphSession door therefore promotes a fully restored prefix-cache hit, not a genuinely cold request. The aborted hold released cleanly with both GPUs at 0 MiB and the port clear.
- 2026-08-13: Corrected the primary ladder to the mechanism-isolating full-hit decode shape: each fresh server receives one unscored frozen 4,860+60 cold seed, one unscored width-matched 4,860+16 full-hit wave, then one scored 4,860+512 full-hit wave. This both makes GraphSession observable at EAGER c=1 and removes cold-prefill work from the decode comparison. End-to-end output rate remains alongside sustained decode rate so the per-request graph capture tax is retained. The separate frozen mixed90 cells remain the actual 90%-hit/10%-miss money shape.
- 2026-08-13: Attempt 2 proved GraphSession activation on both models, then stopped at the first Q27 EAGER c=16 cell because request 11 selected early EOS at token 11 on the frozen sellgate prompt. That is the defect under study, not a throughput sample; the driver rejected it rather than comparing unequal generated work. Cleanup again returned both GPUs to 0 MiB.
- 2026-08-13: The throughput ladder now uses a cacheable long-form chat prompt that explicitly requests more than 512 tokens. Each arm still seeds and fully restores the identical prompt, but the semantic task is designed to make `finish_reason=length` the invariant at all widths. The frozen 4,860+60 prompt remains untouched for the mixed90 money cells, whose short fixed budget is already qualified.
- 2026-08-13: Attempt 3 completed the entire valid campaign in one lock hold from 06:30:17Z through 07:25:46Z. Main: 80/80 points and 1,220/1,220 requests, all full 512-token completions. Mixed90: 40/40 points and 1,000/1,000 requests, all clean at exactly 0.9 cache-hit tokens. Arm order alternated by repetition and the reducer validates the exact point sequence.
- 2026-08-13: Result: request-visible c=1 repair cost is 12.183% on Q27 and 7.234% on Q35 (sustained decode 13.753% and 10.524%). The mixed sold shape is FLAT at c=4 and both knees; its four median costs range from a 0.205% repaired win to a 0.273% repaired loss, with every min..max interval overlapping.
- 2026-08-13: The same-hold pre-release battery passed: kernel-check ALL GREEN (100 cells, 5 skipped), run-gen MATCH on both models, and run-spec K=1..8 8/8 PASS on both models. All five commands exited 0.
- 2026-08-13: The first post-battery reduction rejected 15 isolated 1 MiB GPU1 readings. Re-audit found they are not a startup cluster: they span 06:31:01Z..07:21:34Z, but each is a one-sample P8/180 MHz/0% reading coincident with a same-time one-MiB GPU0 reading. The reducer now tolerates only that exact signature, for at most two seconds continuously and 0.5% of the campaign; any larger allocation, utilization, non-P8 state, higher clock, sustained run, or frequent recurrence fails with timestamped evidence. Final reduction reports zero GPU1 active samples across 13,075 observations and a one-sample maximum bookkeeping run.
- 2026-08-13: Pure CPU re-reduction re-read 2,680 main rows, 3,621 mixed rows, and all 26,150 thermal rows. Attempt 3 remains valid. The exactness result is promoted to the headline: gscost hit Q27 EAGER early EOS at token 11 within eleven requests; the later pre-repair cachesize token-11 match made the class seven independent triggers. The PRO-only recommendation remains REPAIRED, with any future cross-rig promotion gated by `Engine::sm_count()` rather than a bare env default.
- 2026-08-13: Tenant cleanup verified live at 07:28:35Z: both GPUs P8/0 MiB/0%, no compute applications, `/tmp/memra-gpu.lock` free, and port 18468 clear. Raw evidence, deterministic summary, manifest, and RESULTS.md are committed; no board edit, merge, tag, push, or release was performed.
