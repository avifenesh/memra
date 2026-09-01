# Batched-decode row-walk progress

## 2026-08-11 — lane opened

- Branch/worktree: `lane/cx-rowwalk` in `wt-cx-rowwalk`, base `34d6330e`.
- Development gate: local RTX 5090; every GPU block must hold `/tmp/memra-gpu.lock`.
- Conditional performance rig: box1 only after local gates are green and the remote lock is free.
- Target: the Step3.5/Step3.7 `B>1` attention state walk in `decode_batch.rs`, where each
  `(layer,row)` currently appends KV, materializes one Q row, launches FA decode, and copies one
  attention row back.
- Starting receipt: `research/eagerpar-20260810/RESULTS.md` removed the arithmetic-free Q/A row
  copies only for `B=1` and measured +4.381% at c=1, +5.496% at c=2, and +3.718% at c=4 while
  preserving the batched numeric class.
- Candidate contract: reduce the `B=2..8` per-row issue walk without changing KV append order,
  SWA `physical_rows`, attention reduction/fold order, or any surrounding FP operation order.
- Kill gate: split-vs-unsplit decode logits must remain bit-identical for B=1/2/4/8; any changed
  golden output or unsupported mixed-layout dispatch rejects the arm.
- Required closeout: local compile and strict decode-batch battery, one-hash golden receipt,
  interleaved x5 c=2..8 A/B if box1 becomes available, MoESD `T_T(B,1)` versus `T_T(B,gamma)`
  instrumentation, raw logs, and a final verdict in `RESULTS.md`.
- Constraints acknowledged: no `cargo fmt`; no GPU command outside the relevant flock; no push,
  merge, tag, or release; preserve unrelated work; commit at least every 15 minutes.

Next: map the exact row-walk allocations, kernel ABI, SWA/KV metadata, and existing strict-gate
harness before choosing the smallest exact candidate.

## 2026-08-11 — packed-row view candidate compiles

- The existing multi-row `seqs_v4` arm is not a direct Step3.7 answer: it is stamped for hd256
  v4 attention and assumes each cache starts at logical row zero, while Step3.7 uses hd128 and
  its SWA layers expose rebased contiguous ranges through `physical_rows()`.
- Chosen candidate: retain the per-session append and FA main/combine launches, but pass Q and
  attention row views from the existing packed `[B, ...]` buffers. The kernel still receives a
  row-base pointer, so its geometry, split selection, KV view, and FP program are unchanged.
- The same view entry now serves both the generic unpackable-KV fallback and the Step35 walk.
  Removed work is limited to two D2D row copies and two temporary row allocations per
  `(layer,row)`.
- `cargo check -p memra-engine --lib`: PASS with CUDA 13.1, auto-detected sm_120a. No formatting
  command was run.

Next: add a focused copied-row versus packed-row bit-identity cell, build the gate binaries, then
run the local locked kernel and strict multi-session batteries before any performance decision.

## 2026-08-11 — focused exactness cell added

- `kernel-check` now compares the former copied-row program against packed Q/O row views at the
  actual Step3.7 attention geometry (hd128, 64 query heads, 8 KV heads), B=4, depth 257.
- The cell drives both arms through the same quantized KV cache and requires zero differing output
  bits; it is an exact pointer/view contract check, not a tolerance test.
- `cargo check -p memra-engine --bin kernel-check`: PASS. No GPU result exists yet.

Next: build release gate binaries, then acquire the local GPU lock for the focused kernel gate and
a strict multi-session model battery.

## 2026-08-11 — first locked kernel run green, wrapper receipt incomplete

- Release `kernel-check` ran to completion under the local GPU lock and printed `ALL GREEN`.
- The new hd128/B=4 cell reported `fa_decode packed-row views vs copied rows ... bitdiff=0 OK`.
- This is not the final core receipt: the outer zsh rejected Bash's uppercase `PIPESTATUS` after
  the test completed, so the wrapper failed to record the numeric command exit and post-run GPU
  census. The full raw kernel stream is retained under `raw/local-kernel-check/` as diagnostic
  evidence and the wrapper will be rerun explicitly under Bash.
- Concurrent-state caveat: the start census recorded a ColBERT Python process using 1,390 MiB.
  That does not affect the bit-identity finding, but this window is not eligible for performance
  evidence.

Next: rerun the unchanged release binary with corrected Bash bookkeeping, then move to the strict
model gate only after a complete exit/post-state receipt exists.

## 2026-08-11 — local kernel gate complete

- Corrected Bash wrapper rerun: `kernel-check` **ALL GREEN**, numeric exit `0`.
- Focused contract: copied rows versus packed row views at hd128/B=4 had `bitdiff=0`.
- The receipt includes pre/post GPU state and compute-app censuses under one lock hold. The same
  ColBERT process occupied 1,390 MiB throughout; this remains correctness-only evidence.
- Release binary SHA-256: `c49da014596c01a8001e890a47b5d537978b7148044aa556ca1cf9a02744ea53`.

Next: run `decode-batch-gate` on the local 27B model with the generic multi-row FA arm disabled so
the changed unpackable-KV fallback is non-vacuously exercised at B=1/2/4/8 in strict mode.

## 2026-08-11 — local strict multi-session battery green

- Local Qwen3.6 27B NVFP4, strict equalized composition, `MEMRA_BATCH_FA=0`, 24 steps each:
  B=1/2/4/8 all **ALL GREEN** with numeric exit `0` per arm and aggregate exit `0`.
- Gate 1 kept B=1 logits bit-identical to `decode_step_h`; gates 2/3 passed isolated-stream and
  device-sampling identity at every requested width.
- Non-vacuity: disabling the generic seqs FA arm forces B>1 full-attention rows through the changed
  unpackable-KV fallback; no alternate B>1 FA call site remains in that branch.
- The pre/post census again records the resident 1,390 MiB ColBERT process. The battery is an
  exactness receipt only; no timing is used.

Next: complete the repo-law local `run-gen` argmax and `run-spec` K=1..8 gates, then inspect box1
identity and lock availability without starting work unless all local gates remain green.

## 2026-08-11 — local generation/spec gates green

- Qwen3.6 27B `run-gen`: prefill/decode argmax **MATCH** and batched-prime/tokenwise argmax
  **MATCH**; numeric exit `0`.
- Same trunk plus the local trimmed MTP draft, `run-spec` K=1..8: eight self-consistency passes and
  aggregate `SELF-CONSISTENCY PASS`; numeric exit `0`.
- Together with the kernel and strict B=1/2/4/8 receipts, the local 5090 development gate is green.
  The persistent ColBERT allocation is recorded and these cells remain correctness-only.

Next: preflight box1 identity, current lock owner, source/artifact availability, and unrelated GPU
work. Start no build or benchmark unless the remote lock is free and the host matches the lane.

## 2026-08-11 — box1 correctly identified, GPU lock busy

- Corrected remote preflight reached `<private-host-redacted>` and found the pinned Step3.7 trunk and MTP
  artifacts at the expected byte sizes.
- `/tmp/memra-gpu.lock` is **busy**. Remote PIDs 560166/560421/560422/560486 hold it, including a
  live `run-gen`; no build, model load, or GPU command was started by this lane.
- The first preflight file is retained but explicitly invalid for lock/GPU state: a quoting error
  caused that portion to execute locally. `box1-preflight-rerun/` is the authoritative receipt and
  exited `0`.

Next: prepare the exact PP-2 strict/golden/perf harness locally and audit the MoESD rider while the
box is occupied. Re-probe atomically before any remote action; if it remains busy at closeout, name
the PP-2 A/B as the explicit next cell rather than substituting local timing.

## 2026-08-11 — box1 remains occupied; MoESD rider boundary pinned

- Atomic re-probe at `2026-08-11T03:21:19Z`: box1 is still **busy**, now under the
  `cx-sigrouter` lane. Its `memra-server` PID 561519 holds 12,810 MiB on GPU 0 and 1,036 MiB on
  GPU 1; this lane again started no build, model load, or GPU command.
- The rowwalk c=2/4/8 live cell can measure ordinary batched decode (`gamma=1`). It cannot
  truthfully supply `T_T(B,gamma)` for B=4/8: the production speculative scheduler exposes only
  a reduced two-warm-session PP-2 pair path, and no `moesd-gate` standalone B-by-gamma harness
  exists in this source tree.
- The separate `moesd-harness-20260811` design explicitly requires that standalone verify bin,
  including per-layer expert-union telemetry, and prohibits substituting live serving traffic.
  Therefore ordinary c=4/8 throughput will not be mislabeled as batched-verify target time.
- Raw receipts: `raw/box1-reprobe-1/` for remote ownership/state and
  `raw/moesd-rider-audit/` for the exact scheduler/API/design contracts.

Next: finish a pinned, syntax-checked box1 correctness/golden/perf driver for the next free slot,
then re-probe once more. If the lock remains occupied, close this lane with a correctness-green,
performance-unmeasured verdict and name both the PP-2 rowwalk A/B and B-by-gamma MoESD harness as
separate next cells.

## 2026-08-11 — deferred box1 cell is executable and pinned

- Added `gates-box1.sh`: separate bounded `correctness` and `golden` lock blocks. It pins the
  staged source and every release-binary hash, runs kernel-check plus the PP-2 B=1/2/4/8 exactness
  battery and repo-law generation/spec gates, then produces a source-and-binary-bound one-hash
  golden receipt in the second block.
- Added `perf-box1.sh`: it refuses to start without that exact candidate golden receipt, pins both
  candidate and eagerpar baseline sources/binaries, alternates arm order for five fresh-process
  rounds, records c=2/4/8 live points, continuous 500 ms thermals, request rows, non-vacuous
  Step35 B>1 dispatch logs, and a deterministic summary.
- The performance summary schema records the MoESD B-by-gamma fields as unavailable with the
  exact standalone-harness reason. It does not coerce end-to-end plain decode into a target-forward
  timing.
- Both drivers pass `bash -n`, `shellcheck`, and `git diff --check`; neither has been executed on a
  GPU because box1 remains owned by the other lane.

Next: atomically re-probe box1. If free, stage this exact commit, isolated-build the five release
binaries, run `gates-box1.sh correctness`, then `golden`, then `perf-box1.sh`. If busy, write the
final HOLD/READY verdict with this sequence as the explicit next cell.

## 2026-08-11 — final closeout branch selected

- Final clean remote preflight at `2026-08-11T03:29:13Z`: box1 is still **busy**. The active
  `memra-opti2` server holds the GPUs and the `sigrouter` driver is already queued on the same lock.
- The lane brief's busy-box branch now applies. This lane did not join the queue, touch the active
  server, stage source, build remotely, or load the Step3.7 model.
- Final verdict written to `RESULTS.md`: **HOLD — correctness-green and ready for box1; performance
  unmeasured.** No promotion/kill claim, perf-board move, merge, tag, or release is authorized.
- The PP-2 exactness/golden/interleaved c=2/4/8 A/B and the standalone MoESD B-by-gamma matrix are
  named as separate next cells. Ordinary live throughput is not used as a target-efficiency proxy.

Final local release rebuild was cache-clean and exited `0` under CUDA 13.1/sm_120a; all four
binary hashes remained unchanged. The 50-file raw manifest is complete. Next: verify the full
diff/worktree and commit the final lane evidence. No further GPU work is needed in this closeout.

## 2026-08-11 — continuation: box1 acquired and candidate staged

- Continuation brief received after `cx-opti2` released box1. An atomic nonblocking acquisition at
  `2026-08-11T04:39:27Z` reconfirmed host `<private-host-redacted>`, both RTX PRO 6000 GPUs idle at P8,
  26 C, 0 MiB used, and no compute applications.
- Staged the exact committed candidate `fc3a00c939f5deabad2266fbf41609160863ef6f` into the new,
  isolated `~/memra-cx-rowwalk` checkout via an incremental verified git bundle. The pre-existing
  eagerpar baseline remains untouched at source `711fbcaaef54491d22488a84d40b7fc35e5a58dd` with
  server SHA-256 `43ad098d46bb26d644ba0b742d92f3f014d9287ac72e8a0edb8ebf9dac3ba608`.
- No formatting command ran. No origin push, merge, tag, release, or perf-board edit occurred.

Next: build the five pinned release binaries in the isolated candidate checkout, record all
hashes, then run the committed correctness, golden, and interleaved N=5 decision drivers.

## 2026-08-11 — box1 release build pinned

- The first wrapper attempt selected `memra-server` from the `memra-engine` package and therefore
  exited before compilation with Cargo's quoted `no bin target named memra-server` error. That
  failed receipt is retained; it supplied no binary or GPU evidence.
- The corrected fail-closed wrapper built `memra-server` from its own package and the four engine
  gate binaries under one lock hold. Source remained exactly
  `fc3a00c939f5deabad2266fbf41609160863ef6f`; CUDA 13.2 and sm_120a were auto-detected; the clean
  release build completed at `2026-08-11T04:46:20Z`.
- Candidate binary SHA-256 pins: server
  `59037f8b46e723b7e4509ddd1c2d3496ba0385b6fdcb75b6036b2fbee09cb964`, kernel-check
  `7ec9d06f7d92ecec3e1066b1055c304bb46b552129e12d1fb9457e3f62bd19fb`, decode-batch-gate
  `53ae8931bfb21a988d59dab70d71c802426ae1d5882da9963780a1a8eeb83da7`, run-gen
  `7225b37f95fc8785fba7649079cc6fe3aab9a339d2eec30afcac862984bc8413`, and run-spec
  `466601c6d0e142774ed4c72026418f85d68cdf86b4c4e193648ee08d19cc1051`.

Next: run `gates-box1.sh correctness` with the recorded source and four gate-binary hashes. Only a
fully green PP-2 battery advances to the fresh-boot one-hash golden receipt.

## 2026-08-11 — box1 PP-2 exactness green

- `kernel-check` completed **ALL GREEN** and the changed hd128/B=4 copied-row versus packed-row
  contract remained bit-identical (`bitdiff=0`).
- The pinned Step3.7 PP-2 split/unsplit battery passed B=1/2/4/8, two split reps per width, with
  zero differing logit bits, zero failing arms, and the B=8 epilogue check green.
- `run-gen` reported prefill/decode and batched-prime/tokenwise argmax **MATCH**. `run-spec` passed
  self-consistency for every K=1..8. Every command exit was `0`; the one-lock correctness block
  released at `2026-08-11T04:52:38Z` with `result=PASS`.
- Source, artifacts, and all four binary hashes matched the preflight pins. The raw receipt includes
  before/after GPU state, each gate stream, numeric exits, and the aggregate driver log.

Next: run a separate fresh-boot server block against the pinned one-hash golden, retain its
source/binary-bound receipt, then permit the performance driver to consume only that receipt.

## 2026-08-11 — candidate one-hash golden green

- A fresh candidate server boot at source
  `fc3a00c939f5deabad2266fbf41609160863ef6f` and server SHA-256
  `59037f8b46e723b7e4509ddd1c2d3496ba0385b6fdcb75b6036b2fbee09cb964` produced the locked
  golden SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
- Receipt: exactness `match`, 1/1 request successful, one golden match, zero divergences, zero
  errors. The block released its lock at `2026-08-11T04:54:11Z` with `result=PASS` and no fatal
  server signatures.
- `golden-receipt.json` binds the source, candidate binary, expected golden, and full request
  summary. This is the exact prerequisite consumed by the performance preflight.

Next: run the committed interleaved N=5 current/candidate driver under one lock hold, alternating
order over c=2/4/8 with fresh servers, complete request rows, and 500 ms thermal sampling.

## 2026-08-11 — final box1 decision: PROMOTE

- The one-lock interleaved run completed all 10 fresh server arms and deterministically accepted
  40 points / 150 request rows with zero errors, shedding, or short completions.
- N=5 medians, current -> candidate: c=2 `119.197 -> 120.557 tok/s` (**+1.142%**), c=4
  `144.673 -> 146.696` (**+1.398%**), c=8 `162.001 -> 164.235` (**+1.379%**). Every paired
  comparison was positive at every width (15/15).
- Thermal regime: 1,010 samples at 500 ms, 31–45 C, no artificial cooldown. Each server log has
  the live Step3.5 B>1 dispatch marker and the fatal-signature scan is clean.
- Per the lane law, exactness-green plus a positive result changes the final verdict from HOLD to
  **PROMOTE**. `RESULTS.md` now reports only measured gains and keeps MoESD B-by-gamma timing
  explicitly separate and unmeasured.

## 2026-08-11 — lane complete

- PROMOTE closeout and all imported box1 evidence landed in `ce0d47b1`.
- The final 123-file raw manifest verifies in full. `RESULTS.md` contains no remaining HOLD/KILL verdict,
  and the worktree was clean after the decision commit.
- Final box1 readback at `2026-08-11T05:07:51Z`: lock freely acquired, both GPUs P8 at 27 C with
  0 MiB used, and no compute applications. The lane left no server or queued lock owner behind.
- No merge, push, tag, release, generated-board update, or formatting command was performed.

Lane complete; the promoted implementation is ready for the owner/integration lane.
