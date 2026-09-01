# cx-budgetsize progress

- Lane: `lane/cx-budgetsize`
- Base: `7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d` (`v0.81.3`)
- Scope: derive the default prefix-cache budget from loaded model geometry and boot-time free VRAM; expose budget and pinned admission skips; correct the serving docs; preserve explicit `MEMRA_PREFIX_CACHE_MB` behavior.
- Constraints: no insert/evict restructuring; no live serve box; no scored work on box1 GPU1; local serving cells take `/tmp/memra-5090.lock`; raw output is tee'd before parsing; no merge, tag, push, board edit, `cargo fmt`, or `--no-verify`.

## Status

- [x] Worktree/branch isolation and clean baseline verified.
- [x] Inspect current cache geometry, boot-memory plumbing, metrics, docs, and neighboring lane conflict.
- [x] Add derivation tests and refusal metrics/logging.
- [x] Reconcile provisional entry sizes with the committed `cx-cachesize` receipt.
- [x] Reproduce arm A and measure arm B plus c=64 admission/VRAM behavior on the locked local 5090.
- [x] Run the post-fix workspace cargo test suite.
- [x] Run the full GPU correctness battery (named gates green; extra clean-window postcheck dirty).
- [x] Record the final raw-log manifest and commit the lane.

## A/B arm identity

- **Arm A is deliberately metrics-instrumented, not pristine v0.81.3.** Its source is
  `093a214a9e1bc7170dd655bb417b0fd7fc6d13c8`: tag `v0.81.3` at
  `7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d` plus only the refusal-counter publication
  patch. The prefix-cache budget logic remains the v0.81.3 naked 256 MiB default. This lets
  arm A prove `prefix_cache_skips_budget == prefix_cache_misses`; the refusal-observability
  patch (counters/publication plus the one-time first-refusal warning) is intentionally common
  to both arms and is not part of the A/B mechanism comparison.
- **Arm B is the derived-budget candidate** at
  `13b4918ee5cc69b73bd045c036440d065303fd9a`, which corrects the live-found MTP/NextN
  geometry overcount. Its budget derivation is the measured mechanism under test; its refusal
  metrics are the same instrumentation as arm A.
- Both release binaries were rebuilt after the reclaimed-`/tmp` incident with a disk-backed
  `TMPDIR`, copied to immutable arm-specific paths, and their SHA-256 hashes recorded here and
  in `RESULTS.md` before any eligible serving cell for that binary runs.
- Arm A build receipt: source `093a214a9e1bc7170dd655bb417b0fd7fc6d13c8`, binary
  `target/bench-binaries/cx-budgetsize/arm-a-093a214a9-memra-server`, 52,775,640 bytes,
  SHA-256 `ec0c2fed4aa25fa904ab072fc2af53cee34dbee7c352d0eefb257c52f88a2a2f`.
  It was rebuilt successfully with `TMPDIR=target/lane-tmp/baseline` (an absolute disk-backed
  path at invocation) and an isolated `CARGO_TARGET_DIR`; raw compiler output is
  `raw/baseline-build/build-success.log`.
- Excluded initial-candidate receipt: source `772235e526b64f6ed2f02aa7cca853b9d858e299`, binary
  `target/bench-binaries/cx-budgetsize/arm-b-772235e52-memra-server`, 52,789,328 bytes,
  SHA-256 `8cf97fb0771caee87ac73b86186c7127ca91d3942c1dad3212f00d33d49e4840`.
  It was rebuilt successfully from a clean exact-commit detached worktree with disk-backed
  `TMPDIR=target/lane-tmp/candidate` (an absolute path at invocation) and an isolated
  `CARGO_TARGET_DIR`; raw compiler output is `raw/candidate-build.log`. Its one live cell is
  excluded because it found the MTP/NextN geometry overcount.
- Final arm B build receipt: source `13b4918ee5cc69b73bd045c036440d065303fd9a`, binary
  `target/bench-binaries/cx-budgetsize/arm-b-13b4918ee-memra-server`, 52,789,376 bytes,
  SHA-256 `29f1a64e8935bfc5b97ea1e9b6cf02e5fd4b562c05dd844b2b6566a53f9b77a8`.
  It was rebuilt successfully from the clean exact source commit with disk-backed
  `TMPDIR=target/lane-tmp/candidate-13b4918ee` (an absolute path at invocation) and an isolated
  `CARGO_TARGET_DIR`; raw compiler output is `raw/candidate-build-after-mtp-fix.log`.

## Evidence ledger

- 2026-08-13: initial branch was clean at the base commit above; no scored run had started.
- 2026-08-13: orchestrator checkpoint `11705075a` preserved the in-flight implementation,
  documentation, frozen protocol/replay harness, targeted receipts, and full CPU test log after
  the harness process-group sweep. Resume audit confirmed no local serving measurement exists yet.
- 2026-08-13: `cx-cachesize` checkpoint `9c5b628f4` commits its two-pass exact entry probes.
  The six `prefix_cache_bytes` deltas match this lane's geometry tests exactly: Q27
  278,528,000 / 301,215,744 / 400,162,816 B and Q35 103,874,560 / 110,964,480 /
  141,885,440 B at 4,096 / 4,860 / 8,192 tokens. Its capacity sweep is still incomplete and
  is not used by this lane.
- 2026-08-13: checkpoint CPU receipts pass: workspace `cargo test` (including memra-server
  223/223), 16 focused prefix tests, both operator-metrics authorization tests, and all six
  geometry points. No completed test or measurement was repeated on resume.
- 2026-08-13: resume audit corrected two incomplete checkpoint details before GPU work: the
  c=64 arm is now accepted by the local runner, and only the first permanent whole-budget
  refusal emits the one-time loud warning (temporary pinned pressure cannot consume it).
- 2026-08-13: orchestrator steering committed the formerly uncommitted baseline metrics patch as
  `093a214a9`. The protocol now explicitly chooses a deliberately metrics-instrumented arm A;
  prior failed builds are environmental non-verdicts (`Disk quota exceeded` / `LLVM ERROR: out
  of memory` on the old `/tmp` state) and must not supply a benchmark binary.
- 2026-08-13: arm A rebuilt cleanly from `093a214a9` in 3m25s after `/tmp` recovery, using a
  disk-backed `TMPDIR`; its frozen executable hash is recorded above. No serving cell had run at
  the time of the hash.
- 2026-08-13: arm B rebuilt cleanly from exact commit `772235e52` in 4m19s under the same
  disk-backed-TMPDIR policy; its frozen executable hash is retained above for the excluded
  diagnostic that found the overcount.
- 2026-08-13: rechecked the clean `cx-cachesize` worktree at `685d860c6`; its committed
  `research/cachesize-20260813/RESULTS.md` now carries the same six exact entry-byte receipts used
  here. The capacity sweep remains pending there, but it is unrelated to this lane's geometry.
- 2026-08-13: [CUDA 13.1's current vendor documentation](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-compiler-driver-nvcc/index.html#keeping-intermediate-phase-files)
  confirms that Linux `nvcc` uses `TMPDIR` for temporary intermediate files and otherwise falls
  back to `/tmp`. This matches the successful disk-backed rebuilds and the failure mode retained
  in the old raw logs.
- 2026-08-13: local arm A repetition 1 reproduced the defect: five cold requests, five misses,
  five `prefix_cache_skips_budget`, zero hits/inserts/evictions/defers/OOM parks, and identical
  greedy output hashes. The first-warning line appeared exactly once and all five ordinary skip
  lines remain visible. Raw clock samples were 210..1192 MHz (N=181).
- 2026-08-13: A1's first runner invocation exited after its PASS summary because the clock `awk`
  compared the post-`gsub` field lexically. The raw samples never escaped the cap. The checker now
  coerces with `$7 + 0`; `clock-validation-recovery.log`, an empty server-failure scan, and
  `recovery-audit.log` make A1 eligible without rerunning completed measurement.
- 2026-08-13: the first candidate diagnostic (`02-b-r1`, source `772235e52`) passed replay behavior
  but exposed a geometry overcount and is excluded from the final B arm. Boot derived 415,367,168 B
  per full-context entry while the retained snapshot receipt is 400,162,816 B. The 15,204,352 B
  difference is exactly 1,856 B/token x 8,192: `cache_bytes_per_token(cfg)` included Q27's one
  MTP/NextN block (`65 total - 1 nextn = 64 trunk`), while `HybridModel` and the prefix snapshot
  retain only trunk-layer state. The final derivation must use `[0, n_trunk)` and be rebuilt.
- 2026-08-13: the derivation now uses the exact trunk range for both full-attention KV and fixed
  recurrent state. Focused tests pass: both MTP/NextN exclusion and all six measured Q27/Q35
  geometry points (2/2), plus the two-entry/free-VRAM clamp test (1/1).
- 2026-08-13: final arm B rebuilt successfully from exact commit `13b4918ee` in 4m16s using a
  fresh isolated target and disk-backed `TMPDIR`. Its frozen hash is recorded above; the next live
  boot must report the exact 400,162,816 B entry and 800,325,632 B two-entry request.
- 2026-08-13: final arm B repetition 1 passed. Boot reported exactly 400,162,816 B per Q27
  `MEMRA_CTX=8192` entry and an 800,325,632 B two-entry request, below 9,957,277,696 B boot driver
  free. Replay produced one miss/insert then four hits (4/5), zero budget/pinned skips, evictions,
  admission defers, or OOM parks; hit TTFT median was 2.115 ms. Clock samples were 210..1192 MHz
  (N=92), and the greedy output hash matches arm A.
- 2026-08-13: arm A repetition 2 passed with the same defect signature (0 hits, 5 misses,
  5 budget skips, 0 inserts/evictions/defers/OOM parks), the same greedy output hash, and
  210..1192 MHz across 181 clock samples.
- 2026-08-13: final arm B repetition 2 passed with the same corrected boot geometry and
  4/5 hit signature, zero skips/evictions/defers/OOM parks, a 2.305 ms median hit TTFT, the same
  greedy output hash, and 210..1192 MHz across 92 clock samples.
- 2026-08-13: the post-MTP-fix workspace `cargo test` completed successfully, including
  memra-server 224/224, memra-engine 82 passed with the one hardware-only test ignored, and all
  crate/doc-test suites. The complete tee'd receipt is `raw/cargo-test-after-mtp-fix.log`.
- 2026-08-13: arm A repetition 3 passed with the same reproduced defect signature: 0/5 hits,
  five misses and five budget skips, zero inserts/evictions/defers/OOM parks, and the same greedy
  output hash. All 181 clock samples remained within 210..1200 MHz.
- 2026-08-13: two immediate post-checkpoint B3 launches correctly failed closed before creating
  output because neighboring lanes won the GPU lock. The runner now accepts the established
  `MEMRA_5090_LOCK_HELD=1` outer-lock contract, allowing bounded `flock` queueing without changing
  any measurement behavior; its post-lock no-compute-process preflight remains authoritative.
- 2026-08-13: final arm B repetition 3 passed with exact 400,162,816 B / 800,325,632 B boot
  geometry, one miss/insert then four hits, zero skips/evictions/defers/OOM parks, and the common
  greedy output hash. Hit TTFT median was 2.131 ms; all 96 samples were 210..1192 MHz. The
  alternating naked-default comparison is now complete at three eligible boots per arm.
- 2026-08-13: the explicit `MEMRA_PREFIX_CACHE_MB=4096` arm-A control passed with one
  miss/insert then four hits, zero skips/evictions/defers/OOM parks, and the frozen output hash.
  The paired arm-B explicit control is running under the same uninterrupted lock hold.
- 2026-08-13: the paired explicit arm-B control also passed. Request shape, cached-token sequence,
  output SHA-256, and all counter deltas are byte-identical across the two 4,096 MiB controls;
  both clock captures remained within 210..1192 MHz. The explicit compatibility criterion is met.
- 2026-08-13: the first derived-default c=64 Q27 full-shape stress cell is a retained **FAIL**, not
  an environmental exclusion: 64/64 requests completed and all were cache hits, with zero cache
  evictions, session-count defers, or OOM parks, but `admission_vram_defers=7072`. The 24,463 MiB
  card reached 23,448 MiB used / 536 MiB free; logs quote the gate admitting 14 active sessions,
  then queueing because effective free 752..660 MiB was below a 465 MiB session cost plus the
  465 MiB admission reserve. All 239 clock samples were 210..1192 MHz and the server-failure scan
  is empty. Every concurrent stream also differed from the cold seed (multiple hashes); that is
  recorded, not assigned to the budget mechanism without a matched control. Acceptance criterion
  5 is currently RED and the cell will not enter a passing reduction.
- 2026-08-13: a matched c=64 control is frozen next: arm A with today's explicit 4,096 MiB cache
  behavior, the same retained 301,215,744 B entry, and the identical barrier workload. This cannot
  turn the literal zero-deferral criterion green; it distinguishes a derived-default regression
  from the existing full-shape concurrency/admission class.
- 2026-08-13: the final-code named correctness battery passed: local-ci exit 0, kernel-check
  ALL GREEN (106/1 skipped), Q35 run-spec 8/8, correctness stage GREEN, serve-smoke 0 failed,
  standing short-request c=64 ALL GREEN, served-spec acceptance 1/0, and separate Q27/Q35 run-gen
  MATCH. The enclosing wrapper's stricter no-compute-process postcheck failed because
  `sxc-refresh-colbert.service` joined mid-battery with a 1,390 MiB CUDA process despite this lane's
  lock. `raw/gates/POSTCHECK.md` preserves the process/start-time evidence. These are exactness
  gates, so no timing conclusion is taken; the receipt is not described as a clean window.
- 2026-08-13: the matched explicit-4,096-MiB c=64 classifier was not run because the daily
  `sxc-refresh-colbert.service` rebuild continued to hold a 1,390 MiB CUDA process outside this
  lane's flock and may run to its two-hour cap. The literal lane verdict is already NO-GO from
  nonzero c=64 VRAM defers; the control could classify novelty but cannot change that criterion.
  Its fail-closed harness mode is committed for a future clean window rather than contaminating
  scored evidence.
- 2026-08-13: the fail-closed reducer emitted `raw/reduction/summary.json` and exited 1 exactly as
  designed for the c=64 acceptance failure. Independent JSON validation confirms overall FAIL,
  c=64 zero-deferral FAIL, and arm-B N=3 geometry/hit PASS; reducer stderr is empty.
- 2026-08-13: `raw/MANIFEST.sha256` seals 158 files; all entries verify and the manifest SHA-256
  is `91a648293d1bbc9895d2389251dd66ebb235c5ca645ad930f910f17bd338f73c`. The lane deliverable is
  complete at an evidence-backed NO-GO; no merge, tag, push, board edit, or live-box action follows.
