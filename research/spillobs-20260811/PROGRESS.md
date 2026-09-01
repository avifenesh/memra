# Spill config observability progress

Date: 2026-08-11
Branch: `lane/cx-spillobs`
Base: `96afb32e`

## Scope

- Surface invalid spill configuration through `MEMRA_SPILL_STATS` with a dedicated
  config-fallback counter.
- Preserve the existing warning-and-default behavior: invalid `MEMRA_SPILL_IO` still boots the
  mmap backend, and invalid pread depth still uses its current default.
- Keep runtime backend/read failure accounting distinct from configuration-parse fallback
  accounting in code and `docs/FLAGS.md`.
- Add a focused unit test for invalid `MEMRA_SPILL_IO` counter increment plus mmap boot.
- Run the tiny local RTX 5090 `run-gen` argmax gate and record the spill-model smoke as
  `N=1`, `window_clean`; no performance window is required because dispatch behavior is unchanged.

## Constraints

- No fail-closed behavior on the main `MEMRA_*` spill path.
- No merge, tag, push, formatting sweep, or performance-board change.
- Preserve unrelated work and commit only this lane.

## Status

- [x] Read the queued lane brief and project instructions.
- [x] Verified the clean dedicated worktree and branch.
- [x] Committed this ledger first as `422f4496`, before implementation edits.
- [x] Traced spill configuration, statistics emission, mmap construction, and test seams.
- [x] Implemented the process-global config-fallback counter without decode-path cost.
- [x] Added and passed the focused invalid-`MEMRA_SPILL_IO` unit test.
- [x] Reconciled `docs/FLAGS.md`.
- [x] Passed `run-gen` argmax MATCH on the local RTX 5090 and recorded the `N=1`, `window_clean`
      spill-model smoke.
- [x] Proved the live `MEMRA_SPILL_STATS` snapshot exposes `config_fallbacks=1` while the
      separate runtime `fallbacks` counter remains zero.
- [x] Inspected the final diff; the implementation/evidence commit leaves the branch for
      orchestrator review.

## Evidence

- Production path: invalid `MEMRA_SPILL_IO` and `MEMRA_SPILL_PREAD_DEPTH` each increment one
  process-global relaxed atomic when their `OnceLock` resolver warns and substitutes its existing
  default. The stats read happens only in the existing diagnostics path; decode dispatch is
  unchanged.
- Focused test: `cargo test -p memra-engine --lib
  spill_pread::tests::invalid_spill_io_is_counted_and_uses_mmap -- --exact` — PASS. The test runs
  the real environment-backed `configured_mode()` in a fresh child process and asserts counter
  `0 -> 1` plus `SpillIoMode::Mmap`. Receipt: `raw/unit-test-invalid-io.log`.
- Spill module: 6 PASS, 0 FAIL, 1 pre-existing CUDA-only ignored test. Receipt:
  `raw/spill-pread-tests.log`.
- Compile/docs hygiene: `cargo check -p memra-server` PASS; `tools/check-flags.sh` reports no new
  drift beyond the existing baseline. Receipts: `raw/cargo-check-server.log`,
  `raw/check-flags.log`.
- Local RTX 5090 spill gate: `N=1`, `window_clean`, no performance window. Qwen3.6-35B-A3B
  IQ4_XS ran with disk tier forced, pinned fraction 0, eight SLRU slots, and
  `MEMRA_SPILL_IO=wroker`. It placed 30,720 expert blocks / 14,514 MiB in mmap, warned and selected
  mmap, then reported `prefill argmax=1178 decode argmax=1178 ... MATCH`; exit 0. GPU was 0% at
  entry and no other compute app was present in the pre/post queries. Receipt:
  `raw/run-gen-invalid-io.log`.
- Live stats smoke: one plain server request under the same forced mmap configuration, `N=1`,
  `window_clean` (`compute_apps_pre=none`, `compute_apps_post=none`). Request and post-request
  health passed; the snapshot was `reads=0 bytes=0 errors=0 short_reads=0 config_fallbacks=1
  fallbacks=0 buffer_waits=0 ring_full=0`. Receipts: `raw/server-invalid-io-driver.log`,
  `raw/server-invalid-io.log`, `raw/server-invalid-io-response.json`.
- No perf claim, board update, formatting sweep, merge, tag, or push was performed.
