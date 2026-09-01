# Metrics-token and pinned-budget hardening progress

Date: 2026-08-11
Branch: `lane/cx-metricstoken`
Base: `714a5c2d`

## Scope

- Prevent completion bearers in API-keyring deployments from observing process-wide metrics
  without the operator metrics token.
- Preserve the legacy single-tenant/no-keyring metrics behavior.
- Clamp `MEMRA_SPILL_PINNED_FRAC` to its documented range, count invalid configuration through
  the existing spill `config_fallbacks` seam, and log the resolved pinnable-RAM budget.
- Add focused regression tests and reconcile the serving/flags documentation.

## Constraints

- No merge, tag, push, formatting sweep, or performance-board change.
- No GPU gates are required for these server/configuration-only fixes.
- Preserve unrelated work and commit only this lane.

## Status

- [x] Read the queued lane brief and project instructions.
- [x] Verified the clean dedicated worktree and branch from current `main`.
- [x] Committed this ledger first as `f7fd88bf`, before implementation edits.
- [x] Traced the current metrics authorization/scoping and spill fallback seams.
- [x] Removed every process-wide counter from the keyring completion view and made the
      process-wide yield view require the operator metrics token.
- [x] Preserved the existing single-key completion-domain and no-key loopback behavior.
- [x] Bounded `MEMRA_SPILL_PINNED_FRAC`, warned once and counted invalid values through the
      existing `config_fallbacks`, and surfaced `free_pinnable_ram` in both load paths.
- [x] Reconciled serving, flag, and RunPod operator documentation.
- [x] Passed focused tests, `cargo check`, and the flag-drift check.
- [x] Inspected the final diff; the implementation/evidence commit is the next operation.

## Evidence

- Server metrics filter: `cargo test -p memra-server metrics -- --nocapture` — 7 passed,
  0 failed. The regression asserts a keyring completion bearer gets exactly its own `tenants`
  and `adsd_suspect_total` rows, with no process-wide keys, and receives 403 from
  `/yield/metrics`. It also pins the legacy single-key cumulative/yield behavior. Receipt:
  `raw/server-metrics-tests.log`.
- Spill fraction filter: `cargo test -p memra-engine --lib pinned_frac -- --nocapture` — 2
  passed, 0 failed. Valid finite `(0,1]` inputs are accepted; zero, negative, greater-than-one,
  NaN, infinity, and non-numeric values are rejected. Fresh child processes prove both an
  out-of-range value and a parse failure warn, resolve to 0.60 without failing, and increment
  the shared counter exactly once. Receipt: `raw/spill-pinned-frac-tests.log`.
- Existing spill-observability module: `cargo test -p memra-engine --lib spill_pread::tests --
  --nocapture` — 6 passed, 0 failed, 1 pre-existing CUDA-only test ignored. Receipt:
  `raw/spill-pread-tests.log`.
- Compile gate: `cargo check -p memra-server` — PASS. Receipt: `raw/cargo-check-server.log`.
- Flag-doc gate: `tools/check-flags.sh` — no new drift beyond the frozen baseline. Receipt:
  `raw/check-flags.log`.
- No GPU/performance claim, formatting sweep, board change, merge, tag, or push was performed.
