# Self-review: integ29 (B day 25: the health, readiness and lifecycle fault gate; phase=warming)

Author's review of the full diff `main..lane/spill-integ29-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/health.rs`: a `PHASE_WARMING` between LOADING and IDLE; `mark_warming()` moves LOADING to
  WARMING by `compare_exchange` and is a no-op from any other phase (a serving worker is never demoted; the test
  covers IDLE, BUSY and the respawn path); `live()` reports warming as not live with the reason; `phase_name` names it.
- `crates/memra-server/src/worker.rs`: `run_boot_calibration` takes the shared health and marks warming after the last
  skip return and before the probe generation (one call site, asserted by the extended source-order test); the skip
  paths leave the phase alone, so readiness there is unchanged (the gate's arm (c) records that as documented behaviour,
  not a pass).
- `tools/health-fault-gate.sh`: six arms plus (a2), boots and fixed-door faults, assertions on `/readyz`, `/health`
  and the request path; no new fault hook; scratch cleaned. `tools/local-ci.sh` runs it after the smoke with a skip
  row in `docs/FLAGS.md`. `docs/SERVING.md` phase table and `docs/TESTING.md` updated.
- Research: DAY25, receipts from both cards, the two labelled harness failures, STATE, INDEX row; the lead record
  section; this file; battery receipts.

## What I checked
- No numeric change anywhere; the phase is bookkeeping read by the health routes.
- Every environment variable the gate sets has a `docs/FLAGS.md` row (flags census clean, and #615 now refuses an
  unknown name at boot, which the gate's boots would have hit).
- The arms are deterministic on this rig (two identical local runs, one identical target-card run in every asserted
  field); the timing fields are labelled `not a timing claim` or `N=1`.
- The gate's `/healthz` naming in the issue is corrected to the routes that exist (`/health`, `/livez`, `/readyz`).
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check), the local
  5090 serve-smoke, and the gate itself run once more on this tree.

## Review round 1 (revuto, addressed in the integ)
- Default port 8186 collided with `serve-gemma4-batch-gate.sh`; moved to 8189 (unused across `tools/` by census), the
  `memra_port_guard` line unchanged; gate re-run on this tree at 8189, same verdicts.

## What I did not do
- No release-battery decision (#526): the gate is an input; the owner decides what the battery requires.
- The step-OOM and client-disconnect arms stay pre-registered.
