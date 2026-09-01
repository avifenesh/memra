# cx-sec2 progress

- Branch: `lane/cx-sec2`
- Baseline: `2339b5f9`
- Scope: metrics tenant isolation and per-tenant ADSD acceptance-collapse telemetry.
- Constraints: CPU-only; detection only; lossless verification unchanged; no `cargo fmt`.
- Status: complete. Final PASS verdict and verification evidence are in `RESULTS.md`.

## Checkpoints

- 2026-08-11: Created the lane ledger alone in commit `d8202955` before implementation.
- 2026-08-11: Made `MEMRA_METRICS_TOKEN` exclusive for both metrics routes. Without it,
  keyring callers receive global counters with only their own `tenants` row; the legacy
  single-key domain and no-key loopback development behavior remain unchanged. Focused
  metrics suite: 5 passed, 0 failed.
- 2026-08-11: Added bounded request-level acceptance windows per `(model, tenant)` and a
  larger rolling model baseline. A sustained 3-sigma, 20-point collapse emits one latched
  `[adsd-suspect]` log and increments tenant-scoped `adsd_suspect_total`; it never changes
  verification, routing, caching, or rate limits. Synthetic collapse/noise tests: 2 passed.
- 2026-08-11: Documented the exclusive scrape-token/operator choice, completion-key tenant
  scoping, ADSD thresholds, and manual-response posture across the serving, flags, and RunPod
  operator surfaces. Full `cargo test -p memra-server`: 172 passed, 0 failed. Perf-board and
  flag-drift checks passed (`check-flags` reports only the frozen known drift set).
- 2026-08-11: Final consistency and lane-isolation checks passed; recorded the PASS verdict
  in `RESULTS.md`.
