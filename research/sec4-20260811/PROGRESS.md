# cx-sec4 progress

- Branch/worktree: `lane/cx-sec4` at `/home/avifenesh/projects/wt-cx-sec4`
- Starting revision: `ae558f55188d5aa93787481a0f7ea3182ce3b49f`
- Scope: CPU-only serving hardening follow-ups from sec2.
- ADSD: prevent a suspect tenant from diluting its own model baseline, account for baseline sampling error, and add a boiling-frog latch regression.
- Metrics: restrict capacity/VRAM gauges and aggregate speculation metrics to operator scope while preserving tenant token rows, safe global counters, and no-key loopback visibility.
- Required gate: `cargo test -p memra-server`.
- Constraints: no GPU work, no `cargo fmt`, no unrelated cleanup.

## Status

- [x] Lane brief and repository law read.
- [x] ADSD regression and fix.
- [x] Metrics isolation regression and fix.
- [x] Serving documentation updated.
- [x] Full CPU test gate passed.
- [x] Final verdict recorded in `RESULTS.md`.

## Evidence

- Boiling-frog regression reproduced the pre-fix latch clear, then passed after model baselines
  became other-tenant-only and the z-score adopted pooled two-proportion sampling error.
- `cargo test -p memra-server adsd_detector -- --nocapture`: 3 passed, 0 failed.
- Tenant-scope regression reproduced `active_sessions` exposure, then passed after current
  capacity/VRAM, background-job, batch-size, and aggregate speculation state became operator-only.
- `cargo test -p memra-server metrics_ -- --nocapture`: 6 passed, 0 failed; dedicated metrics
  token and no-key loopback retain operator visibility, completion credentials do not.
- `cargo test -p memra-server`: 174 passed, 0 failed, 0 ignored.
- Final verdict: PASS; see `RESULTS.md`.
