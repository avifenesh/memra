# cx-sec1 progress

## Scope

Harden `memra-server` authentication and keyring persistence for the four confirmed external-review findings in the lane brief:

1. Protect `/metrics` and `/yield/metrics` when keys are configured or the server is exposed beyond loopback.
2. Refuse an unauthenticated non-loopback bind unless `MEMRA_ALLOW_OPEN_BIND=1` is explicitly set.
3. Remove short-circuit secret comparisons from static-key and keyring authentication.
4. Make key generation/revocation persistence atomic and mode `0640`.

## Constraints

- Worktree: `/home/avifenesh/projects/wt-cx-sec1`
- Branch: `lane/cx-sec1`
- CPU-only lane; no GPU validation.
- Never run `cargo fmt`.
- Preserve unrelated work and stage only lane files.
- Required gate: `cargo test -p memra-server`.

## Plan and verification

- [x] Add focused regression tests for metrics authorization, exposed-bind refusal, constant-time credential paths, and concurrent-reload-safe atomic rewrites.
- [x] Implement the smallest server/auth changes that make those tests pass.
- [x] Correct the systemd template and document `MEMRA_METRICS_TOKEN` and `MEMRA_ALLOW_OPEN_BIND`.
- [x] Run focused tests followed by the full `memra-server` package test suite.
- [x] Record the final evidence and verdict in `RESULTS.md`.

## Checkpoints

- 2026-08-11: Lane brief and repository law read; branch/worktree confirmed clean; initial plan recorded before source inspection.
- 2026-08-11 03:44 +03:00: Replaced request-time static/ring secret equality with fixed-length SHA-256 digest comparison; key generation now creates mode `0640`; revocation writes `keys.toml.tmp`, fsyncs it, and renames over the live ring. Focused auth suite: 12 passed, 0 failed, including concurrent hot-reload/rewrite stress.
- 2026-08-11 03:49 +03:00: Added pre-model-load bind validation and metrics authorization state. Both metrics routes now return 401 without a bearer when keyed, accept normal API keys or `MEMRA_METRICS_TOKEN`, remain open only for no-key loopback development, and stay locked on an explicitly overridden public bind. Focused handler tests: 4 passed, 0 failed.
- 2026-08-11: Switched the systemd server template to loopback, corrected `keys.toml`, documented the direct-bind override and scrape token, and removed stale deployment documentation claiming metrics were intentionally unauthenticated.
- 2026-08-11 03:55 +03:00: Integrated authenticated scrapes into the fleet meter, RunPod provisioner, and trial health matrix. The provisioner generates a dedicated unprinted token into its mode-`0640` environment file. `bash -n`, `shellcheck`, fleet-meter help, and the RunPod dry run passed.
- 2026-08-11 03:58 +03:00: Final consistency pass made live economics/runbook scrapes bearer-aware and rejects an empty `MEMRA_API_KEY` instead of treating it as a configured public-bind credential. Focused metrics tests, Python help, shell syntax, and ShellCheck remain green.
- 2026-08-11 04:00 +03:00: Final battery passed: 167/167 `memra-server` tests, debug build, generated perf-board check, lane-only diff check, and a real no-key `0.0.0.0` boot refusal (`exit 1`, expected FATAL before model load). Verdict recorded in `RESULTS.md`.
