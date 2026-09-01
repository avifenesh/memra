# Docs batch 3 progress

Lane: `cx-docsync3`
Started: 2026-08-11
Scope: docs plus the non-fatal flag-drift lint requested in `~/.lanectl/inbox/cx-docsync3.md`

## Deliverables

- [x] Rewrite `SECURITY.md` scope for shipped server authentication, metering, and tenant isolation.
- [x] Add `MEMRA_SIG_ROUTER` and correct the stale sigmoid-router comment in `docs/FLAGS.md` / `hybrid_forward.rs`.
- [x] Add the dated pre-fix pointer to `research/ppaudit-20260811/AUDIT.md`.
- [x] Add `tools/check-flags.sh`, wire its warning into `tools/local-ci.sh`, and record the current uncovered list.
- [x] Add the ADSD operational runbook paragraph to `docs/SERVING.md`.
- [x] Append the Windowed-MTP survey addendum to `research/spec-landscape-20260810/SURVEY.md`.

## Evidence and validation

- [x] Preserve the exact receipt paths named by the lane brief.
- [x] Do not edit PERF marker blocks or run `cargo fmt`.
- [x] Run focused documentation/script checks and inspect the final diff.

## Commit log

| Commit | Contents |
| --- | --- |
| `58797caa` | Initial ledger; committed before implementation edits. |
| `0a22e4b4` | Security, router, audit, serving, and survey documentation updates. |
| `b85ac622` | Flag-drift checker, non-fatal local-CI warning, and current uncovered-name baseline. |
