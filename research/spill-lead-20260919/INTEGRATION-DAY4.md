# Generic-spill pre-work — day-4 integration record (lead: agent-c07799, 2026-09-19)

PR #518 (days 1–3 + review fixes) MERGED to `main` @ `61be8b0d` on green CI (owner-authorized, no
human review). CI rounds: publish census → serde_json inference (dsv4_sampler canary; model.rs test)
→ three automated-review findings fixed with fail→pass regressions (fairness prune watermark; bank
tail record clamp; `valid_bytes` compared by `require`, framed `io_bytes` telemetry-only).

Branch `lane/spill-integ2-20260919` from `main@61be8b0d`, merged day-4 lanes: A `989ef282`
(lazy sharded catalog, lease-safe GC, telemetry adapter) → B `be31aff0` (opaque packed-record
materializer, recompute-vs-load in admission, HostPrefix patch v2 unapplied) → C `47885c07` (SLRU
CPU decision model vs source oracle, typed `BankSource::install`, ready-view ownership map, Hy3 guard
patch v2 unapplied, full conversion NO-GO) → D `ec887fd9` (`peer::test_support::FakePeerCapacity<G>`
seam, container-hardened resumable bootstrap, `RIG-DAY1.md` with lane runners, battery `--validate`).
Conflicts: B's `scheduler.rs` imports (merged by hand) and `tests.rs` (append-only union, 0 duplicate
fns); C, D clean.

Integrated CPU battery: **223 tests pass, 0 fail** (`--no-fail-fast`); clippy `-D warnings`, fmt,
`diff --check`, flags census, Linux cross-target check, publish census, `py_compile`, bootstrap
`bash -n` — all OK. Not run: native build (nvcc) — CI on the next PR is that check — any GPU cell.
