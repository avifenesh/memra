# Self-review: integ27 (C day 19: memra#617 argv refusal; B day 24: memra#476 predictive admission terms, #524 gap)

Author's review of the full diff `main..lane/spill-integ27-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/argv.rs` (new) and one hunk in `lib.rs`: `argv::validate` runs as the first statement of
  `serve_with`, before `--version`, the first environment read and any device work; an unknown token is a refusal
  naming the token, exit 2 (the usage class `auth::run_cli` already uses). The accepted set is read from the parser:
  `--version`, `-V`, `--gen-key <tenant>` with `--lane`, `--rate-limit`, `--keys`, `--revoke-key <prefix>` with
  `--keys`. A key modifier without a key command is refused; a missing value falls through to `run_cli` as before.
  Tests: 7 unit, 7 booting the real binary. I checked every tracked launcher under `tools/` and `docs/`: none passes
  `memra-server` an argument outside that set. `docs/SERVING.md` states the rule in one sentence.
- `crates/memra-server/src/admit_predict.rs`: `RequestCharge::from_physical_cost(cost, ctx(C), A, D, ctx(P+L+8))`
  splits the physical cost into its terms (saturating, each term bounded by what lies beyond the context cap) and
  `total()` books `ctx_hat + A + W + D`. `worker.rs`: both predictor sites (verdict and shadow booking) take the charge
  instead of `kv_hat_ring`; the verdict reads the cold cost, the booking the final adjusted cost, which is what the
  real book takes. No new numeric program: nothing in a forward changes; only what the shadow predictor books. Lock
  tests `admit_predict_shadow_wiring` and `verdict_line_locks_fields` unchanged and green. `docs/FLAGS.md`: the two
  existing admission door rows now describe the fuller charge; no new `MEMRA_*` read (census clean).
- Research: C DAY19 (the serverdoor cells, the arena scoping note, the DFlash pre-registration), B DAY24 (the
  arithmetic, before and after receipts, the #524 reading and the pre-registered fault gate), STATE files, INDEX rows,
  the lead record section with ruling 28, this file, battery receipts (the C-only tree's run under `-ctree`, the
  combined tree's run as the receipt).

## What I checked
- The argv refusal cannot reach a served request: it precedes the environment read and the device, and the
  key-lifecycle CLI keeps its own usage exit. The real-binary tests boot with a bogus flag and assert the refusal text.
- The predictor change books more, never less, than before on every path (`total() >= kv_hat_ring` by construction
  since every added term is non-negative), so the shadow verdict can only tighten; the enforcing door stays OFF by
  design and its cell is named as owed.
- The pre-registered clause B that failed before and after is reported as a failure of the clause, not relaxed; its
  cause (pool high-water at inflight 0) is outside the predictor.
- INDEX.md: both sides' rows kept, no parent row lost, the marker census clean.
- Battery on the combined tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector
  pytest, engine CPU lib tests, server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check) and
  the local 5090 serve-smoke.

## What I did not do
- No target-card cell of my own; #617's refusal is argument admission before any device statement.
- The arena lease handoff and the health fault gate stay scoped and pre-registered.
