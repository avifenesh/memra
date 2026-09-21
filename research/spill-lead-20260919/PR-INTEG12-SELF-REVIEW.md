# integ12 self-review (lead, 2026-09-21)

Read in full: `.github/workflows/ci.yml` (`portable-suites` job), `tools/portable-suites.sh`, `tools/test_portable_suites.sh`,
`tools/skip-census.tsv` additions, `tools/local-ci.sh` `cpu_chain()` change, `tools/tier-battery.py` `lock_table`,
`tools/tier-rig-bootstrap.sh`, `crates/memra-tier/tests/battery/private_lock.py` and the touched battery tests,
`crates/memra-tier/tests/reclaim/fault.rs` (strict replays).

## Findings
1. **Teeth are real.** The wrapper reds on planted failures in each of the three crates and names the failed target
   per crate; an undeclared SKIP reds before cargo runs; the planted copy builds in its own target dir after the
   shared-target reuse finding. Build and clippy alone cannot satisfy the job.
2. **Skips explicit, never GPU qualification.** `skip-census.py` gates on declared skips with a zero budget and a
   minimum pass count; step names and the wrapper's last line say so.
3. **Lock seam is test-only.** `MEMRA_TIER_BATTERY_LOCK_DIR` changes the directory the two canonical lock names live
   in; the names themselves are unchanged (lock-name rule kept), production callers untouched, the legacy gates keep
   their literals, and a receipt produced under the seam refuses to validate without it (no laundering of a private
   lock into a real receipt). Proof ran with both real paths held inside a private `/tmp`, so the rig locks were never
   taken.
4. **Strict replays.** The committed-receipt replay in `fault.rs` no longer skips silently when a receipt is absent;
   a missing receipt is a failing test, which matches the receipt-hygiene finding of integ10 and 11.
5. **Nits (not blocking).** Two raw cargo logs are pinned as boundary false positives (`live_fingerprint` on cargo's
   test-binary path); the rule could exclude `target/debug/deps/` paths, a policy edit for a later PR. The
   `research/**` docs-only classing in `tools/ci-change-class.sh` versus test-time `research/` reads is queued.

6. **Revuto round, both fixed on the lane (`0e9e30b31`).** The static census now covers `crates/<crate>/tests` as
   well as `src` (it found and declared memra-tokenizer's four artifact-gated skips), and the teeth plant the
   undeclared SKIP under `tests/` too. The private lock seam is no longer honoured by the environment alone:
   `--execute` and `--dry-run` refuse under `MEMRA_TIER_BATTERY_LOCK_DIR` without `--private-lock-dir-for-tests`, the
   flag without the seam refuses, one loud stderr line names the private directory, `lock.json` carries the seam, and
   every validate path refuses a capture whose seam is not the validating process's own. Red arms for each.

## Verification this review relied on
integ12 CPU batteries (`integration-day12/integ12-cpu-battery/` before the review fixes, `integ12-cpu-battery-2/` after): fmt, the portable-suites wrapper, the teeth script,
memra-server suite, clippy `-D warnings` (incl. memra-cli), censuses, collector pytest, ci.yml YAML load, perf board,
`git diff --check`. D's own receipts under `research/spill-d-20260919/day13/`. No GPU work in this PR.
