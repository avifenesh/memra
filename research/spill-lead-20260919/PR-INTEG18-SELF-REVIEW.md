# integ18 self-review (lead, 2026-09-21)

Read in full: D day 14 diffs to `tools/portable-suites.sh`, `tools/ci-portable.sh` (forward), `tools/local-ci.sh`,
`tools/check-workflow-keys.py`, `tools/test_workflow_keys.sh`, `tools/test_portable_suites.sh` (arm 3),
`tools/test_gpu_ci.py` (wiring arm), `tools/hooks/pre-push` (workflow-file census arm), `docs/CI.md`, `docs/TESTING.md`,
`.github/workflows/ci.yml` after the post-hotfix merge; D's DAY14.md census; receipts spot-checked.

## Findings
1. **One executor, provably.** Teeth arm 3 pins exactly one `portable-suites` job key, no live `cargo test` on the three
   crates outside the wrapper anywhere in ci.yml or local-ci.sh, and a `ci-portable.sh` that forwards and runs no cargo
   of its own; `gpu-ci.yml` is untouched and its dispatch prerequisites are named as draft #566.
2. **The guard sits where it can act.** `check-workflow-keys.py` uses a loader that raises on duplicate keys (the
   thing `safe_load` hides), runs in the `gates` job and as an unconditional pre-push arm with no skip switch; proven on
   main's broken file. The hook arm is the one that can stop a broken ci.yml from reaching origin.
3. **Nothing weakened.** Census budget 0, floor 300, the collector suite under held locks with its unittest floor, the
   gpu-ci controls under a floor (11 tests, floor 9), `--locked` added to the wrapper; teeth counts went up (18 to 22).
4. **Post-hotfix merge clean.** One job, one gpu-ci step; D's branch carried the same removal hunk as #600 and the
   identical step text, so the merge produced no duplicate.
5. **Nits (not blocking).** The boundary pin for another raw cargo log (`live_fingerprint` on the test-binary path) is
   the same false-positive class as day 13; a policy exclusion for `target/debug/deps/` paths is a small later PR.
   D's untracked day-11 `build/` receipt dir is a lane housekeeping item.

## Verification this review relied on
integ18 CPU battery (`integration-day12/integ18-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `check-workflow-keys.py`, both teeth scripts, the gpu-ci tests under the floor, perf board,
diff-check. No GPU work in this PR.
