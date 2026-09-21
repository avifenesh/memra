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

6. **Revuto round, fixed on the lane (`57c550ffb`).** The workflow-key checker no longer imports PyYAML: a stdlib
   line-based walker over block-style YAML (keys by indentation, `- ` scopes, block scalars swallowed, comments and
   markers skipped) with a stated scope enforced as exit 2 "cannot answer" (flow mappings, merge keys, anchors, tags,
   tabs), never green; exit 1 stays the duplicate refusal; the hook prints the matching message for each class; the
   `gates` step needs no install line. Cross-checked against a strict PyYAML loader on all 8 tracked YAML files
   (8 of 8 agree); teeth run the fifteen verdicts twice, with and without a PyYAML shadow (30 ok). Receipt on main's
   broken file with and without PyYAML: the same duplicate-key line, rc 1.

## Verification this review relied on
integ18 CPU batteries (`integration-day12/integ18-cpu-battery/` before the review fix, `-2/` after): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `check-workflow-keys.py`, both teeth scripts, the gpu-ci tests under the floor, perf board,
diff-check. No GPU work in this PR.
