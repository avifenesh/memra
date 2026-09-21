# integ14 self-review (lead, 2026-09-21)

Read in full: B day 14 `crates/memra-server/src/worker.rs` diff (`room_victim_with`, the preflight reclaimable set,
`prefix_insert_refused_oversize`, `prefix_insert_refused_leased`, the four prefix-cache unit tests),
`worker/host_glm.rs` (4 lines), `tools/prefix-newest-turn-fits-gate.py` (skimmed for the verdict clauses and the
calibration boot), `docs/TESTING.md` section, `verify-day14.py`; receipts spot-checked.

## Findings
1. **Victim selection only.** The change keeps the SLRU policy and its budget; it excludes the entry being inserted
   from its own victim set and, when the newest turn still does not fit, takes protected entries oldest first.
   Snapshot capture and restore are untouched, and the gate proves it: completion digests are identical across base
   and fix on every turn and equal to the cache-off boot's cold completions.
2. **Refusals are typed lines, never silent cold turns.** Oversized entries print the exact budget line (pinned by a
   unit test); entries that cannot fit beside leased bytes print their own. The card never hit either (every turn
   fits by construction of the shape), which the record says rather than implies.
3. **Gate honesty.** Two rounds of the gate were wrong about V3 and are kept as failed cells with the byte deltas that
   exposed them; round 3 asserts state identity after every turn. `refused_or_skipped=1` on base is the base's own
   `snapshot skipped` line, not a refusal of the gate.
4. **Nits (not blocking).** The recorded per-token device retention (67,200 B per token with the cache idle) is a real
   observation outside this PR's subject; it belongs to #536's family and should get an issue. The gate's V4 text sha
   binds to this artifact's outputs; another artifact is its own gate run.

## Verification this review relied on
integ14 CPU battery (`integration-day12/integ14-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `verify-day14.py` (B), perf board, diff-check; local 5090 serve-smoke on this tree
(`integ14-serve-smoke-5090/`). B's target-card twin gate, serve-smoke and cache-meter on the fix. This rig cannot run
the model gates.
