# integ20 self-review (lead, 2026-09-21)

Scope: B days 17 to 19 (#602) and A day 13 (the pinned-kind seam and its A/B), one perf battery on the combined tree.

Read in full: `crates/memra-engine/src/spec.rs` (the two republish conditions, 20 lines), the
`crates/memra-server/src/worker.rs` days 17 to 19 diff (`seed_capture_boundary`, `seed_boundary_inside_prompt`,
`Session::seed_at`, the spec `capture_at` change, the `/metrics` counter, the unit tests), `crates/memra-server/src/lib.rs`
(5 lines), the four gates' changes (`spec-on-cache-hit-gate.sh`, `prefix-newest-turn-fits-gate.py`,
`prefix-restore-identity-gate.py`, `prefix-evict-reclaim-gate.py`, `kv-host-tenant-reclaim-gate.sh`), SERVING and TESTING,
the decision-record paragraph; DAY17 to DAY19 and receipts spot-checked.

## Findings
1. **The fix is at the capture, not the prime.** Both capture sites (prompt-end seed and spec boundary) publish at a
   grid-aligned length with a typed refusal under the floor; the prime programs are unchanged (the twin gate's cold
   digests equal day 17's on all twelve turns); `cached_tokens` reports the aligned length, so billing follows bytes.
2. **Engine change is a guard, not a program.** `spec.rs` gates the prompt-end republish on a grid multiple and keeps
   the later prime stop that the old `min()` dropped; no kernel or numeric path moved; the perf battery ran green on the
   lane and its rows are in this tree.
3. **Gates got stricter, never looser.** Identity and grid are verdict clauses in the twin gate (V5, V6); a new restore
   identity gate with five restore points; the #379 gate's identity law untouched and green on both arms, its accounting
   clauses restated to the aligned lengths with the reason beside each, plus a new on-grid full-cover pair so the
   `restore-full-cover` site stays exercised. The evict-reclaim gate's V3 bug (retention not counted) was fixed in the
   gate and both arms pass; lane A's gate reads the leader's `capture_len`.
4. **Both cards.** Twin and restore gates PASS on the 5090 and the target card; serve-smoke and cache-meter 0 failed;
   `local-ci.sh` correctness and `--perf` exit 0 on the lane. N=1 per cell, executed-not-qualified.
5. **Honesty kept.** Day 18 was refused integration because the #379 gate went red; the record says so. The 5090
   confirmation of the policy A/B remains a no-verdict (its precondition was this defect). The near-tie reading of the
   target card's 28/28 is in the decision record.
6. **Nits (not blocking).** `prefix_fanout_groups` takes the raw in-batch LCP as its capture length (inspection only,
   queued). The incident (a SIGTERM to another lane's process) is recorded with the rule; no data was corrupted.

7. **A day 13, the seam is a seam.** `PinnedKind` defaults to today's flag bits (a CPU cell pins them), `alloc_host` is
   unchanged and delegates, the backing goes through the existing `malloc_host` FFI, copy sites are untouched, and no
   env read exists; the gate selects the arm. The A/B was pre-registered, ran N=5 per arm per order in one lock hold with
   the regime stated, and its verdict line is the card's, not a default; the record keeps sitting 1's inconclusive and
   failed runs. Ruling 22 makes the default a per-device arm after the 5090 cell.

## Verification this review relied on
integ20 CPU batteries (`integration-day12/integ20-cpu-battery/` for B alone, `-2/` with A day 13) and the full `tools/local-ci.sh --perf` on the combined tree (`integ20-local-ci-perf/`): fmt, portable suites, memra-server 758 tests, clippy,
censuses, collector pytest, `verify-day19.py`, perf board, diff-check, all rc=0; B's local `local-ci.sh --perf` on the
lane (rows in this tree); B's target-card and 5090 gates. This rig cannot run the model gates beyond the #379 gate,
which ran green here.
