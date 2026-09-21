# Self-review: integ25 (B days 22 and 23 on memra#427 under #614's mechanism; C day 18 MoE slot door inputs; conflict-marker census)

Author's review of the full diff `main..lane/spill-integ25-20260921`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/bin/qwen_a4_width_walk.rs` and `hash_micro.rs` (+ their `Cargo.toml` `[[bin]]` rows):
  two diagnostics. The width walk digests every projection of a 16-row and a 17-row prime chunk under the plain
  program and names any tensor that differs (0 of 497 on main's mechanism); hash-micro times the engine's
  `memra_tier::contracts::checksum` over cached pinned, write-combined pinned and heap bytes. Neither touches an
  engine path, a flag or a kernel. I read both: they call public engine entry points only.
- `docs/TESTING.md`: one pointer sentence to the width walk inside #614's continuation-gate paragraph.
- `tools/check-conflict-markers.sh`, `tools/test_conflict_markers.sh`, the pre-push hook block, the CI gates step:
  the marker census (lead). `research/INDEX.md` and `research/tune-data/perf-ci.jsonl`: one stray diff3 marker line
  removed from each (both from other sessions' rebases on main); every INDEX row of every parent is present and every
  perf-ci line parses as JSON.
- Research: B DAY22/DAY23, `PRIME-MIN-T-DECISION.md` with its "Superseded by #614" section, receipts from both cards;
  C DAY18, door docs, receipts; the lead record and this file; battery receipts on the final tree and, under
  `-btree`, the superseded run on B's own mechanism.

## What I checked
- The lane's `prefill_rows` mechanism is gone: `git grep prefill_rows -- crates/memra-engine` and
  `batched_tier_admits` return nothing. The engine's only #427 mechanism is main's `small_m_tier_max()`.
- No new `MEMRA_*` read (flags census clean). Lock names in new scripts: the two canonical ones only.
- The census refuses all four marker kinds and passes on this tree; on its first run it found the perf-ci marker,
  which is why the data suffixes are in scope. Receipt logs and raw dirs are excluded on purpose (a receipt may capture
  a diff); the teeth cover that exclusion.
- The gates B ran on the merged tree are main's mechanism under test, both cards, verbatim in the record: width walk,
  continuation gate, kernel-check (target card with the 9B staged so the required cell ran), hit gate, twin gate,
  run-gen argmax and run-spec K=1..8 on 16-token and 4112-token prompts.
- Final battery on this tree: fmt, portable suites, memra-server suite (green this time; on B's earlier tree the
  scheduler-sensitive darklane yield test read `left: 1 right: 2` once under load and passed 3 of 3 alone), clippy,
  censuses, collector pytest, engine CPU lib tests, engine clippy `-D warnings`, perf board, diff-check, the marker
  census and its teeth, workflow keys, em-dash scan: all rc=0. Local 5090 serve-smoke on this tree in the receipts.
- Public boundary `check`: 0 new. INDEX.md carries no marker line of any kind.

## Issue actions on merge (lead)
- Close #427 on #614's mechanism plus B's two-card receipts; note the step35 split confirmation as owed.
- File the hygiene finding from C day 18 (memra-server rejects no unknown argument) as an issue.

## What I did not do
- No GPU cell of my own beyond the smoke. The MoE slot door's decision stays for its 2026-10-04 review; the
  HOSTPREFIX door's for 2026-10-05.
