# v0.73 release-wave preparation

Lane: `lane/cx-v073prep`

Frozen audit target: `cbe25b75e95f9aed8863771b625e12c35b016286`

Release range: `v0.72.0..cbe25b75`

## Checklist

- [x] Verify the dedicated branch, clean worktree, and frozen target commit.
- [x] Read `CLAUDE.md`, `docs/RELEASING.md`, `tools/changelog.sh`, and
  `research/tune-data/current-board.json` before audit work.
- [x] Check `~/.lanectl/inbox/cx-v073prep.md` at lane start (file absent).
- [x] Run the changelog generator over `v0.72.0..cbe25b75` and reconcile its public-prefix output
  with the complete merged commit train.
- [x] Trace and summarize the v0.73 mechanisms named in the lane brief: leverC grouped prefill,
  prefill pipelining, primebatch, specplace, prefix dedup and pinning, dynamic microchunks, TTFT,
  archkit, serve-ready receipts, kpolicy, concprefill, and the API-USAGE privacy rule.
- [x] Write `CHANGELOG-DRAFT.md`, preserving changelog-script grouping while adding release-ready
  context and evidence pointers.
- [x] Audit `README.md`, `docs/PERFORMANCE.md`, `docs/FLAGS.md`, and `docs/SERVING.md` against code,
  flags, receipts, research verdicts, and merged documentation.
- [x] Record every confirmed drift item with `file:line` evidence and fix only pure-documentation
  drift outside generated `PERF-*` blocks.
- [x] Run the generated-board drift check and compare post-v0.72 board-moving evidence with
  `current-board.json`; report any missing number without editing the board.
- [x] Draft `RELEASE-NOTES-DRAFT.md` in the v0.72.0 release-note format.
- [x] Re-check the lane inbox at final-audit time (file still absent at 2026-08-08 23:49 +03:00;
  the work did not cross an hourly boundary).
- [x] Run scoped verification and inspect the final diff; checkpoint all intended lane files in the
  closing release-prep commit.

## Constraints

- Docs and audit only: no tag, main merge, origin push, Rust toolchain change, or GPU job.
- Never hand-edit generated `PERF-*` marker blocks.
- Do not update `current-board.json` in this lane; report board-moving omissions for the owner.
- Preserve unrelated work and stage only `research/v073-prep-20260808/` plus confirmed pure-doc fixes.

## Audit result

- Changelog floor: 130 commits in range; `tools/changelog.sh` includes 67 subjects and filters
  49 research/plumbing subjects plus 14 merges. Exact stdout SHA-256 is recorded in
  `CHANGELOG-DRAFT.md`.
- Docs: confirmed and fixed stale Step35 qualification, chunk/tick status, missing v0.73 serving
  mechanisms, prefix-entry eviction semantics, duplicate prime-chunk flag truth, flag counts, and
  the stale RunPod K=3 statement. Full `file:line` ledger: `DOCS-DRIFT.md`.
- Board: no missing board-moving number identified. `current-board.json` and SVGs are unchanged;
  the generated surfaces pass `tools/update-perf-board.py --check`.
- Release notes: drafted in the v0.72.0 section/footer format, with serve-ready and concprefill
  evidence added manually because changelog logic intentionally filters their `data:`/merge rows.
