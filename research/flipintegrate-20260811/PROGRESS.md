# cx-flipintegrate progress

- 2026-08-12: Started on `lane/cx-flipintegrate` from `ba3e70c9a`; worktree clean.
- Scope: docs and research data only; no code, merge, tag, push, formatting pass, or generated-board update.
- 2026-08-12: Fetched box1 inputs read-only. Dual source contains flip commit `e94699eba`; qmatvec candidate is `4f15557c` against baseline `49f5002d`.
- 2026-08-12: Preserved the previously untracked qmatvec PRO raw battery byte-for-byte beside its promotion reduction; the dual raw battery remains owned by the flip branch.
- 2026-08-12: Reduced the qmatvec exactness, N=8/arm timing, and NCU mechanism receipts; drafted both performance notes and copied the flip-owned flag semantics verbatim.
- 2026-08-12: Board review recommends no tracked-cell move: the dual result is a PRO-pair serving cell and qmatvec is PRO micro-only; neither matches a generated 5090/H100 cell.
- 2026-08-12: Validation passed: fetched qmatvec raw files are byte-identical, numeric reductions recompute, the two flag rows match box1 verbatim, all generated PERF blocks and `current-board.json` are unchanged, and `git diff --check` is clean.
- Out-of-scope handoff: box1 `crates/memra-engine/src/decode_batch.rs` still has an internal comment saying the dual arm is default OFF; canonical `pp.rs`, runtime behavior, gates, and the copied public docs all say Auto/default ON. No code edit made in this lane.
- Status: scoped artifacts complete and ready for the required local commit.
