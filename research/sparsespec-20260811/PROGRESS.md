# SparseSpec-L feasibility progress

Date: 2026-08-11
Branch: `lane/cx-sparsespec`
Base: `e7aed855`

## Scope

- Determine whether SparseSpec-L verifies sparsified-KV drafts with the full target distribution;
  treat losslessness as the decisive gate.
- Map train-free self-drafting and recallable sparse KV onto memra's house own-trim MTP regime
  without displacing that baseline.
- Assess the entropy-based speculation-length controller separately from the full sparse-KV method.
- Deliver only `FEASIBILITY.md`; do not implement, build, benchmark, use a GPU, merge, tag, push,
  or format the tree.

## Status

- [x] Read the lane brief and `/home/avifenesh/projects/bw24/CLAUDE.md` first.
- [x] Confirmed a clean worktree on `lane/cx-sparsespec` at current-main base `e7aed855`.
- [x] Trace memra's own-trim MTP baseline, fixed-K control, verification gate, and GGUF drafter attachment.
- [x] Verify the SparseSpec-L algorithm and related public implementations against primary sources.
- [x] Write and citation-audit the feasibility verdict.
- [x] Leave the branch unpushed for orchestrator review.

## Receipt

- Changed only the two research-lane documents under `research/sparsespec-20260811/`.
- Ran only CPU-side read/search and citation checks; no formatter, build, benchmark, or GPU command.
- Did not merge, tag, or push; branch remains `lane/cx-sparsespec` for orchestrator review.
