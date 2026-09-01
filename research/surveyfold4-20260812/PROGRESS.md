# cx-surveyfold4 progress

- Branch/worktree: `lane/cx-surveyfold4` at
  `/home/avifenesh/projects/wt-cx-surveyfold4`, from `main` at `79c3c0b27`.
- Scope: documentation-only fold for PNM heterogeneous serving, FluxMoE expert paging,
  SS-MoE self-speculative drafting, and RotaryQuant role-based mixed precision.
- Constraints: append-only addenda in the existing house format; no rank changes, code,
  formatting, merge, tag, or push; commit the completed lane.

## Work log

- [x] Confirm a clean dedicated worktree and create this progress receipt first.
- [x] Locate and preserve the exact §10 and xshare/spill addendum formats.
- [x] Verify all four papers against current primary web records and record retrieval date.
- [x] Append the four requested short sections without changing existing rankings.
- [x] Audit the docs-only diff and commit the lane without bypassing hooks.

## Source and target receipt

- Retrieved 2026-08-12: arXiv 2608.03555v1 (KARAT/PNM), arXiv 2604.02715v2
  (FluxMoE), ACM DOI 10.1145/3774904.3792218 (SS-MoE), and arXiv 2608.08081v1
  (RotaryQuant).
- `research/spec-landscape-20260810/PROGRESS-xshare.md` identifies `SURVEY.md` as the
  XShare addendum surface; `research/spillobs-20260811/PROGRESS.md` supplies the existing
  expert-spill observability vocabulary. The four entries therefore append to the same survey
  surface; no separate literature-addendum file exists in the current tree.
- PNM is recorded only as a KV-versus-expert spill taxonomy; FluxMoE gets an explicit
  KV-starved-concurrency versus hot-reuse regime boundary; SS-MoE preserves the full-expert
  verifier fence; RotaryQuant does not alter the frozen Hy3 five-arm study.
