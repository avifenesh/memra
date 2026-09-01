# cx-xshare progress

Status: complete — ready for orchestrator review

- Confirmed dedicated branch `lane/cx-xshare` starts from `96afb32e` with a clean worktree.
- Read the lane brief, `/home/avifenesh/projects/bw24/CLAUDE.md`, and the full existing
  `research/spec-landscape-20260810/SURVEY.md` in the mandated order.
- Verified XShare (arXiv 2602.07265) from the primary paper: its modular proxy maximizes summed
  gating score over a live-batch expert cover; its speculative variant composes per-request
  correlation-aware covers before batch aggregation; and its executed top-k is reranked inside
  that restricted cover.
- Added a short XShare addendum beside ScoutAttention/SparDA. The transferable piece is an
  analysis-only optimistic expert-cover curve (an upper bound on union amortization) for the
  MoESD harness plus grouped-dispatch capacity/pricing.
- Made the doctrine fence explicit: XShare changes which target experts execute, so it is
  research-arm-only and must never alter the verifier's native expert set under the
  frozen-artifact/backend rule.
- Constraints: documentation research only; no code, GPU, model bytes, merge, tag, push,
  formatting, generated performance-board changes, or release operations.
