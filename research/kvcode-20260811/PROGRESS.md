# Speculative KV Coding feasibility progress

Date: 2026-08-11
Branch: `lane/cx-kvcode`
Base: `0e890ccf`

## Scope

- Determine whether Speculative KV Coding reconstructs cached keys and values bit-exactly or is
  bounded-lossy; treat that distinction as the decisive gate.
- Map the proposal onto memra's current per-layer/head KV representation, paging, GGUF KV-cache
  format, and prefix-cache behavior using file-and-line citations.
- Pin predictor size, token-path work, execution placement, and applicable serving shapes from
  quoted public primary sources.
- Deliver only `FEASIBILITY.md`; do not implement, build, benchmark, run GPU work, merge, tag,
  push, or format the tree.

## Status

- [x] Read the lane brief and `/home/avifenesh/projects/bw24/CLAUDE.md` first.
- [x] Confirmed a clean worktree on `lane/cx-kvcode` at current-main base `0e890ccf`.
- [x] Traced memra KV storage, paging, GGUF KV-format, and prefix-cache surfaces.
- [x] Verified the proposal against current public primary sources.
- [x] Wrote the cited feasibility verdict and audited every claim.
- [x] Left the branch unpushed for orchestrator review.
