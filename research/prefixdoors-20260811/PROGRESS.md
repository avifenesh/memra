# cx-prefixdoors — progress log

Lane: `lane/cx-prefixdoors`, created from `main` at `0e890ccf`.
Deliverable: `FEASIBILITY.md` — scope HiRadix and CacheBlend as extensions to
memra's shipped PrefixCache, with losslessness and tenant isolation as the
decisive questions.

## Constraints

- CPU-only, read-only research; documentation artifacts are the only writes.
- No GPU, build, merge, tag, push, formatting command, or performance claim.
- Every feasibility claim must cite a memra file and line or a quoted external
  primary source.

## Work log

- [x] Read the lane brief and `/home/avifenesh/projects/bw24/CLAUDE.md` first.
- [x] Confirmed a clean dedicated branch/worktree at the requested base commit.
- [ ] Map PrefixCache keying, LCP matching, eviction, paged reuse, and metrics.
- [ ] Research HiRadix and CacheBlend from current public primary sources.
- [ ] Write and citation-audit `FEASIBILITY.md`; leave the branch for review.

Next: capture exact memra implementation receipts, then evaluate each external
door against the existing PoolKey and paged-KV contracts.
