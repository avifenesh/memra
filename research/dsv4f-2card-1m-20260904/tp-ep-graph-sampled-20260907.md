# DSV4 TP/EP paired-MoE graph: removed door receipt

Decision: NO-GO. The flat rank-local MoE graph island is removed; this receipt is
technical evidence, not a serving or public performance claim.

## Matched A/B

- One model load on the two-card RTX PRO 6000 development pair.
- 256 source-token prime, then 256 vendor-shaped sampled tokens: temperature 1,
  top-p 1, top-k 0, seed 20260907.
- Five fresh-state pairs, alternating graph-to-eager and eager-to-graph.
- All five pairs were eligible and non-looping. Generated tokens, final logits,
  cache digests and hidden digests were frozen-identical.
- Eager decode mean: `37.77 tok/s` (`6,777,508,739 ns`).
- Graph decode mean: `38.03 tok/s` (`6,732,397,396 ns`).
- Delta: approximately `+0.67%`, flat for the owner decision.

The graph captured only local route/GU/down/AR/shared-tail work. Attention,
cache and indexer remained eager. This is not evidence for a future full-round
graph that also captures the attention and transport schedule.

## Binding

- Candidate source: `606adad4bea367b8a1df451cf21ea4a6027fdad9`.
- Candidate binary SHA-256: `0cc724299701a17979ada946cdb4315a3feb0230848792782e820afd63fc6a33`.
- Raw log SHA-256: `0613b40cfab4426e7b6aa55da5a8b9a5aec328b54dbc525c0f23c7a2d92b7288`.
- Source tape SHA-256: `f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded`.
- Raw log and full candidate patch are retained in the companion ops/archive
  namespace; no customer or production claim is made from this cell.

## Historical baseline boundary

The older `24.3093 tok/s` value is the corrected serial receipt from source
`a4cd89c72`, binary `9d49...`, raw log `2cf7...`. The worker candidate was
`23.7555 tok/s` from source `eef64ffd3f...`, binary `81cd43a1...`, and is not
the 24.3093 baseline. The current 37.77 tok/s result is a different later
binary/configuration window. The cross-window change from roughly 24 to 37 is
unattributed here: no source/configuration guess, clock claim or power claim is
being made.
