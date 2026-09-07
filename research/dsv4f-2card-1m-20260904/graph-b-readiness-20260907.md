# Retained attention-tail graph readiness

Implemented, default OFF, not GPU-qualified. The existing full-layer capture
instrument remains separate. Graph-B captures the post-C4 sink-attention,
output projection, attention HC-post, FFN HC-pre and FFN normalization body.
Routing, matrix EP, C4 gather, PP transfer and commit retain their eager path.

Admission requires a closed native device t=1 matrix state with FP8 dense,
device top-k and current context/capacity at least 8192. Stable device position
contents change without changing graph identity. The key includes actual
attention slots, layer/stage, pointers and math arms, including grouped wo_a;
compressed-block counts are deliberately not a key field. Coarse-attention
slot changes get bounded variants; pointer replacement invalidates the layer's
retained variants. Graphs drop before workspace and cache storage.

`set_graph_b_for_state`, `clear_graph_b_for_state` and
`graph_b_stats_for_state` are explicit per-state diagnostic APIs. The stats
include actual captures/replays/invalidations, per-variant node/kernel census,
and wo_a replay nodes counted from retained kernel names after successful
graph launches. The existing eager FFI counter is not fabricated on replay.
The first capture executes once and its cost remains inside the rate gate.

Validation on the implemented source:

- Release library check: PASS.
- Release library CPU tests: 413 passed, zero failed; 15 CUDA tests ignored.
- Release library clippy with `-D warnings`: PASS.
- Rustfmt and diff checks: PASS.

The sampled full-model harness is being extended. It must hold existing
half2/grouped-wo_a/indexer winners fixed, leave short context inert, check all
43 layers through real variant/census entries and assert
`captures + replays == 43 * decode_steps`. It must also check
`wo_a_eager_calls + wo_a_graph_replay_nodes == 43 * decode_steps`, plus
complete sampled-token/final-logit/committed-cache identity. Actual captures
depend on coarse-attention slot variants; 43 is not a hardcoded capture count.
GPU sanitizer, identity, coverage and performance verdicts remain pending.
