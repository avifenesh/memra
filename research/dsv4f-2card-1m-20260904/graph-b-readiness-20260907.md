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

The actual DSV4 config has `sliding_window=128` (`DeepSeekV4Config` and the
DeepSeek-V4 model pack), with `HD=512`. Therefore the live slot classes are
window-only `128`, fine/indexer `128 + min(index_topk, n_blocks)`, and coarse
ratio-128 `128 + n_blocks` (approximately 192..194 over the 8K→8K+256
window). C4's `640` value capacity is the maximum `128 + 512` bound, not a
512-slot live window.

The C4 source audit confirms that base `C4HostStore::gather` is not a CPU row
gather: after host scalar checks and pointer lookup it launches the GPU
`memra_dsv4_c4_gather` kernel, which reads the device index list and mapped
pinned-host rows. It remains an eager Graph-B boundary because producer D2H,
live-row/high-water state and shape parameters are outside the retained body;
that is an implementation boundary, not a categorical CPU/mapped-host
limitation. The recent-sidecar path additionally performs a host
`bind_runtime` prelude before its GPU kernel.

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

The first model run reached 63 retained variants but stopped on a coverage
checker bug: lowercased driver names were compared with case-sensitive C++
template encodings. R2 preserves symbol case, prints the raw census before
assertions, and adds three Rust binary regression tests. The library suite is
now 414 passed / 15 ignored; the reader has 26 CPU tests. R2 model binary
SHA256 `e617b6663119ed80ef19e0a8571e0279b59651c9f4ab5136e66ac5e47a261136`.
The incomplete R1 run is not an 8K exactness or performance verdict.
