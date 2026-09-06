# Plain decode: GU-M1 and per-layer projection graphs

The development baseline is DSV4's matrix request program with whole-expert
EP, FP8 dense weights, tiled index/sink, device routes, active host C4,
GU fusion and m1 down, with route/mirror host validation disabled for timing.
Normal DSV4 decode does not enable the earlier graph probes automatically.

Two default-disabled, explicit gate candidates are included:

- `set_moe_f16g_gu_m1_tc_for_gate` selects a valid-row specialization of the
  fused GU visitor. It retains the f16-MMA order and final epilogue and skips
  duplicate A loads and the two warps computing unused output rows.
  `moe_f16g_gu_m1_tc_dispatches` counts calls at the actual shared EP/non-EP
  dispatch. `cuda_gu_m1_matches_gu_reference` compares all H and routed output
  bits; the 2x RTX PRO 6000 component runs each passed with zero memcheck errors.
- `arm_stateless_prefix_graph_probe_for_state` captures HC pre and Q/KV
  projections, norms, RoPE and QAT in all 43 layers. Device positions are
  refreshed before each round. Capture ends before transient KV writes, so
  compression, indexing, C4 gathering, EP exchange and commit execute normally.
  Per-state graph ownership preserves all source/scratch lifetimes.

`dsv4_plain_perf_gate` compares each candidate and their combination using one
model load, the existing pinned source and 256/8192-token snapshots. It emits
only sampled plain timing rows, six per arm in ABBA order, after a full
output/logits/committed-KV identity comparison. Each row asserts the expected
both-rank GU-M1 count and all-layer graph capture/replay count. Total decode
wall includes capture and excludes restore; timestamps also allow the
post-capture token interval to be reported separately. Explicit parameter
checks run before model loading. No 1M capacity run is repeated.

The model test binary SHA256 is
`37e0792c89d8c40c76a61b240937e3e6104fcf7a279c1970a161bfba0930cb3e`;
the component test SHA256 is
`b4f4e06e6cd64cb75ec8b06c526473b24e6263296302cb91682a1649f2a80a7e`.
Full-model results are pending this checkpoint. No serving defaults change.

Current external reference: FlashInfer commit
`6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f`,
`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py`,
documents a persistent small-M W4A16 path. It is a design reference only;
no third-party kernels or runtime libraries were added.
