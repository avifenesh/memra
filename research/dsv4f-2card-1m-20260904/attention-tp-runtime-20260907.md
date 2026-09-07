# Gate-only attention TP2 vertical slice

This connects actual head-parallel attention to the all-layer expert-ID EP walk. It is not a serving default, performance result or full-width numerical identity claim.

For the 64-head, 8-output-group configuration, each rank uses:

| Projection | Local shape | Checkpoint partition |
| --- | --- | --- |
| Q_b | 16384 x 1024 | 32 contiguous output heads |
| wo_a | 4096 x 4096 | 4 contiguous output groups |
| wo_b | 4096 x 4096 | Half of each input row, repacked with its FP32 scale columns |

Geometry is derived and validated from normalized configuration. Each rank receives its own sink-head offset. Shared latent KV, compressors and indexer remain replicated. Full FP8 source planes are retained beside the packed planes in this first slice; no VRAM-saving claim is made.

The paired step runs both `attention_verify_dev` producers, reduces their 4096-element output partials with the existing native out-of-place rank-order primitive, then enters `post_attention_moe_verify_dev` for attention HC post and FFN preparation. The existing expert join/shared tail follows. A head-sharded request cannot use the single-rank block wrapper or silently fall back to replicated attention.

The numerical class is `dsv4_attention_wo_b_input_split_f32_rank_reduce`. The two actual GPU partials and both joined outputs are preserved separately from MoE buffers. The gate reads the final layer after each successful token and requires every joined f32 bit to equal canonical CPU `rank0 + rank1`; it does not compare that sum to the old full-width accumulation or invent a tolerance.

Attention and MoE reuse the same sticky reduction error plane. The six negative cells inject 40043/40044 immediately after a selected real attention join and require unchanged cache/position plus a poisoned retry. The injection is armed before the walk and consumed through the internal AR-state helper, without recursively acquiring the walk lock.

`set_attention_tp_for_gate` is intentionally OFF. The `dsv4_tp_ep_gate` selector is `MEMRA_DSV4_ATTENTION_TP_GATE=1`, with the ordinary plain all-layer matrix-EP gate settings otherwise unchanged. The existing default arm retains the previous numerical sequence. Batch, drafter and serving qualification remain outside this slice; MoE graph composition is a subsequent source composition.

Source `8a9184ab4ab135c0e0e27a0c4eeb05898bde72b6` builds remotely, its two geometry CPU tests pass, and scoped release Clippy passes with warnings denied. The controller finished with status 0 at 2026-09-07 17:17:24 UTC. Binary SHA256: `0bc3c9bc0b9b00e2f816a8d3be4ec18b215ab062f12a423e0f314959dd702572`. Target model execution remains pending. No local CI or GPU work was performed. Pushes use the owner-authorized `MEMRA_SKIP_PERF_CI=1`; hosted checks and target runtime gates are not waived.
