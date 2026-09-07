# Gate-only attention TP2 vertical slice

This connects actual head-parallel attention to the all-layer expert-ID EP walk. It is not a serving default, performance result or full-width numerical identity claim.

For the 64-head, 8-output-group configuration, each rank uses:

| Projection | Local shape | Checkpoint partition |
| --- | --- | --- |
| Q_b | 16384 x 1024 | 32 contiguous output heads |
| wo_a | 4096 x 4096 | 4 contiguous output groups |
| wo_b | 4096 x 4096 | Half of each input row, repacked with its FP32 scale columns |

Geometry is derived and validated from normalized configuration. Each rank receives its own sink-head offset. Shared latent KV, compressors and indexer remain replicated. Full FP8 source planes are retained beside the packed planes in this first slice; no VRAM-saving claim is made.

The first slice uses the component-qualified per-group `wo_a` GEMV. A packed four-group `grouped_m1` launch is a different shape and remains disabled even if a caller arms the old full-attention grouped accelerator. Its distinct-input/canary component gate is still pending; generic projection packing does not qualify it.

The paired step runs both `attention_verify_dev` producers, reduces their 4096-element output partials with the existing native out-of-place rank-order primitive, then enters `post_attention_moe_verify_dev` for attention HC post and FFN preparation. The existing expert join/shared tail follows. A head-sharded request cannot use the single-rank block wrapper or silently fall back to replicated attention.

The numerical class is `dsv4_attention_wo_b_input_split_f32_rank_reduce`. The two actual GPU partials and both joined outputs are preserved separately from MoE buffers. The gate reads the final layer after each successful token and requires every joined f32 bit to equal canonical CPU `rank0 + rank1`; it does not compare that sum to the old full-width accumulation or invent a tolerance.

Attention and MoE reuse the same sticky reduction error plane. The six negative cells inject 40043/40044 immediately after a selected real attention join and require unchanged cache/position plus a poisoned retry. The injection is armed before the walk and consumed through the internal AR-state helper, without recursively acquiring the walk lock.

`set_attention_tp_for_gate` is intentionally OFF. The `dsv4_tp_ep_gate` selector is `MEMRA_DSV4_ATTENTION_TP_GATE=1`, with the ordinary plain all-layer matrix-EP gate settings otherwise unchanged. The existing default arm retains the previous numerical sequence. Batch, drafter and serving qualification remain outside this slice; MoE graph composition is a subsequent source composition.

Source `8a9184ab4ab135c0e0e27a0c4eeb05898bde72b6` builds remotely, its two geometry CPU tests pass, and scoped release Clippy passes with warnings denied. Binary SHA256: `0bc3c9bc0b9b00e2f816a8d3be4ec18b215ab062f12a423e0f314959dd702572`.

The model gate passed on the two RTX PRO 6000 Blackwell devices at 2026-09-07 17:36:52 UTC, controller status 0. Eight actual final-layer joins (four positions, two fresh-state repeats) matched canonical CPU f32 addition of both actual GPU partials bit-for-bit. Both rank cache/hidden planes stayed equal, and all six attention-specific refusal cells passed rollback and poisoned-retry assertions. Each repeat recorded 172 attention calls per rank, 172 attention reductions, 344 total attention-plus-expert reductions and 344 local expert calls. New-class output SHA256: `bd13bc55fd7fe7a6dc1829b036200a768d6d344b94ca45d8747da76b5835d7f2`; raw gate-log SHA256: `2b8f6dfd1d80d5d30f7adc83b7611edd8c48d05b9591f8f09427e52f44080e97`. Companion receipt namespace: `attention-tp-model-8a918-r1`.

That binary's gate never arms the global grouped `wo_a` control, so the measured configuration used the qualified per-group projection. Later source `41fd34f7e18cfe04b3e4dbd98185ee4e7ba837e0` makes this restriction explicit for every head-sharded caller and adds a no-grouped-dispatch assertion; its final-head checks remain required. These are internal-consistency and native-join correctness receipts, not full-width identity, model-quality, serving or throughput claims.

No local CI, build or GPU work was performed. Pushes use the owner-authorized `MEMRA_SKIP_PERF_CI=1`; hosted checks and final target gates are not waived. The source remains intentionally default OFF.
