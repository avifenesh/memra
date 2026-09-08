# GLM KDA verify rows mechanism, 2026-09-09

Source: `dcfeab7c738912a150ebbfea277112724bb99de4`. Verdict: **SUPERSEDED MECHANISM**. rev: 2026-09-23

## Current PP verify dispatch

`crates/memra-engine/src/glm_spec.rs:228` makes `MEMRA_GLM5_VERIFY_BATCH`
default ON. At :1379 the batched arm requires t>1; :1481 calls
`kda_verify_rows_cached` once per KDA layer with t=K+1. The shared range body
at :1357 is used by the PP stages too; this is not a separate single-device
optimization that PP omits. Explicit =0 selects the old per-row rollback seam.

`src/kda.rs:1103` passes `ConvArm::Prefill` and `KdaStash::Rows` to the cached
mixer. :444-456 applies the six input projections through rows-exact matmuls
when the fused group declines. :531-546 issues three t-row
`memra_kda_conv_silu_f32` kernels, then three t-row
`memra_kda_conv_ring_roll_f32` kernels. :565 folds the two L2 norms into one
`l2_norm2_f32`; :610 folds forget gate and beta into one
`memra_kda_gate_beta_f32`. :648 issues **one** `memra_kda_scan_s128` over all t
rows. The output gate projection, gated RMSNorm and output projection follow.
Projection launch counts depend on weight class and width and are not inferred
from a matmul API count. This source census counts the state/core kernels only:
3 conv + 3 roll + 1 L2 pair + 1 gate/beta + 1 scan = 9 per KDA layer per round,
excluding output norm, projections, snapshots and copies. The proposed scan
changes none of these counts.

`cu/kda.cu:217-263` is the operative kernel body, not just a comment:
`s_shard[4]` loads the recurrent state before the `for (t<T)` loop (:229),
the loop at :233 applies each row in order, and the final state stores after
it at :259. Per row: per-channel decay; memory dot and warp reduction; delta;
state update; readout dot and warp reduction. Launch at `src/kda.rs:1537`
uses grid (64,1,32), block (32,4,1), no dynamic shared memory. Four state
floats per lane remain resident across rows. The public wrapper is :267.

| t | Current scan launches/layer | Proposed launches/layer | Current full-trunk scan launches | Launches removed |
|---|---:|---:|---:|---:|
| 2 | 1 | 1 | 34 | 0 |
| 4 | 1 | 1 | 34 | 0 |
| 7 | 1 | 1 | 34 | 0 |

The model has 45 trunk layers: 34 KDA and 11 MLA. The prior cells' 42 is the
routed-FFN count, not the KDA count. No 42-layer KDA oracle can be claimed.

## Reject and state custody

`glm_spec.rs:1464-1476` snapshots pre-round SSM state once per layer. The
rows stash (`kda.rs:233-258`) retains raw q/k/v, normalized scan inputs and
the pre-round convolution ring; it does not clone SSM state after every row.
Full acceptance leaves the advanced state in place (`glm_spec.rs:2225`).
Partial acceptance calls `kda_verify_rollback_rows` (:2234).
`kda.rs:1140-1218` restores the old ring, re-rolls the kept raw rows and issues
one scan with T=keep from the pre-round snapshot, then swaps SSM buffers.
This already reconstructs the accepted prefix without t separate scan launches.

## Qwen packed GDN template and numeric discipline

`docs/KERNELS.md:123-125` documents packed conv/prep/scan. The operative
`cu/hybrid.cu:1549-1610` keeps state across rows too, but emits intermediate
snapshots and resolves canonical/alternate state from a replay-refreshed
pointer table. That fixes the old GDN verify per-row launch loop. GLM already
has the corresponding one-launch scan and uses input replay for rollback.
Copying Qwen's snapshot slab would add a different memory tradeoff, not remove
the hypothesized t launches. KDA decay is per channel; GDN decay is per head.
Do not transplant GDN's scalar-decay algebra into KDA.

`tests/glm5_verify_batch_gpu.rs:148` already defines the synthetic scan-chain
byte gate, checking outputs and final state plus a swapped-row red arm.
It is prior test source, not a freshly run real-input oracle. No arithmetic,
compiler flags or cross-file twins change in this cell. A new UT/WY algebraic
chunk transform would change summation order and needs its own declared
numeric class and real-input oracle; it is not the requested exact launch fold.
The previously shelved `MEMRA_KDA_CHUNKED` is described in `kda.rs:13-28` and
is not revived here.

## Decision

The requested single t-row register-state scan is already the default program.
No `MEMRA_GLM5_KDA_VERIFY_ROWS` env read, dispatch or twin is introduced.
Its proposed default OFF / decide-by 2026-09-23 is closed before implementation,
recorded in the Removed doors ledger as never introduced. Testing a renamed
copy or =0 rollback seam as if it were today's baseline would misprice this
candidate. A future KDA optimization needs a distinct measured mechanism.
