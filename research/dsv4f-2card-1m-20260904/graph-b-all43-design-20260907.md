# Graph-B all-43 design, fixed 8K sampled plain

Status: design only. No `dsv4_gpu.rs` edit, build, or GPU run in this lane.

Source checkpoint audited: DSV4 worktree `86efe899ac631be869f4d74b592377c738f4ec16`
(the lane was `b261ee6e` when this audit began). The separate Memra root `main`
used for provenance is `6464a1604da85c83707b7c033103668fda24b7bd`.

The pinned DeepSeek-V4-Flash artifact/config has `sliding_window=128`, not
512 (`crates/memra-gguf/src/model_packs/deepseek_v4/mod.rs` and
`crates/memra-gguf/src/dsv4.rs`); the runtime DSV4 constant is
`WIN=128` in `crates/memra-engine/src/dsv4_c4.rs`. C4's `HD=512` is the
head-width row size. Consequently the maximum live gather shape is
`window 128 + index top-k 512 = 640`; 640 is the preallocated capacity, not
the normal window-only live slot count.

## Segment boundary

Graph-B is one retained graph per trunk layer, on the layer's owning PP-stage
stream. It is intentionally not a graph for the whole layer and does not cross
either CUDA context.

Start immediately after the eager `C4HostStore::gather` decision in
`crates/memra-engine/src/dsv4_gpu.rs:13009-13025`, when the two stable pointers
`attention_kv` and `attention_indices` are available. Capture through the code
ending immediately before `self.moe_verify_dev(...)` at
`dsv4_gpu.rs:13201`:

```text
sink_attn_dec_mq[_f32acc/_tiled]
rope_o
cvt_o
wo_a grouped m1 or the existing per-group GEMV loop
wo_b GEMM
hc_post(attention)
hc_pre(ffn)
rmsnorm(ffn) -> vws.xf
```

The capture ends before matrix routing/EP. `moe_verify_dev` and its output
become the eager island between Graph-B and the future post-MoE segment.
This is materially larger than the removed ten/fourteen-kernel expert-only
graph: it owns the attention tail, output projection and both hyperconnection
normalization halves for all 43 layers, while retaining explicit EP/C4/PP
boundaries.

## What is eager and why

- C4 gather remains before capture. The source audit verified that the base
  `C4HostStore::gather` path does not scan rows on the CPU: after scalar shape
  checks and pointer acquisition it launches `memra_dsv4_c4_gather`, whose GPU
  kernel reads the device index list and mapped pinned-host rows. It is kept
  eager here because the producer D2H, live-row/high-water state and shape
  scalars still belong to the surrounding compressor transaction; this is an
  implementation boundary, not a categorical CPU or mapped-host barrier.
  `C4Gather.values/indices` must be pre-sized for the 8K shape and overwritten
  in place each token. A shape/pointer change invalidates Graph-B.
- `moe_verify_dev` remains eager. Matrix EP executes owner/peer copies, event
  record/wait, peer gate/up/down, return copy and owner merge in
  `dsv4_ep.rs:610-691`; it is a two-context boundary, not a node inside the
  owner-stage graph.
- PP stage transfers remain eager. The caller's stage transition performs the
  peer hidden-state copy and boundary event wait in `dsv4_gpu.rs:14029-14069`.
- Commit/rollback remains eager. Graph-B ends before MoE and does not own the
  compressor/indexer high-water state or committed cache publication.

## Live values versus baked values

The first gate fixes `t=1`, the 8K starting position, `idx_stride`, top-k,
ratio/overlap class, dense/math/top-k arms, and active C4 mode. `slots` is a
runtime shape bucket, not a universal 512:

```text
ratio=0/window-only:                slots = win = 128
fine/indexer layer:                 slots = 128 + min(ix.topk, n_blocks)
coarse ratio=128, at ~8K:            slots = 128 + n_blocks
                                     n_blocks ~= 64..66 over a 256-token run
                                     => approximately 192..194 live slots
```

The exact formula is the `block_verify_dev` path at
`dsv4_gpu.rs:12580-12684` and `dsv4_gpu.rs:12925-12945`; capture must use the
actual `slots` observed for that layer/position bucket. `idx_stride` is the
preallocated workspace stride (`dsv4_gpu.rs:11567-11571`), not the live
selection width.

Values refreshed before every replay into those same allocations:

- `tok`/`pos_dev` device arrays are updated by the existing per-round HtoD
  writes before the trunk loop (`verify_batch_dev` around `dsv4_gpu.rs:13951-13962`).
- `attention_indices` is rebuilt eagerly; `C4Gather.indices` and `values` are
  overwritten eagerly before Graph-B.
- `vws.x`, `vws.q`, `vws.kv`, `vws.o`, `vws.xf`, `vws.h_b` and all Graph-B
  outputs retain their VerifyWs allocations and addresses.
- `h_in_ptr` is not uniform at a PP boundary: the first layer after the
  transition reads `vws.h_rx`, while later layers read `vws.h_a` (the selection
  is at `dsv4_gpu.rs:12410-12414`). It is therefore a required pointer-key
  field, not a cosmetic alias.
- Position, compressed block count, route IDs, route weights, EP contributions
  and C4 host rows are not Graph-B parameters. They are consumed either by the
  eager prefix/C4/EP islands or through stable device buffers.

The key must include:

```text
(segment=GraphB, stage, layer, t=1, topk, actual_slots, idx_stride,
 ratio, overlap, C4 mode, EP mode, matrix mode, dense arm, math arm,
 device-topk arm, h_in/pos/index/C4/xf/y/output pointer identities)
```

`pos` values, `n_blocks`, route contents and C4 contents must not be included
as scalar key fields. If an allocation is replaced, the pointer key changes and
the retained executable must be discarded/re-captured.

## Required API composition

`dsv4_graph.rs` should expose a segment label/key and a capture helper retaining
the existing event-tracking refusal, node census, upload and one capture-time
execution. `VerifyState` should retain a map of `(stage, layer, GraphB key)` to
the graph plus its workspace keeper. The `block_verify_dev` owner then composes:

1. eager prefix and C4 gather;
2. `capture_segment(GraphB, ...)` on first use or `replay()` on later use;
3. eager `moe_verify_dev`/EP and merge;
4. existing eager post-MoE/commit path.

No node update is needed for the first fixed-shape gate. If a later shape bucket
must reuse an executable, use CUDA's individual kernel/memcpy node update APIs
only for same-context allocations; otherwise recapture. CUDA documents both
individual node updates and topologically identical whole-graph updates, with
same-context restrictions on memcpy node updates:
[CUDA Graph update guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Gate and coverage receipt

The new graph gate should run the actual matrix+EP+C4 sampled plain program at
fixed 8K and `t=1`, with route/mirror validation disabled only as already
required by the matrix performance arm. `C4Gather::ensure` must be called once
for the maximum fixed shape before the first capture: it reserves the bounded
capacity `nq * 640 * HD` value slots (128 window + 512 top-k maximum) and
`nq * idx_stride` index slots
(`dsv4_c4.rs:350-375`). The gate must not let a first replay allocate or replace
those buffers.

It must:

1. Capture Graph-B for all 43 trunk layers, split by owning PP stage and
   actual shape bucket. The expected count is
   `sum(unique(layer, stage, h_in_ptr, actual_slots, arm-key))`, not blindly 43.
   Ratio-0 layers normally contribute one 128-slot variant; fine layers normally
   contribute one `128 + min(topk,n_blocks)` variant; coarse ratio-128 layers
   may contribute three variants around 192..194 slots over the 8K→8K+256
   position window.
2. Emit `GRAPH_CENSUS layer/stage/segment` with kernel and memcpy node names;
   the census must include sink attention, output projection, HC post, HC pre
   and FFN norm nodes, not merely expert kernels.
3. Run eager-versus-Graph-B ABBA for at least three cycles. Compare logits,
   sampled token, route IDs/weights, `xf`, EP contribution/merge output, all
   live C4/cache classes and committed state hashes.
4. Prove the observed variant count and per-bucket replay counts, zero stale
   fallback, zero eager Graph-B preparation after capture, and stable
   `h_in_ptr`/C4/workspace pointer identities. A slot change must select an
   existing variant or recapture; it must never update a 128-slot graph in place
   for a different live-slot shape.
5. Red-test C4 reallocation, EP stream capture status, PP boundary crossing and
   changed `slots/idx_stride`; each must refuse/re-capture, never replay stale
   pointers.

This gate is a full-coverage local segment gate, not yet a full-round graph
claim. The next extension is a post-MoE Graph-C segment. A true ratio/indexer
full-round graph still requires the device `blocks_dev`/phase/high-water state
described in `layer2-graph-state-design-20260907.md`; commit remains outside
until that state is bit-identical and rollback is device-owned.
