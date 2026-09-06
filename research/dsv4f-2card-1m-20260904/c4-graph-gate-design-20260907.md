# Active C4 graph gate design, 2026-09-07

## Verdict

`C4HostStore::gather` is graph-capturable as a component when its gather
workspace is already allocated and the launch shape is fixed. The existing
ignored CUDA test `cuda_recent_c4_preserves_hits_misses_wrap_and_rollback`
captures a recent-sidecar gather and replays it four times while changing the
canonical host contents/tags, then compares graph and eager outputs bitwise.
That is component evidence, not full-layer DSV4 graph qualification.

The full-layer `c4_host` refusal must stay. `block_verify_dev` still runs the
compressor/indexer state machine before gather, and layer 2's host `n_blocks`,
emission phase, store row, and commit/rollback state are not graph-live. A
component-safe gather does not make that layer safe.

## Fail-closed component gate contract

A future `graph-c4-gather-probe` may capture only the gather component after
these predicates pass:

1. `MEMRA_DSV4_PROFILE_C4_ELIDE`/the gate setter is ON and the recent sidecar
   covers the complete admitted C4 capacity (`recent.tags.len() == host.rows`).
   This removes host-publication DMA from the captured body; the recent values
   and absolute tags remain device-resident.
2. The live prefix tags are validated before capture (`tag[row] == row` for
   every `row < live_rows`) and no restore/snapshot/host mutation is allowed
   while the graph is retained.
3. `C4Gather` is allocated before capture and its values/indices buffers are
   large enough for the exact fixed `(nq, slots, stride)` shape. `ensure` and
   all CUDA/runtime binds are outside the capture.
4. The graph key includes `nq`, `slots`, `stride`, `live_rows`,
   `logical_transient`, `transient_rows`, sidecar capacity, and the C4 arm.
   A block emission or top-k shape change must select a different graph or
   fall back eagerly; no scalar is silently baked across shapes.
5. Route/mirror validation is OFF for the surrounding graph experiment, but
   this gate does not capture route construction, compressor writes, EP, PP
   peer/event transport, or commit.

The component receipt should compare one eager gather against capture and
replay with re-emitted recent rows, checking output values, mapped indices,
padding guards, and absolute-tag hit/miss behavior. The existing test is the
model for that receipt.

## Why the full layer remains blocked

With elision OFF, `C4HostStore::write` queues D2H DMA into pinned host memory;
the pointer is stable, but the full layer also has host-controlled compressor
and high-water decisions. With elision ON, recent-sidecar writes are device
ordered, but `gather` still receives shape/high-water scalars from the host and
the surrounding layer still reaches `commit_verify_dev`'s host slot-row upload
and compressor rollback. Therefore the blanket layer refusal is not removed;
only a separately keyed component gate is honest.
