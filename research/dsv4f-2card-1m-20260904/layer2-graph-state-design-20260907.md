# Layer-2 graph state design, 2026-09-07

The first ratio/indexer layer is not graphable by replacing `pos0` with the
existing `pos_dev` pointer alone. The current `cmp_decode_batch_dev` bakes
several pieces of mutable state into host control and pointer arithmetic:

- `pos0 + i`, `pos % ratio`, and `(pos + 1) % ratio` choose pending slots and
  whether a compressor block emits;
- `j = pos / ratio` chooses the compressed-store destination and RoPE position;
- `blocks: &mut usize` is the append-only high-water mark and is updated on the
  host after an emission;
- `emit` is copied into `store[(row0 + j) * d ..]` through a host-computed
  slice; and
- rollback/commit uses the same host high-water mark and host loop for pending
  replay.

The indexer side is only partly ready. Device top-k is already selected for
plain t=1, and the gate-only fine redirect scalar twin exists
(`graph-indexer-scalar-probe`), but the indexer compressor has the same pending
and high-water state machine. Active host C4 adds a host gather/copy target and
must continue to refuse.

## Smallest honest next design

Restrict the first full layer-2 graph to plain matrix t=1, `n_commit=1`, EP
off, no C4, one PP stage, device top-k, and native device math. Add a per-layer
device state bank:

1. `pos_dev[0]` (already present) and `blocks_dev[0]` for each attention and
   indexer compressor.
2. A device `emit_now`/phase scalar or one graph variant per
   `pos % ratio`; the capture cannot branch on the host `(pos+1)%ratio` test.
3. A device compressor step that copies the current row into the pending slot,
   runs pool/norm/rope/quant only on an emission phase, and writes the emitted
   row through `blocks_dev` into the device store.
4. A device high-water increment/read path. After a full t=1 commit the host
   `cache.n_blocks` can be derived from the committed `state.pos`; speculative
   rollback is explicitly out of this first graph shape.
5. The indexer redirect, score, and device top-k must consume the same live
   `pos_dev`/`blocks_dev` state. The existing fine redirect scalar kernel is the
   first component of this chain, not layer qualification.

The graph key must include `(layer, ratio, overlap, phase, C4=off, EP=off,
PP-stage, dense/math/topk arms)`. Capture and replay stop before
`commit_verify_dev`; commit remains eager until the device high-water state and
slot-row write are separately exact.

## Current exact blockers

`cmp_decode_batch_dev` still owns all five host decisions above, and
`cmp_rollback_replay_dev` repeats them for commit. `C4HostStore::gather` is a
host destination/copy graph boundary. EP dispatch and the PP boundary event
cross CUDA contexts/streams. Therefore no source change in the current
`block_verify_dev` call can safely admit layer 2; the next receipt should be a
component bit-identity gate for the device compressor step plus high-water
update, followed by a t=1 layer-2 capture/replay gate.

