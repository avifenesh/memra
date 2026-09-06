# Gate-only device-head graph, 2026-09-06

The next graphable plain-decode segment is the final device head, after the
full trunk has completed eagerly. `head_logits_batch_dev` runs device-only
head mixing, collapse, normalization, and vocabulary dots into persistent
workspace buffers. It does not read or mutate KV/compressor/indexer state,
route metadata, C4, EP, or PP boundary state.

The explicit `graph-head-probe` mode captures this segment on one token and
replays it on the next token. It retains the same exact output/route/KV
comparison used by the two-layer probe. The expected receipt is:

```text
GRAPH_CENSUS head ...
IDENTITY capture output=true route=true kv=true ...
IDENTITY replay output=true route=true kv=true ...
PASS bounded head graph probe capture/commit/replay exactness
```

This is gate-only and compile-checked; no GPU run was performed in this step.
It is not a serving default or a full-round graph. The remaining launch/sync
blockers are the eager trunk's compressor/indexer state machines, host-C4
gather, EP/PP dependencies, and commit's slot-row upload plus compressor
rollback. The head graph cannot remove those launches; it only removes the
head launch chain after them.

Compile evidence:

```text
cargo check -p memra-engine --bin dsv4_graph_probe_gate
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.71s
```
