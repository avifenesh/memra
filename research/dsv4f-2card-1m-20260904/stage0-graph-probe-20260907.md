# PP stage-0 local graph probe, 2026-09-07

The next smallest PP-relevant graph segment is the stage-0 input prefix:
`embed_rows` followed by `repeat_hc`. It has one live input, the refreshed
device token row, and no KV/compressor/indexer/router/EP/PP-boundary/commit
dependency. The explicit `graph-stage0-probe` mode captures and replays those
two kernels while the whole trunk and stage boundary remain eager.

This is the first stage-local launch-collapse candidate after the exact
window-only pair and head probes. It tests the PP finding at the smallest safe
boundary without pretending to graph the cross-device peer copy/event. The
expected gate receipt is:

```text
GRAPH_CENSUS ... kernels=2 ...
IDENTITY capture output=true route=true kv=true ...
IDENTITY replay output=true route=true kv=true ...
PASS bounded stage-0 embed graph probe capture/commit/replay exactness
```

No GPU run was performed. Compile evidence:

```text
cargo check -p memra-engine --bin dsv4_graph_probe_gate
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3m 40s
```

The next larger stage-local graph would need to absorb stage-0 trunk layers,
but ratio/compressor/indexer state blocks that. A cross-stage graph would need
a multi-device graph/event contract and is intentionally not attempted.
