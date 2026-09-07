# DSV4 same-stage multi-layer graph probe, 2026-09-06

The one-layer graph seam now has a smallest multi-layer extension. The gate
selection is a sorted, contiguous pair of ratio-0/window-only layers on the
same PP stage. Each layer keeps its own CUDA graph, but both are captured and
replayed in round order on the same stream. This preserves the real dependency
between layer `i`'s `h_a` output and layer `i+1`'s input without pretending a
single-stream graph owns a PP peer copy or event.

The probe refreshes the persistent token and position device arrays before each
round. Matrix device routing reads the refreshed token array, while the
window-only redirect and RoPE read the refreshed position array. The ring
commit remains eager and is checked after each graph round through `state.pos`,
route buffers, and all live cache classes. The graph arm is diagnostic-only.

The arm refuses EP, PP-crossing selections, compressor/indexer layers, C4
host residency, and route/mirror validation. It does not capture commit,
compressor replay, host copies, or multi-device dependencies. Those remain the
next blockers for a full round executable.

## Compile evidence

`cargo check -p memra-engine --bin dsv4_graph_probe_gate` passed after the
multi-layer patch. The build initially exposed an unrelated active small-M
CUDA edit: `moe_f16g_sk32v_kernel` referenced undefined `DIRECT_OUT`; restoring
its original `Y + ...` destination expression fixed that compile blocker.

## Pair receipt

The release binary `f090628e7afc6c9d113461e5a576044fb4eb7d7214dd673f9a94a7ffe8bcbb07`
passed the explicit `graph-multi-probe` mode on the locked development pair.
Receipt: `graph-multi-probe-20260906.log`, SHA256
`2ca8524bddc7d1b474f7a320ae9f92804ea672280d85464a72bded846d416355`.
Capture and replay output, compact routes, and live-KV classes were all exact;
the process exited 0. The live scalar lines were:

```text
LIVE_SCALARS capture token=1706 pos=256 commit_state_pos=257 selected_layers=[0, 1]
LIVE_SCALARS replay token=337 pos=257 commit_state_pos=258 selected_layers=[0, 1]
```
