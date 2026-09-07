# Graph-B: exact, flat, removed

The full sampled 28-row gate passes token, final-logit and committed-KV
identity. It proves 63 real retained variants, 882 retained kernel nodes,
and 10,902 replays per 255-step 8K row. Each variant has 14 kernels; the 882
count includes alternate coarse-attention shapes and is not a per-step count.
Actual wo_a FFI submissions are 63, plus 10,902 wo_a nodes on successful
graph replays. EP and indexer enqueues remain fully engaged.

Three ABBA cycles, six scored non-looping rows per arm/context, fixed
half2/grouped-wo_a/radix indexer, sampled T=1/p=1/k=0/seed20260906:

| Prompt | Eager tok/s | Graph-B tok/s | Wall delta | Post-capture delta |
|---|---:|---:|---:|---:|
| 256 | 32.20293 | 32.18989 | -0.0405%, inert | -0.0362% |
| 8192 | 28.53459 | 28.41624 | -0.41475% | -0.35762% |

Verdict: flat/no-go. Remove the performance door, state/maps/keys/stats,
dispatch branch, CLI mode and tests. Keep the existing older capture census
instruments and immutable historical receipts. No deployment/default change.
This is not a verdict on a full round or a multi-device EP graph.

Binary SHA256 `e617b6663119ed80ef19e0a8571e0279b59651c9f4ab5136e66ac5e47a261136`;
full log `e67e55db803899ed1e86269fdc06fe994ceffbbc055c15356b1cdccb7aa19ec0`.
Raw namespace `graph-b-model-20260907-r2`; controller exit zero at 06:20:07Z.
The earlier R1 failed only its symbol-case coverage checker and is not a
qualified performance row. R2 fixed the checker, preserved actual driver
symbols and emitted census data before assertions.

Do not attribute changes against older 33.15/30.96 windows to this graph:
the causal comparison is the matched eager-versus-Graph-B pair above.
Removal validation: 411 CPU tests pass, 15 CUDA tests ignored, release
library/binary clippy and diff checks pass. Full-round graph work remains
open; C4 is GPU mapped-host gathering after preallocation, not a categorical
CPU/VRAM barrier, and EP needs an explicit peer-stream capture contract.
