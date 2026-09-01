# Large models on multiple GPUs

| | Recommended use |
|---|---|
| **Start with** | Step 3.7 Flash on the qualified 2× RTX PRO 6000 Blackwell PP-2 path |
| **Topology** | Pipeline parallelism for the published Step configuration; PP-3/PP-4 wavefront is experimental |
| **Validate** | Placement, cross-device transport, admission, and end-to-end output on the exact topology |
| **Read next** | [Step model card](../models/step37-flash.md), [Serving: PP-2](../SERVING.md#pipeline-parallel-pp-2-serving) |

Pipeline stages, tensor parallelism, expert parallelism, and replicas are different execution
shapes. Use the name and gate for the topology actually running; do not call PP or replicas TP.
For the 2–4 card decision procedure and pending qualification matrix, see
[RTX PRO 6000 multi-card placement](../decisions/PRO6000-MULTICARD.md).
