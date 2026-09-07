# Two-device EP explicit graph mechanism

R5 passes memcheck and synccheck on the two RTX PRO 6000 peer devices.
The graph contains two kernels, two peer copies (one in each direction),
two event records and two waits. Two different input generations match
the eager reference, with all input/work/return/output guards checked.
Both reset streams synchronize before temporary host buffers leave scope.

The earlier R3 `cudaMemcpyPeerAsync` stream-capture attempt failed with
CUDA error 900 (cleanup 901). R5 uses explicit graph nodes and per-device
execution contexts. This is a supported working API route on this pair,
not proof that actual model EP, full-layer or full-round capture works.
No runtime default or performance claim follows from this micro-gate.
R4 was a never-run intermediate; R5 includes its reset/lifetime corrections.

Source `tools/dsv4-ep-graph-gate.cu` SHA256:
`853a8e1b168e2eeefa1f66a4bfaec08323eed22099306979d57d78586248abf3`.
R5 binary SHA256:
`124e22eaf3e7c04bca0f62673e6f9b1c4ded947d3409fab1a1ca62d19220dcac`.
Raw namespaces `ep-graph-{memcheck,synccheck}-20260907-r5`, terminal
2026-09-07T07:56:30Z. Both sanitizer reports contain zero errors.

Current primary references:
[kernel execution context](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaKernelNodeParamsV2.html),
[graph parameter and stream constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html).
The next integration must preserve route/input freshness, explicit peer
dependencies, persistent workspace lifetimes and host commit/rollback.
