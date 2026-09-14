# Current composed plain-decode profile

The `dsv4_decode_profile composed` diagnostic fixes the exact tested plain
composition: GU fusion plus M1 work elision, packed GU/down half2, grouped
wo_a, radix index top-k, and route/mirror validation disabled. Prefill and
snapshot/restore happen before these overrides are armed. No speculative
generation or new runtime/environment door is added.

It retains the existing frozen 8K prompt, 96-token sampled stream and
capture of decode steps 32..64. The whole model is drained at capture
boundaries. Actual GU-M1/GU-half2/down-half2/grouped-wo_a/index-topk
enqueue deltas accompany EP transaction counts; final assertions verify
all 96 steps engaged the requested composition. The fixed sampled token
hash must match the existing baseline protocol. Profiled seconds are
instrumented evidence, never a throughput row.

Build and two CPU tests pass; clippy passes with `-D warnings`.
Binary SHA256:
`fffb8c80844a553774ecd87423c345bd88f727c172231905903f9f9eb6c64d56`.
Compressed transport SHA256:
`25542ee9da93b97d554856123cc80cc5d7a193cb917c13ce9834f87570dc05bb`.
Raw namespace `plain-composed-nsys-20260907-r1` finished at
2026-09-07T08:20:12Z, exit 0. The frozen sampled stream passes, with 4,128
EP transactions over 96 steps, GU-M1/GU-half2/down-half2 each 8,256,
grouped wo_a 4,128 and radix top-k 2,016 actual enqueues.

The trace has 32 observed embed anchors and 102,600 kernels. Profiled wall
is 33.88358 ms/step; the union of either device's kernel intervals is
27.16720 ms/step, leaving 6.71638 ms/step without a traced kernel. The
latter includes copies, CPU/API, dependencies and untraced work, not
automatically removable bubbles. These profiled timings are not tok/s.

Aggregate work over both devices per step: GU 8.52258 ms, ordinary/grouped
dense FP8 GEMV 5.23277 ms, down 3.70066 ms, dots 2.65086 ms, HC Sinkhorn
1.66154 ms, RMS 1.60963 ms, sink score/output 3.00529 ms, and C4 gather
1.18306 ms. These overlapping device sums are not additive wall savings.
The 200 D2H calls total 16,891,904 bytes and 1.51082 ms device-copy time,
but 319.22321 ms CPU API duration over the entire capture. Removing those
bytes does not remove the dependencies the host waits for.
