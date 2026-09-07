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
Raw namespace `plain-composed-nsys-20260907-r1`. Execution is in progress;
no new critical-path or model-rate conclusion is recorded yet.
