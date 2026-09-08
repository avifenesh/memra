# TP/EP issue interleave

Status: experimental, default OFF. Decide-by: 2026-09-22. No kernel or numeric change.

At base f80553700, `dsv4_gpu.rs:9553-9603` submits rank 0 attention then rank 1;
`9663-9682` submits the two post-attention/MoE bodies serially and `9712-9734`
submits the shared tails serially. `dsv4_ep.rs:224-333` issues each rank-sum kernel
in rank order. `tp_ar.cu` implements a bounded peer-arrival barrier before the
canonical rank-0 plus rank-1 sum and a second barrier before input reuse.

The private `compose-sampler-diet-nsys-638eaa0-r1` profile contains 32 tokens,
43 layers each and 2,752 AR calls per rank. Pairing the first attention HC dot
kernel by token and layer gives rank-1 minus rank-0 first-start offsets:
p10 49.468 us, median 103.102 us, p90 194.440 us (N=1,376).
Layer 0 median is 130.2195 us (N=32). The parser skips the preceding token's
head and shared tail, anchors layer 0 after embedding, and retains all 43
per-layer distributions. AR start skew median is 49.9435 us; end skew is
0.7155 us. AR duration medians are 58.112 us on rank 0 and 10.768 us on rank 1.
These instrumented rows diagnose issue timing; they are not throughput claims.

The candidate alternates ranks across six attention phases, five post-attention
and expert phases, and three shared-tail phases. Existing full-program callers
use the same bodies with no phase restriction. Rank-local buffers and CUDA stream
order stay unchanged. Both one-shot reductions retain their existing device
barriers. There are no worker threads and no new host synchronization.

Validation plan on two RTX PRO 6000 Blackwell GPUs:

1. Current and interleaved `dsv4_tp_ep_gate`: equal output SHA and all six refusal
   cells, including cache rollback and retry quarantine.
2. NVTX profile-only decode capture: first-kernel offsets and AR pairing, same
   parser as baseline.
3. Fresh-process sampled ABBA new/current/current/new, five rows per process,
   attention TP2, default radix, `sample_plus_forward_envelope`. Repeat with
   `--moe-m1-splitk` on both arms. Require per-arm repeatability and cross-arm
   token identity within each numeric class.

Raw logs, SQLite and nsys reports stay private under
`issue-interleave-<sha7>-r<N>`. Source SHA, binary SHA, controller/build/gate logs,
hardware/process census and the Python validator bind the evidence.

No local cargo, CI or GPU runs: owner prohibition. Pushes use exported
`MEMRA_SKIP_PERF_CI=1` with normal hooks. Hosted CI and remote gates are pending.
