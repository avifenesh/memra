# DFlash prime adapter

5090 exactness and admitted-peer gates passed. Default remains OFF by design.
The overall 0.20 req/s mixed-load latency goal is not qualified: session slots
remain occupied during slow peer decode. See [RESULTS.md](RESULTS.md).

`dflash.rs` owns `DsparkPrimeState`: cold cache/draft KV or a resumed session,
the frozen trunk ranges, final logits, capture state, and `DflashTapPrime`.
`DsparkPrimeWalker` temporarily binds that owned state to immutable execution
context and implements the shared `PrimeWalker` trait. The worker stores owned
state between advances; no self-referential borrow or additional GPU thread.

The core is the former `prime_dflash_taps` body, advanced one existing range at a
time. The same target `prime_cache` call, request-relative remaining extent,
boundary snapshot, pin, and ingestion operations run on both arms. The 256-row
carry and its device buffer survive each yield; final partial ingestion happens
only during finish. Each advance fences ingestion before shared scratch can be
reused. The synchronous cold/resume APIs drain this same walker.

Worker dispatch retains `dspark_on` while the owned caches are inside a pending
prime. Prompt tokens stay queued until successful finalization, then join the
worker's fed/sampler state once. Errors and cancellation cannot park partial
state. Finalization releases carry storage before optional prompt-end capture.

CPU validation: 56 DFlash tests passed, including the production carry oracle
across chunk and capture cuts. See `CPU.md` for the final source-bound check set.
The greedy four-turn chain and c2 pair passed; the pair logged 129 yields and
both outputs matched their own c1 outputs. Taps, features, positions, logits and
actual boundary snapshots matched the serial oracle. The pair's small TTFT was
1.182 seconds versus 0.552 alone. Full mixed p95 improved from 60.452 to 48.042
seconds, with a 3.664-second long TTFT penalty, but later arrivals still wait at
the four-session cap. This is a positive mechanism result with an unresolved
admission/peer-throughput limit, not a shared-card service SLO qualification.
