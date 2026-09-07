# DSV4 scoped rank-worker submission: no-go

The corrected expert-ID TP/EP sampled baseline is 24.3093 tokens/s on the
two-card RTX PRO 6000 development pair. Two 256-token output streams and final
logits/cache/hidden data agree. This is not a PP comparison or a full model
oracle gate. The existing 120 tokens/s plain objective remains open.

The same-binary Nsight capture has 5,504.5 kernel launches per token. Rank 0
arrives at the one-shot join an average 160.14 microseconds before rank 1;
completion after both ranks have arrived averages 12.65 microseconds. Kernel
durations include wait time and the profile is instrumented, so neither the
arrival gap nor CPU API totals are claimed as removable wall time.

The candidate assigns disjoint workspace/cache/checkpoint borrows to two scoped
host workers. Each queues its producer, rank-local one-shot endpoint and tail
in that rank's stream order. The existing device start/end barriers align the
ranks. Both workers join and drain before the borrowed raw endpoint addresses
may expire; error/panic paths set an abort signal and still drain. Refusal words
are sticky and checked before committing either cache plane. The whole-token
mutex prevents cross-request signal reuse. No unsafe Send/Sync assertion is
added, and no per-layer host barrier or extra input/output copy is introduced.

The gate uses the same numerical class and frozen tape as serial TP/EP. It does
not enable DSpark, MTP, host-cache serving or batched prompt execution. Two
same-TP repeats with the strong existing GU/half2/wo_a posture are required.
Token IDs and all final identity hashes are outside the timed interval. The
candidate is default OFF and has no customer-facing switch.

The target model gate completed at 2026-09-07T15:05:59Z. Both 256-token sampled
runs were eligible and matched the frozen corrected serial-TP token, logit,
cache and hidden hashes exactly. Rates were 23.763919 and 23.747080 tok/s,
pooled 23.755496. There is no measured win over the pinned 24.309285 serial-TP
receipt, so the runtime/CLI door, scoped workers and endpoint API are removed.
No PP comparison was run. This does not reject persistent host workers or
paired CUDA graphs, which are different execution mechanisms.

Measured source: `eef64ffd3fad5a8fe93545d9b9b1f94e72d5c8f2`.
Binary SHA256: `81cd43a1598e0ccda4e997cd1aef08b7ac51b779ba24d43d059a0940c5bad1c2`.
Model log SHA256: `e71a83d49ff2f94321fe482eb8aa1a4de666633a826aebc72bb131350491c98e`.
The original coordinator/barrier draft and slot-copy WIP are not winners.

Local CI is owner-waived for September 7, 2026 while the local machine is in
use. Remote targeted build/run receipts are retained; hosted integration gates
remain required before delivery. Unavailable Revuto may be waived only after
independent source review, per the owner's explicit instruction.
