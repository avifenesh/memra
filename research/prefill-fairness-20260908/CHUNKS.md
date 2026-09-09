# Chunk wall and the conditional service bound

One isolated cold request per boot on the pinned 5090 binary in [RESULTS.md](RESULTS.md).
These eight probes use the same long prompts as the mixed cell, vendor-default
sampling and 2,048 maximum output tokens. OFF/ON are interleaved at each chunk size.
The reported phase contains synchronization, so wall time includes completion of
that chunk's device work. Ranges are split into equal-count early/middle/late
thirds, including actual capture cuts and tails. Each arm has one boot here;
these are observed costs, not estimates of population worst-case latency.

Milliseconds; each triple is the early / middle / late median:

| Route and phase | Chunk | OFF medians | ON medians | Maximum OFF / ON |
|---|---:|---|---|---|
| Ornith trunk | 1024 | 121.293 / 186.461 / 248.193 | 121.441 / 186.664 / 248.275 | 280.351 / 280.304 |
| Ornith draft fill | 1024 | 6.548 / 6.547 / 6.547 | 6.548 / 6.548 / 6.549 | 6.558 / 6.562 |
| Ornith trunk | 4096 | 394.458 / 629.954 / 825.828 | 393.482 / 629.345 / 824.963 | 932.512 / 932.941 |
| Ornith draft fill | 4096 | 25.862 / 25.860 / 25.864 | 25.861 / 25.861 / 25.864 | 25.868 / 25.868 |
| Qwen trunk plus ingestion | 1024 | 369.611 / 530.412 / 689.093 | 370.273 / 531.585 / 690.624 | 764.268 / 764.869 |
| Qwen trunk plus ingestion | 4096 | 1441.272 / 1999.700 / 2505.438 | 1447.545 / 2000.830 / 2508.455 | 2747.043 / 2751.677 |

Ornith executes 126 trunk plus 125 draft advances at 1024, and 33 plus 32 at
4096. Each phase totals 127,477 rows. Qwen executes 129 advances at 1024 and 33
at 4096, totaling 131,070 rows. Capture stops account for the extra ranges; they
are frozen identically in OFF and ON. The reducer retains each third's p95 and
maximum in [gpu/reduced.json](gpu/reduced.json).

The maximum ON worker call while priming, including setup or completion around
the chunk, was 280.501 / 933.146 ms for Ornith at 1024 / 4096, and
764.913 / 2751.719 ms for Qwen. First ON calls were respectively
96.882 / 288.183 and 289.438 / 1198.215 ms. No hidden prompt-length hold appeared
outside the walker chunks in these probes. Outer finalization maxima were
0.001 ms on Ornith and 1.554 ms on Qwen. DFlash's nested finalization is included
in the outer measurement and must not be added twice. The 1024 Qwen probes
overlapped nice-19 CPU compilation; other GPU processes were absent.

## Bound

Let C include the maximum long-prime worker call, Q the maximum service quantum
of an eligible peer, N the admitted-session limit, and m the long prime's frozen
advances. Between successive long advances, each peer gets at most one quantum:
one existing prime chunk, one committed spec round, or one plain decode step.
The added long-prime TTFT work is at most `(m-1)*(N-1)*Q` plus scheduler and
admission overhead. A committed spec round may emit several tokens; its surplus
is preserved. This is a count-of-work bound, not a timer that cancels work.

For one long and one admitted small request, the first-token delay is at most
the residual long call C plus the small request's own first-token work, provided
the small work fits its service quantum. A capture split can make the small
request require multiple advances, in which case include each intervening C.
For the measured 1024 envelope, the long-call contribution is about 0.281 seconds
on Ornith and 0.765 seconds on Qwen; at 4096 it grows to 0.934 and 2.752 seconds.
Use ceiling-rounded values, not a whole-prefill average.

With N=4, Ornith's 251 advances allow at most 750 peer quanta before completion;
Qwen's 129 allow 384. Q depends on peer prompt size, decode shape and current
context, so substituting the long chunk's C for Q would be incorrect. The actual
median long TTFT penalties were 1.756 and 3.664 seconds. Requests outside the
active set, two long primes, expensive admission restores and interference from
another process require their own terms. None is covered by an unconditional
one-chunk HTTP TTFT promise.

The c2 byte gate demonstrates the admitted-peer behavior without sampled route
changes: Ornith small TTFT 0.225 seconds alone and 0.236 during the prime; Qwen
0.552 alone and 1.182 during the prime. The full offered-load Qwen p95 remains
48.042 seconds because later requests wait for session slots. Both measurements
belong in the verdict.
