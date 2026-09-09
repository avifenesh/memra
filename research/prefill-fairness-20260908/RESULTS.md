# Prime fairness on one RTX 5090

2026-09-09. The owned walker removes the whole-prime worker hold without changing
the tested per-request bytes or boundary state. Ornith has a clear mixed-load
latency win. Qwen admits and serves early peers promptly, but its four occupied
session slots still leave later arrivals waiting for tens of seconds. This is
not qualification of Qwen at a 0.20 small-request/s latency SLO.

`MEMRA_PRIME_YIELD` remains OFF by design. Proposal for the next owner-batched
rollout: make it ON in the qualified Ornith launcher default, using this receipt.
Do not flip the shared engine default across unqualified routes. Qwen's positive
admitted-peer result retains its adapter, with admission/peer decode throughput
still required for an overall service claim. GLM5 is a separate stacked lane.
No release, fleet change, context reduction or output-budget reduction is included.

## Decision cell

Requested measurement head: `549d5d3eda9bbe6a7de2c6c18483ca9e38ed9040`, whose
runtime code is `41f15a8618c37ba1e1c845ee03ebcbc0049d5189`. Binary SHA-256:
`19db5df71812eabfa914ba99f58ff676e5d6a860174c0a75084dfad19a89d8e4`.
The source/build manifest is [CPU.md](CPU.md). Later harness commits do not change
this binary. Integration with main is reported separately below.

One non-production RTX 5090, one model per boot, full vision profile retained,
context capacity 262,144, four active-session slots. `MEMRA_PRIME_CHUNK=1024`.
One cold long request at time zero and 20 fixed small arrivals at 5..100 seconds:
0.20 small requests/s, 0.21 total requests/s. The small mix is 16 partial-prefix
continuations, two roughly 1k requests and two roughly 2k requests. Eight seed
requests and one calibration are outside the scored window. Client in-flight
ceiling 16, explicit drain, vendor-default sampling with no sampling fields.
No greedy output enters performance rows.

Arms were interleaved OFF/ON three times on Ornith, then OFF/ON three times on
Qwen. Each boot has a distinct nonce, process identity and checked binary hash.
The harness held the GPU lock and checked the empty compute list before every
launch. All 252 scored requests completed: zero OOMs, retries or incomplete drains.
One initial Ornith boot rejected the deployment-only ledger flag before loading;
that setup failure is retained in the private archive and excluded. The corrected
public-binary launcher is the `ornith-r1-off-v2` boot, not a retried request.

Medians of three boot statistics, seconds:

| Route | Small p95 OFF | Small p95 ON | Long TTFT OFF | Long TTFT ON | Long TTFT penalty | Long total OFF | Long total ON |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ornith MTP, 127,477 input | 15.125 | 0.977 | 24.257 | 26.013 | +1.756 | 29.079 | 29.417 |
| Qwen DFlash2, 131,070 input | 60.452 | 48.042 | 68.341 | 72.005 | +3.664 | 76.938 | 76.614 |

The per-boot small p95 range is 15.122..15.127 OFF versus 0.687..4.537 ON for
Ornith, and 60.207..60.505 OFF versus 47.528..54.206 ON for Qwen. Nearest-rank
p95 of 20 requests is the second slowest request. The median small maximum is
20.120 versus 5.168 seconds for Ornith and 65.448 versus 51.969 for Qwen. The
16-request partial class's p95 is its maximum; it must not be confused with the
all-small p95 above. Raw per-request timings, class counts, routes, yields and
queue peaks are in [gpu/reduced.json](gpu/reduced.json) and the per-boot JSONs.

Every Ornith boot served 15 MTP and six plain scored requests. Qwen OFF served
2/2/1 DFlash requests and 19/19/20 plain requests across repetitions; ON served
3/2/3 DFlash and 18/19/18 plain. Timing-dependent route choice is existing policy.
Scored log excerpts show nonzero drafted/accepted work, including DFlash's
`dspark-acc` counter. The MTP `spec_k` field is not DFlash's draft-count field.
ON yield counts were 275/279/267 on Ornith and 140/139/140 on Qwen; OFF was zero.

## What still waits

In Qwen's first ON boot, the first three small requests received a first token
in 0.843, 1.353 and 1.657 seconds. Those requests took 41.42, 67.16 and 62.78
seconds to finish because the bounded peer quantum supplies little decode work
between long chunks. They retain session slots. A later request's 51.738-second
TTFT contains 51.404 seconds of worker queue wait and only 0.322 seconds of prime.
The boot reports 4,093 session-cap deferrals and zero VRAM deferrals. This is
evidence of admission waiting, not evidence that another GPU or a smaller context
is required. The next software direction is peer decode throughput/admission
fairness while preserving the frozen prime tape.

Ornith also has isolated session-cap outliers up to 5.576 seconds. Its median
post-worker-queue p95 falls from 0.861 to 0.495 seconds; Qwen's from 1.656 to
1.245 seconds. These intervals start when worker tokenization begins, not at an
instrumented `active.push`, so they are not an unconditional admission bound.
Never subtract queueing from the headline TTFT result.

## Exactness

[gpu/byte-boundary.json](gpu/byte-boundary.json) records direct comparisons of
content and reasoning bytes, not just HTTP status:

| Gate | Ornith MTP | Qwen DFlash2 |
|---|---|---|
| c1 greedy four-turn OFF/ON chain | 4/4 equal | 4/4 equal |
| c2 greedy long/small pair versus each request's c1 output | 2/2 equal | 2/2 equal |
| Pair yield count | 250 | 129 |
| Pair routes | 2 MTP, 0 plain | 2 DFlash, 0 plain |
| Small c1 / c2 TTFT, seconds | 0.225 / 0.236 | 0.552 / 1.182 |
| Logits and actual boundary capture across yielded/unyielded prime | equal | equal |

The greedy cells explicitly raise `MEMRA_SPEC_GATE_LOW=64` and
`MEMRA_SPEC_GATE_HIGH=65`. They use bounded real prompts, seed 42, temperature 0
and disabled penalties. The four-turn chains exercise prefix-restored follow-ups.
The concurrent pair sends its small request after the first logged long chunk.
No cross-timing sampled-byte comparison is used as a gate.

DFlash reuses the tap-storage oracle: taps, features, positions, target logits,
boundary logits and hashes of the actual snapshot planes. Full oracle-line
multisets match between serial and concurrent execution despite completion order.
MTP also matches actual capture hashes, and its separate
[run-spec cell](gpu/mtp-boundary-check.log) passed 5,961 rows with 11 yields against
the unyielded trunk and a separately advanced cache, followed by K=3
self-consistency against plain target output. The component oracle hashes KV
lengths plus the recurrent planes, position, logits and anchor; it does not claim
to cover unrelated latent-tail routes.

## Chunk wall and bound

The isolated 1024/4096 probes and the worker-service bound are tabulated in
[CHUNKS.md](CHUNKS.md). These are phase-level wall measurements, not whole TTFT
divided by chunk count. Setup, final capture/ingestion and peer quanta are separate
from the chunk body. The bound is conditional on admitted eligible peers and
the observed execution envelope; it is not a promise about requests waiting
outside the active set.

## Provenance and scope

The original trigger was the read-only `mixed-rps-20260908/REPORT.md`: a
127,625-token Ornith cold prime raised small p95 from about 1.0 to 16.2284 seconds;
the separate long-decode row reached about 70 seconds. The current rendered
Ornith template reports 127,477 input tokens, so the original report is context,
not the OFF arm. Qwen uses the frozen capacity-lane 128k request, SHA-256
`324cc7205e985469e1f6903a7ca19bd620e78601fca45a712aed0081cb2d7021`.
The long-decode shape was not rerun and is not qualified by this prefill receipt.
No claim transfers from this 5090 to the co-located PRO 6000 without its own cell.

Public [gpu/manifest.json](gpu/manifest.json) binds compact JSON and selected
trace files. Full raw streams, profiles and requests are banked privately in
Darklanes under the same lane, with transient authentication files excluded.
The rejected global-budget/reordering lane remains a no-go; see
[MECHANISM.md](MECHANISM.md). This change yields an existing chunk tape and does
not introduce mixed forward execution or repartition plain batches.
