# Current sampled engine decode rate, 2026-09-06

The rates below are the pre-sorter control. The subsequent exact-order rewrite
and its same-window comparison are recorded in [sampler-sort.md](sampler-sort.md):
radix reaches 27.19/25.57 plain/DSpark output tok/s at context 256 and
25.08/33.05 at context 8192. No default is promoted.

The current matrix/EP + tiled-indexer/scorer + direct-host C4 program remains
well below the throughput objective. These results are not comparable with
the older PP2 greedy/ignore-EOS public-prompt measurements.

Binary: `37f72a7b2f1cb3438f0f540f4234901863f9ea0ce1e05d8b75fbeae097e53c3b`.
Hardware: two RTX PRO 6000 Blackwell Max-Q cards, no concurrent GPU process.
Pinned source and house artifacts remain unchanged from the lane lock.
The raw model log and 250 ms telemetry are banked in the companion private
lane; SHA256 respectively
`49d71ffb98838bc687c4eac3a8f64dec655efa253ead4f7ba7aa9e2a23d36759` and
`3b78e0b3cb1aad66587ec93b99f17fedd9ecf29be1bb34242ed47a1f04fa952e`.
Controller completed at 06:11:27 UTC, status 0.

## Protocol and results

One load, one fresh source prefill per context, canonical host snapshots and
direct-host restoration before every row. Two warmups (plain, DSpark), then
ABBA x3, six measured rows per mode/context. Every row uses T=1, top_p=1,
top_k=0, seed 20260906, 256 requested outputs and respects EOS. Output identity
is checked across all 14 walks per context. No row looped or stopped early.
Source inputs contain real code once, not repeated-token filler.

| Prompt tokens | Plain output tok/s | DSpark output tok/s | Plain range | DSpark range |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 20.7900 | 19.7868 | 20.7828–20.8016 | 19.6055–19.8405 |
| 8192 | 19.3562 | 23.8815 | 19.2936–19.4935 | 23.7748–24.0414 |

These are median **decode-wall** rates: visible output count divided by wall
from ready logits through generation and the final stream drain. Prefill,
restoration, allocation, hashing, detokenization and output logging are outside
that interval. Plain leaves DSpark weights resident but does not execute the
drafter or maintain its rings, matching a genuinely plain engine loop.

The separately reported post-first-commit rates are plain/DSpark 20.7333/19.8896
at 256 and 19.3034/24.0825 at 8192. These are engine commit timestamps, not HTTP
TTFT or wire ITL. Speculative commits are bursts and their same-timestamp token
gaps are zero; the summary labels them explicitly. NVIDIA's current
[metrics definitions](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html#metrics)
distinguish per-user generation rate from whole-benchmark output throughput.
That page now redirects new benchmarking work toward
[AIPerf](https://github.com/ai-dynamo/aiperf); no external runtime was installed.

The independent `tools/dsv4-decode-rate-summary.py` audit checks schedule,
production mode engagement, output hashes from raw token IDs, loop exclusions,
EOS/budget, timestamp monotonicity, one owned GPU process, and phase-correlated
two-GPU telemetry. All 24 measured rows are eligible. Maximum observed telemetry
gap is 311 ms. Watt/clock data do not establish a power-limited diagnosis.

## Next measured boundary

Plain's first commit from an already-ready logits row takes median 14.60 ms
at 256 and 15.11 ms at 8192. In this harness that interval contains host
sampling, not restoration or a GPU decode step. Source inspection shows
`dsv4_sample_row` sorts the complete vocabulary descending before its ordered
f64 softmax/nucleus/CDF walk. Profile that operation separately, then test an
exact-order optimization; do not change categorical ordering or f64 arithmetic
to obtain a superficially faster seeded stream. GPU decode/verify still needs
its own current phase profile. No default is promoted from these measurements.

The host-C4 HTTP cold/warm/refusal gate is the next capacity-safety cell.
Actual 1M prompts, long HTTP TTFT, c1–c16 scheduling/fairness, persistent live
graphs, broader matrix quality and peak plain/speculative speed remain open.
