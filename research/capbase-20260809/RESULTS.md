# Box1 PP-2 honest capacity baseline on the fixed ctxcharge tip

## Honest capacity table

N=1 per requested cap, with a fresh server for every row, offered concurrency 24, simultaneous
barrier release, short prompts, explicit `max_ctx`, greedy 64-token generations, and continuous
one-second GPU sampling under one exclusive lock. Capacity is the active-session count at the
first admission defer. Where no defer occurred, the result is a measured lower bound, not an
extrapolation.

| explicit `max_ctx` | logged request charge | concurrent sessions before first defer | defer observed | peak GPU 0 used | peak GPU 1 used | max temp GPU 0/1 | completion |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8,192 | 684 MB | **at least 24** | no | 53,937 MiB | 64,371 MiB | 41/43 C | 24/24 |
| 32,768 | 2,737 MB | **at least 24** | no | 76,913 MiB | 88,371 MiB | 42/43 C | 24/24 |
| 131,072 | 10,947 MB | **4** | yes, first defer at 4 active | 66,705 MiB | 77,683 MiB | 40/41 C | 24/24 |
| 262,144 | 21,894 MB | **1** | yes, first defer at 1 active | 66,705 MiB | 77,683 MiB | 41/41 C | 24/24 |

The four distinct plain-path charge lines are the applied-surface proof that request-owned context
charging is active at the fixed tip. The 128k first defer saw 20,535 MB effective free against a
10,947 MB cost plus a 10,947 MB reserve. The 262k first defer saw 31,726 MB effective free against
a 21,894 MB cost plus a 21,894 MB reserve. All 96 streams eventually completed, all failure scans
were empty, and every row reported zero step-OOM parks. Raw receipt:
[`raw/block-capacity-20260809T195046Z/`](raw/block-capacity-20260809T195046Z/).

## 8k-cap burst service order

These are short-prompt requests with explicit `max_ctx=8192`, not 8,000-token prompts. N=1 per
concurrency, with a fresh server per cell, a simultaneous barrier release, greedy 64-token
generations, and continuous one-second GPU sampling under one exclusive lock.

| concurrency | request-start spread | ordered TTFB, seconds | TTFB span | `/metrics` step p50/p99 | completion |
|---:|---:|---|---:|---:|---:|
| 4 | 0.456 ms | 1.288, 1.340, 1.341, 1.341 | **0.053 s** | 43.31/49.31 ms | 4/4 |
| 8 | 0.854 ms | 2.521, 2.626, 2.626, 2.626, 2.627, 2.627, 2.627, 2.627 | **0.106 s** | 81.11/93.18 ms | 8/8 |

There is no approximately 1.2-second per-request serialization staircase in either fixed-tip
receipt. Both cells reported zero admission defers and zero step-OOM parks, and both failure scans
were empty. Maximum temperatures were 34/36 C at c=4 and 38/37 C at c=8 for GPU 0/1. Raw receipt:
[`raw/block-bursts-20260809T195656Z/`](raw/block-bursts-20260809T195656Z/).

## Sustained c=8 mixed 8k-prompt workload

N=1, one fresh server, continuously replenished c=8 for 600.001 seconds. Every request supplied
exactly 8,000 prompt ids, explicit `max_ctx=8192`, greedy sampling, and a 128-token generation.
This is end-to-end aggregate **output-token** throughput including repeated 8k prefills; it is not
a decode-only throughput number.

| window | output tokens completed | aggregate output tok/s | `/metrics` step p50/p99 | admitted / completed / active at boundary |
|---:|---:|---:|---:|---:|
| 600.001 s | 3,072 | **5.12** | **57.39/57.39 ms** | 32 / 24 / 8 |

The exact boundary scrape is a 30-second rolling step window and landed during the fourth-wave
prefill. The after-drain scrape was 64.14/64.73 ms and is retained alongside it to expose that
phase sensitivity. After the boundary, all 32/32 requests drained cleanly for 4,096 total output
tokens. Median request wall time was 178.661 seconds. There were zero admission defers and zero
step-OOM parks; the failure scan was empty. Peak GPU memory was 49,553/59,827 MiB and maximum
temperature was 50/54 C for GPU 0/1. Raw receipt:
[`raw/block-sustained-20260809T195942Z/`](raw/block-sustained-20260809T195942Z/).

## One 262k park plus c=4 8k pressure

**FAIL to confirm reclaim-on-defer: the requested pressure did not produce a defer.** This is a
negative mechanism receipt, not a server or request failure.

N=1, one fresh server. An explicit-262k-cap request completed, was charged 21,894 MB, and left one
plain continuation entry. A simultaneous c=4 burst then used explicit 8k caps, a logged 684 MB
charge per request, and greedy 64-token generations. All 4/4 requests completed with ordered TTFB
1.218, 1.271, 1.271, and 1.271 seconds, a 0.053-second span, zero step-OOM parks, and an empty
failure scan.

The server recorded zero VRAM defers and no `reclaim-on-defer` event. Its final continuation-pool
eviction counter was 2, but neither eviction was identified as reclaim-on-defer, so that counter
does not prove the parked session was evicted by the requested mechanism. The cell was not
reshaped with additional artificial pressure. Peak GPU memory was 57,745/68,371 MiB and maximum
temperature was 38/38 C for GPU 0/1. Raw verdict:
[`park-summary.json`](raw/block-park-20260809T201459Z/park262k-pressure-c4-8k/park-summary.json).
Complete raw receipt:
[`raw/block-park-20260809T201459Z/`](raw/block-park-20260809T201459Z/).

## Run identity and evidence

- Lane: `lane/cx-capbase`; runtime source remained fixed at `d2d6e6d1`. No runtime code was changed.
- Host: box1 cloud pair, two RTX PRO 6000 GPUs; `MEMRA_PP_STAGES=2`,
  `MEMRA_PP_DEVICES=0,1`, `MEMRA_CTX=262144`; PP placement selected `K=0`.
- Release binary SHA-256:
  `a5f068a3ce8dd84800c26d9e50978f7fcfd8b8d758d873daa8d7ee73221c2352`.
- All three staged IQ4_XS shards and the Q8_0 draft passed fresh SHA-256 verification against
  manifest SHA-256 `4c22bdce378de2c365cdcbf3ce6dcf94d9dd690b0058e5fb01e3fb71a5b29312`.
  Build and artifact verification receipt:
  [`raw/provision-d2d6e6d1/`](raw/provision-d2d6e6d1/).
- Each bounded GPU block held `/tmp/memra-gpu.lock`; every long run was detached; client, server,
  analyzer, GPU, metrics, process, artifact, and failure-scan evidence was retained. Every reported
  cell is a single run (N=1), not a median.
- `~/.lanectl/inbox/cx-capbase.md` was absent at intake and immediately before every bounded block.
- Nothing was pushed, tagged, merged, or released from this lane.
