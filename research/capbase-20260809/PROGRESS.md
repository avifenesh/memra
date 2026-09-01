# capbase box1 progress

Lane `lane/cx-capbase`, base `d2d6e6d1`, started 2026-08-09.

## Contract

- Measurement-only lane: no runtime code changes, no fixes, no origin push, no tag.
- Box1: `ubuntu@<rented-box-ip>`, two RTX PRO 6000 GPUs.
- Serving shape: `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,1`, `MEMRA_CTX=262144`.
- Every bounded GPU block holds `/tmp/memra-gpu.lock`; long runs are detached and all output is tee'd.
- Raw evidence lands under `research/capbase-20260809/raw/`; each result states N and thermal regime.
- Any observed defect is captured and reported, not fixed in this lane.

## Cells

- [x] Capacity at requested 8k, 32k, 128k, and 262k: offered c=24, first-defer active count, completion, and per-GPU peak `nvidia-smi`.
- [x] 8k burst c=4 and c=8: ordered TTFB and total span.
- [x] Sustained c=8: 8k request cap, 128-token generations, 10 minutes; aggregate tok/s and `/metrics` step p50/p99.
- [x] Full-262k park plus c=4 8k pressure: burst completed, but the reclaim-on-defer gate failed because no defer occurred.
- [x] `RESULTS.md` begins with the honest capacity table.

## Live status

- Checkout started clean at the requested fixed tip `d2d6e6d1`.
- `~/.lanectl/inbox/cx-capbase.md` was absent at lane start and immediately before the capacity block.
- Prior `val256-20260809` result and committed harnesses read. Its requested-128k row is obsolete because it was charged at 262k.
- Remote release build from exact runtime tip `d2d6e6d1` passed; binary SHA-256 is
  `a5f068a3ce8dd84800c26d9e50978f7fcfd8b8d758d873daa8d7ee73221c2352`.
- All three staged IQ4_XS shards and the Q8_0 draft passed a fresh SHA-256 verification against
  manifest `4c22bdce378de2c365cdcbf3ce6dcf94d9dd690b0058e5fb01e3fb71a5b29312`.

## Capacity block — complete

N=1 per requested cap, four fresh servers under one exclusive lock, c=24 barrier release,
greedy 64-token generations, and continuous one-second per-cell GPU sampling. All 96 streams
completed, every failure scan was empty, and every cell reported zero step-OOM parks.

| explicit `max_ctx` | result before first defer | request cost | peak GPU 0 | peak GPU 1 |
|---:|---:|---:|---:|---:|
| 8,192 | **at least 24**; no defer, 24 sampled active | 684 MB | 53,937 MiB | 64,371 MiB |
| 32,768 | **at least 24**; no defer, 24 sampled active | 2,737 MB | 76,913 MiB | 88,371 MiB |
| 131,072 | **4** | 10,947 MB | 66,705 MiB | 77,683 MiB |
| 262,144 | **1** | 21,894 MB | 66,705 MiB | 77,683 MiB |

The distinct plain-path charge lines confirm the request-owned fix on the fixed tip. Raw receipt:
`raw/block-capacity-20260809T195046Z/`.

## 8k burst timing block — complete

N=1 per concurrency with a fresh server per cell under one exclusive lock. The requests used
explicit `max_ctx=8192`, greedy sampling, and 64-token generations. Both failure scans were empty;
all 12 requests completed with zero admission defers and zero step-OOM parks.

| concurrency | ordered TTFB, seconds | span | `/metrics` step p50 / p99 |
|---:|---|---:|---:|
| 4 | 1.288, 1.340, 1.341, 1.341 | **0.053 s** | 43.31 / 49.31 ms |
| 8 | 2.521, 2.626, 2.626, 2.626, 2.627, 2.627, 2.627, 2.627 | **0.106 s** | 81.11 / 93.18 ms |

The request-release spreads were 0.456 ms and 0.854 ms. Maximum temperatures were 34/36 C at
c=4 and 38/37 C at c=8 for GPU 0/1. The lane inbox was absent immediately before this block.
Raw receipt: `raw/block-bursts-20260809T195656Z/`.

## Sustained c=8 block — complete

N=1, one fresh server under one exclusive lock, continuously replenished c=8 for an exact
600.001-second window. Every request used exactly 8,000 prompt tokens, explicit
`max_ctx=8192`, greedy sampling, and a 128-token generation. The fixed window completed 3,072
output tokens for **5.12 aggregate output tok/s**; 8 requests remained active at the boundary.
After a clean drain, all 32/32 admitted requests had completed (4,096 output tokens total), with
zero admission defers and zero step-OOM parks.

The exact window-end `/metrics` scrape reported step p50/p99 of **57.39/57.39 ms**. This metric
is a 30-second rolling window and the boundary scrape landed during the fourth-wave prefill; the
after-drain scrape reported **64.14/64.73 ms**, which captures the completed wave. Both scrapes
are retained in the raw JSONL. Maximum temperatures were 50/54 C for GPU 0/1 under continuous
one-second sampling. The lane inbox was absent immediately before this block. Raw receipt:
`raw/block-sustained-20260809T195942Z/`.

## Full-262k park plus c=4 8k pressure — measured, gate FAIL

N=1, one fresh server under one exclusive lock. The explicit-262k request completed and left one
plain continuation entry. Its logged request charge was 21,894 MB. The simultaneous c=4 burst
then used explicit `max_ctx=8192`, a logged 684 MB charge per request, greedy sampling, and
64-token generations. All 4/4 burst requests completed; ordered TTFB was 1.218, 1.271, 1.271,
and 1.271 seconds (0.053-second span), with zero step-OOM parks and an empty failure scan.

The requested reclaim-on-defer proof did **not** occur: the server logged zero VRAM defers and
no `reclaim-on-defer` event. The final continuation-pool eviction counter was 2, but neither
eviction was identified as reclaim-on-defer, so that counter cannot establish the requested
ordering or parked-session eviction. The gate therefore records FAIL without adding artificial
pressure. Peak GPU memory was 57,745/68,371 MiB and maximum temperature was 38/38 C for GPU 0/1
under continuous one-second sampling. The lane inbox was absent immediately before this block.
Raw receipt: `raw/block-park-20260809T201459Z/`.
