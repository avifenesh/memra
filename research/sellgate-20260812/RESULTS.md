# Q35 + Q27 sold-cap deployment gate — eu-west PRO pair

Date: 2026-08-12

Rig: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, one target-only server per physical GPU

Scored runtime source: `79c3c0b2779101c7de89d6f822b9392d03e71702`

## Verdict

**WEEK-1 GO on Q27. Q27 is SELLABLE at c=4; Q35 is NOT at c=4.** This satisfies the
week-1 rule that at least one model shape pass on the simultaneous two-server pair, but it does
not qualify the originally proposed Q35+Q27 pair as a two-model offer.

The c=4 two-second bars below are first-content TTFT, not full-response latency. Full-response
percentiles are published separately and no cold or p99 sub-two-second promise is made.

| Model | Standard exactness | Serial cache exactness | Required base cells | c=4 hit TTFT p95 | c=4 all-traffic TTFT p50 | c=4 cached-token reconciliation | Clean throughput knee / headroom | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q27 | PASS | PASS | **40/40** | **269.077 ms** | **18.568 ms** | **437,400 = 437,400 = 437,400** | c=12 / **200%** above c=4 | **SELLABLE** |
| Q35 | PASS | PASS | **35/40** | 98.449 ms | 7.487 ms | **437,400 = 437,400 = 437,400** | c=8 / 100% above c=4 | **NOT at c=4** |

The three accounting values are client `cached_tokens`, engine `cached_tokens_in`, and engine
`prefix_cache_hit_tokens`. Q35 meets the c=4 latency and cached-token bars in isolation, but a cap
of four must remain correct at every admitted depth up to four. Its mixed c=2 cell failed in all
five repetitions, so the lower-width integrity failure rejects the shape.

## Customer one-page envelope — Q27 only

This is the exact envelope eligible for `tiyuvta.ai/inference`. It is the frozen 4,860-prompt-token
plus 60-completion-token workload (exactly 81:1), at c=4/model while Q35 served its paired cells on
the other GPU. Each row pools five interleaved cells. The 90%-hit population is 100 requests:
90 full-prefix hits and 10 real misses. The pure-cold population is a separate 100 requests.
Percentiles are nearest-rank except p50, which is the population median.

### First-content TTFT

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | **18.568 ms** | 19.225 ms | **299.678 ms** | 1,513.906 ms | **2,910.545 ms** |
| Cache hits only | 90 | 18.511 ms | 18.901 ms | 19.718 ms | **269.077 ms** | 299.678 ms |
| Misses inside the 90%-hit mix | 10 | 1,514.071 ms | 1,516.048 ms | 2,910.545 ms | 2,911.259 ms | 2,911.259 ms |
| Pure cold arm | 100 | **5,670.258 ms** | 5,671.995 ms | 5,673.627 ms | 5,675.001 ms | **5,676.113 ms** |

The honest customer statement is therefore: typical cache-hit TTFT is tens of milliseconds,
cache-hit p95 is 269.077 ms, and mixed all-traffic p50 is 18.568 ms. The 10% miss class pushes the
mixed p99 to 2.911 seconds; a fully cold c=4 request population is about 5.67 seconds to first
content.

### Full-response latency for 60 completion tokens

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | **1,081.945 ms** | 2,485.567 ms | **2,529.531 ms** | 2,579.119 ms | **3,973.229 ms** |
| Cache hits only | 90 | 1,081.761 ms | 2,484.287 ms | 2,487.337 ms | 2,488.923 ms | 3,883.321 ms |
| Misses inside the 90%-hit mix | 10 | 2,575.732 ms | 2,579.478 ms | 3,973.229 ms | 3,973.348 ms | 3,973.348 ms |
| Pure cold arm | 100 | **6,732.607 ms** | 6,733.983 ms | 6,735.405 ms | 6,736.514 ms | **6,736.936 ms** |

### Inter-token latency

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 18.015 ms | 37.096 ms | 41.842 ms | 41.864 ms | 65.732 ms |
| Cache hits only | 90 | 18.015 ms | 37.112 ms | 41.843 ms | 41.866 ms | 65.733 ms |
| Misses inside the 90%-hit mix | 10 | 18.007 ms | 18.024 ms | 18.024 ms | 18.036 ms | 18.036 ms |
| Pure cold arm | 100 | 17.993 ms | 18.010 ms | 30.720 ms | 30.724 ms | 30.727 ms |

### Rate and accounting envelope

| Q27 c=4 measurement | N=5 median or exact total |
|---|---:|
| Mixed output throughput | **144.552 completion tok/s** |
| Mixed requests/s | 2.409 |
| Mixed billed prompt rate | 11,708.673 prompt tok/s |
| Mixed computed prompt rate | 1,170.867 prompt tok/s |
| Pure-cold output throughput | 35.646 completion tok/s |
| c=4 mixed prompt / cached / completion tokens | 486,000 / **437,400** / 6,000 |
| Engine cached counters | `cached_tokens_in=437,400`; `prefix_cache_hit_tokens=437,400` |
| Cache hits / misses | 90 / 10 |
| Session defers / VRAM defers / OOM parks | 0 / 0 / 0 |
| Prefix-cache budget / observed c=4 peak | 4,096 MiB / 4,021.664 MiB |

Both model servers were active in the same c=4 windows. Pair-window throughput, measured from the
shared release barrier until the slower model drained, was **289.103 completion tok/s median**
across five repetitions (287.605–289.560 tok/s). This is not the sum of independently timed model
rates.

## Q27 capacity headroom

Every Q27 row below is clean and contains N=5 cells / 100 requests per arm. Mixed throughput rose
through c=12 and fell at c=16, so c=12 is the highest consecutive clean rising width. c=24 was not
run, exactly as required by the frozen stop rule.

| c/model | Cold output tok/s | 90%-hit output tok/s | Mixed hit TTFT p95 | Mixed all TTFT p50 | Mixed all TTFT p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.546 | 65.631 | 2.082 ms | 1.826 ms | 1,406.399 ms |
| 2 | 31.754 | 96.602 | 14.763 ms | 14.455 ms | 1,494.440 ms |
| **4 sold cap** | **35.646** | **144.552** | **269.077 ms** | **18.568 ms** | **2,910.545 ms** |
| 8 | 37.222 | 175.061 | 571.736 ms | 25.230 ms | 2,957.562 ms |
| **12 measured knee** | **37.698** | **186.306** | 1,170.597 ms | 31.565 ms | 2,980.087 ms |
| 16 stop point | 37.509 | 183.020 | 601.283 ms | 319.155 ms | 3,063.470 ms |

The sold cap is four versus a measured knee of twelve: **200% concurrency headroom** by the frozen
definition, comfortably above the 25% bar. The knee supplies 28.9% more mixed output throughput
than c=4; c=16 is 1.8% below c=12 and therefore does not extend capacity.

Across the full six-width scored campaign, Q27 completed **1,200/1,200 requests** at exactly 60
tokens and all 60 cells were clean. Its **2,624,400** scored cached tokens reconciled exactly in
client usage, `cached_tokens_in`, and `prefix_cache_hit_tokens`. All Q27 cells had zero prompt- or
cache-token drift, zero admission defers, and zero OOM parks.

## Why Q35 is NOT at c=4

Q35's five c=4 mixed cells themselves were clean: 403.117 output tok/s median, hit TTFT p95
98.449 ms, all-traffic TTFT p50/p90/p99 7.487/103.031/908.552 ms, mixed-miss TTFT
p50/p95 522.834/1,008.444 ms, and pure-cold TTFT p50/p99 1,939.552/1,963.589 ms. Its c=4
437,400 cached tokens also reconciled exactly in both engine counters.

The shape still fails because **all five mixed c=2 cells were invalid**. Seven cached requests
ended with `finish_reason=stop` at 17 or 25 completion tokens instead of the frozen 60. The five
windows reported response completion totals of 1,165 / 1,122 / 1,165 / 1,165 / 1,122 versus engine
`tokens_out` totals of 1,164 / 1,120 / 1,164 / 1,164 / 1,120. Cached-token accounting remained
exact, but response/output exactness did not. One more cached request failed the same way in the
fifth c=12 mixed window. Q35 therefore finished 35/40 required base cells and 54/60 total cells
clean; fast c=4 latency cannot erase a reproducible lower-width integrity failure.

The next private-pair action is Q27-replicas qualification on both cards. The failed shape's card
goes to **RESEARCH (SOTA training), OR listing continues on the passing card**. Q122 is not an
automatic fallback. A passing one-card Q27 customer at $2,750/month is $90.41/day and clears the
corrected $63/day expansion trigger; the $5,500/month two-card offer requires the second Q27 shape
to pass first.

## Exactness and output-hash boundary

- Both physical GPUs reported `ALL GREEN (95 cells, 13 skipped)` from `kernel-check` on the scored
  build. Q27 and Q35 each passed `run-gen` prefill/decode plus batched-prime/tokenwise argmax MATCH,
  and each produced exactly eight K=1..8 `run-spec` self-consistency PASS rows plus the overall
  PASS sentinel.
- The serial partial-prefix gate passed N=3 per model. Each reconciled 27,702 cached tokens exactly
  in client usage and both engine counters, with six hits, six misses, and byte-identical
  cold/partial/full-hit output under the same c=1 decode composition.
- At c=4, Q27 produced two cold output SHA-256 classes (75/25 requests). Its 90 cached responses
  produced those exact same two classes (79/11), with no cache-only class. Q35's clean c=4 cells
  likewise introduced no cache-only output class.
- Every c=4 hot output differs from the serial c=1 golden, which is retained as 90 observed
  comparison mismatches per model. This is the repository's documented batched-prime near-tie
  class: cold and cached c=4 share the same output classes, while serial cache exactness and the
  standard argmax/spec batteries pass. This receipt does **not** claim byte identity across
  different batching compositions.

## Pinned inputs

| Input | SHA-256 |
|---|---|
| Runtime source | `79c3c0b2779101c7de89d6f822b9392d03e71702` |
| `memra-server` | `3d6ff26c047b8dc59d1a865f4c5b4a889ddbe0b3df39206510b7d1938a4076e2` |
| `kernel-check` | `433cd4f5b3a840101325011043197d71fd6e092d4be2a4b76bcc5ee269b27be4` |
| `run-gen` | `51ae25223d9baedf43b2185b004babb440b0822763e2ce2c1f168e748064fe23` |
| `run-spec` | `6013fc96c8204c1e0460faf316ea4c75be7a4959c9a34088afc91d5241b14fb2` |
| Q27 artifact | `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517` |
| Q27 external draft | `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581` |
| Q27 embedded `tokenizer.chat_template` (7,764 bytes) | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| Q35 artifact | `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf` |
| Q35 external draft | `ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a` |
| Q35 embedded `tokenizer.chat_template` (8,057 bytes) | `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c` |
| Workload lock | `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34` |
| Canonical scored prompt IDs | `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb` |
| Prompt qualification manifest | `f99d4b24c01776e516146e0f310e1012890c0d6fe9c13b9fa611f75dca416dd3` |
| Standard exactness prompt | `ce404f9ec20c6aab37220a2428254c6f7dc59286f1620d9060bb30e9d5ad9027` |

The main and draft artifact hashes were recomputed from the complete files before both correctness
and scoring. Template hashes are over the exact raw GGUF metadata strings, not rendered text.

## Method and evidence boundary

- The detached campaign held `/tmp/memra-gpu.lock` from 03:32:00Z through its sealed PASS at
  04:00:57Z. Both target servers stayed live together through 120 cells and 2,400 scored request
  rows. Arms alternated, base-width order rotated, and every width used N=5 without artificial
  cooldown or clock changes.
- Eight isolated hot namespaces carried one fixed prompt identity qualified before scoring at
  c=1/2/4/8 on both models: 90/90 pilot requests reached 60 tokens with no errors, cache credit,
  or counter drift. Unique namespaces supplied real misses. This controls prompt length and content
  but is not a diverse customer-prompt corpus.
- Scored calls use the frozen prompt token IDs directly. The standard gates exercise chat-template
  rendering and the embedded templates are pinned above, but this receipt is not a chat/tools/
  structured-output acceptance soak. It also does not replace a sustained customer-specific
  workload acceptance test.
- The 4,096 MiB cache budget covers the eight-entry hot working set plus active cold churn. Q27's
  scored peak was 4,021.664 MiB. Evictions were expected from unique cold namespaces and never
  removed the protected hot hit shape.
- Continuous 250 ms telemetry observed GPU0/GPU1 maxima of 68/52 C, 510.85/407.48 W, and
  29,043/25,713 MiB used. There were no captured OOM, CUDA-error, panic, segmentation, or fatal
  markers in either server log.

## Receipts

- Machine-readable verdict and every reduced cold/mixed level: [`summary.json`](summary.json).
- Sealed final JSONL, server logs, metrics, thermal traces, and manifest:
  [`raw/campaign-scored/`](raw/campaign-scored/). Its 31-file manifest verifies and has SHA-256
  `ef8b08e7e8dea069fc6dd59baec7aaa298c70487ac0edec237666e36a6520498`.
- Sealed correctness battery: [`raw/gates/`](raw/gates/). Its 12-file manifest verifies and has
  SHA-256 `08b89461323975f5c78491615824ab2737ee3f143c6c316e8f452d80a7c2ec42`.
- Sealed raw template extraction: [`raw/template-hashes/`](raw/template-hashes/). Its manifest
  verifies and has SHA-256 `bb43117f58f447610682138b4c4e9b56f973c3bb7e03bbd6b346f334cf9abb8c`.
- Prompt qualification and the two excluded pre-score attempts remain under
  [`raw/prompt-pilot/`](raw/prompt-pilot/), [`raw/campaign-attempt1/`](raw/campaign-attempt1/),
  and [`raw/campaign-attempt2/`](raw/campaign-attempt2/), with their exclusion reasons in
  [`PROGRESS.md`](PROGRESS.md).

No runtime code, generated performance board, README number, merge, tag, push, or formatting
surface changed in this lane.
