# Q35 + Q27 fresh sold-cap requalification — eu-west PRO pair

Date: 2026-08-12

Rig: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, one target-only server per physical GPU

Scored runtime source: `ac6ef049b8661008c0da91f4747f68f4dabdaa04`

## Verdict

**PAIR QUALIFIED: Q27 and Q35 are both SELLABLE at c=4.**

The two-second bars are first-content TTFT, not full-response latency. Full-response percentiles are published separately; no cold or p99 sub-two-second promise is made.

| Model | Standard exactness | Serial cache exactness | Required base cells | c=4 hit TTFT p95 | c=4 all-traffic TTFT p50 | c=4 cached-token reconciliation | Clean throughput knee / headroom | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q27 | PASS | PASS | **40/40** | **21.565 ms** | **18.660 ms** | **437,400 = 437,400 = 437,400** | c=12 / **200%** above c=4 | **SELLABLE** |
| Q35 | PASS | PASS | **40/40** | **101.203 ms** | **7.678 ms** | **437,400 = 437,400 = 437,400** | c=32 / **700%** above c=4 | **SELLABLE** |

## ⚠ Q27 REGRESSION FLAGS ABOVE 2%

**3 published Q27 metrics regressed by more than 2%.** Every regression is listed individually; none is averaged away.

| Metric | Original | Fresh | Regression |
|---|---:|---:|---:|
| `q27.capacity.c1.mixed90.ttft_all.p50_ms` | 1.826098 | 1.922927 | **5.302%** |
| `q27.capacity.c8.mixed90.ttft_all.p50_ms` | 25.229613 | 25.746039 | **2.047%** |
| `q27.capacity.c12.mixed90.ttft_all.p50_ms` | 31.565457 | 32.698285 | **3.589%** |

## Explicit q35bug regression matrix — mixed c=2 x5

These are the exact frozen campaign cells, surfaced explicitly for both models.

| Model | Rep | Requests OK | Response tokens | Engine `tokens_out` | Short | Cached drift | Clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q27 | 1 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q27 | 2 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q27 | 3 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q27 | 4 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q27 | 5 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q35 | 1 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q35 | 2 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q35 | 3 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q35 | 4 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |
| Q35 | 5 | 20/20 | 1,200 | 1,200 | 0 | 0 | PASS |

## Customer one-page envelope — Q27

Frozen workload: 4,860 prompt tokens plus 60 completion tokens (81:1), c=4/model, with Q35 active on the other GPU. Each row pools five interleaved cells; mixed traffic is 90 full-prefix hits and 10 real misses, and pure cold is a separate 100-request population.

### First-content TTFT

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 18.660 ms | 19.202 ms | 301.698 ms | 1,512.781 ms | 2,912.474 ms |
| Cache hits only | 90 | 18.573 ms | 19.025 ms | 19.771 ms | 21.565 ms | 301.698 ms |
| Misses inside the 90%-hit mix | 10 | 1,513.497 ms | 1,517.461 ms | 2,912.474 ms | 2,912.643 ms | 2,912.643 ms |
| Pure cold arm | 100 | 5,670.162 ms | 5,672.616 ms | 5,673.802 ms | 5,674.197 ms | 5,675.570 ms |

Typical mixed TTFT is 18.660 ms; cache-hit p95 is 21.565 ms. The 10% miss class puts mixed p99 at 2,912.474 ms; pure-cold c=4 p50 is 5,670.162 ms.

### Full-response latency for 60 completion tokens

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 1,083.654 ms | 2,487.477 ms | 2,529.733 ms | 2,579.538 ms | 3,976.973 ms |
| Cache hits only | 90 | 1,083.564 ms | 2,473.601 ms | 2,487.960 ms | 2,489.442 ms | 3,886.770 ms |
| Misses inside the 90%-hit mix | 10 | 2,577.457 ms | 2,581.850 ms | 3,976.973 ms | 3,977.008 ms | 3,977.008 ms |
| Pure cold arm | 100 | 6,733.786 ms | 6,735.588 ms | 6,736.963 ms | 6,737.519 ms | 6,737.719 ms |

### Inter-token latency

| Q27 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 18.045 ms | 37.116 ms | 41.854 ms | 41.868 ms | 65.773 ms |
| Cache hits only | 90 | 18.046 ms | 37.646 ms | 41.865 ms | 41.870 ms | 65.784 ms |
| Misses inside the 90%-hit mix | 10 | 18.040 ms | 18.052 ms | 18.056 ms | 18.058 ms | 18.058 ms |
| Pure cold arm | 100 | 18.018 ms | 18.041 ms | 30.741 ms | 30.748 ms | 30.750 ms |

### Rate and accounting envelope

| Q27 c=4 measurement | N=5 median or exact total |
|---|---:|
| Mixed output throughput | **144.462 completion tok/s** |
| Mixed requests/s | 2.408 |
| Mixed billed prompt rate | 11,701.386 prompt tok/s |
| Mixed computed prompt rate | 1,170.139 prompt tok/s |
| Pure-cold output throughput | 35.638 completion tok/s |
| c=4 mixed prompt / cached / completion tokens | 486,000 / **437,400** / 6,000 |
| Engine cached counters | `cached_tokens_in=437,400`; `prefix_cache_hit_tokens=437,400` |
| Cache hits / misses | 90 / 10 |
| Session defers / VRAM defers / OOM parks | 0 / 0 / 0 |
| Prefix-cache budget / observed c=4 peak | 4,096 MiB / 4,021.664 MiB |

Both servers were active in the same c=4 windows. Pair-window throughput, measured from the shared release barrier until the slower model drained, was **288.923 completion tok/s median** across five repetitions.

### Q27 capacity headroom

| c/model | Cold output tok/s | 90%-hit output tok/s | Mixed hit TTFT p95 | Mixed all TTFT p50 | Mixed all TTFT p99 | Clean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.542 | 65.615 | 2.098 ms | 1.923 ms | 1,406.916 ms | PASS |
| 2 | 31.752 | 96.558 | 14.939 ms | 14.490 ms | 1,492.210 ms | PASS |
| **4 sold cap** | 35.638 | 144.462 | 21.565 ms | 18.660 ms | 2,912.474 ms | PASS |
| 8 | 37.186 | 175.015 | 573.134 ms | 25.746 ms | 2,947.616 ms | PASS |
| **12 measured knee** | 37.691 | 186.216 | 592.804 ms | 32.698 ms | 3,010.725 ms | PASS |
| 16 | 37.496 | 182.761 | 601.889 ms | 321.490 ms | 3,067.277 ms | PASS |
| 24 | 37.673 | 186.116 | 613.320 ms | 332.300 ms | 4,609.506 ms | PASS |
| 32 | 37.925 | 192.598 | 1,185.358 ms | 903.568 ms | 6,033.025 ms | PASS |
| 48 | 37.708 | 188.011 | 1,495.225 ms | 1,484.800 ms | 7,639.438 ms | PASS |

The clean throughput knee is c=12, or **200% headroom** above the sold cap of four. Across the full campaign this model completed **2,400 requests** and 90/90 cells clean; cached tokens reconcile 5,248,800 = 5,248,800 = 5,248,800.

## Customer one-page envelope — Q35

Frozen workload: 4,860 prompt tokens plus 60 completion tokens (81:1), c=4/model, with Q27 active on the other GPU. Each row pools five interleaved cells; mixed traffic is 90 full-prefix hits and 10 real misses, and pure cold is a separate 100-request population.

### First-content TTFT

| Q35 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 7.678 ms | 9.410 ms | 105.684 ms | 534.979 ms | 932.952 ms |
| Cache hits only | 90 | 7.623 ms | 8.743 ms | 10.896 ms | 101.203 ms | 105.684 ms |
| Misses inside the 90%-hit mix | 10 | 535.851 ms | 547.548 ms | 932.952 ms | 1,034.606 ms | 1,034.606 ms |
| Pure cold arm | 100 | 1,955.707 ms | 1,963.363 ms | 1,978.655 ms | 2,013.664 ms | 2,018.873 ms |

Typical mixed TTFT is 7.678 ms; cache-hit p95 is 101.203 ms. The 10% miss class puts mixed p99 at 932.952 ms; pure-cold c=4 p50 is 1,955.707 ms.

### Full-response latency for 60 completion tokens

| Q35 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 412.490 ms | 898.530 ms | 911.126 ms | 939.526 ms | 1,432.201 ms |
| Cache hits only | 90 | 408.017 ms | 886.288 ms | 899.533 ms | 904.657 ms | 1,403.840 ms |
| Misses inside the 90%-hit mix | 10 | 934.153 ms | 946.518 ms | 1,432.201 ms | 1,439.987 ms | 1,439.987 ms |
| Pure cold arm | 100 | 2,333.352 ms | 2,348.660 ms | 2,383.363 ms | 2,401.440 ms | 2,405.565 ms |

### Inter-token latency

| Q35 c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 90%-hit mix, all traffic | 100 | 6.783 ms | 13.499 ms | 15.114 ms | 15.170 ms | 22.003 ms |
| Cache hits only | 90 | 6.785 ms | 13.536 ms | 15.115 ms | 15.282 ms | 23.626 ms |
| Misses inside the 90%-hit mix | 10 | 6.759 ms | 6.783 ms | 6.871 ms | 8.462 ms | 8.462 ms |
| Pure cold arm | 100 | 6.465 ms | 6.580 ms | 11.132 ms | 11.222 ms | 11.391 ms |

### Rate and accounting envelope

| Q35 c=4 measurement | N=5 median or exact total |
|---|---:|
| Mixed output throughput | **394.157 completion tok/s** |
| Mixed requests/s | 6.569 |
| Mixed billed prompt rate | 31,926.717 prompt tok/s |
| Mixed computed prompt rate | 3,192.672 prompt tok/s |
| Pure-cold output throughput | 102.436 completion tok/s |
| c=4 mixed prompt / cached / completion tokens | 486,000 / **437,400** / 6,000 |
| Engine cached counters | `cached_tokens_in=437,400`; `prefix_cache_hit_tokens=437,400` |
| Cache hits / misses | 90 / 10 |
| Session defers / VRAM defers / OOM parks | 0 / 0 / 0 |
| Prefix-cache budget / observed c=4 peak | 4,096 MiB / 4,021.311 MiB |

Both servers were active in the same c=4 windows. Pair-window throughput, measured from the shared release barrier until the slower model drained, was **288.923 completion tok/s median** across five repetitions.

### Q35 capacity headroom

| c/model | Cold output tok/s | 90%-hit output tok/s | Mixed hit TTFT p95 | Mixed all TTFT p50 | Mixed all TTFT p99 | Clean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.183 | 189.906 | 1.800 ms | 1.502 ms | 422.540 ms | PASS |
| 2 | 92.985 | 283.858 | 6.408 ms | 5.945 ms | 530.719 ms | PASS |
| **4 sold cap** | 102.436 | 394.157 | 101.203 ms | 7.678 ms | 932.952 ms | PASS |
| 8 | 106.870 | 471.798 | 206.373 ms | 13.085 ms | 1,050.243 ms | PASS |
| 12 | 107.293 | 475.742 | 217.675 ms | 113.850 ms | 1,079.832 ms | PASS |
| 16 | 107.527 | 480.054 | 315.293 ms | 122.110 ms | 1,095.764 ms | PASS |
| 24 | 108.236 | 493.022 | 318.941 ms | 219.753 ms | 1,632.350 ms | PASS |
| **32 measured knee** | 108.337 | 500.281 | 653.080 ms | 321.499 ms | 2,170.765 ms | PASS |
| 48 | 107.983 | 493.729 | 1,050.581 ms | 512.606 ms | 2,794.964 ms | PASS |

The clean throughput knee is c=32, or **700% headroom** above the sold cap of four. Across the full campaign this model completed **2,400 requests** and 90/90 cells clean; cached tokens reconcile 5,248,800 = 5,248,800 = 5,248,800.

## Exactness and pinned inputs

- Both physical GPUs passed the full kernel checker. Q27 and Q35 each passed prefill/decode and batched-prime/tokenwise argmax MATCH plus `run-spec` K=1..8.
- Both serial partial-prefix gates passed N=3 with byte-identical cold/partial/full output and exact client/engine cached-token reconciliation.
- At c=4, each model's cache-hit output hashes are a subset of its cold output hashes. This does not claim identity across different batching compositions.

| Input | SHA-256 |
|---|---|
| Runtime source | `ac6ef049b8661008c0da91f4747f68f4dabdaa04` |
| `memra-server` | `53b31fc0ba7b09d597a8bff3ce210ac91e5473d9460404a72e293f6ad7003761` |
| `kernel-check` | `3c3b9dcb00992ffb929aa52dd6a1a5d0ad1b6edf80c6cf21f445a682dad8a20a` |
| `run-gen` | `10c0840f3173b08b3d9757a4ab88c1d59cda69866082db30ea67c44567123b96` |
| `run-spec` | `2dd70158c959ed4110c8f777e8afce4bc08b366e87462ac6a0bf9b182b622b76` |
| Q27 artifact | `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517` |
| Q27 external draft | `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581` |
| Q27 embedded `tokenizer.chat_template` (7,764 bytes) | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| Q35 artifact | `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf` |
| Q35 external draft | `ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a` |
| Q35 embedded `tokenizer.chat_template` (8,057 bytes) | `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c` |
| Workload lock | `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34` |
| Canonical scored prompt IDs | `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb` |

## Method and receipts

- One uninterrupted `/tmp/memra-gpu.lock` hold ran from 2026-08-12T07:04:24Z through the sealed PASS at 2026-08-12T08:02:35Z; the gateway soak queued behind it.
- Both target servers stayed live together through 180 cells and 4,800 scored requests. Arms alternated, base-width order rotated, and every width used N=5 without artificial cooldown or clock changes.
- Thermal maxima by GPU: `{"0": {"max_memory_used_mib": 43219.0, "max_power_w": 509.79, "max_temperature_c": 66.0, "max_utilization_percent": 100.0}, "1": {"max_memory_used_mib": 31151.0, "max_power_w": 438.4, "max_temperature_c": 53.0, "max_utilization_percent": 99.0}}`.
- Campaign manifest: `068f6202a3300bfedfdc8d657205c30e3b3d80c5d12be774726dc4e808592e97` (31 files verified).
- Correctness manifest: `276ca2a95abb3037bdcfc730b403813e1ecb49e975296bc137019d630fc82f5c` (12 files verified).
- Machine-readable verdict: [`summary.json`](summary.json); explicit regression matrix and Q27 comparison: [`analysis.json`](analysis.json); sealed raw evidence: [`raw/campaign/`](raw/campaign/) and [`raw/gates/`](raw/gates/).

No runtime code, generated performance board, README number, merge, tag, push, or formatting surface changed in this lane.
