# Qwen code-only v2 held-out result

The first all-format held-out qualifier stopped at 4K prose after
8,192 reasoning tokens with no final answer. That failure is retained
separately. This version qualifies requested **code** only on a new
tuple-utility task, then compares the selected development C against
fixed K=3/C=0 and fixed K=2/C=0 on six untouched task families.

Selected development arm: `c015` (pmin=0.15, zero-draft=off).
Code qualification: 4/4 covered; prose qualification diagnostic: 4/4 covered.

| Code prompt tokens | Pairs | All arms code-covered | K=3/C=0 tok/s | Selected tok/s | Change vs K=3/C=0 (95% interval) | K=2/C=0 tok/s | Change vs K=2/C=0 (95% interval) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 6 | 6 | 174.42 | 173.23 | -0.68% [-5.81%, +6.39%] | 160.70 | +7.79% [+1.73%, +15.72%] |
| 1,024 | 6 | 6 | 159.11 | 155.96 | -1.98% [-3.36%, +1.11%] | 151.09 | +3.22% [+0.38%, +9.39%] |
| 4,096 | 6 | 6 | 135.77 | 129.00 | -4.98% [-6.85%, -1.76%] | 132.74 | -2.81% [-8.01%, +3.06%] |
| 16,384 | 6 | 6 | 88.01 | 108.76 | +23.57% [-5.13%, +54.31%] | 82.73 | +31.45% [+1.51%, +70.60%] |

Pooled across code lengths (the same six topics recur at each length,
so no pooled bootstrap interval is attached):

| Pairs | K=3/C=0 tok/s | Selected tok/s | Change vs K=3/C=0 | K=2/C=0 tok/s | Change vs K=2/C=0 | Selected confidence-shortened rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 125.88 | 127.17 | +1.03% | 120.91 | +5.17% | 1011/11036 |

Matched loop exclusions: 0.
The primary code rows retain capped outputs and format misses; the
machine-readable report includes the all-arms-format-covered subset,
paired gains, output-length ratios, latency ratios, and prose diagnostics.
Intervals are pointwise whole-scenario resamples of this synthetic
corpus and do not adjust for selecting C on the earlier grid.

Both versions use the same pinned Qwen NVFP4+Q5_K artifact, full embedded
MTP head, sampled 0.7/20/0.95 decode, one non-production RTX 5090, and
fresh native caches. This is a fixed-cutoff test, not live C learning,
warm-session KV, HTTP/concurrency qualification, or functional code
correctness.
