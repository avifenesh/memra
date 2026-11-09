| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off, armed at end | evictions preflight / LRU | OOM lines | new prompts: rows, errors, TTFT p50 / p95 s | repeats: rows, hits, cached tokens, TTFT p50 s | repeat hits per 300 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| steady-1-A | 2 (0, 0) | 114 / 108.0 (57 / 54.2) | 0 / 0, no | 7 / 0 | 0 | 75, 0, 53.5 / 74.3 | 25, 7, 185216, 53.7 | 2/8 / 4/7 / 1/7 / 0/3 | 164/164 |
| steady-2-B | 26 (26, 0) | 86 / 81.5 (7 / 6.8) | 12 / 12, no | 11 / 0 | 0 | 87, 0, 35.6 / 63.8 | 29, 27, 717696, 1.5 | 9/9 / 8/8 / 7/9 / 3/3 | 164/164 |
| steady-3-B | 26 (26, 0) | 87 / 82.4 (7 / 6.8) | 13 / 13, no | 10 / 0 | 0 | 87, 0, 35.1 / 64.0 | 29, 27, 717696, 1.5 | 9/9 / 8/8 / 7/9 / 3/3 | 164/164 |
