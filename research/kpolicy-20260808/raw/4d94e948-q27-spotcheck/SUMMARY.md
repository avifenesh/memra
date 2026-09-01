# K-policy matrix summary

Rows: 36. N=3 independent server boots per cell.
Primary rate is completion tokens divided by client-observed request wall time.

| model | class | K | net tok/s median (range) | server tok/s median | acceptance median | prompt tok | cached tok |
|---|---|---:|---:|---:|---:|---:|---:|
| q27 | cached-long | 0 | 72.92 (72.87-73.24) | 73.37 | plain | 5493 | 5411 |
| q27 | cached-long | 2 | 124.47 (124.44-124.57) | 127.20 | 66.36% | 5496 | 5477 |
| q27 | cached-long | 3 | 116.21 (116.11-116.41) | 118.61 | 45.06% | 5494 | 5475 |
| q27 | cached-long | 5 | 100.05 (99.78-100.60) | 101.80 | 28.30% | 5495 | 5476 |
| q27 | cold-long | 0 | 39.19 (39.08-39.30) | 39.50 | plain | 5411 | 0 |
| q27 | cold-long | 2 | 52.22 (52.09-52.33) | 52.76 | 79.00% | 5411 | 0 |
| q27 | cold-long | 3 | 55.46 (55.38-55.53) | 56.08 | 77.78% | 5411 | 0 |
| q27 | cold-long | 5 | 55.71 (55.69-55.87) | 56.31 | 60.00% | 5411 | 0 |
| q27 | cold-short | 0 | 72.40 (72.23-72.45) | 73.07 | plain | 28 | 0 |
| q27 | cold-short | 2 | 136.30 (135.67-136.34) | 138.62 | 79.00% | 28 | 0 |
| q27 | cold-short | 3 | 143.57 (142.60-143.65) | 146.23 | 67.46% | 28 | 0 |
| q27 | cold-short | 5 | 142.95 (141.96-143.14) | 145.53 | 53.71% | 28 | 0 |

## Per-class ordering

- q27 cached-long: K=2 124.47, K=3 116.21, K=5 100.05, K=0 72.92
- q27 cold-long: K=5 55.71, K=3 55.46, K=2 52.22, K=0 39.19
- q27 cold-short: K=3 143.57, K=5 142.95, K=2 136.30, K=0 72.40
