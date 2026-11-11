| cell | B in flight s | A gaps overlapping B: n, p50 ms, p99 ms, max ms | A gaps before B: n, p50 ms, p99 ms, max ms |
|---|---|---|---|
| fix-itl-1 | 1.832 | 4, 34.1, 1854.4, 1854.4 | 47, 55.0, 78.4, 78.4 |
| fix-itl-2 | 1.832 | 4, 34.1, 1856.2, 1856.2 | 47, 55.2, 78.4, 78.4 |
| fix-itl-trace | 1.831 | 4, 34.2, 1855.9, 1855.9 | 47, 55.4, 78.5, 78.5 |
| main-itl-1 | 7.947 | 4, 34.1, 7975.8, 7975.8 | 47, 55.3, 78.6, 78.6 |
| main-itl-2 | 7.948 | 4, 34.2, 7977.1, 7977.1 | 47, 55.2, 79.0, 79.0 |

fix-itl-trace: 480 SLRU pairs layer calls, modes ['Pinned'], lock hold per call p50 1.08 ms, p99 1.68 ms, max 18.34 ms, sum 549.6 ms
