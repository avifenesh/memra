| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 54277 | 6 of 6 | 443 | 220.78 | 598 | 0 (0) | 0 | 0 | 131/131 |
| mixed-2-B | 54277 | 6 of 6 | 265 | 225.84 | 587 | 0 (0) | 0 | 0 | 127/127 |
| mixed-3-B | 54277 | 6 of 6 | 193 | 224.94 | 588 | 0 (0) | 0 | 0 | 127/127 |
| mixed-4-A | 54277 | 6 of 6 | 447 | 226.58 | 596 | 0 (0) | 0 | 0 | 130/130 |
| mixed-5-A | 54277 | 6 of 6 | 403 | 227.02 | 588 | 0 (0) | 0 | 0 | 130/130 |
| mixed-6-B | 54277 | 6 of 6 | 188 | 225.32 | 595 | 0 (0) | 0 | 0 | 127/127 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
