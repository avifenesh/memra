| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 54277 | 6 of 6 | 291 | 235.04 | 604 | 0 (0) | 0 | 0 | 135/135 |
| mixed-2-B | 54277 | 6 of 6 | 287 | 235.38 | 586 | 0 (0) | 0 | 0 | 135/135 |
| mixed-3-B | 54277 | 6 of 6 | 230 | 228.48 | 616 | 0 (0) | 0 | 0 | 134/134 |
| mixed-4-A | 54277 | 6 of 6 | 387 | 229.71 | 616 | 0 (0) | 0 | 0 | 135/135 |
| mixed-5-A | 54277 | 6 of 6 | 336 | 231.56 | 598 | 0 (0) | 0 | 0 | 135/135 |
| mixed-6-B | 54277 | 6 of 6 | 346 | 230.82 | 600 | 0 (0) | 0 | 0 | 135/135 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
