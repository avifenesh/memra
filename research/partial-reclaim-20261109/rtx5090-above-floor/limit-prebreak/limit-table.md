| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 4949 | 5 (0, 0) | 25 / 12.6 (9 / 4.5) | 0 / 0 | 0 / 1 | 0 | 0 of 4, 29.0 / 29.1 | 2 of 6, 57920, 7.75 | 42/42 |

Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, free before -> after MB, outcome):

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
