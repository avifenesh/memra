| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 4949 | 6 of 6 | 1319 | 273.18 | 985 | 0 (0) | 52 | 0 | 150/150 |
| mixed-2-B | 4949 | 6 of 6 | 1334 | 272.08 | 988 | 0 (0) | 52 | 0 | 149/149 |
| mixed-3-B | 4949 | 6 of 6 | 1336 | 271.99 | 991 | 0 (0) | 52 | 0 | 149/149 |
| mixed-4-A | 4949 | 6 of 6 | 1334 | 271.94 | 990 | 0 (0) | 52 | 0 | 148/148 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
