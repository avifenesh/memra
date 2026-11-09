| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-B | 4949 | 15 (13, 1) | 28 / 14.5 (4 / 2.1) | 6 / 6 | 0 / 1 | 0 | 0 of 4, 28.8 / 28.8 | 1 of 6, 28960, 7.61 | 0/0 |

Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, free before -> after MB, outcome):
- limit-1-B: 4 entries, 2060 MB for 1805 MB (818 owed, 3 priming); kept 5 (2421 MB); free 2047 -> 4107 MB; covered
- limit-1-B: 2 entries, 927 MB for 826 MB (483 owed, 0 priming); kept 7 (3640 MB); free 2690 -> 3618 MB; covered
- limit-1-B: 3 entries, 1494 MB for 1351 MB (818 owed, 1 priming); kept 4 (2146 MB); free 2165 -> 3659 MB; covered
- limit-1-B: 2 entries, 1073 MB for 974 MB (1293 owed, 2 priming); kept 2 (1073 MB); free 2542 -> 3615 MB; covered
- limit-1-B: 2 entries, 1073 MB for 1018 MB (1768 owed, 3 priming); kept 0 (0 MB); free 2498 -> 3571 MB; covered
- limit-1-B: 3 entries, 1609 MB for 1185 MB (483 owed, 0 priming); kept 1 (536 MB); free 2332 -> 3941 MB; covered
- limit-1-B: 1 entries, 536 MB for 795 MB (1293 owed, 2 priming); kept 0 (0 MB); free 2721 -> 3258 MB; covered
- limit-1-B: 3 entries, 1609 MB for 1265 MB (483 owed, 0 priming); kept 1 (536 MB); free 2251 -> 3861 MB; covered
- limit-1-B: 1 entries, 536 MB for 983 MB (1293 owed, 2 priming); kept 0 (0 MB); free 2533 -> 3070 MB; covered
- limit-1-B: 2 entries, 1073 MB for 768 MB (483 owed, 0 priming); kept 2 (1073 MB); free 2272 -> 3345 MB; covered
- limit-1-B: 2 entries, 1073 MB for 765 MB (483 owed, 1 priming); kept 1 (483 MB); free 2275 -> 3348 MB; covered
- limit-1-B: 1 entries, 483 MB for 1034 MB (1891 owed, 2 priming); kept 0 (0 MB); free 2006 -> 2489 MB; still short, nothing unleased left
- limit-1-B: 2 entries, 965 MB for 486 MB (483 owed, 0 priming); kept 1 (483 MB); free 2554 -> 3519 MB; covered

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
