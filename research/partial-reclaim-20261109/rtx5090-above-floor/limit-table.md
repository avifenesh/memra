| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 4949 | 6 (0, 0) | 30 / 15.2 (9 / 4.5) | 0 / 0 | 0 / 1 | 0 | 0 of 4, 29.6 / 29.7 | 0 of 6, 0, 7.86 | 42/42 |
| limit-2-B | 4949 | 14 (12, 0) | 28 / 14.4 (4 / 2.1) | 7 / 7 | 0 / 1 | 0 | 0 of 4, 29.8 / 29.9 | 1 of 6, 28960, 7.64 | 42/42 |
| limit-3-B | 4949 | 16 (13, 0) | 33 / 16.9 (1 / 2.1) | 8 / 7 | 0 / 1 | 0 | 0 of 4, 28.4 / 28.5 | 0 of 6, 0, 7.62 | 42/42 |
| limit-4-A | 4949 | 6 (0, 0) | 25 / 12.8 (9 / 4.5) | 0 / 0 | 0 / 1 | 0 | 0 of 4, 28.9 / 28.9 | 1 of 6, 28960, 7.52 | 42/42 |

Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, free before -> after MB, outcome):
- limit-2-B: 4 entries, 2060 MB for 1805 MB (818 owed, 3 priming); kept 5 (2421 MB); free 2047 -> 4107 MB; covered
- limit-2-B: 2 entries, 927 MB for 826 MB (483 owed, 0 priming); kept 7 (3640 MB); free 2690 -> 3618 MB; covered
- limit-2-B: 3 entries, 1494 MB for 1351 MB (818 owed, 1 priming); kept 4 (2146 MB); free 2165 -> 3659 MB; covered
- limit-2-B: 2 entries, 1073 MB for 974 MB (1293 owed, 2 priming); kept 2 (1073 MB); free 2542 -> 3615 MB; covered
- limit-2-B: 2 entries, 1073 MB for 1018 MB (1768 owed, 3 priming); kept 0 (0 MB); free 2498 -> 3571 MB; covered
- limit-2-B: 3 entries, 1609 MB for 1185 MB (483 owed, 0 priming); kept 1 (536 MB); free 2332 -> 3941 MB; covered
- limit-2-B: 1 entries, 536 MB for 795 MB (1293 owed, 2 priming); kept 0 (0 MB); free 2721 -> 3258 MB; covered
- limit-2-B: 3 entries, 1609 MB for 1265 MB (483 owed, 0 priming); kept 1 (536 MB); free 2251 -> 3861 MB; covered
- limit-2-B: 1 entries, 536 MB for 515 MB (1355 owed, 2 priming); kept 0 (0 MB); free 2525 -> 3062 MB; covered
- limit-2-B: 2 entries, 1019 MB for 607 MB (483 owed, 0 priming); kept 2 (965 MB); free 2433 -> 3452 MB; covered
- limit-2-B: 3 entries, 1448 MB for 1999 MB (1891 owed, 2 priming); kept 0 (0 MB); free 2449 -> 3897 MB; covered
- limit-2-B: 2 entries, 965 MB for 486 MB (483 owed, 0 priming); kept 1 (483 MB); free 2554 -> 3519 MB; covered
- limit-3-B: 4 entries, 2060 MB for 1805 MB (818 owed, 3 priming); kept 5 (2421 MB); free 2047 -> 4107 MB; covered
- limit-3-B: 2 entries, 927 MB for 826 MB (483 owed, 0 priming); kept 7 (3640 MB); free 2690 -> 3618 MB; covered
- limit-3-B: 3 entries, 1494 MB for 1351 MB (818 owed, 1 priming); kept 4 (2146 MB); free 2165 -> 3659 MB; covered
- limit-3-B: 2 entries, 1073 MB for 974 MB (1293 owed, 2 priming); kept 2 (1073 MB); free 2542 -> 3615 MB; covered
- limit-3-B: 2 entries, 1073 MB for 1018 MB (1768 owed, 3 priming); kept 0 (0 MB); free 2498 -> 3571 MB; covered
- limit-3-B: 3 entries, 1609 MB for 1185 MB (483 owed, 0 priming); kept 1 (536 MB); free 2332 -> 3941 MB; covered
- limit-3-B: 1 entries, 536 MB for 795 MB (1293 owed, 2 priming); kept 0 (0 MB); free 2721 -> 3258 MB; covered
- limit-3-B: 3 entries, 1609 MB for 1265 MB (483 owed, 0 priming); kept 1 (536 MB); free 2251 -> 3861 MB; covered
- limit-3-B: 1 entries, 536 MB for 983 MB (1293 owed, 2 priming); kept 0 (0 MB); free 2533 -> 3070 MB; covered
- limit-3-B: 4 entries, 2146 MB for 1668 MB (483 owed, 0 priming); kept 0 (0 MB); free 1848 -> 3994 MB; covered
- limit-3-B: 2 entries, 1019 MB for 755 MB (483 owed, 0 priming); kept 2 (965 MB); free 2285 -> 3304 MB; covered
- limit-3-B: 4 entries, 1931 MB for 2640 MB (2345 owed, 3 priming); kept 0 (0 MB); free 2244 -> 4175 MB; covered
- limit-3-B: 2 entries, 927 MB for 504 MB (483 owed, 0 priming); kept 7 (3341 MB); free 2653 -> 3580 MB; covered

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
