| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.5 / 87.4 | 0 of 16, 0, 10.46 | 92/92 |
| limit-2-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.0 / 88.1 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-3-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.0 / 88.0 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-4-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.6 / 87.5 | 0 of 16, 0, 10.47 | 92/92 |

Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, free before -> after MB, outcome):
- limit-2-B: 8 entries, 7768 MB for 7553 MB (7114 owed, 6 priming); kept 49 (46428 MB); free 4606 -> 12374 MB; covered
- limit-2-B: 2 entries, 1895 MB for 1861 MB (8074 owed, 7 priming); kept 47 (44533 MB); free 4099 -> 5994 MB; covered
- limit-2-B: 3 entries, 2843 MB for 2041 MB (9035 owed, 8 priming); kept 44 (41691 MB); free 3919 -> 6762 MB; covered
- limit-2-B: 2 entries, 1895 MB for 1273 MB (9995 owed, 9 priming); kept 42 (39796 MB); free 4687 -> 6582 MB; covered
- limit-2-B: 2 entries, 1895 MB for 1545 MB (10990 owed, 10 priming); kept 40 (37901 MB); free 4507 -> 6402 MB; covered
- limit-2-B: 2 entries, 1895 MB for 1798 MB (11987 owed, 11 priming); kept 38 (36006 MB); free 4254 -> 6149 MB; covered
- limit-3-B: 8 entries, 7768 MB for 7553 MB (7114 owed, 6 priming); kept 49 (46428 MB); free 4606 -> 12374 MB; covered
- limit-3-B: 2 entries, 1895 MB for 1861 MB (8074 owed, 7 priming); kept 47 (44533 MB); free 4099 -> 5994 MB; covered
- limit-3-B: 3 entries, 2843 MB for 2041 MB (9035 owed, 8 priming); kept 44 (41691 MB); free 3919 -> 6762 MB; covered
- limit-3-B: 2 entries, 1895 MB for 1273 MB (9995 owed, 9 priming); kept 42 (39796 MB); free 4687 -> 6582 MB; covered
- limit-3-B: 2 entries, 1895 MB for 1545 MB (10990 owed, 10 priming); kept 40 (37901 MB); free 4507 -> 6402 MB; covered
- limit-3-B: 2 entries, 1895 MB for 1798 MB (11987 owed, 11 priming); kept 38 (36006 MB); free 4254 -> 6149 MB; covered

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
