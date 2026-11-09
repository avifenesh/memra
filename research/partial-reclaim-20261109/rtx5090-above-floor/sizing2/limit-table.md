| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-B | 4949 | 16 (15, 0) | 32 / 16.7 (4 / 2.1) | 14 / 14 | 0 / 4 | 0 | 0 of 4, 28.3 / 28.3 | 0 of 6, 0, 7.20 | 0/0 |

Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, free before -> after MB, outcome):
- limit-1-B: 2 entries, 1030 MB for 558 MB (483 owed, 0 priming); kept 7 (3670 MB); free 2959 -> 3989 MB; covered
- limit-1-B: 2 entries, 1030 MB for 759 MB (483 owed, 0 priming); kept 6 (3176 MB); free 2758 -> 3788 MB; covered
- limit-1-B: 2 entries, 1030 MB for 960 MB (483 owed, 0 priming); kept 5 (2682 MB); free 2556 -> 3587 MB; covered
- limit-1-B: 3 entries, 1609 MB for 1161 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2355 -> 3965 MB; covered
- limit-1-B: 2 entries, 1073 MB for 665 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2851 -> 3924 MB; covered
- limit-1-B: 2 entries, 1073 MB for 705 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2811 -> 3884 MB; covered
- limit-1-B: 2 entries, 1073 MB for 746 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2771 -> 3844 MB; covered
- limit-1-B: 2 entries, 1073 MB for 786 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2731 -> 3803 MB; covered
- limit-1-B: 2 entries, 1073 MB for 826 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2690 -> 3763 MB; covered
- limit-1-B: 2 entries, 1073 MB for 823 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2693 -> 3766 MB; covered
- limit-1-B: 2 entries, 1073 MB for 823 MB (483 owed, 0 priming); kept 3 (1609 MB); free 2693 -> 3766 MB; covered
- limit-1-B: 4 entries, 2146 MB for 1728 MB (1315 owed, 1 priming); kept 1 (536 MB); free 2126 -> 4272 MB; covered
- limit-1-B: 1 entries, 536 MB for 664 MB (1830 owed, 2 priming); kept 0 (0 MB); free 2339 -> 2875 MB; covered
- limit-1-B: 2 entries, 927 MB for 558 MB (483 owed, 0 priming); kept 4 (2000 MB); free 2959 -> 3886 MB; covered
- limit-1-B: 2 entries, 927 MB for 703 MB (483 owed, 0 priming); kept 4 (2146 MB); free 2813 -> 3741 MB; covered

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
