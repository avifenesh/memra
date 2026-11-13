| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 4.846, 4.918, 4.923 | 4.965 | 546.4 | e482c0ab7d65 |
| run2-B | B | 5.567, 5.601, 5.607 | 5.738 | 622.3 | 02b39a7236fd |
| run3-B | B | 5.586, 5.627, 5.630 | 5.669 | 625.3 | 02b39a7236fd |
| run4-A | A | 4.930, 4.973, 4.979 | 5.015 | 552.6 | e482c0ab7d65 |
| run5-A | A | 4.944, 4.991, 4.999 | 5.035 | 554.5 | e482c0ab7d65 |
| run6-B | B | 5.643, 5.675, 5.681 | 5.721 | 630.6 | 02b39a7236fd |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 4.918, 4.973, 4.991 | 4.973 | 552.6 | 1.000 |
| B | 5.601, 5.627, 5.675 | 5.627 | 625.3 | 1.132 |

generated ids identical across every run of every arm: False
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 21:01:31 21:01:59
