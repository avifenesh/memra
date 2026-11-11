| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 4.872, 4.962, 4.968 | 5.005 | 551.3 | e482c0ab7d65 |
| run2-B | B | 4.912, 4.948, 4.950 | 5.071 | 549.8 | 02b39a7236fd |
| run3-B | B | 4.919, 4.965, 4.968 | 5.008 | 551.7 | 02b39a7236fd |
| run4-A | A | 4.974, 5.012, 5.015 | 5.054 | 556.9 | e482c0ab7d65 |
| run5-A | A | 4.978, 5.027, 5.026 | 5.068 | 558.4 | e482c0ab7d65 |
| run6-B | B | 4.965, 5.000, 4.999 | 5.042 | 555.4 | 02b39a7236fd |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 4.962, 5.012, 5.026 | 5.012 | 556.9 | 1.000 |
| B | 4.948, 4.965, 4.999 | 4.965 | 551.7 | 0.991 |

generated ids identical across every run of every arm: False
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 21:52:30 21:52:58
