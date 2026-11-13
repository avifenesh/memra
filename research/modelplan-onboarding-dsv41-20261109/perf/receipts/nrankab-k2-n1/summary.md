| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 4.854, 4.924, 4.934 | 4.972 | 547.1 | e482c0ab7d65 |
| run2-B | B | 5.509, 5.558, 5.567 | 5.606 | 617.6 | 02b39a7236fd |
| run3-B | B | 5.548, 5.593, 5.598 | 5.644 | 621.4 | 02b39a7236fd |
| run4-A | A | 4.926, 4.971, 4.965 | 5.012 | 551.7 | e482c0ab7d65 |
| run5-A | A | 4.946, 4.979, 4.986 | 5.030 | 553.3 | e482c0ab7d65 |
| run6-B | B | 5.630, 5.671, 5.670 | 5.714 | 630.0 | 02b39a7236fd |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 4.924, 4.965, 4.979 | 4.965 | 551.7 | 1.000 |
| B | 5.558, 5.593, 5.670 | 5.593 | 621.4 | 1.126 |

generated ids identical across every run of every arm: False
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 21:01:31 21:01:59
