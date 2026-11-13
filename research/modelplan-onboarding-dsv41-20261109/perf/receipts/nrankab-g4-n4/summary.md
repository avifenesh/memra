| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 6.446, 6.667, 6.716 | 6.793 | 740.8 | ce4ee1953f40 |
| run2-B | B | 6.411, 6.622, 6.691 | 6.809 | 735.8 | ce4ee1953f40 |
| run3-B | B | 6.386, 6.630, 6.689 | 6.846 | 736.7 | ce4ee1953f40 |
| run4-A | A | 6.449, 6.639, 6.737 | 6.796 | 737.7 | ce4ee1953f40 |
| run5-A | A | 6.440, 6.681, 6.724 | 6.814 | 742.3 | ce4ee1953f40 |
| run6-B | B | 6.457, 6.666, 6.705 | 6.841 | 740.7 | ce4ee1953f40 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 6.667, 6.639, 6.681 | 6.667 | 740.8 | 1.000 |
| B | 6.622, 6.630, 6.666 | 6.630 | 736.7 | 0.994 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 11:39:20 11:39:48
