| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 7.152, 7.078 | 7.286 | 355.7 | afe570f7ef12 |
| run2-B | B | 7.681, 7.630 | 7.836 | 382.8 | 1a648efed55d |
| run3-C | C | 7.588, 7.605 | 7.694 | 379.8 | 1a648efed55d |
| run4-C | C | 7.626, 7.640 | 7.725 | 381.6 | 1a648efed55d |
| run5-B | B | 7.595, 7.618 | 7.706 | 380.3 | 1a648efed55d |
| run6-A | A | 7.119, 7.132 | 7.207 | 356.3 | afe570f7ef12 |
| run7-A | A | 7.117, 7.129 | 7.216 | 356.1 | afe570f7ef12 |
| run8-B | B | 7.636, 7.648 | 7.736 | 382.1 | 1a648efed55d |
| run9-C | C | 7.622, 7.642 | 7.723 | 381.6 | 1a648efed55d |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 7.115, 7.126, 7.123 | 7.123 | 356.1 | 1.000 |
| B | 7.656, 7.607, 7.642 | 7.642 | 382.1 | 1.073 |
| C | 7.596, 7.633, 7.632 | 7.632 | 381.6 | 1.071 |

generated ids identical across every run of every arm: False
slowdown: 9 runs, 0 flagged (cards 01)
page cache: 0 runs with a drop inside; last drops seen 21:52:30 21:52:58
