| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 5.828, 5.890, 5.900 | 6.000 | 654.4 | c0287c9fd93d |
| run2-B | B | 5.828, 5.904, 5.914 | 6.000 | 656.0 | c0287c9fd93d |
| run3-B | B | 5.837, 5.906, 5.918 | 6.000 | 656.3 | c0287c9fd93d |
| run4-A | A | 5.832, 5.906, 5.909 | 6.017 | 656.2 | c0287c9fd93d |
| run5-A | A | 5.852, 5.912, 5.923 | 6.030 | 656.9 | c0287c9fd93d |
| run6-B | B | 5.844, 5.918, 5.937 | 6.011 | 657.5 | c0287c9fd93d |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 5.890, 5.906, 5.912 | 5.906 | 656.2 | 1.000 |
| B | 5.904, 5.906, 5.918 | 5.906 | 656.3 | 1.000 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 17:09:50 17:10:20
