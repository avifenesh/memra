| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 6.931, 7.002, 7.009 | 7.109 | 778.0 | c0287c9fd93d |
| run2-B | B | 6.867, 6.960, 6.984 | 7.061 | 773.3 | c0287c9fd93d |
| run3-B | B | 6.930, 7.023, 7.024 | 7.139 | 780.3 | c0287c9fd93d |
| run4-A | A | 6.995, 7.088, 7.074 | 7.168 | 786.0 | c0287c9fd93d |
| run5-A | A | 6.994, 7.058, 7.069 | 7.172 | 784.2 | c0287c9fd93d |
| run6-B | B | 7.001, 7.121, 7.135 | 7.241 | 791.3 | c0287c9fd93d |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 7.002, 7.074, 7.058 | 7.058 | 784.2 | 1.000 |
| B | 6.960, 7.023, 7.121 | 7.023 | 780.3 | 0.995 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 17:09:50 17:10:20
