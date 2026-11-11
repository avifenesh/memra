| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 6.688, 6.790, 6.803 | 6.896 | 754.4 | c0287c9fd93d |
| run2-B | B | 7.049, 7.177, 7.185 | 7.289 | 797.5 | 57ac6dfb7761 |
| run3-B | B | 7.099, 7.198, 7.206 | 7.299 | 799.8 | 57ac6dfb7761 |
| run4-A | A | 6.755, 6.839, 6.841 | 6.940 | 759.9 | c0287c9fd93d |
| run5-A | A | 6.820, 6.919, 6.922 | 7.014 | 768.8 | c0287c9fd93d |
| run6-B | B | 7.161, 7.254, 7.267 | 7.358 | 806.1 | 57ac6dfb7761 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 6.790, 6.839, 6.919 | 6.839 | 759.9 | 1.000 |
| B | 7.177, 7.198, 7.254 | 7.198 | 799.8 | 1.052 |

generated ids identical across every run of every arm: False
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 21:01:31 21:01:59
