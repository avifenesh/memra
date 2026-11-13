| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 7.109, 7.104 | 7.200 | 355.3 | afe570f7ef12 |
| run2-B | B | 6.835, 6.845 | 6.927 | 342.0 | 1a648efed55d |
| run3-B | B | 6.838, 6.850 | 6.938 | 342.2 | 1a648efed55d |
| run4-A | A | 7.110, 7.116 | 7.201 | 355.7 | afe570f7ef12 |
| run5-A | A | 7.114, 7.127 | 7.208 | 356.0 | afe570f7ef12 |
| run6-B | B | 6.852, 6.866 | 6.951 | 342.9 | 1a648efed55d |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 7.106, 7.113, 7.120 | 7.113 | 355.7 | 1.000 |
| B | 6.840, 6.844, 6.859 | 6.844 | 342.2 | 0.962 |

generated ids identical across every run of every arm: False
slowdown: 6 runs, 0 flagged (cards 01)
page cache: 0 runs with a drop inside; last drops seen 21:52:30 21:52:58
