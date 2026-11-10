| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 4.692, 4.918, 4.942 | 5.012 | 546.5 | ce4ee1953f40 |
| run2-B | B | 4.679, 4.915, 4.925 | 5.016 | 546.1 | ce4ee1953f40 |
| run3-B | B | 4.711, 4.920, 4.950 | 5.018 | 546.6 | ce4ee1953f40 |
| run4-A | A | 4.675, 4.915, 4.941 | 5.010 | 546.1 | ce4ee1953f40 |
| run5-A | A | 4.695, 4.909, 4.939 | 5.007 | 545.4 | ce4ee1953f40 |
| run6-B | B | 4.674, 4.901, 4.917 | 4.999 | 544.5 | ce4ee1953f40 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 4.918, 4.915, 4.909 | 4.915 | 546.1 | 1.000 |
| B | 4.915, 4.920, 4.901 | 4.915 | 546.1 | 1.000 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 0)
