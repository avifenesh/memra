| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 10.780, 10.005, 10.030 | 12.189 | 501.5 | 5486f13b7289 |
| run2-B | B | 9.809, 9.793, 9.802 | 10.021 | 490.1 | 5486f13b7289 |
| run3-B | B | 9.907, 9.790, 9.766 | 11.216 | 489.5 | 5486f13b7289 |
| run4-A | A | 10.192, 9.982, 9.974 | 10.386 | 499.1 | 5486f13b7289 |
| run5-A | A | 10.086, 9.968, 9.986 | 10.320 | 499.3 | 5486f13b7289 |
| run6-B | B | 9.865, 9.833, 9.778 | 10.056 | 491.6 | 5486f13b7289 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 10.030, 9.982, 9.986 | 9.986 | 499.3 | 1.000 |
| B | 9.802, 9.790, 9.833 | 9.802 | 490.1 | 0.982 |

generated ids identical across every run of every arm: True
