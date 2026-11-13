| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 7.172, 7.150 | 7.259 | 358.1 | afe570f7ef12 |
| run2-B | B | 7.101, 7.094 | 7.191 | 354.9 | afe570f7ef12 |
| run3-B | B | 7.101, 7.092 | 7.192 | 354.8 | afe570f7ef12 |
| run4-A | A | 7.169, 7.152 | 7.256 | 358.0 | afe570f7ef12 |
| run5-A | A | 7.169, 7.159 | 7.255 | 358.2 | afe570f7ef12 |
| run6-B | B | 7.044, 7.040 | 7.128 | 352.1 | afe570f7ef12 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 7.161, 7.161, 7.164 | 7.161 | 358.1 | 1.000 |
| B | 7.097, 7.096, 7.042 | 7.096 | 354.8 | 0.991 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 01)
page cache: 0 runs with a drop inside; last drops seen 18:53:12 18:53:40
