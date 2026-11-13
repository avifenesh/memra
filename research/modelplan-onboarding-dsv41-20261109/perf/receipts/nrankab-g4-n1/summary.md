| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 4.655, 4.863, 4.887 | 4.954 | 540.4 | ce4ee1953f40 |
| run2-B | B | 4.626, 4.863, 4.875 | 4.951 | 540.3 | ce4ee1953f40 |
| run3-B | B | 4.621, 4.861, 4.901 | 4.960 | 540.2 | ce4ee1953f40 |
| run4-A | A | 4.664, 4.876, 4.906 | 4.967 | 541.8 | ce4ee1953f40 |
| run5-A | A | 4.666, 4.890, 4.910 | 4.980 | 543.3 | ce4ee1953f40 |
| run6-B | B | 4.628, 4.887, 4.902 | 4.972 | 543.0 | ce4ee1953f40 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 4.863, 4.876, 4.890 | 4.876 | 541.8 | 1.000 |
| B | 4.863, 4.861, 4.887 | 4.863 | 540.3 | 0.997 |

generated ids identical across every run of every arm: True
slowdown: 6 runs, 0 flagged (cards 0)
page cache: 0 runs with a drop inside; last drops seen 11:39:20 11:39:48
