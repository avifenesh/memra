| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 12.307, 12.186, 12.201 | 15.404 | 610.1 | 5486f13b7289 |
| run2-B | B | 12.136, 12.107, 11.993 | 14.438 | 605.3 | 5486f13b7289 |
| run3-B | B | 11.973, 11.978, 12.021 | 15.026 | 598.9 | 5486f13b7289 |
| run4-A | A | 12.410, 12.251, 12.217 | 15.417 | 612.6 | 5486f13b7289 |
| run5-A | A | 12.397, 12.193, 12.253 | 15.971 | 612.6 | 5486f13b7289 |
| run6-B | B | 12.024, 12.079, 12.071 | 13.369 | 603.5 | 5486f13b7289 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 12.201, 12.251, 12.253 | 12.251 | 612.6 | 1.000 |
| B | 12.107, 11.978, 12.071 | 12.071 | 603.5 | 0.985 |

generated ids identical across every run of every arm: True
