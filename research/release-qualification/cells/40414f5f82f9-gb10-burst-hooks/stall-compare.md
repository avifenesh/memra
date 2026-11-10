| clients | think ms | long tokens in 30 s (lane fix / main v0.141.0) | longest gap s (lane fix) | short requests | short p50 s | short p95 s | demotions | text vs lane fix | text vs main v0.141.0 |
|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 2 | 0 | 74 (74 / 1) | 0.96 (0.94) | 130 | 0.472 | 0.519 | 0 | prefix-equal | prefix-equal |
| 4 | 0 | 57 (56 / 1) | 1.79 (1.77) | 160 | 0.835 | 0.938 | 1 | prefix-equal | prefix-equal |
| 4 | 200 | 50 (50 / 1) | 1.56 (1.56) | 136 | 0.731 | 0.747 | 0 | prefix-equal | prefix-equal |

verdict: PASS (the long request advances in every cell, longest gap under 2 s, texts prefix-equal)
