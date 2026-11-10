| run | arm | step ms, median of each round | step ms p95 | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 10.059, 7.491, 7.752 | 14.567 | 969.0 | 3e23de40e48e |
| run2-B | B | 7.384, 7.642, 7.681 | 7.862 | 955.3 | 3e23de40e48e |
| run3-B | B | 7.391, 7.660, 7.714 | 8.617 | 957.5 | 3e23de40e48e |

A 7.752 ms/step (969.0 us/layer, spread 7.491-10.059); B 7.651 ms/step (956.4 us/layer, spread 7.384-7.714); B/A 0.987; generated ids identical across every run: True
