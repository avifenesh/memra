| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.696, 0.696, 0.696 | 0.747, 0.748, 0.749 | 0.696 | 0.748 | 1.075 | 0.00% | 0.31% | NO OVERLAP |
| 20 | decode at 4096 | 0.683, 0.683, 0.682 | 0.726, 0.727, 0.725 | 0.683 | 0.726 | 1.064 | 0.16% | 0.19% | NO OVERLAP |
| 21 | decode at 4096 | 0.568, 0.569, 0.569 | 0.631, 0.631, 0.630 | 0.569 | 0.631 | 1.109 | 0.09% | 0.25% | NO OVERLAP |
| 2 | prefill 4096 | 43.372, 43.313, 43.325 | 39.819, 39.797, 39.724 | 43.325 | 39.797 | 0.919 | 0.14% | 0.24% | NO OVERLAP |
| 20 | prefill 4096 | 40.972, 40.911, 40.923 | 37.527, 37.501, 37.476 | 40.923 | 37.501 | 0.916 | 0.15% | 0.14% | NO OVERLAP |
| 21 | prefill 4096 | 39.481, 39.441, 39.437 | 36.155, 36.146, 36.142 | 39.441 | 36.146 | 0.916 | 0.11% | 0.04% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.693, 0.695, 0.693 | 0.746, 0.749, 0.748 | 0.693 | 0.748 | 1.079 | 0.29% | 0.40% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.245, 1.246, 1.246 | 1.348, 1.345, 1.347 | 1.246 | 1.347 | 1.082 | 0.12% | 0.20% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 106 max 174, gpu0_temp_c median 75 max 78, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 54 max 68, cotenant_power_w median 280 max 283
