| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.589, 0.592, 0.594 | 0.596, 0.595, 0.597 | 0.592 | 0.596 | 1.006 | 0.74% | 0.35% | NO OVERLAP |
| 20 | decode at 4096 | 0.568, 0.568, 0.567 | 0.569, 0.573, 0.573 | 0.568 | 0.573 | 1.010 | 0.11% | 0.72% | NO OVERLAP |
| 21 | decode at 4096 | 0.474, 0.472, 0.472 | 0.478, 0.481, 0.479 | 0.472 | 0.479 | 1.015 | 0.40% | 0.71% | NO OVERLAP |
| 2 | prefill 4096 | 34.485, 34.012, 34.091 | 34.520, 33.827, 34.213 | 34.091 | 34.213 | 1.004 | 1.39% | 2.03% | OVERLAP |
| 20 | prefill 4096 | 32.465, 32.055, 32.137 | 31.794, 31.937, 32.241 | 32.137 | 31.937 | 0.994 | 1.27% | 1.40% | OVERLAP |
| 21 | prefill 4096 | 31.426, 30.919, 31.015 | 30.692, 30.813, 31.102 | 31.015 | 30.813 | 0.993 | 1.64% | 1.33% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.589, 0.594, 0.588 | 0.592, 0.591, 0.598 | 0.589 | 0.592 | 1.005 | 0.93% | 1.18% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.040, 1.040, 1.041 | 1.050, 1.049, 1.051 | 1.040 | 1.050 | 1.009 | 0.09% | 0.13% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 98 max 118, gpu0_temp_c median 67 max 73, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 37 max 41, cotenant_power_w median 269 max 280
