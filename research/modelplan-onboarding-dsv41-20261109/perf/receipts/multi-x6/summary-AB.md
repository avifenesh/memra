| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.581, 0.583, 0.585 | 0.594, 0.595, 0.597 | 0.583 | 0.595 | 1.021 | 0.77% | 0.55% | NO OVERLAP |
| 20 | decode at 4096 | 0.559, 0.558, 0.560 | 0.571, 0.570, 0.572 | 0.559 | 0.571 | 1.021 | 0.30% | 0.42% | NO OVERLAP |
| 21 | decode at 4096 | 0.467, 0.470, 0.469 | 0.481, 0.483, 0.486 | 0.469 | 0.483 | 1.031 | 0.66% | 0.95% | NO OVERLAP |
| 2 | prefill 4096 | 34.338, 34.540, 34.621 | 34.428, 34.423, 34.701 | 34.540 | 34.428 | 0.997 | 0.82% | 0.81% | OVERLAP |
| 20 | prefill 4096 | 32.392, 32.522, 32.640 | 32.390, 32.441, 32.719 | 32.522 | 32.441 | 0.998 | 0.76% | 1.01% | OVERLAP |
| 21 | prefill 4096 | 31.244, 31.465, 31.524 | 31.246, 31.298, 31.539 | 31.465 | 31.298 | 0.995 | 0.89% | 0.93% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.584, 0.581, 0.583 | 0.593, 0.594, 0.594 | 0.583 | 0.594 | 1.019 | 0.45% | 0.29% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.027, 1.027, 1.028 | 1.054, 1.055, 1.061 | 1.027 | 1.055 | 1.027 | 0.15% | 0.66% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 105 max 128, gpu0_temp_c median 76 max 80, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 47 max 49, cotenant_power_w median 248 max 280
