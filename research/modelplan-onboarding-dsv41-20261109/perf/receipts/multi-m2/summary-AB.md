| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.588, 0.591, 0.592 | 0.550, 0.549, 0.551 | 0.591 | 0.550 | 0.932 | 0.81% | 0.38% | NO OVERLAP |
| 20 | decode at 4096 | 0.565, 0.566, 0.565 | 0.527, 0.528, 0.529 | 0.565 | 0.528 | 0.934 | 0.11% | 0.30% | NO OVERLAP |
| 21 | decode at 4096 | 0.472, 0.475, 0.475 | 0.435, 0.436, 0.436 | 0.475 | 0.436 | 0.918 | 0.70% | 0.23% | NO OVERLAP |
| 2 | prefill 4096 | 34.010, 34.118, 34.191 | 33.854, 33.998, 34.295 | 34.118 | 33.998 | 0.996 | 0.53% | 1.30% | OVERLAP |
| 20 | prefill 4096 | 31.758, 32.139, 32.221 | 31.906, 32.033, 32.334 | 32.139 | 32.033 | 0.997 | 1.44% | 1.34% | OVERLAP |
| 21 | prefill 4096 | 30.685, 31.003, 31.117 | 30.826, 30.938, 31.229 | 31.003 | 30.938 | 0.998 | 1.39% | 1.30% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.587, 0.591, 0.589 | 0.551, 0.553, 0.551 | 0.589 | 0.551 | 0.935 | 0.78% | 0.38% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.038, 1.044, 1.043 | 0.960, 0.961, 0.963 | 1.043 | 0.961 | 0.922 | 0.61% | 0.34% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 82 max 127, gpu0_temp_c median 64 max 73, gpu0_sm_mhz median 2272 max 2325, gpu0_fan median 31 max 42, cotenant_power_w median 280 max 281
