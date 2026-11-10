| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.531, 0.534, 0.536 | 0.536, 0.537, 0.539 | 0.534 | 0.537 | 1.005 | 0.79% | 0.69% | OVERLAP |
| 20 | decode at 4096 | 0.514, 0.512, 0.514 | 0.512, 0.514, 0.514 | 0.514 | 0.514 | 1.000 | 0.39% | 0.45% | OVERLAP |
| 21 | decode at 4096 | 0.419, 0.420, 0.421 | 0.419, 0.419, 0.421 | 0.420 | 0.419 | 1.000 | 0.36% | 0.48% | OVERLAP |
| 2 | prefill 4096 | 34.411, 34.915, 34.938 | 34.693, 34.749, 35.002 | 34.915 | 34.749 | 0.995 | 1.51% | 0.89% | OVERLAP |
| 20 | prefill 4096 | 32.576, 32.850, 32.935 | 32.692, 32.819, 33.012 | 32.850 | 32.819 | 0.999 | 1.09% | 0.97% | OVERLAP |
| 21 | prefill 4096 | 31.467, 31.685, 31.790 | 31.535, 31.665, 31.845 | 31.685 | 31.665 | 0.999 | 1.02% | 0.98% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.537, 0.537, 0.532 | 0.533, 0.531, 0.532 | 0.537 | 0.532 | 0.990 | 1.01% | 0.47% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.934, 0.931, 0.936 | 0.932, 0.932, 0.935 | 0.934 | 0.932 | 0.998 | 0.51% | 0.32% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 104 max 134, gpu0_temp_c median 76 max 83, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 46 max 52, cotenant_power_w median 280 max 281
