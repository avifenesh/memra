| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.674, 0.671, 0.672 | 0.703, 0.702, 0.702 | 0.672 | 0.702 | 1.043 | 0.43% | 0.23% | NO OVERLAP |
| 20 | decode at 4096 | 0.648, 0.649, 0.649 | 0.667, 0.668, 0.668 | 0.649 | 0.668 | 1.029 | 0.11% | 0.15% | NO OVERLAP |
| 21 | decode at 4096 | 0.552, 0.553, 0.553 | 0.587, 0.587, 0.588 | 0.553 | 0.587 | 1.063 | 0.11% | 0.19% | NO OVERLAP |
| 2 | prefill 4096 | 33.898, 34.131, 34.243 | 33.934, 33.993, 34.312 | 34.131 | 33.993 | 0.996 | 1.01% | 1.11% | OVERLAP |
| 20 | prefill 4096 | 31.932, 32.089, 32.224 | 31.987, 31.984, 32.351 | 32.089 | 31.987 | 0.997 | 0.91% | 1.15% | OVERLAP |
| 21 | prefill 4096 | 30.792, 30.972, 31.090 | 30.793, 30.858, 31.170 | 30.972 | 30.858 | 0.996 | 0.96% | 1.22% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.670, 0.670, 0.669 | 0.704, 0.704, 0.705 | 0.670 | 0.704 | 1.052 | 0.12% | 0.21% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.194, 1.194, 1.192 | 1.247, 1.248, 1.250 | 1.194 | 1.248 | 1.045 | 0.19% | 0.22% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 100 max 156, gpu0_temp_c median 74 max 83, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 46 max 52, cotenant_power_w median 280 max 285
