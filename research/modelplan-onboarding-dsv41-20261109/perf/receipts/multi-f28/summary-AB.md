| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.619, 0.623, 0.618 | 0.674, 0.671, 0.672 | 0.619 | 0.672 | 1.086 | 0.79% | 0.43% | NO OVERLAP |
| 20 | decode at 4096 | 0.594, 0.596, 0.594 | 0.648, 0.649, 0.649 | 0.594 | 0.649 | 1.091 | 0.45% | 0.11% | NO OVERLAP |
| 21 | decode at 4096 | 0.502, 0.503, 0.502 | 0.552, 0.553, 0.553 | 0.502 | 0.553 | 1.100 | 0.18% | 0.11% | NO OVERLAP |
| 2 | prefill 4096 | 33.823, 34.171, 34.257 | 33.898, 34.131, 34.243 | 34.171 | 34.131 | 0.999 | 1.27% | 1.01% | OVERLAP |
| 20 | prefill 4096 | 31.800, 32.150, 32.282 | 31.932, 32.089, 32.224 | 32.150 | 32.089 | 0.998 | 1.50% | 0.91% | OVERLAP |
| 21 | prefill 4096 | 30.660, 31.015, 31.100 | 30.792, 30.972, 31.090 | 31.015 | 30.972 | 0.999 | 1.42% | 0.96% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.616, 0.620, 0.621 | 0.670, 0.670, 0.669 | 0.620 | 0.670 | 1.081 | 0.71% | 0.12% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.090, 1.089, 1.089 | 1.194, 1.194, 1.192 | 1.089 | 1.194 | 1.096 | 0.06% | 0.19% | NO OVERLAP |

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
