| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.562, 0.562, 0.564 | 0.559, 0.555, 0.557 | 0.562 | 0.557 | 0.992 | 0.37% | 0.70% | NO OVERLAP |
| 20 | decode at 4096 | 0.540, 0.539, 0.536 | 0.532, 0.532, 0.535 | 0.539 | 0.532 | 0.988 | 0.74% | 0.49% | NO OVERLAP |
| 21 | decode at 4096 | 0.444, 0.443, 0.446 | 0.437, 0.439, 0.439 | 0.444 | 0.439 | 0.988 | 0.52% | 0.48% | NO OVERLAP |
| 2 | prefill 4096 | 33.926, 34.269, 34.346 | 34.065, 34.160, 34.377 | 34.269 | 34.160 | 0.997 | 1.23% | 0.91% | OVERLAP |
| 20 | prefill 4096 | 32.020, 32.290, 32.367 | 32.164, 32.240, 32.465 | 32.290 | 32.240 | 0.998 | 1.07% | 0.93% | OVERLAP |
| 21 | prefill 4096 | 30.892, 31.163, 31.221 | 30.982, 31.085, 31.259 | 31.163 | 31.085 | 0.997 | 1.05% | 0.89% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.560, 0.565, 0.560 | 0.555, 0.553, 0.557 | 0.560 | 0.555 | 0.991 | 0.95% | 0.79% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.980, 0.984, 0.986 | 0.972, 0.973, 0.972 | 0.984 | 0.972 | 0.988 | 0.61% | 0.08% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 91 max 125, gpu0_temp_c median 69 max 75, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 43 max 45, cotenant_power_w median 280 max 288
