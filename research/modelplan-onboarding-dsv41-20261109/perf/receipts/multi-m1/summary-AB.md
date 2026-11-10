| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.591, 0.592, 0.594 | 0.578, 0.575, 0.577 | 0.592 | 0.577 | 0.974 | 0.57% | 0.55% | NO OVERLAP |
| 20 | decode at 4096 | 0.567, 0.569, 0.567 | 0.559, 0.559, 0.557 | 0.567 | 0.559 | 0.986 | 0.39% | 0.38% | NO OVERLAP |
| 21 | decode at 4096 | 0.473, 0.472, 0.472 | 0.463, 0.462, 0.464 | 0.472 | 0.463 | 0.980 | 0.23% | 0.39% | NO OVERLAP |
| 2 | prefill 4096 | 34.101, 33.782, 33.920 | 34.330, 34.581, 34.058 | 33.920 | 34.330 | 1.012 | 0.94% | 1.52% | OVERLAP |
| 20 | prefill 4096 | 32.167, 31.875, 32.007 | 32.418, 31.780, 32.069 | 32.007 | 32.069 | 1.002 | 0.91% | 1.99% | OVERLAP |
| 21 | prefill 4096 | 31.110, 30.738, 30.860 | 31.300, 30.637, 30.985 | 30.860 | 30.985 | 1.004 | 1.21% | 2.14% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.593, 0.590, 0.591 | 0.578, 0.578, 0.577 | 0.591 | 0.578 | 0.979 | 0.66% | 0.21% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.040, 1.041, 1.040 | 1.022, 1.020, 1.023 | 1.040 | 1.022 | 0.982 | 0.14% | 0.25% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 92 max 115, gpu0_temp_c median 64 max 71, gpu0_sm_mhz median 2272 max 2287, gpu0_fan median 32 max 38, cotenant_power_w median 151 max 280
