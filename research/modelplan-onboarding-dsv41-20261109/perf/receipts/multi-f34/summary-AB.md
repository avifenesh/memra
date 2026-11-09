| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.694, 0.696, 0.694 | 0.619, 0.619, 0.620 | 0.694 | 0.619 | 0.891 | 0.32% | 0.15% | NO OVERLAP |
| 20 | decode at 4096 | 0.682, 0.685, 0.682 | 0.597, 0.596, 0.598 | 0.682 | 0.597 | 0.875 | 0.41% | 0.32% | NO OVERLAP |
| 21 | decode at 4096 | 0.567, 0.570, 0.568 | 0.504, 0.503, 0.503 | 0.568 | 0.503 | 0.886 | 0.44% | 0.26% | NO OVERLAP |
| 2 | prefill 4096 | 42.638, 43.042, 43.140 | 34.057, 34.144, 34.347 | 43.042 | 34.144 | 0.793 | 1.17% | 0.85% | NO OVERLAP |
| 20 | prefill 4096 | 40.293, 40.623, 40.767 | 32.090, 32.126, 32.301 | 40.623 | 32.126 | 0.791 | 1.17% | 0.66% | NO OVERLAP |
| 21 | prefill 4096 | 38.909, 39.240, 39.330 | 30.898, 31.021, 31.168 | 39.240 | 31.021 | 0.791 | 1.07% | 0.87% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.693, 0.695, 0.696 | 0.621, 0.616, 0.618 | 0.695 | 0.618 | 0.889 | 0.49% | 0.78% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.244, 1.248, 1.244 | 1.090, 1.091, 1.090 | 1.244 | 1.090 | 0.876 | 0.38% | 0.13% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 102 max 127, gpu0_temp_c median 72 max 78, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 46 max 47, cotenant_power_w median 251 max 280
