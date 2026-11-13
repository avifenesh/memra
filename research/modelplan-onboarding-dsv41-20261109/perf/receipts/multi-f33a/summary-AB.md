| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.620, 0.617, 0.618 | 0.617, 0.617, 0.619 | 0.618 | 0.617 | 0.998 | 0.49% | 0.41% | OVERLAP |
| 20 | decode at 4096 | 0.597, 0.594, 0.596 | 0.596, 0.597, 0.596 | 0.596 | 0.596 | 1.001 | 0.42% | 0.07% | OVERLAP |
| 21 | decode at 4096 | 0.504, 0.503, 0.503 | 0.500, 0.501, 0.502 | 0.503 | 0.501 | 0.994 | 0.02% | 0.32% | NO OVERLAP |
| 2 | prefill 4096 | 34.285, 33.950, 34.028 | 33.669, 33.907, 34.139 | 34.028 | 33.907 | 0.996 | 0.99% | 1.39% | OVERLAP |
| 20 | prefill 4096 | 32.392, 31.972, 32.058 | 31.737, 31.879, 32.143 | 32.058 | 31.879 | 0.994 | 1.31% | 1.27% | OVERLAP |
| 21 | prefill 4096 | 31.173, 30.820, 30.913 | 30.588, 30.723, 30.977 | 30.913 | 30.723 | 0.994 | 1.14% | 1.27% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.620, 0.618, 0.618 | 0.617, 0.616, 0.618 | 0.618 | 0.617 | 0.999 | 0.29% | 0.32% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.093, 1.091, 1.091 | 1.092, 1.092, 1.092 | 1.091 | 1.092 | 1.001 | 0.22% | 0.04% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 92 max 117, gpu0_temp_c median 66 max 73, gpu0_sm_mhz median 2272 max 2317, gpu0_fan median 34 max 41, cotenant_power_w median 217 max 281
