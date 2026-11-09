| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.621, 0.616, 0.618 | 0.617, 0.622, 0.617 | 0.618 | 0.617 | 0.999 | 0.73% | 0.84% | OVERLAP |
| 20 | decode at 4096 | 0.596, 0.595, 0.598 | 0.597, 0.596, 0.596 | 0.596 | 0.596 | 1.001 | 0.47% | 0.05% | OVERLAP |
| 21 | decode at 4096 | 0.503, 0.502, 0.504 | 0.503, 0.503, 0.502 | 0.503 | 0.503 | 0.999 | 0.54% | 0.14% | OVERLAP |
| 2 | prefill 4096 | 34.290, 34.236, 34.315 | 34.137, 34.217, 34.255 | 34.290 | 34.217 | 0.998 | 0.23% | 0.34% | OVERLAP |
| 20 | prefill 4096 | 32.108, 32.192, 32.273 | 32.138, 32.181, 32.257 | 32.192 | 32.181 | 1.000 | 0.51% | 0.37% | OVERLAP |
| 21 | prefill 4096 | 30.966, 31.061, 31.056 | 30.971, 31.050, 31.042 | 31.056 | 31.042 | 1.000 | 0.31% | 0.25% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.621, 0.617, 0.621 | 0.620, 0.617, 0.621 | 0.621 | 0.620 | 0.998 | 0.60% | 0.61% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.090, 1.090, 1.091 | 1.092, 1.092, 1.091 | 1.090 | 1.092 | 1.001 | 0.10% | 0.10% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 97 max 153, gpu0_temp_c median 73 max 79, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 50 max 55, cotenant_power_w median 219 max 280
