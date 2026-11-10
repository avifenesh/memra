| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.555, 0.695, 0.695, 0.554, 0.554 | 0.533, 0.535, 0.533 | 0.555 | 0.533 | 0.960 | 25.53% | 0.43% | NO OVERLAP |
| 20 | decode at 4096 | 0.528, 0.684, 0.683, 0.530, 0.529 | 0.513, 0.512, 0.513 | 0.530 | 0.513 | 0.968 | 29.39% | 0.16% | NO OVERLAP |
| 21 | decode at 4096 | 0.437, 0.570, 0.569, 0.436, 0.437 | 0.419, 0.418, 0.420 | 0.437 | 0.419 | 0.957 | 30.46% | 0.48% | NO OVERLAP |
| 2 | prefill 4096 | 34.522, 43.715, 43.703, 34.523, 34.618 | 34.410, 34.527, 34.644 | 34.618 | 34.527 | 0.997 | 26.55% | 0.68% | OVERLAP |
| 20 | prefill 4096 | 32.394, 41.299, 41.301, 32.548, 32.585 | 32.461, 32.482, 32.600 | 32.585 | 32.482 | 0.997 | 27.34% | 0.43% | OVERLAP |
| 21 | prefill 4096 | 31.292, 39.819, 39.817, 31.383, 31.453 | 31.315, 31.371, 31.468 | 31.453 | 31.371 | 0.997 | 27.11% | 0.49% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.554, 0.694, 0.695, 0.552, 0.549 | 0.534, 0.534, 0.531 | 0.554 | 0.534 | 0.964 | 26.41% | 0.56% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.967, 1.246, 1.244, 0.967, 0.968 | 0.930, 0.932, 0.932 | 0.968 | 0.932 | 0.963 | 28.91% | 0.14% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 110 max 132, gpu0_temp_c median 75 max 78, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 52 max 53, cotenant_power_w median 280 max 284
