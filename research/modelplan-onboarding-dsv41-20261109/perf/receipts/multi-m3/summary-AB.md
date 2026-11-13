| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.594, 0.591, 0.590 | 0.564, 0.561, 0.563 | 0.591 | 0.563 | 0.953 | 0.64% | 0.55% | NO OVERLAP |
| 20 | decode at 4096 | 0.570, 0.568, 0.567 | 0.535, 0.535, 0.539 | 0.568 | 0.535 | 0.942 | 0.46% | 0.67% | NO OVERLAP |
| 21 | decode at 4096 | 0.476, 0.476, 0.476 | 0.446, 0.448, 0.446 | 0.476 | 0.446 | 0.937 | 0.04% | 0.56% | NO OVERLAP |
| 2 | prefill 4096 | 34.413, 34.480, 34.533 | 34.344, 34.391, 34.623 | 34.480 | 34.391 | 0.997 | 0.35% | 0.81% | OVERLAP |
| 20 | prefill 4096 | 32.294, 32.468, 32.535 | 32.358, 32.428, 32.546 | 32.468 | 32.428 | 0.999 | 0.74% | 0.58% | OVERLAP |
| 21 | prefill 4096 | 31.190, 31.359, 31.420 | 31.213, 31.308, 31.417 | 31.359 | 31.308 | 0.998 | 0.73% | 0.65% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.593, 0.588, 0.590 | 0.561, 0.564, 0.560 | 0.590 | 0.561 | 0.951 | 0.85% | 0.71% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.046, 1.048, 1.049 | 0.979, 0.984, 0.985 | 1.048 | 0.984 | 0.939 | 0.31% | 0.63% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 102 max 131, gpu0_temp_c median 74 max 79, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 45 max 48, cotenant_power_w median 280 max 284
