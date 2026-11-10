| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.555, 0.553, 0.551 | 0.558, 0.557, 0.554 | 0.553 | 0.557 | 1.007 | 0.72% | 0.68% | OVERLAP |
| 20 | decode at 4096 | 0.531, 0.529, 0.527 | 0.530, 0.529, 0.530 | 0.529 | 0.530 | 1.001 | 0.62% | 0.15% | OVERLAP |
| 21 | decode at 4096 | 0.438, 0.436, 0.439 | 0.437, 0.437, 0.437 | 0.438 | 0.437 | 0.996 | 0.78% | 0.02% | OVERLAP |
| 2 | prefill 4096 | 35.328, 34.628, 34.444 | 35.094, 34.652, 34.472 | 34.628 | 34.652 | 1.001 | 2.55% | 1.80% | OVERLAP |
| 20 | prefill 4096 | 33.066, 32.480, 32.378 | 32.689, 32.604, 32.473 | 32.480 | 32.604 | 1.004 | 2.12% | 0.66% | OVERLAP |
| 21 | prefill 4096 | 31.871, 31.331, 31.265 | 31.521, 31.440, 31.255 | 31.331 | 31.440 | 1.003 | 1.93% | 0.85% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.553, 0.550, 0.554 | 0.556, 0.555, 0.554 | 0.553 | 0.555 | 1.005 | 0.71% | 0.34% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.972, 0.969, 0.968 | 0.972, 0.970, 0.965 | 0.969 | 0.970 | 1.001 | 0.43% | 0.72% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 116 max 257, gpu0_temp_c median 77 max 89, gpu0_sm_mhz median 2220 max 2355, gpu0_fan median 57 max 67, cotenant_power_w median 280 max 288
