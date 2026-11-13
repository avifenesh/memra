| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.553, 0.553, 0.554 | 0.564, 0.570, 0.568 | 0.553 | 0.568 | 1.029 | 0.34% | 1.13% | NO OVERLAP |
| 20 | decode at 4096 | 0.528, 0.529, 0.530 | 0.536, 0.533, 0.534 | 0.529 | 0.534 | 1.009 | 0.21% | 0.43% | NO OVERLAP |
| 21 | decode at 4096 | 0.437, 0.438, 0.438 | 0.442, 0.443, 0.443 | 0.438 | 0.443 | 1.012 | 0.23% | 0.20% | NO OVERLAP |
| 2 | prefill 4096 | 34.543, 34.776, 34.776 | 34.563, 34.717, 34.782 | 34.776 | 34.717 | 0.998 | 0.67% | 0.63% | OVERLAP |
| 20 | prefill 4096 | 32.549, 32.738, 32.716 | 32.635, 32.711, 32.772 | 32.716 | 32.711 | 1.000 | 0.58% | 0.42% | OVERLAP |
| 21 | prefill 4096 | 31.446, 31.587, 31.602 | 31.514, 31.550, 31.636 | 31.587 | 31.550 | 0.999 | 0.49% | 0.39% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.552, 0.556, 0.553 | 0.563, 0.562, 0.564 | 0.553 | 0.563 | 1.018 | 0.83% | 0.44% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.965, 0.969, 0.969 | 0.979, 0.979, 0.976 | 0.969 | 0.979 | 1.010 | 0.46% | 0.28% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 110 max 131, gpu0_temp_c median 78 max 81, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 49 max 51, cotenant_power_w median 280 max 289
