| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.552, 0.548, 0.552 | 0.552, 0.557, 0.555 | 0.552 | 0.555 | 1.006 | 0.74% | 0.85% | OVERLAP |
| 20 | decode at 4096 | 0.530, 0.530, 0.529 | 0.531, 0.532, 0.533 | 0.530 | 0.532 | 1.004 | 0.09% | 0.32% | NO OVERLAP |
| 21 | decode at 4096 | 0.437, 0.438, 0.438 | 0.437, 0.440, 0.439 | 0.438 | 0.439 | 1.003 | 0.30% | 0.77% | OVERLAP |
| 2 | prefill 4096 | 34.486, 34.733, 34.774 | 34.566, 34.593, 34.811 | 34.733 | 34.593 | 0.996 | 0.83% | 0.71% | OVERLAP |
| 20 | prefill 4096 | 32.434, 32.729, 32.717 | 32.550, 32.611, 32.804 | 32.717 | 32.611 | 0.997 | 0.90% | 0.78% | OVERLAP |
| 21 | prefill 4096 | 31.378, 31.588, 31.637 | 31.427, 31.517, 31.658 | 31.588 | 31.517 | 0.998 | 0.82% | 0.73% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.553, 0.554, 0.553 | 0.552, 0.553, 0.553 | 0.553 | 0.553 | 1.000 | 0.27% | 0.18% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.967, 0.971, 0.971 | 0.966, 0.972, 0.969 | 0.971 | 0.969 | 0.998 | 0.41% | 0.62% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 110 max 134, gpu0_temp_c median 76 max 80, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 46 max 50, cotenant_power_w median 280 max 290
