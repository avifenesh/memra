| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.583, 0.583, 0.584 | 0.579, 0.584, 0.581 | 0.583 | 0.581 | 0.995 | 0.26% | 0.84% | OVERLAP |
| 20 | decode at 4096 | 0.560, 0.562, 0.562 | 0.557, 0.558, 0.556 | 0.562 | 0.557 | 0.990 | 0.32% | 0.27% | NO OVERLAP |
| 21 | decode at 4096 | 0.465, 0.468, 0.468 | 0.466, 0.468, 0.470 | 0.468 | 0.468 | 1.000 | 0.68% | 0.77% | OVERLAP |
| 2 | prefill 4096 | 33.845, 34.279, 34.381 | 33.960, 34.134, 34.441 | 34.279 | 34.134 | 0.996 | 1.56% | 1.41% | OVERLAP |
| 20 | prefill 4096 | 31.904, 32.289, 32.391 | 32.025, 32.180, 32.408 | 32.289 | 32.180 | 0.997 | 1.51% | 1.19% | OVERLAP |
| 21 | prefill 4096 | 30.791, 31.161, 31.284 | 30.953, 31.030, 31.283 | 31.161 | 31.030 | 0.996 | 1.58% | 1.06% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.584, 0.583, 0.586 | 0.578, 0.579, 0.580 | 0.584 | 0.579 | 0.991 | 0.60% | 0.21% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.026, 1.029, 1.034 | 1.025, 1.025, 1.029 | 1.029 | 1.025 | 0.996 | 0.78% | 0.44% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 100 max 122, gpu0_temp_c median 71 max 76, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 40 max 45, cotenant_power_w median 278 max 282
