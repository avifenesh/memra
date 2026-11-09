| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.618, 0.623, 0.620 | 0.631, 0.633, 0.630 | 0.620 | 0.631 | 1.018 | 0.76% | 0.52% | NO OVERLAP |
| 20 | decode at 4096 | 0.595, 0.594, 0.595 | 0.607, 0.610, 0.610 | 0.595 | 0.610 | 1.025 | 0.22% | 0.46% | NO OVERLAP |
| 21 | decode at 4096 | 0.503, 0.503, 0.504 | 0.503, 0.503, 0.502 | 0.503 | 0.503 | 0.999 | 0.28% | 0.32% | OVERLAP |
| 2 | prefill 4096 | 34.122, 34.234, 34.344 | 34.318, 34.379, 34.448 | 34.234 | 34.379 | 1.004 | 0.65% | 0.38% | OVERLAP |
| 20 | prefill 4096 | 32.139, 32.276, 32.226 | 32.408, 32.399, 32.506 | 32.226 | 32.408 | 1.006 | 0.42% | 0.33% | NO OVERLAP |
| 21 | prefill 4096 | 30.949, 31.141, 31.098 | 31.025, 31.011, 31.111 | 31.098 | 31.025 | 0.998 | 0.62% | 0.32% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.618, 0.616, 0.619 | 0.632, 0.631, 0.631 | 0.618 | 0.631 | 1.022 | 0.39% | 0.21% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.094, 1.091, 1.089 | 1.103, 1.106, 1.106 | 1.091 | 1.106 | 1.014 | 0.49% | 0.30% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 98 max 200, gpu0_temp_c median 71 max 89, gpu0_sm_mhz median 2280 max 2355, gpu0_fan median 44 max 60, cotenant_power_w median 280 max 286
