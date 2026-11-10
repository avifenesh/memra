| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.550, 0.550, 0.552 | 0.551, 0.551, 0.550 | 0.550 | 0.551 | 1.001 | 0.47% | 0.18% | OVERLAP |
| 20 | decode at 4096 | 0.527, 0.528, 0.529 | 0.530, 0.528, 0.529 | 0.528 | 0.529 | 1.001 | 0.34% | 0.44% | OVERLAP |
| 21 | decode at 4096 | 0.436, 0.435, 0.436 | 0.436, 0.435, 0.438 | 0.436 | 0.436 | 1.000 | 0.16% | 0.67% | OVERLAP |
| 2 | prefill 4096 | 34.248, 33.852, 33.984 | 34.487, 33.732, 34.106 | 33.984 | 34.106 | 1.004 | 1.17% | 2.21% | OVERLAP |
| 20 | prefill 4096 | 32.282, 31.867, 32.026 | 32.424, 31.768, 32.095 | 32.026 | 32.095 | 1.002 | 1.29% | 2.04% | OVERLAP |
| 21 | prefill 4096 | 31.235, 30.779, 30.883 | 31.471, 30.660, 30.980 | 30.883 | 30.980 | 1.003 | 1.47% | 2.62% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.554, 0.551, 0.550 | 0.557, 0.550, 0.552 | 0.551 | 0.552 | 1.002 | 0.74% | 1.29% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.962, 0.964, 0.966 | 0.966, 0.964, 0.966 | 0.964 | 0.966 | 1.002 | 0.42% | 0.17% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 98 max 122, gpu0_temp_c median 66 max 72, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 34 max 39, cotenant_power_w median 175 max 280
