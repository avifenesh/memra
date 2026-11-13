| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.647, 0.645, 0.654 | 0.625, 0.626, 0.628 | 0.647 | 0.626 | 0.966 | 1.44% | 0.38% | NO OVERLAP |
| 20 | decode at 4096 | 0.622, 0.623, 0.630 | 0.603, 0.603, 0.604 | 0.623 | 0.603 | 0.968 | 1.19% | 0.20% | NO OVERLAP |
| 21 | decode at 4096 | 0.532, 0.534, 0.537 | 0.510, 0.509, 0.512 | 0.534 | 0.510 | 0.955 | 0.92% | 0.63% | NO OVERLAP |
| 2 | prefill 4096 | 35.064, 35.334, 35.407 | 35.154, 35.246, 35.517 | 35.334 | 35.246 | 0.997 | 0.97% | 1.03% | OVERLAP |
| 20 | prefill 4096 | 33.125, 33.346, 33.536 | 33.202, 33.272, 33.500 | 33.346 | 33.272 | 0.998 | 1.23% | 0.90% | OVERLAP |
| 21 | prefill 4096 | 31.890, 32.146, 32.282 | 32.002, 32.035, 32.249 | 32.146 | 32.035 | 0.997 | 1.22% | 0.77% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.644, 0.644, 0.650 | 0.626, 0.627, 0.628 | 0.644 | 0.627 | 0.973 | 0.92% | 0.24% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.152, 1.148, 1.156 | 1.107, 1.108, 1.110 | 1.152 | 1.108 | 0.962 | 0.69% | 0.28% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 105 max 158, gpu0_temp_c median 78 max 83, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 51 max 52, cotenant_power_w median 280 max 280
