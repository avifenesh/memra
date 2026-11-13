| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.614, 0.615, 0.615 | 0.590, 0.592, 0.592 | 0.615 | 0.592 | 0.961 | 0.31% | 0.20% | NO OVERLAP |
| 20 | decode at 4096 | 0.593, 0.593, 0.593 | 0.569, 0.568, 0.564 | 0.593 | 0.568 | 0.959 | 0.08% | 0.77% | NO OVERLAP |
| 21 | decode at 4096 | 0.500, 0.500, 0.501 | 0.478, 0.475, 0.474 | 0.500 | 0.475 | 0.950 | 0.18% | 0.84% | NO OVERLAP |
| 2 | prefill 4096 | 34.330, 34.567, 34.422 | 34.349, 34.429, 34.404 | 34.422 | 34.404 | 1.000 | 0.69% | 0.23% | OVERLAP |
| 20 | prefill 4096 | 32.366, 32.369, 32.461 | 32.389, 32.394, 32.381 | 32.369 | 32.389 | 1.001 | 0.29% | 0.04% | OVERLAP |
| 21 | prefill 4096 | 31.232, 31.259, 31.289 | 31.247, 31.187, 31.234 | 31.259 | 31.234 | 0.999 | 0.18% | 0.19% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.614, 0.614, 0.614 | 0.589, 0.593, 0.596 | 0.614 | 0.593 | 0.966 | 0.15% | 1.16% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.095, 1.091, 1.095 | 1.044, 1.043, 1.038 | 1.095 | 1.043 | 0.953 | 0.33% | 0.58% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 104 max 123, gpu0_temp_c median 73 max 77, gpu0_sm_mhz median 2272 max 2325, gpu0_fan median 43 max 45, cotenant_power_w median 171 max 280
