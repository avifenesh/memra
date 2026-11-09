| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.645, 0.645, 0.647 | 0.644, 0.645, 0.645 | 0.645 | 0.645 | 0.999 | 0.22% | 0.28% | OVERLAP |
| 20 | decode at 4096 | 0.623, 0.621, 0.622 | 0.621, 0.620, 0.620 | 0.622 | 0.620 | 0.998 | 0.21% | 0.13% | NO OVERLAP |
| 21 | decode at 4096 | 0.532, 0.529, 0.530 | 0.531, 0.531, 0.530 | 0.530 | 0.531 | 1.000 | 0.53% | 0.23% | OVERLAP |
| 2 | prefill 4096 | 36.422, 36.220, 36.064 | 36.182, 36.124, 35.961 | 36.220 | 36.124 | 0.997 | 0.99% | 0.61% | OVERLAP |
| 20 | prefill 4096 | 34.329, 34.238, 34.115 | 34.101, 34.076, 33.905 | 34.238 | 34.076 | 0.995 | 0.62% | 0.57% | NO OVERLAP |
| 21 | prefill 4096 | 32.945, 32.868, 32.718 | 32.871, 32.830, 32.678 | 32.868 | 32.830 | 0.999 | 0.69% | 0.59% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.644, 0.645, 0.642 | 0.640, 0.643, 0.642 | 0.644 | 0.642 | 0.998 | 0.51% | 0.37% | OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.144, 1.145, 1.145 | 1.146, 1.145, 1.145 | 1.145 | 1.145 | 1.000 | 0.09% | 0.12% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 110 max 186, gpu0_temp_c median 77 max 88, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 56 max 70, cotenant_power_w median 280 max 281
