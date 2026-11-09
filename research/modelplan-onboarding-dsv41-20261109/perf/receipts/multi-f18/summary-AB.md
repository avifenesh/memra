| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.663, 0.659, 0.659 | 0.645, 0.645, 0.647 | 0.659 | 0.645 | 0.979 | 0.53% | 0.22% | NO OVERLAP |
| 20 | decode at 4096 | 0.639, 0.638, 0.638 | 0.623, 0.621, 0.622 | 0.638 | 0.622 | 0.974 | 0.11% | 0.21% | NO OVERLAP |
| 21 | decode at 4096 | 0.551, 0.548, 0.550 | 0.532, 0.529, 0.530 | 0.550 | 0.530 | 0.965 | 0.51% | 0.53% | NO OVERLAP |
| 2 | prefill 4096 | 37.157, 36.656, 36.560 | 36.422, 36.220, 36.064 | 36.656 | 36.220 | 0.988 | 1.63% | 0.99% | NO OVERLAP |
| 20 | prefill 4096 | 34.980, 34.625, 34.603 | 34.329, 34.238, 34.115 | 34.625 | 34.238 | 0.989 | 1.09% | 0.62% | NO OVERLAP |
| 21 | prefill 4096 | 33.665, 33.318, 33.340 | 32.945, 32.868, 32.718 | 33.340 | 32.868 | 0.986 | 1.04% | 0.69% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.660, 0.661, 0.662 | 0.644, 0.645, 0.642 | 0.661 | 0.644 | 0.975 | 0.27% | 0.51% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.178, 1.179, 1.181 | 1.144, 1.145, 1.145 | 1.179 | 1.145 | 0.970 | 0.24% | 0.09% | NO OVERLAP |

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
