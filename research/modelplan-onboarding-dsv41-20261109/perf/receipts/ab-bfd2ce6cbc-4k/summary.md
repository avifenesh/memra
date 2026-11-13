| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 1.050, 1.057, 1.053 | 1.016, 1.018, 1.018 | 1.053 | 1.018 | 0.966 | 0.68% | 0.19% | NO OVERLAP |
| 20 | decode at 4096 | 1.023, 1.019, 1.022 | 0.977, 0.978, 0.979 | 1.022 | 0.978 | 0.956 | 0.43% | 0.19% | NO OVERLAP |
| 21 | decode at 4096 | 0.890, 0.891, 0.892 | 0.852, 0.850, 0.853 | 0.891 | 0.852 | 0.957 | 0.28% | 0.35% | NO OVERLAP |
| 2 | prefill 4096 | 60.711, 61.112, 61.319 | 46.495, 46.686, 47.074 | 61.112 | 46.686 | 0.764 | 0.99% | 1.24% | NO OVERLAP |
| 20 | prefill 4096 | 58.836, 59.145, 59.333 | 44.525, 44.603, 45.040 | 59.145 | 44.603 | 0.754 | 0.84% | 1.15% | NO OVERLAP |
| 21 | prefill 4096 | 56.810, 57.203, 57.388 | 42.602, 42.656, 43.105 | 57.203 | 42.656 | 0.746 | 1.01% | 1.18% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 115 max 139, gpu0_temp_c median 74 max 80, gpu0_sm_mhz median 2280 max 2355, gpu0_fan median 46 max 49, cotenant_power_w median 280 max 280
