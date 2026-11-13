| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.696, 0.700, 0.696 | 0.662, 0.660, 0.661 | 0.696 | 0.661 | 0.950 | 0.59% | 0.21% | NO OVERLAP |
| 20 | decode at 4096 | 0.684, 0.683, 0.683 | 0.639, 0.639, 0.638 | 0.683 | 0.639 | 0.935 | 0.16% | 0.14% | NO OVERLAP |
| 21 | decode at 4096 | 0.569, 0.569, 0.568 | 0.550, 0.551, 0.550 | 0.569 | 0.550 | 0.967 | 0.16% | 0.13% | NO OVERLAP |
| 2 | prefill 4096 | 43.828, 43.851, 43.958 | 36.670, 36.907, 37.075 | 43.851 | 36.907 | 0.842 | 0.29% | 1.10% | NO OVERLAP |
| 20 | prefill 4096 | 41.085, 41.499, 41.566 | 34.769, 34.993, 35.117 | 41.499 | 34.993 | 0.843 | 1.16% | 0.99% | NO OVERLAP |
| 21 | prefill 4096 | 39.641, 40.018, 40.078 | 33.472, 33.652, 33.784 | 40.018 | 33.652 | 0.841 | 1.09% | 0.92% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.695, 0.691, 0.695 | 0.663, 0.660, 0.659 | 0.695 | 0.660 | 0.951 | 0.58% | 0.70% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.244, 1.247, 1.247 | 1.180, 1.183, 1.184 | 1.247 | 1.183 | 0.949 | 0.21% | 0.29% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 156 max 179, gpu0_temp_c median 85 max 88, gpu0_sm_mhz median 2325 max 2347, gpu0_fan median 65 max 69, cotenant_power_w median 280 max 282
