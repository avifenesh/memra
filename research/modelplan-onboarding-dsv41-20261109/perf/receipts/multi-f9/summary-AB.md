| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.695, 0.696, 0.696 | 0.663, 0.663, 0.663 | 0.696 | 0.663 | 0.953 | 0.19% | 0.06% | NO OVERLAP |
| 20 | decode at 4096 | 0.684, 0.685, 0.683 | 0.639, 0.638, 0.638 | 0.684 | 0.638 | 0.933 | 0.20% | 0.20% | NO OVERLAP |
| 21 | decode at 4096 | 0.570, 0.570, 0.569 | 0.551, 0.549, 0.549 | 0.570 | 0.549 | 0.964 | 0.23% | 0.46% | NO OVERLAP |
| 2 | prefill 4096 | 44.035, 43.773, 43.783 | 36.919, 36.831, 36.929 | 43.783 | 36.919 | 0.843 | 0.60% | 0.26% | NO OVERLAP |
| 20 | prefill 4096 | 41.454, 41.318, 41.366 | 34.913, 34.881, 34.965 | 41.366 | 34.913 | 0.844 | 0.33% | 0.24% | NO OVERLAP |
| 21 | prefill 4096 | 39.946, 39.824, 39.877 | 33.584, 33.576, 33.627 | 39.877 | 33.584 | 0.842 | 0.31% | 0.15% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.695, 0.694, 0.695 | 0.659, 0.664, 0.663 | 0.695 | 0.663 | 0.954 | 0.16% | 0.71% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.248, 1.244, 1.247 | 1.184, 1.181, 1.182 | 1.247 | 1.182 | 0.948 | 0.36% | 0.21% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 191 max 203, gpu0_temp_c median 86 max 89, gpu0_sm_mhz median 2317 max 2355, gpu0_fan median 71 max 73, cotenant_power_w median 280 max 282
