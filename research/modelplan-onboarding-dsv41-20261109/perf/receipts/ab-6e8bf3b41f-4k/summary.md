| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 3.987, 4.002, 3.993 | 1.052, 1.059, 1.051 | 3.993 | 1.052 | 0.263 | 0.38% | 0.82% | NO OVERLAP |
| 20 | decode at 4096 | 3.392, 3.397, 3.388 | 1.017, 1.019, 1.020 | 3.392 | 1.019 | 0.301 | 0.26% | 0.26% | NO OVERLAP |
| 21 | decode at 4096 | 2.960, 2.967, 2.966 | 0.888, 0.889, 0.890 | 2.966 | 0.889 | 0.300 | 0.23% | 0.28% | NO OVERLAP |
| 2 | prefill 4096 | 57.009, 57.461, 57.736 | 60.442, 60.343, 60.950 | 57.461 | 60.442 | 1.052 | 1.27% | 1.00% | NO OVERLAP |
| 20 | prefill 4096 | 55.184, 55.560, 55.699 | 58.438, 58.445, 58.996 | 55.560 | 58.445 | 1.052 | 0.93% | 0.96% | NO OVERLAP |
| 21 | prefill 4096 | 53.279, 53.641, 53.860 | 56.346, 56.525, 56.965 | 53.641 | 56.525 | 1.054 | 1.08% | 1.09% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 110 max 137, gpu0_temp_c median 68 max 75, gpu0_sm_mhz median 2280 max 2347, gpu0_fan median 36 max 43, cotenant_power_w median 280 max 281
