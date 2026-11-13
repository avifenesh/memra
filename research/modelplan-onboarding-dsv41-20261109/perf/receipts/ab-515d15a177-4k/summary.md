| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 1.018, 1.021, 1.015 | 0.764, 0.762, 0.762 | 1.018 | 0.762 | 0.749 | 0.62% | 0.31% | NO OVERLAP |
| 20 | decode at 4096 | 0.978, 0.979, 0.978 | 0.726, 0.726, 0.727 | 0.978 | 0.726 | 0.742 | 0.06% | 0.10% | NO OVERLAP |
| 21 | decode at 4096 | 0.853, 0.852, 0.853 | 0.609, 0.609, 0.609 | 0.853 | 0.609 | 0.714 | 0.15% | 0.03% | NO OVERLAP |
| 2 | prefill 4096 | 47.159, 47.370, 47.434 | 42.931, 42.972, 43.213 | 47.370 | 42.972 | 0.907 | 0.58% | 0.65% | NO OVERLAP |
| 20 | prefill 4096 | 45.141, 45.325, 45.349 | 40.551, 40.612, 40.753 | 45.325 | 40.612 | 0.896 | 0.46% | 0.50% | NO OVERLAP |
| 21 | prefill 4096 | 43.156, 43.350, 43.399 | 39.135, 39.184, 39.346 | 43.350 | 39.184 | 0.904 | 0.56% | 0.54% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 120 max 138, gpu0_temp_c median 81 max 84, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 52 max 54, cotenant_power_w median 280 max 280
