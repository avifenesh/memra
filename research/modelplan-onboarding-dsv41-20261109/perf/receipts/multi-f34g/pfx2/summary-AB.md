| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | prefill 32768 | 360.657, 365.479, 366.543 | 278.859, 280.386, 281.774 | 365.479 | 280.386 | 0.767 | 1.61% | 1.04% | NO OVERLAP |
| 20 | prefill 32768 | 372.886, 377.536, 378.433 | 285.164, 286.146, 286.951 | 377.536 | 286.146 | 0.758 | 1.47% | 0.62% | NO OVERLAP |
| 21 | prefill 32768 | 305.879, 308.810, 309.584 | 239.594, 239.961, 240.890 | 308.810 | 239.961 | 0.777 | 1.20% | 0.54% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| 2 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 20 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 21 | prefill 32768 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 127 max 198, gpu0_temp_c median 80 max 86, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 54 max 56, cotenant_power_w median 279 max 281
