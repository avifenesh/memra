| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | prefill 32768 | 279.074, 284.915, 286.415 | 286.512, 288.080, 291.864 | 284.915 | 288.080 | 1.011 | 2.58% | 1.86% | NO OVERLAP |
| 20 | prefill 32768 | 291.803, 297.106, 298.303 | 297.984, 299.890, 303.125 | 297.106 | 299.890 | 1.009 | 2.19% | 1.71% | OVERLAP |
| 21 | prefill 32768 | 238.571, 242.697, 243.814 | 239.444, 241.428, 243.906 | 242.697 | 241.428 | 0.995 | 2.16% | 1.85% | OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| 2 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 20 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 21 | prefill 32768 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 98 max 200, gpu0_temp_c median 71 max 89, gpu0_sm_mhz median 2280 max 2355, gpu0_fan median 44 max 60, cotenant_power_w median 280 max 286
