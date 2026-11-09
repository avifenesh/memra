| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | prefill 32768 | 366.771, 370.390, 371.429 | 285.008, 286.127, 287.938 | 370.390 | 286.127 | 0.773 | 1.26% | 1.02% | NO OVERLAP |
| 20 | prefill 32768 | 377.673, 383.326, 383.444 | 296.588, 297.391, 298.850 | 383.326 | 297.391 | 0.776 | 1.51% | 0.76% | NO OVERLAP |
| 21 | prefill 32768 | 309.507, 313.435, 313.552 | 242.533, 242.901, 243.829 | 313.435 | 242.901 | 0.775 | 1.29% | 0.53% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| 2 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 20 | prefill 32768 | 1 distinct | 1 distinct | yes |
| 21 | prefill 32768 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 101 max 204, gpu0_temp_c median 82 max 89, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 58 max 61, cotenant_power_w median 280 max 282
