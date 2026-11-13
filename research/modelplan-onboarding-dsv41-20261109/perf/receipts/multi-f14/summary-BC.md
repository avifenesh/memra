| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.662, 0.660, 0.661 | 0.700, 0.700, 0.700 | 0.661 | 0.700 | 1.058 | 0.21% | 0.13% | NO OVERLAP |
| 20 | decode at 4096 | 0.639, 0.639, 0.638 | 0.677, 0.676, 0.677 | 0.639 | 0.677 | 1.059 | 0.14% | 0.21% | NO OVERLAP |
| 21 | decode at 4096 | 0.550, 0.551, 0.550 | 0.580, 0.580, 0.579 | 0.550 | 0.580 | 1.053 | 0.13% | 0.14% | NO OVERLAP |
| 2 | prefill 4096 | 36.670, 36.907, 37.075 | 36.828, 36.904, 37.167 | 36.907 | 36.904 | 1.000 | 1.10% | 0.92% | OVERLAP |
| 20 | prefill 4096 | 34.769, 34.993, 35.117 | 34.881, 34.949, 35.225 | 34.993 | 34.949 | 0.999 | 0.99% | 0.98% | OVERLAP |
| 21 | prefill 4096 | 33.472, 33.652, 33.784 | 33.570, 33.644, 33.897 | 33.652 | 33.644 | 1.000 | 0.92% | 0.97% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.663, 0.660, 0.659 | 0.700, 0.698, 0.699 | 0.660 | 0.699 | 1.059 | 0.70% | 0.26% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.180, 1.183, 1.184 | 1.248, 1.248, 1.251 | 1.183 | 1.248 | 1.055 | 0.29% | 0.25% | NO OVERLAP |

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
