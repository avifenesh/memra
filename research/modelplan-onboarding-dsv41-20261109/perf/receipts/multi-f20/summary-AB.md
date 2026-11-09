| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.643, 0.645, 0.643 | 0.648, 0.647, 0.648 | 0.643 | 0.648 | 1.007 | 0.26% | 0.15% | NO OVERLAP |
| 20 | decode at 4096 | 0.619, 0.622, 0.621 | 0.624, 0.624, 0.625 | 0.621 | 0.624 | 1.005 | 0.45% | 0.16% | NO OVERLAP |
| 21 | decode at 4096 | 0.529, 0.532, 0.532 | 0.534, 0.532, 0.534 | 0.532 | 0.534 | 1.003 | 0.49% | 0.24% | NO OVERLAP |
| 2 | prefill 4096 | 35.839, 36.050, 36.132 | 35.209, 35.322, 35.511 | 36.050 | 35.322 | 0.980 | 0.81% | 0.85% | NO OVERLAP |
| 20 | prefill 4096 | 33.798, 34.008, 34.133 | 33.266, 33.350, 33.553 | 34.008 | 33.350 | 0.981 | 0.99% | 0.86% | NO OVERLAP |
| 21 | prefill 4096 | 32.575, 32.839, 32.926 | 32.046, 32.106, 32.343 | 32.839 | 32.106 | 0.978 | 1.07% | 0.93% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.641, 0.641, 0.642 | 0.649, 0.648, 0.646 | 0.641 | 0.648 | 1.011 | 0.11% | 0.51% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.143, 1.145, 1.147 | 1.150, 1.150, 1.150 | 1.145 | 1.150 | 1.005 | 0.35% | 0.03% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 97 max 164, gpu0_temp_c median 77 max 84, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 50 max 53, cotenant_power_w median 280 max 280
