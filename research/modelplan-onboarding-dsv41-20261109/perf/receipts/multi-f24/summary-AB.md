| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.694, 0.696, 0.694 | 0.618, 0.618, 0.620 | 0.694 | 0.618 | 0.891 | 0.24% | 0.42% | NO OVERLAP |
| 20 | decode at 4096 | 0.683, 0.682, 0.683 | 0.593, 0.595, 0.594 | 0.683 | 0.594 | 0.870 | 0.19% | 0.42% | NO OVERLAP |
| 21 | decode at 4096 | 0.569, 0.571, 0.569 | 0.502, 0.504, 0.504 | 0.569 | 0.504 | 0.885 | 0.35% | 0.30% | NO OVERLAP |
| 2 | prefill 4096 | 42.674, 43.556, 43.683 | 34.316, 34.467, 34.812 | 43.556 | 34.467 | 0.791 | 2.32% | 1.44% | NO OVERLAP |
| 20 | prefill 4096 | 40.696, 41.215, 41.294 | 32.346, 32.479, 32.740 | 41.215 | 32.479 | 0.788 | 1.45% | 1.21% | NO OVERLAP |
| 21 | prefill 4096 | 39.268, 39.726, 39.795 | 31.234, 31.351, 31.595 | 39.726 | 31.351 | 0.789 | 1.33% | 1.15% | NO OVERLAP |
| (2,) | whole decode step at 4104 | 0.695, 0.694, 0.694 | 0.617, 0.621, 0.618 | 0.694 | 0.618 | 0.890 | 0.19% | 0.66% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.244, 1.247, 1.244 | 1.093, 1.092, 1.092 | 1.244 | 1.092 | 0.877 | 0.28% | 0.08% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 99 max 172, gpu0_temp_c median 78 max 84, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 48 max 53, cotenant_power_w median 280 max 281
