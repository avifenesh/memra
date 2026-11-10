| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.565, 0.563, 0.562 | 0.555, 0.552, 0.552 | 0.563 | 0.552 | 0.981 | 0.55% | 0.54% | NO OVERLAP |
| 20 | decode at 4096 | 0.540, 0.539, 0.538 | 0.529, 0.529, 0.531 | 0.539 | 0.529 | 0.983 | 0.26% | 0.26% | NO OVERLAP |
| 21 | decode at 4096 | 0.446, 0.447, 0.445 | 0.437, 0.438, 0.437 | 0.446 | 0.437 | 0.980 | 0.34% | 0.14% | NO OVERLAP |
| 2 | prefill 4096 | 34.794, 34.459, 34.372 | 34.405, 34.513, 34.439 | 34.459 | 34.439 | 0.999 | 1.22% | 0.31% | OVERLAP |
| 20 | prefill 4096 | 32.400, 32.369, 32.359 | 32.426, 32.421, 32.296 | 32.369 | 32.421 | 1.002 | 0.13% | 0.40% | OVERLAP |
| 21 | prefill 4096 | 31.242, 31.211, 31.231 | 31.317, 31.324, 31.140 | 31.231 | 31.317 | 1.003 | 0.10% | 0.59% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.561, 0.563, 0.563 | 0.553, 0.552, 0.553 | 0.563 | 0.553 | 0.982 | 0.36% | 0.24% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 0.985, 0.984, 0.982 | 0.968, 0.969, 0.964 | 0.984 | 0.968 | 0.983 | 0.30% | 0.45% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 100 max 254, gpu0_temp_c median 73 max 82, gpu0_sm_mhz median 2272 max 2355, gpu0_fan median 54 max 66, cotenant_power_w median 280 max 283
