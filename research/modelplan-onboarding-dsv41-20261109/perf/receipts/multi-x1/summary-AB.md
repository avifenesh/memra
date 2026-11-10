| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |
|---|---|---|---|---|---|---|---|---|---|
| 2 | decode at 4096 | 0.615, 0.616, 0.616 | 0.583, 0.583, 0.584 | 0.616 | 0.583 | 0.947 | 0.28% | 0.26% | NO OVERLAP |
| 20 | decode at 4096 | 0.591, 0.591, 0.592 | 0.560, 0.562, 0.562 | 0.591 | 0.562 | 0.951 | 0.32% | 0.32% | NO OVERLAP |
| 21 | decode at 4096 | 0.500, 0.499, 0.500 | 0.465, 0.468, 0.468 | 0.500 | 0.468 | 0.937 | 0.12% | 0.68% | NO OVERLAP |
| 2 | prefill 4096 | 34.620, 34.419, 34.515 | 33.845, 34.279, 34.381 | 34.515 | 34.279 | 0.993 | 0.58% | 1.56% | NO OVERLAP |
| 20 | prefill 4096 | 31.806, 32.452, 32.396 | 31.904, 32.289, 32.391 | 32.396 | 32.289 | 0.997 | 1.99% | 1.51% | OVERLAP |
| 21 | prefill 4096 | 30.726, 31.310, 31.264 | 30.791, 31.161, 31.284 | 31.264 | 31.161 | 0.997 | 1.87% | 1.58% | OVERLAP |
| (2,) | whole decode step at 4104 | 0.614, 0.614, 0.613 | 0.584, 0.583, 0.586 | 0.614 | 0.584 | 0.951 | 0.10% | 0.60% | NO OVERLAP |
| (20, 21) | whole decode step at 4104 | 1.087, 1.094, 1.093 | 1.026, 1.029, 1.034 | 1.093 | 1.029 | 0.942 | 0.61% | 0.78% | NO OVERLAP |

| layer | phase | hashes A | hashes B | identical |
|---|---|---|---|---|
| (2,) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (2,) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| (20, 21) | decode steps at 4096 | 1 distinct | 1 distinct | yes |
| (20, 21) | whole decode steps at 4104 | 1 distinct | 1 distinct | yes |
| 2 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 20 | prefill 4096 | 1 distinct | 1 distinct | yes |
| 21 | prefill 4096 | 1 distinct | 1 distinct | yes |

Telemetry (250 ms, all runs): gpu0_power_w median 100 max 122, gpu0_temp_c median 71 max 76, gpu0_sm_mhz median 2272 max 2347, gpu0_fan median 40 max 45, cotenant_power_w median 278 max 282
