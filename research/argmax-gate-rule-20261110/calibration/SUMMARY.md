# Calibration cells

`tools/floor_rows.sh` over the banked logs: each cell's whole-prompt table through the gate's
`--judge --calibrate`. Percentiles are nearest rank over the agreeing rows' gap shift. Known rows
are the table's own flips above its floor, with their two margins.

| device | model | decode | rows | agreeing | flips | most flips in 12 rows | p50 | p99 | p99.9 | floor | set by row | known rows: row (margins) | probe sha256 | receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GB10 | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | batch | 2045 | 1856 | 189 | 6 | 0.2364 | 1.7071 | 4.0128 | 4.4502 | 306 | none | 6e6650071f22 | `calibration/gb10/ornith-batch/probe.log` |
| NVIDIA GB10 | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | eager | 2045 | 1854 | 191 | 6 | 0.2189 | 1.6159 | 3.0134 | 3.5812 | 727 | none | 6e6650071f22 | `calibration/gb10/ornith-eager/probe.log` |
| NVIDIA GB10 | Qwen3.8-27B-NVFP4-Q5K-mtp | eager | 2045 | 1963 | 82 | 4 | 0.0847 | 0.6671 | 1.2674 | 1.5695 | 1198 | 1512 (1.807 + 1.207) | 6e6650071f22 | `calibration/gb10/q38-eager/probe.log` |
| NVIDIA GB10 | Qwen3.5-9B-NVFP4-MTP-GGUF | batch | 2045 | 1881 | 164 | 5 | 0.1644 | 1.5625 | 3.0952 | 4.6464 | 1197 | none | 6e6650071f22 | `calibration/gb10/qwen35-9b-batch/probe.log` |
| NVIDIA GB10 | Qwen3.5-9B-NVFP4-MTP-GGUF | eager | 2045 | 1882 | 163 | 5 | 0.1570 | 1.5317 | 2.8857 | 4.1477 | 1197 | none | 6e6650071f22 | `calibration/gb10/qwen35-9b-eager/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | gemma-4-26B_q4_0-it | eager | 2100 | 1850 | 250 | 6 | 0.2777 | 1.9179 | 3.4214 | 3.5565 | 1458 | 150 (0.902 + 3.072), 943 (0.510 + 4.836), 981 (3.051 + 0.766), 1013 (3.082 + 0.615), 1117 (3.200 + 0.581), 1540 (2.084 + 2.331) | 723a3f7ff8dd | `calibration/rtx5090-laptop/gemma-4-26b-eager/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | gemma-4-31B_q4_0-it | eager | 2100 | 1986 | 114 | 4 | 0.1387 | 1.1919 | 3.3530 | 6.3987 | 594 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/gemma-4-31b-eager/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | batch | 2045 | 1874 | 171 | 5 | 0.1994 | 1.5819 | 3.3391 | 3.9557 | 740 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/ornith-batch/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | eager | 2045 | 1846 | 199 | 5 | 0.1972 | 1.5224 | 2.2653 | 3.5582 | 306 | 41 (1.472 + 2.899), 1809 (4.079 + 0.070) | 723a3f7ff8dd | `calibration/rtx5090-laptop/ornith-eager/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Qwen3.8-27B-NVFP4-Q5K-mtp | batch | 2045 | 2005 | 40 | 3 | 0.0455 | 0.3636 | 0.6629 | 3.0876 | 1199 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/q38-batch/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Qwen3.8-27B-NVFP4-Q5K-mtp | eager | 2045 | 2003 | 42 | 4 | 0.0468 | 0.3399 | 0.9602 | 3.7303 | 1198 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/q38-eager/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Qwen3.5-9B-NVFP4-MTP-GGUF | batch | 2045 | 1889 | 156 | 6 | 0.1509 | 1.4126 | 2.9690 | 3.8715 | 909 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/qwen35-9b-batch/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Qwen3.5-9B-NVFP4-MTP-GGUF | eager | 2045 | 1893 | 152 | 6 | 0.1466 | 1.4247 | 2.8103 | 4.0104 | 909 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/qwen35-9b-eager/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | batch | 2045 | 1864 | 181 | 5 | 0.1948 | 1.5043 | 2.7631 | 3.6951 | 1169 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/ornith-batch/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | eager | 2045 | 1872 | 173 | 7 | 0.1994 | 1.3699 | 2.8478 | 3.9178 | 306 | 41 (1.472 + 2.899) | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/ornith-eager/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Qwen3.8-27B-NVFP4-Q5K-mtp | batch | 2045 | 2004 | 41 | 4 | 0.0495 | 0.3706 | 1.2220 | 4.2419 | 1198 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/q38-batch/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Qwen3.8-27B-NVFP4-Q5K-mtp | eager | 2045 | 2001 | 44 | 3 | 0.0481 | 0.3938 | 1.1395 | 3.9595 | 1198 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/q38-eager/probe.log` |
