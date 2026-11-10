# Calibration cells

`tools/floor_rows.sh` over the banked logs: each cell's whole-prompt table through the gate's
`--judge --calibrate`. Percentiles are nearest rank over the agreeing rows' gap shift. Known rows
are the table's own flips above its floor, with their two margins.

| device | model | decode | rows | agreeing | flips | most flips in 12 rows | p50 | p99 | p99.9 | floor | set by row | known rows: row (margins) | probe sha256 | receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | batch | 2045 | 1874 | 171 | 5 | 0.1994 | 1.5819 | 3.3391 | 3.9557 | 740 | none | 723a3f7ff8dd | `calibration/rtx5090-laptop/ornith-batch/probe.log` |
| NVIDIA GeForce RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | eager | 2045 | 1846 | 199 | 5 | 0.1972 | 1.5224 | 2.2653 | 3.5582 | 306 | 41 (1.472 + 2.899), 1809 (4.079 + 0.070) | 723a3f7ff8dd | `calibration/rtx5090-laptop/ornith-eager/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | batch | 2045 | 1864 | 181 | 5 | 0.1948 | 1.5043 | 2.7631 | 3.6951 | 1169 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/ornith-batch/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | eager | 2045 | 1872 | 173 | 7 | 0.1994 | 1.3699 | 2.8478 | 3.9178 | 306 | 41 (1.472 + 2.899) | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/ornith-eager/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Qwen3.8-27B-NVFP4-Q5K-mtp | batch | 2045 | 2004 | 41 | 4 | 0.0495 | 0.3706 | 1.2220 | 4.2419 | 1198 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/q38-batch/probe.log` |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | Qwen3.8-27B-NVFP4-Q5K-mtp | eager | 2045 | 2001 | 44 | 3 | 0.0481 | 0.3938 | 1.1395 | 3.9595 | 1198 | none | 723a3f7ff8dd | `calibration/rtxpro6000-maxq/q38-eager/probe.log` |
