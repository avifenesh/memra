# Argmax margin gate: candidate rules on existing receipts

65600 rows from `collect.py`.

## 1. The old predicate on every flip

| receipts | distinct flips | old rule says unexplained | smallest spread / smaller margin | largest margin at a flip | largest margin sum at a flip |
|---|---|---|---|---|---|
| tables | 27 | 0 | 1.406 | 14.386 | 16.867 |
| kv-quant | 1282 | 0 | 1.036 | 15.805 | 18.260 |

All 1309 distinct flips: 0 unexplained, smallest spread / smaller margin 1.036, smallest spread / mean margin 1.004 (the bound is 1).

## 2. Populations: what agreeing rows show per (model, prompt, pair, device)

`max gap shift` and `max spread` are over agreeing rows. Identical rows from repeated runs count once.

| tree | model | prompt | pair | device | tables | agreeing rows | gap shift p99 | p99.9 | max gap shift | spread p99 | max spread | distinct flips | largest margin sum at a flip | largest margin at a flip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-32768 | prefill:batch | gb10 | 1 | 2000 | 1.057 | 2.234 | 2.832 (row 31433) | 1.169 | 3.295 | 48 | 1.176 | 0.703 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-32768 | prefill:eager | gb10 | 1 | 1992 | 1.044 | 2.527 | 2.762 (row 31433) | 1.156 | 2.489 | 56 | 1.073 | 0.668 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-32768 | prefill:prime | gb10 | 2 | 1997 | 1.004 | 2.350 | 2.721 (row 31433) | 1.012 | 2.301 | 51 | 2.242 | 1.580 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-32768 | prime:batch | gb10 | 1 | 2004 | 0.694 | 1.373 | 2.906 (row 31821) | 0.808 | 5.119 | 44 | 0.587 | 0.445 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-32768 | prime:eager | gb10 | 1 | 2000 | 0.613 | 1.193 | 1.940 (row 31821) | 0.797 | 2.260 | 48 | 1.939 | 1.580 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-65535 | prefill:batch | gb10 | 1 | 2014 | 0.749 | 1.589 | 2.375 (row 63665) | 0.978 | 6.690 | 34 | 1.691 | 1.099 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-65535 | prefill:prime | gb10 | 1 | 2015 | 0.768 | 1.414 | 2.326 (row 65010) | 1.065 | 3.774 | 33 | 1.046 | 0.725 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | csv-65535 | prime:batch | gb10 | 1 | 2018 | 0.647 | 1.773 | 3.406 (row 63665) | 0.831 | 8.021 | 30 | 1.474 | 0.792 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-32768 | prefill:batch | gb10 | 1 | 2029 | 4.297 | 8.532 | 9.762 (row 31193) | 7.444 | 19.564 | 19 | 11.800 | 9.770 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-32768 | prefill:eager | gb10 | 1 | 2030 | 4.453 | 8.200 | 11.202 (row 31951) | 6.660 | 20.590 | 18 | 13.347 | 11.002 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-32768 | prefill:prime | gb10 | 2 | 2034 | 3.968 | 6.547 | 10.667 (row 31971) | 6.139 | 19.595 | 14 | 16.587 | 11.880 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-32768 | prime:batch | gb10 | 1 | 2040 | 4.427 | 8.567 | 9.215 (row 31193) | 6.657 | 19.679 | 8 | 17.082 | 11.880 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-32768 | prime:eager | gb10 | 1 | 2031 | 4.754 | 8.933 | 10.135 (row 31951) | 5.551 | 11.859 | 17 | 18.260 | 11.880 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-65535 | prefill:batch | gb10 | 1 | 2039 | 3.222 | 6.746 | 10.788 (row 63740) | 4.614 | 17.889 | 9 | 8.502 | 5.798 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-65535 | prefill:prime | gb10 | 1 | 2041 | 2.851 | 7.987 | 11.052 (row 63740) | 4.175 | 9.256 | 7 | 8.870 | 8.459 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl-65535 | prime:batch | gb10 | 1 | 2041 | 2.558 | 6.098 | 10.404 (row 64016) | 4.496 | 18.541 | 7 | 6.425 | 6.015 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl_user-32768 | prefill:batch | gb10 | 1 | 2047 | 8.968 | 12.303 | 14.360 (row 31724) | 22.999 | 28.308 | 1 | 16.580 | 15.805 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl_user-32768 | prefill:prime | gb10 | 1 | 2048 | 8.551 | 14.018 | 14.942 (row 30836) | 23.187 | 28.651 | 0 | 0.000 | 0.000 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | jsonl_user-32768 | prime:batch | gb10 | 1 | 2047 | 9.159 | 14.337 | 15.448 (row 32043) | 23.216 | 31.371 | 1 | 9.223 | 8.448 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-32768 | prefill:batch | gb10 | 1 | 1928 | 0.964 | 1.378 | 1.480 (row 32616) | 0.894 | 1.658 | 120 | 1.322 | 1.172 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-32768 | prefill:eager | gb10 | 1 | 1920 | 0.911 | 1.309 | 1.749 (row 32341) | 0.888 | 1.900 | 128 | 1.189 | 1.031 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-32768 | prefill:prime | gb10 | 2 | 1915 | 0.853 | 1.169 | 1.616 (row 31554) | 0.877 | 2.053 | 133 | 2.444 | 1.426 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-32768 | prime:batch | gb10 | 1 | 1942 | 0.916 | 1.341 | 1.634 (row 32310) | 0.939 | 1.859 | 106 | 1.912 | 1.518 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-32768 | prime:eager | gb10 | 1 | 1936 | 0.921 | 1.310 | 1.815 (row 31300) | 1.010 | 1.702 | 112 | 1.704 | 1.441 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-65535 | prefill:batch | gb10 | 1 | 1966 | 0.810 | 1.234 | 3.126 (row 65440) | 0.876 | 14.684 | 82 | 0.965 | 0.788 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-65535 | prefill:prime | gb10 | 1 | 1975 | 0.705 | 1.219 | 1.412 (row 65471) | 0.775 | 5.201 | 73 | 1.607 | 1.004 |
| kv-quant | Qwen3.8-27B-NVFP4-Q5K-mtp | prose-65535 | prime:batch | gb10 | 1 | 1969 | 0.694 | 1.361 | 2.984 (row 65440) | 0.789 | 12.403 | 79 | 1.175 | 1.060 |
| tables | ? | unknown | prefill:eager | unknown | 1 | 11 |  |  | 0.444 (row 1909) |  | 0.294 | 1 | 0.443 | 0.363 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | T=2048 | prefill:eager | rtxpro6000 | 1 | 11 |  |  | 0.613 (row 2039) |  | 0.505 | 1 | 0.278 | 0.240 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | gb10 | 20 | 12 |  |  | 0.617 (row 2045) |  | 0.877 | 0 | 0.000 | 0.000 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtx5090 | 17 | 12 |  |  | 0.593 (row 2047) |  | 1.091 | 0 | 0.000 | 0.000 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtxpro6000 | 9 | 11 |  |  | 0.569 (row 2040) |  | 0.471 | 1 | 0.288 | 0.240 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | gb10 | 39 | 12 |  |  | 0.914 (row 2040) |  | 0.792 | 0 | 0.000 | 0.000 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtx5090 | 38 | 12 |  |  | 0.981 (row 2040) |  | 0.742 | 0 | 0.000 | 0.000 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtxpro6000 | 16 | 11 |  |  | 0.613 (row 2039) |  | 0.505 | 1 | 0.278 | 0.240 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | deep-32774 | prefill:batch | gb10 | 3 | 16 |  |  | 1.440 (row 32769) |  | 1.620 | 2 | 1.165 | 0.614 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | deep-32774 | prefill:eager | gb10 | 2 | 11 |  |  | 1.008 (row 32769) |  | 1.004 | 1 | 1.429 | 0.879 |
| tables | Qwen3.5-9B-NVFP4-MTP-GGUF | board-2048 | prefill:eager | rtx5090 | 2 | 24 |  |  | 0.454 (row 2028) |  | 0.782 | 0 | 0.000 | 0.000 |
| tables | Qwen3.5-9B-Q4_K_M | p4-16390 | prefill:batch | gb10 | 1 | 12 |  |  | 2.097 (row 16379) |  | 2.694 | 0 | 0.000 | 0.000 |
| tables | Qwen3.5-9B-Q4_K_M | p4-16390 | prefill:eager | gb10 | 1 | 12 |  |  | 2.205 (row 16379) |  | 2.838 | 0 | 0.000 | 0.000 |
| tables | Qwen3.6-27B-NVFP4-Q4_K_M-mtp | T=2048 | prefill:eager | rtxpro6000 | 1 | 23 |  |  | 0.214 (row 2046) |  | 0.232 | 1 | 0.087 | 0.068 |
| tables | Qwen3.6-27B-NVFP4-Q4_K_M-mtp | board-2048 | prefill:eager | rtx5090 | 1 | 24 |  |  | 0.375 (row 2027) |  | 0.257 | 0 | 0.000 | 0.000 |
| tables | Qwen3.6-27B-Q8_0 | T=2048 | prefill:eager | rtxpro6000 | 2 | 46 |  |  | 0.266 (row 2035) |  | 0.258 | 2 | 0.131 | 0.100 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | T=2048 | prefill:batch | gb10 | 1 | 3 |  |  | 0.842 (row 2046) |  | 0.709 | 1 | 0.216 | 0.126 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | T=2048 | prefill:eager | gb10 | 4 | 15 |  |  | 0.894 (row 2046) |  | 0.683 | 1 | 0.307 | 0.180 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | gb10 | 20 | 12 |  |  | 0.526 (row 2047) |  | 0.558 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtx5090 | 17 | 12 |  |  | 0.254 (row 2047) |  | 0.177 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtxpro6000 | 9 | 12 |  |  | 0.128 (row 2045) |  | 0.137 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | gb10 | 39 | 12 |  |  | 0.408 (row 2047) |  | 0.408 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtx5090 | 38 | 12 |  |  | 0.205 (row 2044) |  | 0.146 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtxpro6000 | 16 | 12 |  |  | 0.128 (row 2038) |  | 0.113 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16386 | prefill:eager | gb10 | 1 | 11 |  |  | 9.845 (row 16375) |  | 11.911 | 1 | 4.386 | 4.141 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:batch | gb10 | 3 | 18 |  |  | 5.999 (row 16385) |  | 6.413 | 0 | 0.000 | 0.000 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:eager | gb10 | 3 | 17 |  |  | 7.229 (row 16385) |  | 7.157 | 1 | 16.867 | 14.386 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | p4-16390 | prefill:eager | gb10 | 3 | 17 |  |  | 1.039 (row 16378) |  | 1.065 | 1 | 0.295 | 0.266 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | p4-16k | prefill:eager | gb10 | 14 | 9 |  |  | 0.596 (row 23475) |  | 0.362 | 3 | 0.233 | 0.202 |
| tables | gemma-4-12b-it-qat-q4_0 | board-2048 | prefill:eager | sm170 | 3 | 12 |  |  | 0.771 (row 2091) |  | 3.835 | 0 | 0.000 | 0.000 |
| tables | gemma-4-26B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 2 | 10 |  |  | 0.778 (row 2092) |  | 1.526 | 2 | 1.313 | 1.300 |
| tables | gemma-4-26B_q4_0-it | board-2048 | prefill:eager | sm170 | 4 | 21 |  |  | 0.648 (row 2102) |  | 2.002 | 3 | 0.835 | 0.821 |
| tables | gemma-4-31B-it-shipQ6K-downQ6K | board-2048 | prefill:eager | rtxpro6000 | 1 | 10 |  |  | 1.689 (row 2100) |  | 3.746 | 2 | 0.492 | 0.293 |
| tables | gemma-4-31B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 2 | 11 |  |  | 0.272 (row 2094) |  | 0.445 | 1 | 0.600 | 0.377 |
| tables | gemma-4-31B_q4_0-it | board-2048 | prefill:eager | sm170 | 2 | 11 |  |  | 0.415 (row 2094) |  | 0.524 | 1 | 0.682 | 0.459 |
| tables | gemma-4-E4B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 1 | 12 |  |  | 0.335 (row 2099) |  | 0.304 | 0 | 0.000 | 0.000 |
| tables | ornith-1.0-9b-Q8_0 | board-2048 | prefill:eager | rtx5090 | 3 | 24 |  |  | 0.165 (row 2030) |  | 0.204 | 0 | 0.000 | 0.000 |

## 3. Candidate rules on the kv-quant gate pairs

Floors come from the agreeing rows of the calibration set: a control (prose, csv) calibrates itself; the JSONL sets take the prose set of the same depth and pair. Each cell is flips failing / flips. Rules: A = a margin above the spread floor (R1 with the spread statistic); B = A, or the flip row's spread above the spread floor (R1 + R2 on the flip row); C = margin sum above the gap-shift floor; D = a margin above the gap-shift floor.

| set | pair | floor from | agreeing rows | flips | A p99 (floor) | A p99.9 (floor) | A max (floor) | B p99 (floor) | B p99.9 (floor) | B max (floor) | C p99 (floor) | C p99.9 (floor) | C max (floor) | D p99 (floor) | D p99.9 (floor) | D max (floor) | margin sums at the flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| csv-32768 | prefill:batch | csv-32768 | 2000 | 48 | 0 (1.17) | 0 (2.09) | 0 (3.29) | 2 (1.17) | 0 (2.09) | 0 (3.29) | 1 (1.06) | 0 (2.23) | 0 (2.83) | 0 (1.06) | 0 (2.23) | 0 (2.83) | max 1.18 |
| csv-32768 | prefill:eager | csv-32768 | 1992 | 56 | 0 (1.16) | 0 (2.12) | 0 (2.49) | 1 (1.16) | 0 (2.12) | 0 (2.49) | 1 (1.04) | 0 (2.53) | 0 (2.76) | 0 (1.04) | 0 (2.53) | 0 (2.76) | max 1.07 |
| csv-65535 | prefill:batch | csv-65535 | 2014 | 34 | 1 (0.98) | 0 (3.11) | 0 (6.69) | 2 (0.98) | 0 (3.11) | 0 (6.69) | 2 (0.75) | 1 (1.59) | 0 (2.37) | 2 (0.75) | 0 (1.59) | 0 (2.37) | max 1.69 |
| jsonl-32768 | prefill:batch | prose-32768 | 1928 | 19 | 10 (0.89) | 9 (1.32) | 9 (1.66) | 10 (0.89) | 10 (1.32) | 10 (1.66) | 10 (0.96) | 10 (1.38) | 10 (1.48) | 10 (0.96) | 9 (1.38) | 9 (1.48) | 0.01 0.01 0.01 0.02 0.02 0.02 0.03 0.04 0.05 1.97 5.56 6.73 6.81 6.83 7.13 9.15 9.24 9.43 11.80 |
| jsonl-32768 | prefill:eager | prose-32768 | 1920 | 18 | 11 (0.89) | 11 (1.52) | 11 (1.90) | 11 (0.89) | 11 (1.52) | 11 (1.90) | 11 (0.91) | 11 (1.31) | 11 (1.75) | 11 (0.91) | 11 (1.31) | 11 (1.75) | 0.02 0.02 0.03 0.03 0.03 0.04 0.04 6.21 7.99 8.41 10.66 11.11 11.36 11.76 12.61 12.68 12.75 13.35 |
| jsonl-65535 | prefill:batch | prose-65535 | 1966 | 9 | 3 (0.88) | 1 (4.82) | 0 (14.68) | 3 (0.88) | 3 (4.82) | 2 (14.68) | 3 (0.81) | 3 (1.23) | 3 (3.13) | 3 (0.81) | 3 (1.23) | 2 (3.13) | 0.02 0.02 0.03 0.03 0.04 0.41 4.75 5.39 8.50 |
| jsonl_user-32768 | prefill:batch | prose-32768 | 1928 | 1 | 1 (0.89) | 1 (1.32) | 1 (1.66) | 1 (0.89) | 1 (1.32) | 1 (1.66) | 1 (0.96) | 1 (1.38) | 1 (1.48) | 1 (0.96) | 1 (1.38) | 1 (1.48) | 16.58 |
| prose-32768 | prefill:batch | prose-32768 | 1928 | 120 | 3 (0.89) | 0 (1.32) | 0 (1.66) | 4 (0.89) | 1 (1.32) | 1 (1.66) | 7 (0.96) | 0 (1.38) | 0 (1.48) | 3 (0.96) | 0 (1.38) | 0 (1.48) | max 1.32 |
| prose-32768 | prefill:eager | prose-32768 | 1920 | 128 | 3 (0.89) | 0 (1.52) | 0 (1.90) | 6 (0.89) | 1 (1.52) | 0 (1.90) | 7 (0.91) | 0 (1.31) | 0 (1.75) | 3 (0.91) | 0 (1.31) | 0 (1.75) | max 1.19 |
| prose-65535 | prefill:batch | prose-65535 | 1966 | 82 | 0 (0.88) | 0 (4.82) | 0 (14.68) | 2 (0.88) | 0 (4.82) | 0 (14.68) | 2 (0.81) | 0 (1.23) | 0 (3.13) | 0 (0.81) | 0 (1.23) | 0 (3.13) | max 0.97 |

| rule | floor quantile | control flips failed (false fails) | JSONL flips failed |
|---|---|---|---|
| A | p99 | 7 / 468 | 25 / 47 |
| A | p99.9 | 0 / 468 | 22 / 47 |
| A | max | 0 / 468 | 21 / 47 |
| B | p99 | 17 / 468 | 25 / 47 |
| B | p99.9 | 2 / 468 | 25 / 47 |
| B | max | 1 / 468 | 24 / 47 |
| C | p99 | 20 / 468 | 25 / 47 |
| C | p99.9 | 1 / 468 | 25 / 47 |
| C | max | 0 / 468 | 25 / 47 |
| D | p99 | 8 / 468 | 25 / 47 |
| D | p99.9 | 0 / 468 | 24 / 47 |
| D | max | 0 / 468 | 23 / 47 |

## 4. The gate's 12-row window on the same rows

Each 2,048-row set cut into 170 windows of 12 consecutive rows. Floor F = the largest gap shift at an agreeing row of the calibration set (section 3). Old verdict: flips <= 1. New: a window fails on a flip whose margin sum is above F (wide flip), on an agreeing row whose gap shift is above F (wide row), or on more near ties than the budget.

| set | pair | floor from | F | windows | windows with 0/1/2/3/4+ flips | old rule fails | new: wide flip | new: wide row | new: either | near ties > 1 | > 2 | > 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| csv-32768 | prefill:batch | csv-32768 | 2.832 | 170 | 132/30/6/2/0 | 8 | 0 | 0 | 0 | 8 | 2 | 0 |
| csv-32768 | prefill:eager | csv-32768 | 2.762 | 170 | 125/36/7/2/0 | 9 | 0 | 0 | 0 | 9 | 2 | 0 |
| csv-65535 | prefill:batch | csv-65535 | 2.375 | 170 | 138/30/2/0/0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| jsonl-32768 | prefill:batch | prose-32768 | 1.480 | 170 | 154/14/1/1/0 | 2 | 9 | 60 | 62 | 1 | 1 | 0 |
| jsonl-32768 | prefill:eager | prose-32768 | 1.749 | 170 | 152/18/0/0/0 | 0 | 11 | 57 | 58 | 0 | 0 | 0 |
| jsonl-65535 | prefill:batch | prose-65535 | 3.126 | 170 | 162/7/1/0/0 | 1 | 3 | 18 | 19 | 1 | 0 | 0 |
| jsonl_user-32768 | prefill:batch | prose-32768 | 1.480 | 170 | 169/1/0/0/0 | 0 | 1 | 154 | 154 | 0 | 0 | 0 |
| prose-32768 | prefill:batch | prose-32768 | 1.480 | 170 | 83/59/23/5/0 | 28 | 0 | 0 | 0 | 28 | 5 | 0 |
| prose-32768 | prefill:eager | prose-32768 | 1.749 | 170 | 74/69/22/5/0 | 27 | 0 | 0 | 0 | 27 | 5 | 0 |
| prose-65535 | prefill:batch | prose-65535 | 3.126 | 170 | 105/51/11/3/0 | 14 | 0 | 0 | 0 | 14 | 3 | 0 |

## 5. How many calibration rows a floor needs

Controls only. Floor = the largest gap shift at an agreeing row inside a block of N consecutive rows; tested on every 12-row window outside the block. `expected` is the order-statistics rate 12 / (N + 12) for exchangeable rows.

| N | blocks | mean floor | test windows | windows with an agreeing row over the floor | with a flip over the floor | either (false fail) | expected |
|---|---|---|---|---|---|---|---|
| 12 | 96 | 0.468 | 16224 | 54.3% | 7.8% | 56.1% | 50.0% |
| 48 | 96 | 0.824 | 15936 | 24.8% | 3.5% | 26.3% | 20.0% |
| 96 | 96 | 1.133 | 15552 | 12.4% | 2.0% | 13.5% | 11.1% |
| 256 | 48 | 1.456 | 7140 | 5.6% | 0.9% | 6.2% | 4.5% |
| 512 | 24 | 1.679 | 3060 | 3.2% | 0.4% | 3.5% | 2.3% |
| 1024 | 12 | 2.068 | 1020 | 0.8% | 0.0% | 0.8% | 1.2% |

## 6. Every distinct 12-row table, re-judged

One line per distinct table (identical tables from repeated runs are merged; `runs` counts them). `needs F >=` is the smallest floor under which the table passes the two per-row clauses: the largest margin sum at a flip and the largest gap shift at an agreeing row. Old verdict: the table part of the gate at 031f146690 (flip count against the budget row; the per-flip clause never fires).

| tree | model | prompt | pair | device | runs | first receipt | flips / old budget | old verdict | flip clause needs F >= | row clause needs F >= | max spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tables | ? | unknown | prefill:eager | unknown | 1 | research/hf-argmax-margin-20260905/glm4o6-margin-table.txt | 1 / 1 | PASS | 0.443 (row 1920: 0.363 + 0.080, spread 0.404) | 0.444 (row 1909: 8.134 vs 8.578, spread 0.294) | 0.404 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | T=2048 | prefill:eager | rtxpro6000 | 1 | research/orndecode-20260822/raw/argmax-margin-ornith.log | 1 / 1 | PASS | 0.278 (row 2041: 0.240 + 0.038, spread 0.348) | 0.613 (row 2039: 0.947 vs 0.334, spread 0.505) | 0.505 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | gb10 | 20 | research/release-qualification/cells/03234a773114-gb10-battery-perf-serving/battery/argmax-batch-2/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.batch.log | 0 / 1 | PASS |  | 0.617 (row 2045: 2.051 vs 2.668, spread 0.361) | 0.877 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtx5090 | 17 | research/release-qualification/cells/03234a773114-rtx5090-battery-perf-serving/battery/argmax-batch-2/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.batch.log | 0 / 1 | PASS |  | 0.593 (row 2047: 1.214 vs 0.621, spread 1.091) | 1.091 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtxpro6000 | 9 | research/release-qualification/cells/67d6d927d5f7-rtxpro6000-battery-perf/battery/argmax-batch-2/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.batch.log | 1 / 1 | PASS | 0.288 (row 2041: 0.240 + 0.048, spread 0.460) | 0.569 (row 2040: 7.587 vs 7.018, spread 0.471) | 0.471 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | gb10 | 39 | research/release-qualification/cells/03234a773114-gb10-battery-perf-serving/battery/argmax-1/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.log | 0 / 1 | PASS |  | 0.914 (row 2040: 7.654 vs 8.568, spread 0.792) | 0.792 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtx5090 | 38 | research/release-qualification/cells/03234a773114-rtx5090-battery-perf-serving/battery/argmax-1/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.log | 0 / 1 | PASS |  | 0.981 (row 2040: 7.587 vs 8.568, spread 0.742) | 0.742 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtxpro6000 | 16 | research/release-qualification/candidate-pro-ubuntu-24-04/cells/argmax-1/gate-Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp-board-2048.log | 1 / 1 | PASS | 0.278 (row 2041: 0.240 + 0.038, spread 0.348) | 0.613 (row 2039: 0.947 vs 0.334, spread 0.505) | 0.505 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | deep-32774 | prefill:batch | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit1/am-or-32774-b-base.txt | 1 / 1 | PASS | 1.165 (row 32773: 0.551 + 0.614, spread 0.774) | 1.440 (row 32769: 6.751 vs 8.191, spread 1.222) | 1.400 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | deep-32774 | prefill:batch | gb10 | 2 | research/gb10-split-rung-20261110/gates-unit1/am-or-32774-b.txt | 1 / 1 | PASS | 0.916 (row 32773: 0.551 + 0.366, spread 0.612) | 1.397 (row 32769: 6.751 vs 8.148, spread 1.415) | 1.620 |
| tables | Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp | deep-32774 | prefill:eager | gb10 | 2 | research/gb10-split-rung-20261110/gates-unit1/am-or-32774-e.txt | 1 / 1 | PASS | 1.429 (row 32773: 0.551 + 0.879, spread 0.866) | 1.008 (row 32769: 6.751 vs 7.759, spread 1.004) | 1.004 |
| tables | Qwen3.5-9B-NVFP4-MTP-GGUF | board-2048 | prefill:eager | rtx5090 | 1 | research/q8-argmax-20260806/gate-Qwen3.5-9B-NVFP4-MTP-GGUF-board-2048.log | 0 / 1 | PASS |  | 0.435 (row 2047: 0.100 vs 0.535, spread 0.231) | 0.782 |
| tables | Qwen3.5-9B-NVFP4-MTP-GGUF | board-2048 | prefill:eager | rtx5090 | 1 | research/q8-argmax-20260806/local5090/margin-q9nv-board2048.log | 0 / 2 | PASS |  | 0.454 (row 2028: 1.098 vs 0.644, spread 0.278) | 0.782 |
| tables | Qwen3.5-9B-Q4_K_M | p4-16390 | prefill:batch | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit2/am-9b-p4-16390-b-764cd39b5b.txt | 0 / 1 | PASS |  | 2.097 (row 16379: 10.294 vs 8.198, spread 2.694) | 2.694 |
| tables | Qwen3.5-9B-Q4_K_M | p4-16390 | prefill:eager | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit2/am-9b-p4-16390-e-764cd39b5b.txt | 0 / 1 | PASS |  | 2.205 (row 16379: 10.294 vs 8.089, spread 2.838) | 2.838 |
| tables | Qwen3.6-27B-NVFP4-Q4_K_M-mtp | T=2048 | prefill:eager | rtxpro6000 | 1 | research/q8-argmax-20260806/pod/probe-nv-naked.log | 1 / 2 | PASS | 0.087 (row 2042: 0.068 + 0.018, spread 0.068) | 0.214 (row 2046: 0.420 vs 0.206, spread 0.146) | 0.232 |
| tables | Qwen3.6-27B-NVFP4-Q4_K_M-mtp | board-2048 | prefill:eager | rtx5090 | 1 | research/q8-argmax-20260806/local5090/margin-q27nv-board2048.log | 0 / 2 | PASS |  | 0.375 (row 2027: 5.377 vs 5.752, spread 0.257) | 0.257 |
| tables | Qwen3.6-27B-Q8_0 | T=2048 | prefill:eager | rtxpro6000 | 1 | research/q8-argmax-20260806/pod/probe-q8-naked.log | 1 / 2 | PASS | 0.131 (row 2047: 0.031 + 0.100, spread 0.212) | 0.240 (row 2044: 1.575 vs 1.336, spread 0.177) | 0.258 |
| tables | Qwen3.6-27B-Q8_0 | T=2048 | prefill:eager | rtxpro6000 | 1 | research/q8-argmax-20260806/pod/probe-q8-split8.log | 1 / 2 | PASS | 0.074 (row 2047: 0.031 + 0.043, spread 0.232) | 0.266 (row 2035: 9.071 vs 9.337, spread 0.207) | 0.232 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | T=2048 | prefill:batch | gb10 | 1 | research/kv-quant-repetitive-20261110/runs/repro-ref-batch/stdout.log | 1 / 1 | PASS | 0.216 (row 2044: 0.126 + 0.090, spread 0.194) | 0.842 (row 2046: 8.367 vs 7.525, spread 0.709) | 0.709 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | T=2048 | prefill:eager | gb10 | 3 | research/gb10-prefill-attn-20261107/gates-unit1-final/argmax-gate-raw.log | 0 / 1 | PASS |  | 0.408 (row 2047: 1.209 vs 0.801, spread 0.408) | 0.408 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | T=2048 | prefill:eager | gb10 | 1 | research/kv-quant-repetitive-20261110/runs/repro-ref-eager/stdout.log | 1 / 1 | PASS | 0.307 (row 2044: 0.126 + 0.180, spread 0.252) | 0.894 (row 2046: 8.367 vs 7.473, spread 0.683) | 0.683 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | gb10 | 20 | research/release-qualification/cells/03234a773114-gb10-battery-perf-serving/battery/argmax-batch-5/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.batch.log | 0 / 1 | PASS |  | 0.526 (row 2047: 1.209 vs 0.684, spread 0.558) | 0.558 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtx5090 | 17 | research/release-qualification/cells/03234a773114-rtx5090-battery-perf-serving/battery/argmax-batch-5/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.batch.log | 0 / 1 | PASS |  | 0.254 (row 2047: 0.938 vs 0.684, spread 0.177) | 0.177 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:batch | rtxpro6000 | 9 | research/release-qualification/cells/67d6d927d5f7-rtxpro6000-battery-perf/battery/argmax-batch-5/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.batch.log | 0 / 1 | PASS |  | 0.128 (row 2045: 1.387 vs 1.516, spread 0.086) | 0.137 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | gb10 | 39 | research/release-qualification/cells/03234a773114-gb10-battery-perf-serving/battery/argmax-4/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.log | 0 / 1 | PASS |  | 0.408 (row 2047: 1.209 vs 0.801, spread 0.408) | 0.408 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtx5090 | 38 | research/release-qualification/cells/03234a773114-rtx5090-battery-perf-serving/battery/argmax-4/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.log | 0 / 1 | PASS |  | 0.205 (row 2044: 2.461 vs 2.666, spread 0.119) | 0.146 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | board-2048 | prefill:eager | rtxpro6000 | 16 | research/release-qualification/candidate-pro-ubuntu-24-04/cells/argmax-3/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.log | 0 / 1 | PASS |  | 0.128 (row 2038: 0.498 vs 0.626, spread 0.070) | 0.113 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16386 | prefill:eager | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit1/ref-main-16386.txt | 1 / 1 | PASS | 4.386 (row 16374: 0.245 + 4.141, spread 13.704) | 9.845 (row 16375: 2.926 vs 12.771, spread 11.911) | 13.704 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:batch | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit1/am-27b-16390-b-base.txt | 0 / 1 | PASS |  | 4.035 (row 16385: 14.386 vs 10.351, spread 5.582) | 5.582 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:batch | gb10 | 2 | research/gb10-split-rung-20261110/gates-unit1/am-27b-16390-b.txt | 0 / 1 | PASS |  | 5.999 (row 16385: 14.386 vs 8.387, spread 6.413) | 6.413 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:eager | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit1/am-27b-16390-e-base.txt | 0 / 1 | PASS |  | 7.229 (row 16385: 14.386 vs 7.156, spread 7.157) | 7.157 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | deep-16390 | prefill:eager | gb10 | 2 | research/gb10-split-rung-20261110/gates-unit1/am-27b-16390-e.txt | 1 / 1 | PASS | 16.867 (row 16385: 14.386 + 2.481, spread 16.722) | 0.846 (row 16383: 10.313 vs 11.159, spread 0.606) | 16.722 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | p4-16390 | prefill:eager | gb10 | 1 | research/gb10-split-rung-20261110/gates-unit1/am-27b-p4-16390-e-base.txt | 1 / 1 | PASS | 0.295 (row 16385: 0.266 + 0.030, spread 0.196) | 1.039 (row 16378: 6.050 vs 7.089, spread 1.065) | 1.065 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | p4-16390 | prefill:eager | gb10 | 2 | research/gb10-split-rung-20261110/gates-unit1/am-27b-p4-16390-e.txt | 0 / 1 | PASS |  | 1.039 (row 16378: 6.050 vs 7.089, spread 1.065) | 1.065 |
| tables | Qwen3.8-27B-NVFP4-Q5K-mtp | p4-16k | prefill:eager | gb10 | 14 | research/gb10-prefill-20261106/gemm-v2-gates-final-unit1/argmax16k-V/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-p4-16k.log | 3 / 3 | PASS | 0.233 (row 23472: 0.202 + 0.031, spread 0.138) | 0.596 (row 23475: 2.357 vs 2.954, spread 0.362) | 0.362 |
| tables | gemma-4-12b-it-qat-q4_0 | board-2048 | prefill:eager | sm170 | 3 | research/exec-p1a-gemma-prime-20260919/raw/12b/argmax-new/gate-gemma-4-12b-it-qat-q4_0-board-2048.log | 0 / 1 | PASS |  | 0.771 (row 2091: 1.561 vs 2.332, spread 0.900) | 3.835 |
| tables | gemma-4-26B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 2 | research/gemma-one-program-20261104/raw/argmax-g26-off/gate-gemma-4-26B_q4_0-it-board-2048.log | 2 / 3 | PASS | 1.313 (row 2096: 0.014 + 1.300, spread 2.283) | 0.778 (row 2092: 3.437 vs 4.215, spread 1.526) | 2.283 |
| tables | gemma-4-26B_q4_0-it | board-2048 | prefill:eager | sm170 | 1 | research/exec-p1a-gemma-prime-20260919/raw/26b-a4b/argmax-new/gate-gemma-4-26B_q4_0-it-board-2048.log | 1 / 3 | PASS | 0.378 (row 2091: 0.364 + 0.014, spread 0.744) | 0.648 (row 2102: 0.048 vs 0.695, spread 0.566) | 2.002 |
| tables | gemma-4-26B_q4_0-it | board-2048 | prefill:eager | sm170 | 3 | research/exec-p1a-gemma-prime-20260919/raw/26b-a4b/routerfix/argmax/gate-gemma-4-26B_q4_0-it-board-2048.log | 2 / 3 | PASS | 0.835 (row 2096: 0.014 + 0.821, spread 3.019) | 0.638 (row 2102: 0.057 vs 0.695, spread 0.628) | 3.019 |
| tables | gemma-4-31B-it-shipQ6K-downQ6K | board-2048 | prefill:eager | rtxpro6000 | 1 | research/gemma-vision-20260816/gap-receipts/argmax-gate/banked-invocations.txt | 2 / 3 | PASS | 0.492 (row 2094: 0.199 + 0.293, spread 0.328) | 1.689 (row 2100: 3.130 vs 1.441, spread 3.746) | 3.746 |
| tables | gemma-4-31B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 2 | research/gemma-one-program-20261104/raw/argmax-g31-off/gate-gemma-4-31B_q4_0-it-board-2048.log | 1 / 3 | PASS | 0.600 (row 2101: 0.223 + 0.377, spread 0.421) | 0.272 (row 2094: 0.046 vs 0.317, spread 0.200) | 0.445 |
| tables | gemma-4-31B_q4_0-it | board-2048 | prefill:eager | sm170 | 2 | research/exec-p1a-gemma-prime-20260919/raw/31b/argmax-new/gate-gemma-4-31B_q4_0-it-board-2048.log | 1 / 3 | PASS | 0.682 (row 2101: 0.223 + 0.459, spread 0.504) | 0.415 (row 2094: 0.046 vs 0.460, spread 0.437) | 0.524 |
| tables | gemma-4-E4B_q4_0-it | board-2048 | prefill:eager | rtx5090 | 1 | research/modelplan-onboarding-e4b-20261104/gpu/argmax/gate-gemma-4-E4B_q4_0-it-board-2048.log | 0 / 1 | PASS |  | 0.335 (row 2099: 0.713 vs 0.378, spread 0.304) | 0.304 |
| tables | ornith-1.0-9b-Q8_0 | board-2048 | prefill:eager | rtx5090 | 2 | research/q8-argmax-20260806/gate-ornith-1.0-9b-Q8_0-board-2048.canary.log | 0 / 1 | PASS |  | 0.123 (row 2041: 0.353 vs 0.476, spread 0.183) | 0.183 |
| tables | ornith-1.0-9b-Q8_0 | board-2048 | prefill:eager | rtx5090 | 1 | research/q8-argmax-20260806/local5090/margin-o9q8-board2048.log | 0 / 2 | PASS |  | 0.165 (row 2030: 0.445 vs 0.280, spread 0.110) | 0.204 |
