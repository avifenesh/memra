### L2/prefill (2 instances): wall 1511.618 ms, GPU busy 1506.812 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 585.123 | 38.8% |
| index_score | 258.521 | 17.2% |
| sparse_attn | 152.569 | 10.1% |
| wo_a | 93.499 | 6.2% |
| (unranged) | 91.876 | 6.1% |
| q_b | 58.914 | 3.9% |
| wo_b | 56.165 | 3.7% |
| shared_expert | 51.221 | 3.4% |
| compressor | 45.476 | 3.0% |
| topk | 43.361 | 2.9% |
| router | 23.512 | 1.6% |
| index_q | 18.104 | 1.2% |
| qr | 11.733 | 0.8% |
| combine | 7.602 | 0.5% |
| kv_window | 7.411 | 0.5% |
| kv_append | 0.490 | 0.0% |
| index_keys | 0.485 | 0.0% |
| attn_idx_rows | 0.427 | 0.0% |
| moe_plan | 0.323 | 0.0% |

host API calls: 1791 calls, 1510.750 ms; top: cuStreamSynchronize 1500.098 ms x2, cuMemAllocAsync 8.881 ms x281, cudaLaunchKernel_v7000 1.405 ms x588, cuMemFreeAsync 0.124 ms x279, cuMemsetD8Async 0.119 ms x48, cuMemcpyDtoDAsync_v2 0.067 ms x5

### L20/prefill (2 instances): wall 2023.282 ms, GPU busy 2021.642 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 582.825 | 28.8% |
| index_score | 570.469 | 28.2% |
| topk | 166.934 | 8.3% |
| sparse_attn | 158.272 | 7.8% |
| wo_a | 100.988 | 5.0% |
| candidates | 97.020 | 4.8% |
| (unranged) | 94.281 | 4.7% |
| q_b | 60.562 | 3.0% |
| wo_b | 57.998 | 2.9% |
| shared_expert | 53.000 | 2.6% |
| router | 24.634 | 1.2% |
| index_q | 18.077 | 0.9% |
| qr | 12.055 | 0.6% |
| combine | 7.986 | 0.4% |
| kv_window | 7.623 | 0.4% |
| compressor | 6.002 | 0.3% |
| kv_append | 1.164 | 0.1% |
| index_keys | 0.987 | 0.0% |
| attn_idx_rows | 0.423 | 0.0% |
| moe_plan | 0.345 | 0.0% |

host API calls: 1817 calls, 2022.492 ms; top: cuStreamSynchronize 2014.941 ms x2, cuMemAllocAsync 5.913 ms x280, cudaLaunchKernel_v7000 1.297 ms x602, cuMemFreeAsync 0.129 ms x278, cuMemsetD8Async 0.100 ms x48, cuMemcpyDtoDAsync_v2 0.058 ms x5

### L21/prefill (2 instances): wall 1167.037 ms, GPU busy 1162.999 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 602.055 | 51.8% |
| sparse_attn | 152.441 | 13.1% |
| (unranged) | 94.383 | 8.1% |
| wo_a | 93.360 | 8.0% |
| q_b | 58.343 | 5.0% |
| wo_b | 55.893 | 4.8% |
| shared_expert | 53.119 | 4.6% |
| router | 24.720 | 2.1% |
| qr | 12.091 | 1.0% |
| combine | 7.999 | 0.7% |
| kv_window | 7.659 | 0.7% |
| attn_idx_rows | 0.413 | 0.0% |
| moe_plan | 0.343 | 0.0% |
| kv_append | 0.180 | 0.0% |

host API calls: 1474 calls, 1166.394 ms; top: cuStreamSynchronize 1157.707 ms x2, cuMemAllocAsync 7.360 ms x245, cudaLaunchKernel_v7000 1.017 ms x466, cuMemsetD8Async 0.105 ms x48, cuMemFreeAsync 0.104 ms x243, cuMemcpyDtoDAsync_v2 0.057 ms x4

### L2/decode (6 instances): wall 0.810 ms, GPU busy 0.793 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.165 | 20.8% |
| topk | 0.118 | 14.9% |
| wo_a | 0.051 | 6.4% |
| compressor | 0.050 | 6.3% |
| kv_append | 0.050 | 6.3% |
| shared_expert | 0.047 | 5.9% |
| router | 0.046 | 5.7% |
| wo_b | 0.045 | 5.6% |
| (unranged) | 0.042 | 5.3% |
| q_b | 0.037 | 4.7% |
| sparse_attn | 0.036 | 4.5% |
| index_q | 0.034 | 4.2% |
| index_score | 0.019 | 2.4% |
| kv_window | 0.014 | 1.8% |
| moe_plan | 0.014 | 1.8% |
| qr | 0.013 | 1.6% |
| index_keys | 0.005 | 0.7% |
| combine | 0.005 | 0.6% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 145 calls, 0.731 ms; top: cuStreamSynchronize 0.534 ms x2, cudaLaunchKernel_v7000 0.151 ms x59, cuMemcpyDtoDAsync_v2 0.021 ms x7, cuMemAllocAsync 0.008 ms x9, cuKernelGetName 0.006 ms x59, cuMemsetD8Async 0.005 ms x3

### L20/decode (6 instances): wall 1.067 ms, GPU busy 1.082 ms

| op | GPU ms | share of busy |
|---|---|---|
| topk | 0.225 | 20.8% |
| experts | 0.168 | 15.5% |
| kv_append | 0.123 | 11.3% |
| candidates | 0.106 | 9.8% |
| wo_a | 0.052 | 4.8% |
| shared_expert | 0.047 | 4.3% |
| wo_b | 0.045 | 4.1% |
| router | 0.044 | 4.1% |
| q_b | 0.043 | 4.0% |
| (unranged) | 0.043 | 3.9% |
| sparse_attn | 0.037 | 3.5% |
| index_q | 0.034 | 3.1% |
| index_score | 0.033 | 3.0% |
| compressor | 0.024 | 2.2% |
| moe_plan | 0.014 | 1.3% |
| kv_window | 0.014 | 1.3% |
| qr | 0.013 | 1.2% |
| index_keys | 0.011 | 1.0% |
| combine | 0.005 | 0.4% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 142 calls, 1.016 ms; top: cuStreamSynchronize 0.855 ms x2, cudaLaunchKernel_v7000 0.132 ms x62, cuMemcpyDtoDAsync_v2 0.015 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuMemAllocAsync 0.002 ms x2

### L21/decode (6 instances): wall 0.641 ms, GPU busy 0.662 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.172 | 26.0% |
| kv_append | 0.120 | 18.1% |
| wo_a | 0.050 | 7.6% |
| shared_expert | 0.048 | 7.3% |
| wo_b | 0.047 | 7.2% |
| (unranged) | 0.045 | 6.9% |
| router | 0.045 | 6.8% |
| q_b | 0.043 | 6.6% |
| sparse_attn | 0.040 | 6.1% |
| moe_plan | 0.015 | 2.2% |
| kv_window | 0.014 | 2.1% |
| qr | 0.013 | 2.0% |
| combine | 0.005 | 0.7% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 107 calls, 0.605 ms; top: cuStreamSynchronize 0.485 ms x2, cudaLaunchKernel_v7000 0.098 ms x46, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

