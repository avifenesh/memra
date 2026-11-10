### L2/prefill (2 instances): wall 2087.110 ms, GPU busy 2086.699 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 821.163 | 39.4% |
| attn_keys/index_score | 446.523 | 21.4% |
| sparse_attn | 186.323 | 8.9% |
| wo_a | 103.548 | 5.0% |
| q_b | 70.870 | 3.4% |
| hc_ffn_pre | 69.925 | 3.4% |
| wo_b | 68.245 | 3.3% |
| shared_expert | 67.666 | 3.2% |
| attn_keys/topk | 51.905 | 2.5% |
| attn_keys/compressor | 50.873 | 2.4% |
| hc_attn_pre | 40.970 | 2.0% |
| router | 27.656 | 1.3% |
| hc_post | 23.426 | 1.1% |
| attn_keys/index_q | 20.154 | 1.0% |
| attn_keys/qr | 13.393 | 0.6% |
| attn_keys/kv_window | 8.191 | 0.4% |
| combine | 7.922 | 0.4% |
| moe_plan | 6.701 | 0.3% |
| attn_keys/kv_append | 0.507 | 0.0% |
| attn_idx_rows | 0.408 | 0.0% |
| attn_keys/index_keys | 0.330 | 0.0% |

host API calls: 2175 calls, 2086.353 ms; top: cuStreamSynchronize 2077.981 ms x1, cuMemAllocAsync 6.632 ms x458, cudaLaunchKernel_v7000 1.351 ms x604, cuMemFreeAsync 0.187 ms x456, cuMemsetD8Async 0.108 ms x48, cuKernelGetName 0.055 ms x604

### L20/prefill (2 instances): wall 2629.545 ms, GPU busy 2629.099 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 881.234 | 33.5% |
| experts | 791.364 | 30.1% |
| sparse_attn | 188.614 | 7.2% |
| attn_keys/topk | 125.374 | 4.8% |
| wo_a | 105.017 | 4.0% |
| attn_keys/candidates | 100.274 | 3.8% |
| q_b | 71.174 | 2.7% |
| hc_ffn_pre | 71.082 | 2.7% |
| wo_b | 68.925 | 2.6% |
| shared_expert | 68.022 | 2.6% |
| hc_attn_pre | 41.140 | 1.6% |
| router | 27.989 | 1.1% |
| hc_post | 23.898 | 0.9% |
| attn_keys/index_q | 20.271 | 0.8% |
| attn_keys/qr | 13.459 | 0.5% |
| attn_keys/kv_window | 8.223 | 0.3% |
| combine | 7.964 | 0.3% |
| moe_plan | 6.891 | 0.3% |
| attn_keys/compressor | 5.916 | 0.2% |
| attn_keys/kv_append | 1.204 | 0.0% |
| attn_keys/index_keys | 0.642 | 0.0% |
| attn_idx_rows | 0.420 | 0.0% |

host API calls: 2201 calls, 2628.232 ms; top: cuStreamSynchronize 2615.531 ms x1, cuMemAllocAsync 10.705 ms x457, cudaLaunchKernel_v7000 1.556 ms x618, cuMemFreeAsync 0.220 ms x455, cuMemsetD8Async 0.115 ms x48, cuKernelGetName 0.060 ms x618

### L21/prefill (2 instances): wall 1492.263 ms, GPU busy 1491.906 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 810.390 | 54.3% |
| sparse_attn | 181.538 | 12.2% |
| wo_a | 97.562 | 6.5% |
| hc_ffn_pre | 70.131 | 4.7% |
| q_b | 69.130 | 4.6% |
| shared_expert | 67.759 | 4.5% |
| wo_b | 66.147 | 4.4% |
| hc_attn_pre | 41.001 | 2.7% |
| router | 27.746 | 1.9% |
| hc_post | 23.587 | 1.6% |
| attn_keys/qr | 13.401 | 0.9% |
| attn_keys/kv_window | 8.197 | 0.5% |
| combine | 7.897 | 0.5% |
| moe_plan | 6.807 | 0.5% |
| attn_idx_rows | 0.433 | 0.0% |
| attn_keys/kv_append | 0.179 | 0.0% |

host API calls: 1826 calls, 1491.646 ms; top: cuStreamSynchronize 1485.446 ms x1, cuMemAllocAsync 4.801 ms x406, cudaLaunchKernel_v7000 1.058 ms x482, cuMemFreeAsync 0.174 ms x404, cuMemsetD8Async 0.101 ms x48, cuKernelGetName 0.044 ms x482

### L2/decode (6 instances): wall 0.973 ms, GPU busy 0.868 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.163 | 18.8% |
| attn_keys/topk | 0.146 | 16.8% |
| wo_b | 0.065 | 7.5% |
| wo_a | 0.051 | 5.8% |
| attn_keys/kv_append | 0.050 | 5.8% |
| attn_keys/compressor | 0.048 | 5.6% |
| sparse_attn | 0.047 | 5.4% |
| shared_expert | 0.044 | 5.0% |
| attn_keys/index_q | 0.043 | 4.9% |
| q_b | 0.037 | 4.3% |
| hc_ffn_pre | 0.037 | 4.2% |
| router | 0.036 | 4.1% |
| hc_attn_pre | 0.028 | 3.2% |
| attn_keys/index_score | 0.016 | 1.9% |
| attn_keys/kv_window | 0.013 | 1.5% |
| attn_keys/qr | 0.012 | 1.4% |
| moe_plan | 0.012 | 1.4% |
| hc_post | 0.007 | 0.9% |
| attn_keys/index_keys | 0.005 | 0.6% |
| combine | 0.004 | 0.5% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.894 ms; top: cuStreamSynchronize 0.672 ms x1, cudaLaunchKernel_v7000 0.148 ms x60, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.235 ms, GPU busy 1.200 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.280 | 23.4% |
| experts | 0.167 | 13.9% |
| attn_keys/kv_append | 0.124 | 10.3% |
| attn_keys/candidates | 0.114 | 9.5% |
| wo_b | 0.062 | 5.2% |
| wo_a | 0.051 | 4.3% |
| sparse_attn | 0.048 | 4.0% |
| q_b | 0.044 | 3.6% |
| shared_expert | 0.043 | 3.6% |
| attn_keys/index_q | 0.042 | 3.5% |
| hc_ffn_pre | 0.037 | 3.1% |
| router | 0.036 | 3.0% |
| attn_keys/compressor | 0.032 | 2.7% |
| attn_keys/index_score | 0.030 | 2.5% |
| hc_attn_pre | 0.028 | 2.3% |
| attn_keys/kv_window | 0.013 | 1.1% |
| moe_plan | 0.013 | 1.0% |
| attn_keys/qr | 0.012 | 1.0% |
| attn_keys/index_keys | 0.010 | 0.8% |
| hc_post | 0.007 | 0.6% |
| combine | 0.004 | 0.4% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.173 ms; top: cuStreamSynchronize 0.974 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.723 ms, GPU busy 0.692 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.167 | 24.1% |
| attn_keys/kv_append | 0.120 | 17.3% |
| wo_b | 0.063 | 9.1% |
| wo_a | 0.053 | 7.6% |
| sparse_attn | 0.049 | 7.1% |
| q_b | 0.044 | 6.3% |
| shared_expert | 0.043 | 6.3% |
| hc_ffn_pre | 0.037 | 5.3% |
| router | 0.036 | 5.2% |
| hc_attn_pre | 0.029 | 4.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.7% |
| hc_post | 0.007 | 1.1% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 191 calls, 0.675 ms; top: cuStreamSynchronize 0.520 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.009 ms x3, cuMemsetD8Async 0.004 ms x3

