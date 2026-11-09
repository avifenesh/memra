### L2/prefill (2 instances): wall 2085.817 ms, GPU busy 2085.444 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 820.503 | 39.3% |
| attn_keys/index_score | 446.114 | 21.4% |
| sparse_attn | 186.168 | 8.9% |
| wo_a | 103.458 | 5.0% |
| q_b | 70.928 | 3.4% |
| hc_ffn_pre | 69.961 | 3.4% |
| wo_b | 68.259 | 3.3% |
| shared_expert | 67.653 | 3.2% |
| attn_keys/topk | 52.011 | 2.5% |
| attn_keys/compressor | 50.786 | 2.4% |
| hc_attn_pre | 40.929 | 2.0% |
| router | 27.669 | 1.3% |
| hc_post | 23.456 | 1.1% |
| attn_keys/index_q | 20.141 | 1.0% |
| attn_keys/qr | 13.382 | 0.6% |
| attn_keys/kv_window | 8.182 | 0.4% |
| combine | 7.898 | 0.4% |
| moe_plan | 6.702 | 0.3% |
| attn_keys/kv_append | 0.504 | 0.0% |
| attn_idx_rows | 0.413 | 0.0% |
| attn_keys/index_keys | 0.329 | 0.0% |

host API calls: 2175 calls, 2085.081 ms; top: cuStreamSynchronize 2076.469 ms x1, cuMemAllocAsync 6.928 ms x458, cudaLaunchKernel_v7000 1.314 ms x604, cuMemFreeAsync 0.184 ms x456, cuMemsetD8Async 0.103 ms x48, cuKernelGetName 0.056 ms x604

### L20/prefill (2 instances): wall 2641.184 ms, GPU busy 2640.762 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 884.184 | 33.5% |
| experts | 794.746 | 30.1% |
| sparse_attn | 189.171 | 7.2% |
| attn_keys/topk | 126.872 | 4.8% |
| wo_a | 106.101 | 4.0% |
| attn_keys/candidates | 100.241 | 3.8% |
| hc_ffn_pre | 71.564 | 2.7% |
| q_b | 71.401 | 2.7% |
| wo_b | 69.086 | 2.6% |
| shared_expert | 68.262 | 2.6% |
| hc_attn_pre | 41.293 | 1.6% |
| router | 28.206 | 1.1% |
| hc_post | 24.321 | 0.9% |
| attn_keys/index_q | 20.338 | 0.8% |
| attn_keys/qr | 13.528 | 0.5% |
| attn_keys/kv_window | 8.264 | 0.3% |
| combine | 8.054 | 0.3% |
| moe_plan | 6.900 | 0.3% |
| attn_keys/compressor | 5.956 | 0.2% |
| attn_keys/kv_append | 1.205 | 0.0% |
| attn_keys/index_keys | 0.650 | 0.0% |
| attn_idx_rows | 0.420 | 0.0% |

host API calls: 2201 calls, 2640.408 ms; top: cuStreamSynchronize 2628.581 ms x1, cuMemAllocAsync 10.045 ms x457, cudaLaunchKernel_v7000 1.376 ms x618, cuMemFreeAsync 0.200 ms x455, cuMemsetD8Async 0.106 ms x48, cuKernelGetName 0.060 ms x618

### L21/prefill (2 instances): wall 1500.701 ms, GPU busy 1500.339 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 814.734 | 54.3% |
| sparse_attn | 182.498 | 12.2% |
| wo_a | 98.467 | 6.6% |
| hc_ffn_pre | 70.649 | 4.7% |
| q_b | 69.498 | 4.6% |
| shared_expert | 68.061 | 4.5% |
| wo_b | 66.372 | 4.4% |
| hc_attn_pre | 41.154 | 2.7% |
| router | 27.940 | 1.9% |
| hc_post | 23.890 | 1.6% |
| attn_keys/qr | 13.460 | 0.9% |
| attn_keys/kv_window | 8.225 | 0.5% |
| combine | 7.953 | 0.5% |
| moe_plan | 6.831 | 0.5% |
| attn_idx_rows | 0.427 | 0.0% |
| attn_keys/kv_append | 0.181 | 0.0% |

host API calls: 1826 calls, 1500.068 ms; top: cuStreamSynchronize 1493.785 ms x1, cuMemAllocAsync 4.608 ms x406, cudaLaunchKernel_v7000 1.330 ms x482, cuMemFreeAsync 0.173 ms x404, cuMemsetD8Async 0.104 ms x48, cuKernelGetName 0.044 ms x482

### L2/decode (6 instances): wall 0.933 ms, GPU busy 0.870 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.164 | 18.8% |
| attn_keys/topk | 0.146 | 16.8% |
| wo_b | 0.066 | 7.6% |
| wo_a | 0.051 | 5.9% |
| attn_keys/kv_append | 0.050 | 5.8% |
| attn_keys/compressor | 0.048 | 5.6% |
| sparse_attn | 0.047 | 5.4% |
| shared_expert | 0.044 | 5.0% |
| attn_keys/index_q | 0.042 | 4.9% |
| hc_ffn_pre | 0.037 | 4.3% |
| q_b | 0.037 | 4.2% |
| router | 0.036 | 4.1% |
| hc_attn_pre | 0.028 | 3.3% |
| attn_keys/index_score | 0.016 | 1.8% |
| attn_keys/kv_window | 0.013 | 1.5% |
| moe_plan | 0.012 | 1.4% |
| attn_keys/qr | 0.012 | 1.4% |
| hc_post | 0.008 | 0.9% |
| attn_keys/index_keys | 0.005 | 0.6% |
| combine | 0.004 | 0.5% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.849 ms; top: cuStreamSynchronize 0.625 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.237 ms, GPU busy 1.202 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.280 | 23.3% |
| experts | 0.167 | 13.9% |
| attn_keys/kv_append | 0.124 | 10.3% |
| attn_keys/candidates | 0.114 | 9.5% |
| wo_b | 0.062 | 5.1% |
| wo_a | 0.052 | 4.3% |
| sparse_attn | 0.048 | 4.0% |
| q_b | 0.044 | 3.6% |
| shared_expert | 0.043 | 3.6% |
| attn_keys/index_q | 0.042 | 3.5% |
| hc_ffn_pre | 0.037 | 3.1% |
| router | 0.036 | 3.0% |
| attn_keys/compressor | 0.032 | 2.7% |
| attn_keys/index_score | 0.030 | 2.5% |
| hc_attn_pre | 0.028 | 2.4% |
| attn_keys/kv_window | 0.013 | 1.1% |
| moe_plan | 0.013 | 1.1% |
| attn_keys/qr | 0.012 | 1.0% |
| attn_keys/index_keys | 0.010 | 0.8% |
| hc_post | 0.007 | 0.6% |
| combine | 0.004 | 0.4% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.174 ms; top: cuStreamSynchronize 0.979 ms x1, cudaLaunchKernel_v7000 0.132 ms x63, cuMemAllocAsync 0.023 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.724 ms, GPU busy 0.694 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.168 | 24.2% |
| attn_keys/kv_append | 0.120 | 17.3% |
| wo_b | 0.063 | 9.1% |
| wo_a | 0.053 | 7.6% |
| sparse_attn | 0.050 | 7.1% |
| q_b | 0.044 | 6.3% |
| shared_expert | 0.043 | 6.2% |
| hc_ffn_pre | 0.037 | 5.3% |
| router | 0.036 | 5.2% |
| hc_attn_pre | 0.029 | 4.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.7% |
| hc_post | 0.007 | 1.1% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 191 calls, 0.676 ms; top: cuStreamSynchronize 0.524 ms x1, cudaLaunchKernel_v7000 0.098 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

