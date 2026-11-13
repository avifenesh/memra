### L2/prefill (2 instances): wall 2072.005 ms, GPU busy 2071.548 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 814.500 | 39.3% |
| attn_keys/index_score | 444.193 | 21.4% |
| sparse_attn | 185.221 | 8.9% |
| wo_a | 102.318 | 4.9% |
| q_b | 70.559 | 3.4% |
| hc_ffn_pre | 69.266 | 3.3% |
| wo_b | 67.898 | 3.3% |
| shared_expert | 67.272 | 3.2% |
| attn_keys/topk | 51.384 | 2.5% |
| attn_keys/compressor | 50.528 | 2.4% |
| hc_attn_pre | 40.717 | 2.0% |
| router | 27.380 | 1.3% |
| hc_post | 23.066 | 1.1% |
| attn_keys/index_q | 20.044 | 1.0% |
| attn_keys/qr | 13.299 | 0.6% |
| attn_keys/kv_window | 8.143 | 0.4% |
| combine | 7.817 | 0.4% |
| moe_plan | 6.684 | 0.3% |
| attn_keys/kv_append | 0.505 | 0.0% |
| attn_idx_rows | 0.428 | 0.0% |
| attn_keys/index_keys | 0.326 | 0.0% |

host API calls: 2175 calls, 2071.099 ms; top: cuStreamSynchronize 2060.924 ms x1, cuMemAllocAsync 8.070 ms x458, cudaLaunchKernel_v7000 1.638 ms x604, cuMemFreeAsync 0.221 ms x456, cuMemsetD8Async 0.122 ms x48, cuKernelGetName 0.065 ms x604

### L20/prefill (2 instances): wall 2626.344 ms, GPU busy 2625.907 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 880.101 | 33.5% |
| experts | 790.695 | 30.1% |
| sparse_attn | 188.397 | 7.2% |
| attn_keys/topk | 125.058 | 4.8% |
| wo_a | 105.127 | 4.0% |
| attn_keys/candidates | 99.931 | 3.8% |
| q_b | 71.086 | 2.7% |
| hc_ffn_pre | 70.895 | 2.7% |
| wo_b | 68.741 | 2.6% |
| shared_expert | 68.027 | 2.6% |
| hc_attn_pre | 41.168 | 1.6% |
| router | 27.946 | 1.1% |
| hc_post | 23.796 | 0.9% |
| attn_keys/index_q | 20.241 | 0.8% |
| attn_keys/qr | 13.450 | 0.5% |
| attn_keys/kv_window | 8.225 | 0.3% |
| combine | 7.975 | 0.3% |
| moe_plan | 6.877 | 0.3% |
| attn_keys/compressor | 5.917 | 0.2% |
| attn_keys/kv_append | 1.203 | 0.0% |
| attn_keys/index_keys | 0.643 | 0.0% |
| attn_idx_rows | 0.407 | 0.0% |

host API calls: 2201 calls, 2625.567 ms; top: cuStreamSynchronize 2614.356 ms x1, cuMemAllocAsync 9.393 ms x457, cudaLaunchKernel_v7000 1.407 ms x618, cuMemFreeAsync 0.203 ms x455, cuMemsetD8Async 0.103 ms x48, cuKernelGetName 0.059 ms x618

### L21/prefill (2 instances): wall 10171.779 ms, GPU busy 1480.200 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 806.514 | 54.5% |
| sparse_attn | 178.914 | 12.1% |
| wo_a | 95.396 | 6.4% |
| hc_ffn_pre | 69.526 | 4.7% |
| q_b | 68.203 | 4.6% |
| shared_expert | 67.587 | 4.6% |
| wo_b | 65.304 | 4.4% |
| hc_attn_pre | 40.980 | 2.8% |
| router | 27.516 | 1.9% |
| hc_post | 23.454 | 1.6% |
| attn_keys/qr | 13.382 | 0.9% |
| attn_keys/kv_window | 8.184 | 0.6% |
| combine | 7.836 | 0.5% |
| moe_plan | 6.808 | 0.5% |
| attn_idx_rows | 0.412 | 0.0% |
| attn_keys/kv_append | 0.183 | 0.0% |

host API calls: 1826 calls, 10171.139 ms; top: cuMemAllocAsync 8729.729 ms x406, cuStreamSynchronize 1439.295 ms x1, cudaLaunchKernel_v7000 1.750 ms x482, cuMemFreeAsync 0.179 ms x404, cuMemsetD8Async 0.109 ms x48, cuKernelGetName 0.047 ms x482

### L2/decode (6 instances): wall 0.932 ms, GPU busy 0.867 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.163 | 18.8% |
| attn_keys/topk | 0.146 | 16.9% |
| wo_b | 0.066 | 7.6% |
| attn_keys/kv_append | 0.050 | 5.8% |
| wo_a | 0.050 | 5.8% |
| attn_keys/compressor | 0.048 | 5.6% |
| sparse_attn | 0.047 | 5.4% |
| shared_expert | 0.043 | 5.0% |
| attn_keys/index_q | 0.043 | 4.9% |
| hc_ffn_pre | 0.037 | 4.3% |
| q_b | 0.037 | 4.3% |
| router | 0.035 | 4.1% |
| hc_attn_pre | 0.028 | 3.3% |
| attn_keys/index_score | 0.016 | 1.9% |
| attn_keys/kv_window | 0.013 | 1.5% |
| moe_plan | 0.012 | 1.4% |
| attn_keys/qr | 0.012 | 1.4% |
| hc_post | 0.007 | 0.8% |
| attn_keys/index_keys | 0.005 | 0.6% |
| combine | 0.004 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.850 ms; top: cuStreamSynchronize 0.619 ms x1, cudaLaunchKernel_v7000 0.152 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.236 ms, GPU busy 1.200 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.280 | 23.3% |
| experts | 0.166 | 13.8% |
| attn_keys/kv_append | 0.125 | 10.4% |
| attn_keys/candidates | 0.114 | 9.5% |
| wo_b | 0.063 | 5.2% |
| wo_a | 0.051 | 4.3% |
| sparse_attn | 0.048 | 4.0% |
| shared_expert | 0.043 | 3.6% |
| q_b | 0.043 | 3.6% |
| attn_keys/index_q | 0.043 | 3.5% |
| hc_ffn_pre | 0.037 | 3.1% |
| router | 0.036 | 3.0% |
| attn_keys/compressor | 0.032 | 2.7% |
| attn_keys/index_score | 0.030 | 2.5% |
| hc_attn_pre | 0.028 | 2.3% |
| attn_keys/kv_window | 0.013 | 1.1% |
| moe_plan | 0.012 | 1.0% |
| attn_keys/qr | 0.012 | 1.0% |
| attn_keys/index_keys | 0.010 | 0.8% |
| hc_post | 0.007 | 0.6% |
| combine | 0.004 | 0.4% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.174 ms; top: cuStreamSynchronize 0.976 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.023 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.721 ms, GPU busy 0.689 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.166 | 24.0% |
| attn_keys/kv_append | 0.121 | 17.5% |
| wo_b | 0.064 | 9.2% |
| wo_a | 0.053 | 7.6% |
| sparse_attn | 0.049 | 7.1% |
| shared_expert | 0.043 | 6.3% |
| q_b | 0.042 | 6.0% |
| hc_ffn_pre | 0.037 | 5.3% |
| router | 0.035 | 5.1% |
| hc_attn_pre | 0.028 | 4.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.7% |
| hc_post | 0.007 | 1.1% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.674 ms; top: cuStreamSynchronize 0.518 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuKernelGetName 0.004 ms x47

