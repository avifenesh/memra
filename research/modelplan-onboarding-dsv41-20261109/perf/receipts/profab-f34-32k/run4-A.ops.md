### L2/prefill (2 instances): wall 380.232 ms, GPU busy 380.039 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 175.507 | 46.2% |
| sparse_attn | 41.328 | 10.9% |
| attn_keys/index_score | 26.809 | 7.1% |
| wo_a | 21.684 | 5.7% |
| shared_expert | 16.516 | 4.3% |
| q_b | 15.816 | 4.2% |
| wo_b | 15.410 | 4.1% |
| hc_ffn_pre | 15.314 | 4.0% |
| attn_keys/compressor | 12.738 | 3.4% |
| hc_attn_pre | 10.092 | 2.7% |
| router | 6.598 | 1.7% |
| hc_post | 5.609 | 1.5% |
| attn_keys/index_q | 4.729 | 1.2% |
| attn_keys/qr | 3.386 | 0.9% |
| attn_keys/topk | 2.593 | 0.7% |
| attn_keys/kv_window | 2.036 | 0.5% |
| combine | 1.911 | 0.5% |
| moe_plan | 1.643 | 0.4% |
| attn_keys/kv_append | 0.127 | 0.0% |
| attn_idx_rows | 0.101 | 0.0% |
| attn_keys/index_keys | 0.091 | 0.0% |

host API calls: 627 calls, 379.933 ms; top: cuStreamSynchronize 377.473 ms x1, cuMemAllocAsync 1.835 ms x134, cudaLaunchKernel_v7000 0.463 ms x172, cuMemFreeAsync 0.066 ms x132, cuMemsetD8Async 0.040 ms x12, cuMemcpyDtoDAsync_v2 0.035 ms x4

### L20/prefill (2 instances): wall 395.378 ms, GPU busy 395.185 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 167.241 | 42.3% |
| attn_keys/index_score | 52.313 | 13.2% |
| sparse_attn | 42.443 | 10.7% |
| wo_a | 22.308 | 5.6% |
| shared_expert | 16.732 | 4.2% |
| hc_ffn_pre | 16.206 | 4.1% |
| q_b | 16.047 | 4.1% |
| wo_b | 15.730 | 4.0% |
| hc_attn_pre | 10.193 | 2.6% |
| router | 6.757 | 1.7% |
| hc_post | 5.797 | 1.5% |
| attn_keys/topk | 5.260 | 1.3% |
| attn_keys/index_q | 4.752 | 1.2% |
| attn_keys/qr | 3.407 | 0.9% |
| attn_keys/candidates | 2.249 | 0.6% |
| attn_keys/kv_window | 2.053 | 0.5% |
| combine | 1.953 | 0.5% |
| moe_plan | 1.730 | 0.4% |
| attn_keys/compressor | 1.495 | 0.4% |
| attn_keys/kv_append | 0.238 | 0.1% |
| attn_keys/index_keys | 0.174 | 0.0% |
| attn_idx_rows | 0.106 | 0.0% |

host API calls: 629 calls, 395.089 ms; top: cuStreamSynchronize 392.605 ms x1, cuMemAllocAsync 1.887 ms x133, cudaLaunchKernel_v7000 0.440 ms x174, cuMemFreeAsync 0.070 ms x131, cuMemsetD8Async 0.041 ms x12, cuMemcpyDtoDAsync_v2 0.027 ms x4

### L21/prefill (2 instances): wall 324.252 ms, GPU busy 324.079 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 166.334 | 51.3% |
| sparse_attn | 41.758 | 12.9% |
| wo_a | 21.790 | 6.7% |
| shared_expert | 16.560 | 5.1% |
| q_b | 16.034 | 4.9% |
| wo_b | 15.339 | 4.7% |
| hc_ffn_pre | 14.652 | 4.5% |
| hc_attn_pre | 10.192 | 3.1% |
| router | 6.539 | 2.0% |
| hc_post | 5.667 | 1.7% |
| attn_keys/qr | 3.414 | 1.1% |
| attn_keys/kv_window | 2.052 | 0.6% |
| combine | 1.920 | 0.6% |
| moe_plan | 1.693 | 0.5% |
| attn_idx_rows | 0.106 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 324.008 ms; top: cuStreamSynchronize 322.604 ms x1, cuMemAllocAsync 0.914 ms x118, cudaLaunchKernel_v7000 0.353 ms x134, cuMemFreeAsync 0.058 ms x116, cuMemsetD8Async 0.035 ms x12, cuMemcpyDtoDAsync_v2 0.029 ms x3

### L2/decode (6 instances): wall 0.757 ms, GPU busy 0.697 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 22.8% |
| wo_b | 0.063 | 9.0% |
| wo_a | 0.050 | 7.2% |
| attn_keys/compressor | 0.048 | 6.9% |
| sparse_attn | 0.047 | 6.7% |
| shared_expert | 0.043 | 6.2% |
| attn_keys/index_q | 0.042 | 6.1% |
| hc_ffn_pre | 0.036 | 5.2% |
| router | 0.035 | 5.0% |
| attn_keys/topk | 0.034 | 4.9% |
| q_b | 0.032 | 4.6% |
| hc_attn_pre | 0.028 | 4.0% |
| attn_keys/kv_append | 0.015 | 2.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.8% |
| attn_keys/index_score | 0.007 | 1.0% |
| hc_post | 0.007 | 1.0% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.674 ms; top: cuStreamSynchronize 0.445 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.030 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.790 ms, GPU busy 0.754 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 21.3% |
| wo_b | 0.064 | 8.5% |
| attn_keys/topk | 0.056 | 7.4% |
| wo_a | 0.050 | 6.6% |
| sparse_attn | 0.047 | 6.2% |
| shared_expert | 0.043 | 5.8% |
| attn_keys/index_q | 0.042 | 5.6% |
| hc_ffn_pre | 0.037 | 4.9% |
| router | 0.036 | 4.8% |
| q_b | 0.032 | 4.3% |
| attn_keys/compressor | 0.032 | 4.3% |
| attn_keys/candidates | 0.029 | 3.9% |
| hc_attn_pre | 0.028 | 3.7% |
| attn_keys/kv_append | 0.027 | 3.5% |
| attn_keys/kv_window | 0.013 | 1.7% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/qr | 0.012 | 1.6% |
| attn_keys/index_keys | 0.010 | 1.4% |
| attn_keys/index_score | 0.009 | 1.2% |
| hc_post | 0.007 | 0.9% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.727 ms; top: cuStreamSynchronize 0.526 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.604 ms, GPU busy 0.572 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 27.9% |
| wo_b | 0.064 | 11.2% |
| wo_a | 0.050 | 8.8% |
| sparse_attn | 0.048 | 8.4% |
| shared_expert | 0.043 | 7.6% |
| hc_ffn_pre | 0.036 | 6.3% |
| router | 0.036 | 6.3% |
| q_b | 0.032 | 5.5% |
| hc_attn_pre | 0.029 | 5.0% |
| attn_keys/kv_append | 0.023 | 4.0% |
| attn_keys/kv_window | 0.013 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| attn_keys/qr | 0.012 | 2.1% |
| hc_post | 0.007 | 1.3% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.556 ms; top: cuStreamSynchronize 0.400 ms x1, cudaLaunchKernel_v7000 0.099 ms x47, cuMemAllocAsync 0.022 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

