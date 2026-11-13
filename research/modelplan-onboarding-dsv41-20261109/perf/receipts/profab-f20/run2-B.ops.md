### L2/prefill (2 instances): wall 36.988 ms, GPU busy 36.880 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.812 | 48.3% |
| sparse_attn | 5.059 | 13.7% |
| wo_a | 2.693 | 7.3% |
| q_b | 1.709 | 4.6% |
| shared_expert | 1.658 | 4.5% |
| wo_b | 1.608 | 4.4% |
| attn_keys/compressor | 1.534 | 4.2% |
| hc_ffn_pre | 1.170 | 3.2% |
| hc_attn_pre | 0.883 | 2.4% |
| router | 0.735 | 2.0% |
| attn_keys/index_q | 0.610 | 1.7% |
| attn_keys/qr | 0.372 | 1.0% |
| hc_post | 0.240 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.6% |
| attn_keys/index_score | 0.208 | 0.6% |
| combine | 0.208 | 0.6% |
| attn_keys/topk | 0.062 | 0.2% |
| attn_keys/index_keys | 0.041 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 36.807 ms; top: cuStreamSynchronize 36.108 ms x1, cuMemAllocAsync 0.380 ms x53, cudaLaunchKernel_v7000 0.223 ms x64, cuMemFreeAsync 0.033 ms x51, cuMemcpyDtoDAsync_v2 0.031 ms x4, cuMemsetD8Async 0.021 ms x3

### L20/prefill (2 instances): wall 35.133 ms, GPU busy 35.016 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.706 | 47.7% |
| sparse_attn | 5.176 | 14.8% |
| wo_a | 2.713 | 7.7% |
| q_b | 1.730 | 4.9% |
| shared_expert | 1.677 | 4.8% |
| wo_b | 1.616 | 4.6% |
| hc_ffn_pre | 1.182 | 3.4% |
| hc_attn_pre | 0.888 | 2.5% |
| router | 0.743 | 2.1% |
| attn_keys/index_q | 0.610 | 1.7% |
| attn_keys/index_score | 0.439 | 1.3% |
| attn_keys/qr | 0.375 | 1.1% |
| attn_keys/compressor | 0.247 | 0.7% |
| hc_post | 0.237 | 0.7% |
| attn_keys/kv_window | 0.234 | 0.7% |
| combine | 0.207 | 0.6% |
| attn_keys/topk | 0.088 | 0.3% |
| attn_keys/index_keys | 0.053 | 0.2% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.030 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 34.967 ms; top: cuStreamSynchronize 34.196 ms x1, cuMemAllocAsync 0.477 ms x52, cudaLaunchKernel_v7000 0.210 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.017 ms x3

### L21/prefill (2 instances): wall 33.943 ms, GPU busy 33.836 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.974 | 50.2% |
| sparse_attn | 5.236 | 15.5% |
| wo_a | 2.713 | 8.0% |
| q_b | 1.713 | 5.1% |
| shared_expert | 1.687 | 5.0% |
| wo_b | 1.617 | 4.8% |
| hc_ffn_pre | 1.183 | 3.5% |
| hc_attn_pre | 0.884 | 2.6% |
| router | 0.744 | 2.2% |
| attn_keys/qr | 0.374 | 1.1% |
| hc_post | 0.240 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.7% |
| combine | 0.208 | 0.6% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 33.800 ms; top: cuStreamSynchronize 33.323 ms x1, cuMemAllocAsync 0.239 ms x46, cudaLaunchKernel_v7000 0.163 ms x47, cuMemFreeAsync 0.029 ms x44, cuMemcpyDtoDAsync_v2 0.021 ms x3, cuMemsetD8Async 0.017 ms x3

### L2/decode (6 instances): wall 0.679 ms, GPU busy 0.617 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.2% |
| wo_b | 0.062 | 10.1% |
| wo_a | 0.049 | 8.0% |
| attn_keys/compressor | 0.049 | 7.9% |
| shared_expert | 0.045 | 7.2% |
| sparse_attn | 0.043 | 7.0% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| hc_ffn_pre | 0.020 | 3.3% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.2% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.006 | 0.9% |
| combine | 0.005 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.594 ms; top: cuStreamSynchronize 0.370 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.641 ms, GPU busy 0.604 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.6% |
| wo_b | 0.065 | 10.8% |
| wo_a | 0.049 | 8.1% |
| shared_expert | 0.044 | 7.4% |
| sparse_attn | 0.043 | 7.1% |
| attn_keys/index_q | 0.034 | 5.6% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.8% |
| attn_keys/compressor | 0.023 | 3.9% |
| hc_ffn_pre | 0.020 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.2% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.005 | 0.7% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_keys/candidates | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.579 ms; top: cuStreamSynchronize 0.380 ms x1, cudaLaunchKernel_v7000 0.132 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.549 ms, GPU busy 0.517 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.1% |
| wo_b | 0.066 | 12.7% |
| wo_a | 0.048 | 9.4% |
| sparse_attn | 0.045 | 8.6% |
| shared_expert | 0.044 | 8.6% |
| q_b | 0.032 | 6.2% |
| router | 0.029 | 5.6% |
| hc_ffn_pre | 0.020 | 3.8% |
| hc_attn_pre | 0.017 | 3.3% |
| attn_keys/kv_window | 0.015 | 2.8% |
| attn_keys/qr | 0.014 | 2.7% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.502 ms; top: cuStreamSynchronize 0.350 ms x1, cudaLaunchKernel_v7000 0.099 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

