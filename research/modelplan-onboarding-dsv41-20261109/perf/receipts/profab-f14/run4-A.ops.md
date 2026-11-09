### L2/prefill (2 instances): wall 38.363 ms, GPU busy 38.231 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.500 | 48.4% |
| sparse_attn | 4.625 | 12.1% |
| wo_a | 2.918 | 7.6% |
| q_b | 1.837 | 4.8% |
| shared_expert | 1.782 | 4.7% |
| wo_b | 1.733 | 4.5% |
| attn_keys/compressor | 1.533 | 4.0% |
| hc_ffn_pre | 1.175 | 3.1% |
| hc_attn_pre | 0.889 | 2.3% |
| router | 0.764 | 2.0% |
| attn_keys/index_q | 0.589 | 1.5% |
| moe_plan | 0.396 | 1.0% |
| attn_keys/qr | 0.380 | 1.0% |
| attn_keys/index_score | 0.312 | 0.8% |
| hc_post | 0.237 | 0.6% |
| attn_keys/kv_window | 0.231 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.063 | 0.2% |
| attn_keys/index_keys | 0.030 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 38.172 ms; top: cuStreamSynchronize 37.437 ms x1, cuMemAllocAsync 0.401 ms x53, cudaLaunchKernel_v7000 0.238 ms x64, cuMemFreeAsync 0.035 ms x51, cuMemcpyDtoDAsync_v2 0.029 ms x4, cuMemsetD8Async 0.020 ms x3

### L20/prefill (2 instances): wall 36.313 ms, GPU busy 36.206 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.239 | 47.6% |
| sparse_attn | 4.706 | 13.0% |
| wo_a | 2.933 | 8.1% |
| q_b | 1.848 | 5.1% |
| shared_expert | 1.788 | 4.9% |
| wo_b | 1.732 | 4.8% |
| hc_ffn_pre | 1.179 | 3.3% |
| hc_attn_pre | 0.890 | 2.5% |
| router | 0.766 | 2.1% |
| attn_keys/index_score | 0.647 | 1.8% |
| attn_keys/index_q | 0.587 | 1.6% |
| moe_plan | 0.408 | 1.1% |
| attn_keys/qr | 0.381 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.6% |
| attn_keys/compressor | 0.218 | 0.6% |
| combine | 0.205 | 0.6% |
| attn_keys/topk | 0.088 | 0.2% |
| attn_keys/candidates | 0.040 | 0.1% |
| attn_keys/kv_append | 0.036 | 0.1% |
| attn_keys/index_keys | 0.034 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 36.154 ms; top: cuStreamSynchronize 35.398 ms x1, cuMemAllocAsync 0.469 ms x52, cudaLaunchKernel_v7000 0.202 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.017 ms x3

### L21/prefill (2 instances): wall 34.967 ms, GPU busy 34.872 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.510 | 50.2% |
| sparse_attn | 4.742 | 13.6% |
| wo_a | 2.938 | 8.4% |
| q_b | 1.843 | 5.3% |
| shared_expert | 1.795 | 5.1% |
| wo_b | 1.727 | 5.0% |
| hc_ffn_pre | 1.179 | 3.4% |
| hc_attn_pre | 0.895 | 2.6% |
| router | 0.769 | 2.2% |
| moe_plan | 0.409 | 1.2% |
| attn_keys/qr | 0.380 | 1.1% |
| hc_post | 0.237 | 0.7% |
| attn_keys/kv_window | 0.231 | 0.7% |
| combine | 0.206 | 0.6% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 34.832 ms; top: cuStreamSynchronize 34.335 ms x1, cuMemAllocAsync 0.249 ms x46, cudaLaunchKernel_v7000 0.170 ms x47, cuMemFreeAsync 0.029 ms x44, cuMemcpyDtoDAsync_v2 0.024 ms x3, cuMemsetD8Async 0.017 ms x3

### L2/decode (6 instances): wall 0.690 ms, GPU busy 0.628 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 25.4% |
| wo_b | 0.062 | 9.9% |
| wo_a | 0.049 | 7.8% |
| attn_keys/compressor | 0.048 | 7.6% |
| shared_expert | 0.044 | 7.1% |
| sparse_attn | 0.040 | 6.4% |
| router | 0.037 | 5.9% |
| attn_keys/index_q | 0.034 | 5.4% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.3% |
| hc_attn_pre | 0.023 | 3.7% |
| attn_keys/kv_window | 0.013 | 2.1% |
| attn_keys/topk | 0.013 | 2.1% |
| attn_keys/qr | 0.012 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/kv_append | 0.006 | 0.9% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.605 ms; top: cuStreamSynchronize 0.378 ms x1, cudaLaunchKernel_v7000 0.151 ms x60, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.651 ms, GPU busy 0.615 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 25.7% |
| wo_b | 0.066 | 10.7% |
| wo_a | 0.049 | 7.9% |
| shared_expert | 0.044 | 7.2% |
| sparse_attn | 0.040 | 6.5% |
| router | 0.037 | 6.0% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.3% |
| hc_attn_pre | 0.023 | 3.8% |
| attn_keys/compressor | 0.022 | 3.6% |
| attn_keys/kv_window | 0.013 | 2.1% |
| moe_plan | 0.013 | 2.0% |
| attn_keys/qr | 0.012 | 2.0% |
| attn_keys/topk | 0.012 | 2.0% |
| attn_keys/index_keys | 0.010 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.004 | 0.6% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.588 ms; top: cuStreamSynchronize 0.387 ms x1, cudaLaunchKernel_v7000 0.135 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.559 ms, GPU busy 0.529 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 29.8% |
| wo_b | 0.066 | 12.4% |
| wo_a | 0.048 | 9.1% |
| shared_expert | 0.044 | 8.4% |
| sparse_attn | 0.041 | 7.8% |
| router | 0.037 | 7.0% |
| q_b | 0.032 | 6.0% |
| hc_ffn_pre | 0.026 | 5.0% |
| hc_attn_pre | 0.024 | 4.5% |
| attn_keys/kv_window | 0.013 | 2.5% |
| attn_keys/qr | 0.013 | 2.4% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.511 ms; top: cuStreamSynchronize 0.357 ms x1, cudaLaunchKernel_v7000 0.101 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

