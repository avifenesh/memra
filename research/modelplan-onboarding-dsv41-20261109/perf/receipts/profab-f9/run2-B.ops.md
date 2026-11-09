### L2/prefill (2 instances): wall 38.269 ms, GPU busy 38.170 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.458 | 48.4% |
| sparse_attn | 4.611 | 12.1% |
| wo_a | 2.951 | 7.7% |
| q_b | 1.828 | 4.8% |
| shared_expert | 1.776 | 4.7% |
| wo_b | 1.718 | 4.5% |
| attn_keys/compressor | 1.527 | 4.0% |
| hc_ffn_pre | 1.174 | 3.1% |
| hc_attn_pre | 0.888 | 2.3% |
| router | 0.765 | 2.0% |
| attn_keys/index_q | 0.589 | 1.5% |
| moe_plan | 0.396 | 1.0% |
| attn_keys/qr | 0.381 | 1.0% |
| attn_keys/index_score | 0.311 | 0.8% |
| hc_post | 0.237 | 0.6% |
| attn_keys/kv_window | 0.232 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.062 | 0.2% |
| attn_keys/index_keys | 0.030 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 38.105 ms; top: cuStreamSynchronize 37.425 ms x1, cuMemAllocAsync 0.373 ms x53, cudaLaunchKernel_v7000 0.216 ms x64, cuMemFreeAsync 0.032 ms x51, cuMemcpyDtoDAsync_v2 0.030 ms x4, cuMemsetD8Async 0.020 ms x3

### L20/prefill (2 instances): wall 36.456 ms, GPU busy 36.335 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.296 | 47.6% |
| sparse_attn | 4.723 | 13.0% |
| wo_a | 2.940 | 8.1% |
| q_b | 1.852 | 5.1% |
| shared_expert | 1.795 | 4.9% |
| wo_b | 1.739 | 4.8% |
| hc_ffn_pre | 1.184 | 3.3% |
| hc_attn_pre | 0.898 | 2.5% |
| router | 0.772 | 2.1% |
| attn_keys/index_score | 0.651 | 1.8% |
| attn_keys/index_q | 0.591 | 1.6% |
| moe_plan | 0.407 | 1.1% |
| attn_keys/qr | 0.382 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.234 | 0.6% |
| attn_keys/compressor | 0.219 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.088 | 0.2% |
| attn_keys/candidates | 0.040 | 0.1% |
| attn_keys/kv_append | 0.037 | 0.1% |
| attn_keys/index_keys | 0.034 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 236 calls, 36.298 ms; top: cuStreamSynchronize 35.523 ms x1, cuMemAllocAsync 0.482 ms x52, cudaLaunchKernel_v7000 0.210 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.022 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 35.129 ms, GPU busy 35.024 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.590 | 50.2% |
| sparse_attn | 4.765 | 13.6% |
| wo_a | 2.942 | 8.4% |
| q_b | 1.849 | 5.3% |
| shared_expert | 1.804 | 5.2% |
| wo_b | 1.738 | 5.0% |
| hc_ffn_pre | 1.183 | 3.4% |
| hc_attn_pre | 0.899 | 2.6% |
| router | 0.770 | 2.2% |
| moe_plan | 0.408 | 1.2% |
| attn_keys/qr | 0.382 | 1.1% |
| hc_post | 0.240 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| combine | 0.206 | 0.6% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.005 | 0.0% |

host API calls: 191 calls, 34.992 ms; top: cuStreamSynchronize 34.494 ms x1, cuMemAllocAsync 0.250 ms x46, cudaLaunchKernel_v7000 0.170 ms x47, cuMemFreeAsync 0.029 ms x44, cuMemcpyDtoDAsync_v2 0.022 ms x3, cuMemsetD8Async 0.018 ms x3

### L2/decode (6 instances): wall 0.691 ms, GPU busy 0.629 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 25.4% |
| wo_b | 0.062 | 9.9% |
| wo_a | 0.049 | 7.8% |
| attn_keys/compressor | 0.048 | 7.6% |
| shared_expert | 0.045 | 7.1% |
| sparse_attn | 0.040 | 6.4% |
| router | 0.037 | 5.9% |
| attn_keys/index_q | 0.034 | 5.3% |
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

host API calls: 233 calls, 0.606 ms; top: cuStreamSynchronize 0.380 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.652 ms, GPU busy 0.615 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 25.7% |
| wo_b | 0.065 | 10.6% |
| wo_a | 0.049 | 7.9% |
| shared_expert | 0.044 | 7.2% |
| sparse_attn | 0.040 | 6.5% |
| router | 0.037 | 6.0% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.4% |
| hc_attn_pre | 0.023 | 3.8% |
| attn_keys/compressor | 0.022 | 3.6% |
| attn_keys/kv_window | 0.013 | 2.1% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/topk | 0.012 | 2.0% |
| attn_keys/index_keys | 0.010 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.4% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.004 | 0.6% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.588 ms; top: cuStreamSynchronize 0.388 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.562 ms, GPU busy 0.530 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 29.9% |
| wo_b | 0.066 | 12.4% |
| wo_a | 0.048 | 9.1% |
| shared_expert | 0.044 | 8.4% |
| sparse_attn | 0.041 | 7.8% |
| router | 0.037 | 7.0% |
| q_b | 0.032 | 6.0% |
| hc_ffn_pre | 0.027 | 5.0% |
| hc_attn_pre | 0.024 | 4.5% |
| attn_keys/kv_window | 0.013 | 2.5% |
| attn_keys/qr | 0.013 | 2.4% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.514 ms; top: cuStreamSynchronize 0.360 ms x1, cudaLaunchKernel_v7000 0.099 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

