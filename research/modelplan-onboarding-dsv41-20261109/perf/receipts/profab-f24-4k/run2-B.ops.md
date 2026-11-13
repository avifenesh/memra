### L2/prefill (2 instances): wall 36.138 ms, GPU busy 36.046 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.885 | 49.6% |
| sparse_attn | 4.261 | 11.8% |
| wo_a | 2.529 | 7.0% |
| q_b | 1.716 | 4.8% |
| shared_expert | 1.666 | 4.6% |
| wo_b | 1.622 | 4.5% |
| attn_keys/compressor | 1.540 | 4.3% |
| hc_ffn_pre | 1.193 | 3.3% |
| hc_attn_pre | 0.897 | 2.5% |
| router | 0.736 | 2.0% |
| attn_keys/index_q | 0.611 | 1.7% |
| attn_keys/qr | 0.374 | 1.0% |
| hc_post | 0.239 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.6% |
| combine | 0.208 | 0.6% |
| attn_keys/index_score | 0.187 | 0.5% |
| attn_keys/topk | 0.063 | 0.2% |
| attn_keys/index_keys | 0.040 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.972 ms; top: cuStreamSynchronize 35.296 ms x1, cuMemAllocAsync 0.377 ms x53, cudaLaunchKernel_v7000 0.209 ms x64, cuMemFreeAsync 0.032 ms x51, cuMemcpyDtoDAsync_v2 0.030 ms x4, cuMemsetD8Async 0.018 ms x3

### L20/prefill (2 instances): wall 34.072 ms, GPU busy 33.948 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.685 | 49.1% |
| sparse_attn | 4.349 | 12.8% |
| wo_a | 2.533 | 7.5% |
| q_b | 1.728 | 5.1% |
| shared_expert | 1.675 | 4.9% |
| wo_b | 1.629 | 4.8% |
| hc_ffn_pre | 1.200 | 3.5% |
| hc_attn_pre | 0.904 | 2.7% |
| router | 0.739 | 2.2% |
| attn_keys/index_q | 0.607 | 1.8% |
| attn_keys/index_score | 0.399 | 1.2% |
| attn_keys/qr | 0.373 | 1.1% |
| hc_post | 0.239 | 0.7% |
| attn_keys/kv_window | 0.234 | 0.7% |
| attn_keys/compressor | 0.212 | 0.6% |
| combine | 0.207 | 0.6% |
| attn_keys/topk | 0.088 | 0.3% |
| attn_keys/index_keys | 0.051 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.031 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.889 ms; top: cuStreamSynchronize 33.032 ms x1, cuMemAllocAsync 0.537 ms x52, cudaLaunchKernel_v7000 0.229 ms x63, cuMemFreeAsync 0.037 ms x50, cuMemcpyDtoDAsync_v2 0.024 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 32.928 ms, GPU busy 32.827 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.952 | 51.6% |
| sparse_attn | 4.397 | 13.4% |
| wo_a | 2.531 | 7.7% |
| q_b | 1.710 | 5.2% |
| shared_expert | 1.682 | 5.1% |
| wo_b | 1.627 | 5.0% |
| hc_ffn_pre | 1.199 | 3.7% |
| hc_attn_pre | 0.906 | 2.8% |
| router | 0.741 | 2.3% |
| attn_keys/qr | 0.372 | 1.1% |
| hc_post | 0.240 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.7% |
| combine | 0.208 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.787 ms; top: cuStreamSynchronize 32.304 ms x1, cuMemAllocAsync 0.241 ms x46, cudaLaunchKernel_v7000 0.165 ms x47, cuMemFreeAsync 0.030 ms x44, cuMemcpyDtoDAsync_v2 0.022 ms x3, cuMemsetD8Async 0.017 ms x3

### L2/decode (6 instances): wall 0.655 ms, GPU busy 0.588 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 27.0% |
| wo_a | 0.049 | 8.3% |
| attn_keys/compressor | 0.048 | 8.2% |
| shared_expert | 0.045 | 7.6% |
| wo_b | 0.044 | 7.5% |
| sparse_attn | 0.036 | 6.0% |
| attn_keys/index_q | 0.034 | 5.8% |
| q_b | 0.032 | 5.4% |
| router | 0.029 | 4.9% |
| hc_ffn_pre | 0.021 | 3.5% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.569 ms; top: cuStreamSynchronize 0.334 ms x1, cudaLaunchKernel_v7000 0.155 ms x60, cuMemAllocAsync 0.030 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.610 ms, GPU busy 0.573 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 27.5% |
| wo_a | 0.049 | 8.6% |
| shared_expert | 0.045 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.031 | 5.5% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.023 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.5% |
| attn_keys/qr | 0.014 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.9% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/index_score | 0.003 | 0.6% |
| attn_keys/candidates | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.546 ms; top: cuStreamSynchronize 0.341 ms x1, cudaLaunchKernel_v7000 0.138 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.523 ms, GPU busy 0.487 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 32.6% |
| wo_a | 0.049 | 10.0% |
| shared_expert | 0.045 | 9.2% |
| wo_b | 0.044 | 9.1% |
| sparse_attn | 0.037 | 7.6% |
| q_b | 0.032 | 6.5% |
| router | 0.029 | 5.9% |
| hc_ffn_pre | 0.021 | 4.2% |
| hc_attn_pre | 0.017 | 3.6% |
| attn_keys/kv_window | 0.014 | 3.0% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.4% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.473 ms; top: cuStreamSynchronize 0.313 ms x1, cudaLaunchKernel_v7000 0.103 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

