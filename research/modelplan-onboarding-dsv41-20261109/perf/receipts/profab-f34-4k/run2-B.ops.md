### L2/prefill (2 instances): wall 35.605 ms, GPU busy 35.530 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.601 | 49.5% |
| sparse_attn | 4.211 | 11.9% |
| wo_a | 2.488 | 7.0% |
| q_b | 1.693 | 4.8% |
| shared_expert | 1.645 | 4.6% |
| wo_b | 1.600 | 4.5% |
| attn_keys/compressor | 1.515 | 4.3% |
| hc_ffn_pre | 1.178 | 3.3% |
| hc_attn_pre | 0.890 | 2.5% |
| router | 0.728 | 2.0% |
| attn_keys/index_q | 0.605 | 1.7% |
| attn_keys/qr | 0.370 | 1.0% |
| hc_post | 0.234 | 0.7% |
| attn_keys/kv_window | 0.226 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.068 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.455 ms; top: cuStreamSynchronize 34.822 ms x1, cuMemAllocAsync 0.349 ms x53, cudaLaunchKernel_v7000 0.204 ms x64, cuMemFreeAsync 0.030 ms x51, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.014 ms x3

### L20/prefill (2 instances): wall 33.699 ms, GPU busy 33.599 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.483 | 49.1% |
| sparse_attn | 4.313 | 12.8% |
| wo_a | 2.499 | 7.4% |
| q_b | 1.720 | 5.1% |
| shared_expert | 1.658 | 4.9% |
| wo_b | 1.608 | 4.8% |
| hc_ffn_pre | 1.186 | 3.5% |
| hc_attn_pre | 0.897 | 2.7% |
| router | 0.733 | 2.2% |
| attn_keys/index_q | 0.602 | 1.8% |
| attn_keys/index_score | 0.398 | 1.2% |
| attn_keys/qr | 0.373 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.231 | 0.7% |
| attn_keys/compressor | 0.214 | 0.6% |
| combine | 0.204 | 0.6% |
| attn_keys/topk | 0.101 | 0.3% |
| attn_keys/index_keys | 0.049 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.532 ms; top: cuStreamSynchronize 32.759 ms x1, cuMemAllocAsync 0.491 ms x52, cudaLaunchKernel_v7000 0.205 ms x63, cuMemFreeAsync 0.031 ms x50, cuMemcpyDtoDAsync_v2 0.020 ms x4, cuMemsetD8Async 0.015 ms x3

### L21/prefill (2 instances): wall 32.539 ms, GPU busy 32.449 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.734 | 51.6% |
| sparse_attn | 4.357 | 13.4% |
| wo_a | 2.505 | 7.7% |
| q_b | 1.693 | 5.2% |
| shared_expert | 1.664 | 5.1% |
| wo_b | 1.610 | 5.0% |
| hc_ffn_pre | 1.181 | 3.6% |
| hc_attn_pre | 0.898 | 2.8% |
| router | 0.734 | 2.3% |
| attn_keys/qr | 0.370 | 1.1% |
| hc_post | 0.237 | 0.7% |
| attn_keys/kv_window | 0.229 | 0.7% |
| combine | 0.207 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.401 ms; top: cuStreamSynchronize 31.921 ms x1, cuMemAllocAsync 0.251 ms x46, cudaLaunchKernel_v7000 0.160 ms x47, cuMemFreeAsync 0.028 ms x44, cuMemcpyDtoDAsync_v2 0.018 ms x3, cuMemsetD8Async 0.015 ms x3

### L2/decode (6 instances): wall 0.656 ms, GPU busy 0.586 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 26.7% |
| wo_a | 0.049 | 8.4% |
| attn_keys/compressor | 0.049 | 8.3% |
| shared_expert | 0.044 | 7.6% |
| wo_b | 0.044 | 7.6% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.8% |
| q_b | 0.032 | 5.4% |
| router | 0.029 | 4.9% |
| hc_ffn_pre | 0.020 | 3.5% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.568 ms; top: cuStreamSynchronize 0.321 ms x1, cudaLaunchKernel_v7000 0.164 ms x60, cuMemAllocAsync 0.032 ms x52, cuMemFreeAsync 0.022 ms x51, cuMemcpyDtoDAsync_v2 0.019 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.612 ms, GPU busy 0.573 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 27.5% |
| wo_a | 0.049 | 8.5% |
| shared_expert | 0.045 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.6% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.023 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/candidates | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.545 ms; top: cuStreamSynchronize 0.331 ms x1, cudaLaunchKernel_v7000 0.143 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.021 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.520 ms, GPU busy 0.484 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 32.3% |
| wo_a | 0.049 | 10.0% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.045 | 9.2% |
| sparse_attn | 0.037 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.029 | 5.9% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 2.9% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.4% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.469 ms; top: cuStreamSynchronize 0.302 ms x1, cudaLaunchKernel_v7000 0.107 ms x47, cuMemAllocAsync 0.024 ms x46, cuMemFreeAsync 0.018 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

