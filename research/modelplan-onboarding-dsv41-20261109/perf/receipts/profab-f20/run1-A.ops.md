### L2/prefill (2 instances): wall 37.496 ms, GPU busy 37.405 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.356 | 49.1% |
| sparse_attn | 4.581 | 12.2% |
| wo_a | 2.672 | 7.1% |
| q_b | 1.816 | 4.9% |
| shared_expert | 1.769 | 4.7% |
| wo_b | 1.728 | 4.6% |
| attn_keys/compressor | 1.524 | 4.1% |
| hc_ffn_pre | 1.179 | 3.2% |
| hc_attn_pre | 0.893 | 2.4% |
| router | 0.728 | 1.9% |
| attn_keys/index_q | 0.630 | 1.7% |
| attn_keys/qr | 0.393 | 1.1% |
| attn_keys/index_score | 0.304 | 0.8% |
| attn_keys/kv_window | 0.240 | 0.6% |
| hc_post | 0.238 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.060 | 0.2% |
| attn_keys/index_keys | 0.041 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 37.336 ms; top: cuStreamSynchronize 36.665 ms x1, cuMemAllocAsync 0.374 ms x53, cudaLaunchKernel_v7000 0.214 ms x64, cuMemFreeAsync 0.029 ms x51, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.019 ms x3

### L20/prefill (2 instances): wall 35.772 ms, GPU busy 35.663 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.230 | 48.3% |
| sparse_attn | 4.705 | 13.2% |
| wo_a | 2.697 | 7.6% |
| q_b | 1.845 | 5.2% |
| shared_expert | 1.791 | 5.0% |
| wo_b | 1.746 | 4.9% |
| hc_ffn_pre | 1.192 | 3.3% |
| hc_attn_pre | 0.896 | 2.5% |
| router | 0.739 | 2.1% |
| attn_keys/index_score | 0.639 | 1.8% |
| attn_keys/index_q | 0.627 | 1.8% |
| attn_keys/qr | 0.397 | 1.1% |
| attn_keys/compressor | 0.246 | 0.7% |
| attn_keys/kv_window | 0.243 | 0.7% |
| hc_post | 0.239 | 0.7% |
| combine | 0.205 | 0.6% |
| attn_keys/topk | 0.084 | 0.2% |
| attn_keys/index_keys | 0.052 | 0.1% |
| attn_keys/kv_append | 0.036 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 35.616 ms; top: cuStreamSynchronize 34.848 ms x1, cuMemAllocAsync 0.476 ms x52, cudaLaunchKernel_v7000 0.208 ms x63, cuMemFreeAsync 0.031 ms x50, cuMemcpyDtoDAsync_v2 0.024 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 34.429 ms, GPU busy 34.305 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.531 | 51.1% |
| sparse_attn | 4.752 | 13.9% |
| wo_a | 2.700 | 7.9% |
| q_b | 1.839 | 5.4% |
| shared_expert | 1.797 | 5.2% |
| wo_b | 1.745 | 5.1% |
| hc_ffn_pre | 1.192 | 3.5% |
| hc_attn_pre | 0.901 | 2.6% |
| router | 0.738 | 2.2% |
| attn_keys/qr | 0.394 | 1.1% |
| attn_keys/kv_window | 0.243 | 0.7% |
| hc_post | 0.237 | 0.7% |
| combine | 0.206 | 0.6% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 34.286 ms; top: cuStreamSynchronize 33.773 ms x1, cuMemAllocAsync 0.254 ms x46, cudaLaunchKernel_v7000 0.179 ms x47, cuMemFreeAsync 0.029 ms x44, cuMemcpyDtoDAsync_v2 0.023 ms x3, cuMemsetD8Async 0.019 ms x3

### L2/decode (6 instances): wall 0.673 ms, GPU busy 0.612 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.3% |
| wo_b | 0.062 | 10.2% |
| wo_a | 0.049 | 8.0% |
| attn_keys/compressor | 0.049 | 8.0% |
| shared_expert | 0.045 | 7.3% |
| sparse_attn | 0.040 | 6.5% |
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
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.593 ms; top: cuStreamSynchronize 0.370 ms x1, cudaLaunchKernel_v7000 0.148 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.636 ms, GPU busy 0.600 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 26.6% |
| wo_b | 0.065 | 10.9% |
| wo_a | 0.049 | 8.2% |
| shared_expert | 0.044 | 7.4% |
| sparse_attn | 0.040 | 6.6% |
| attn_keys/index_q | 0.034 | 5.6% |
| q_b | 0.032 | 5.3% |
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
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.575 ms; top: cuStreamSynchronize 0.378 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.543 ms, GPU busy 0.511 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 31.2% |
| wo_b | 0.065 | 12.8% |
| wo_a | 0.048 | 9.4% |
| shared_expert | 0.044 | 8.7% |
| sparse_attn | 0.041 | 8.1% |
| q_b | 0.032 | 6.2% |
| router | 0.029 | 5.6% |
| hc_ffn_pre | 0.020 | 3.9% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.015 | 2.8% |
| attn_keys/qr | 0.014 | 2.7% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.496 ms; top: cuStreamSynchronize 0.343 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

