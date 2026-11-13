### L2/prefill (2 instances): wall 291.312 ms, GPU busy 291.141 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 128.684 | 44.2% |
| sparse_attn | 34.967 | 12.0% |
| wo_a | 20.236 | 7.0% |
| attn_keys/index_score | 15.010 | 5.2% |
| shared_expert | 13.693 | 4.7% |
| q_b | 13.484 | 4.6% |
| wo_b | 12.939 | 4.4% |
| attn_keys/compressor | 11.671 | 4.0% |
| hc_ffn_pre | 10.341 | 3.6% |
| hc_attn_pre | 8.012 | 2.8% |
| router | 5.746 | 2.0% |
| attn_keys/index_q | 4.459 | 1.5% |
| attn_keys/qr | 3.025 | 1.0% |
| attn_keys/topk | 2.425 | 0.8% |
| hc_post | 2.216 | 0.8% |
| combine | 1.909 | 0.7% |
| attn_keys/kv_window | 1.875 | 0.6% |
| attn_keys/index_keys | 0.138 | 0.0% |
| attn_keys/kv_append | 0.130 | 0.0% |
| attn_idx_rows | 0.103 | 0.0% |
| moe_plan | 0.080 | 0.0% |

host API calls: 627 calls, 290.996 ms; top: cuStreamSynchronize 288.587 ms x1, cuMemAllocAsync 1.826 ms x134, cudaLaunchKernel_v7000 0.436 ms x172, cuMemFreeAsync 0.065 ms x132, cuMemsetD8Async 0.037 ms x12, cuMemcpyDtoDAsync_v2 0.025 ms x4

### L20/prefill (2 instances): wall 296.950 ms, GPU busy 296.780 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.562 | 41.3% |
| sparse_attn | 35.197 | 11.9% |
| attn_keys/index_score | 31.460 | 10.6% |
| wo_a | 20.349 | 6.9% |
| shared_expert | 13.814 | 4.7% |
| q_b | 13.597 | 4.6% |
| wo_b | 13.012 | 4.4% |
| hc_ffn_pre | 10.416 | 3.5% |
| hc_attn_pre | 8.078 | 2.7% |
| router | 5.826 | 2.0% |
| attn_keys/index_q | 4.458 | 1.5% |
| attn_keys/topk | 4.329 | 1.5% |
| attn_keys/qr | 3.027 | 1.0% |
| attn_keys/candidates | 2.446 | 0.8% |
| hc_post | 2.226 | 0.8% |
| combine | 1.913 | 0.6% |
| attn_keys/kv_window | 1.887 | 0.6% |
| attn_keys/compressor | 1.487 | 0.5% |
| attn_keys/index_keys | 0.265 | 0.1% |
| attn_keys/kv_append | 0.243 | 0.1% |
| attn_idx_rows | 0.105 | 0.0% |
| moe_plan | 0.082 | 0.0% |

host API calls: 629 calls, 296.640 ms; top: cuStreamSynchronize 294.144 ms x1, cuMemAllocAsync 1.921 ms x133, cudaLaunchKernel_v7000 0.428 ms x174, cuMemFreeAsync 0.066 ms x131, cuMemsetD8Async 0.037 ms x12, cuMemcpyDtoDAsync_v2 0.024 ms x4

### L21/prefill (2 instances): wall 249.318 ms, GPU busy 249.177 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 120.521 | 48.4% |
| sparse_attn | 35.087 | 14.1% |
| wo_a | 20.307 | 8.1% |
| q_b | 13.700 | 5.5% |
| shared_expert | 13.496 | 5.4% |
| wo_b | 12.976 | 5.2% |
| hc_ffn_pre | 10.253 | 4.1% |
| hc_attn_pre | 8.010 | 3.2% |
| router | 5.611 | 2.3% |
| attn_keys/qr | 3.028 | 1.2% |
| hc_post | 2.211 | 0.9% |
| attn_keys/kv_window | 1.887 | 0.8% |
| combine | 1.880 | 0.8% |
| attn_idx_rows | 0.104 | 0.0% |
| moe_plan | 0.079 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 249.069 ms; top: cuStreamSynchronize 247.702 ms x1, cuMemAllocAsync 0.901 ms x118, cudaLaunchKernel_v7000 0.339 ms x134, cuMemFreeAsync 0.054 ms x116, cuMemsetD8Async 0.034 ms x12, cuMemcpyDtoDAsync_v2 0.024 ms x3

### L2/decode (6 instances): wall 0.683 ms, GPU busy 0.619 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 25.9% |
| wo_a | 0.049 | 8.0% |
| attn_keys/compressor | 0.049 | 7.9% |
| wo_b | 0.045 | 7.2% |
| shared_expert | 0.044 | 7.1% |
| sparse_attn | 0.036 | 5.8% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| attn_keys/topk | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.599 ms; top: cuStreamSynchronize 0.374 ms x1, cudaLaunchKernel_v7000 0.147 ms x60, cuMemAllocAsync 0.030 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.698 ms, GPU busy 0.660 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 24.4% |
| wo_a | 0.050 | 7.6% |
| wo_b | 0.045 | 6.8% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/topk | 0.040 | 6.1% |
| sparse_attn | 0.036 | 5.5% |
| attn_keys/index_q | 0.034 | 5.1% |
| attn_keys/candidates | 0.032 | 4.8% |
| q_b | 0.032 | 4.8% |
| router | 0.030 | 4.5% |
| attn_keys/kv_append | 0.027 | 4.1% |
| attn_keys/compressor | 0.023 | 3.5% |
| hc_ffn_pre | 0.021 | 3.2% |
| hc_attn_pre | 0.017 | 2.5% |
| attn_keys/kv_window | 0.014 | 2.1% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/index_keys | 0.011 | 1.6% |
| attn_keys/index_score | 0.010 | 1.6% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.634 ms; top: cuStreamSynchronize 0.435 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.543 ms, GPU busy 0.511 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.5% |
| wo_a | 0.050 | 9.8% |
| wo_b | 0.045 | 8.7% |
| shared_expert | 0.044 | 8.7% |
| sparse_attn | 0.037 | 7.3% |
| q_b | 0.032 | 6.3% |
| router | 0.030 | 6.0% |
| attn_keys/kv_append | 0.023 | 4.6% |
| hc_ffn_pre | 0.021 | 4.2% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.4% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.495 ms; top: cuStreamSynchronize 0.338 ms x1, cudaLaunchKernel_v7000 0.099 ms x47, cuMemAllocAsync 0.022 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

