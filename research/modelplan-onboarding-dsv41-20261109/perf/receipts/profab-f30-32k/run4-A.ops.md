### L2/prefill (2 instances): wall 291.798 ms, GPU busy 291.608 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 127.851 | 43.8% |
| sparse_attn | 34.855 | 12.0% |
| wo_a | 20.056 | 6.9% |
| attn_keys/index_score | 14.890 | 5.1% |
| shared_expert | 13.622 | 4.7% |
| q_b | 13.388 | 4.6% |
| wo_b | 12.898 | 4.4% |
| attn_keys/compressor | 11.598 | 4.0% |
| hc_ffn_pre | 10.209 | 3.5% |
| hc_attn_pre | 7.938 | 2.7% |
| router | 5.682 | 1.9% |
| attn_keys/topk | 4.861 | 1.7% |
| attn_keys/index_q | 4.358 | 1.5% |
| attn_keys/qr | 2.999 | 1.0% |
| hc_post | 2.214 | 0.8% |
| combine | 1.889 | 0.6% |
| attn_keys/kv_window | 1.862 | 0.6% |
| attn_keys/index_keys | 0.137 | 0.0% |
| attn_keys/kv_append | 0.120 | 0.0% |
| attn_idx_rows | 0.100 | 0.0% |
| moe_plan | 0.078 | 0.0% |

host API calls: 627 calls, 291.454 ms; top: cuStreamSynchronize 288.902 ms x1, cuMemAllocAsync 1.893 ms x134, cudaLaunchKernel_v7000 0.487 ms x172, cuMemFreeAsync 0.071 ms x132, cuMemsetD8Async 0.041 ms x12, cuMemcpyDtoDAsync_v2 0.038 ms x4

### L20/prefill (2 instances): wall 303.958 ms, GPU busy 303.708 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.142 | 40.2% |
| sparse_attn | 35.138 | 11.6% |
| attn_keys/index_score | 31.386 | 10.3% |
| wo_a | 20.236 | 6.7% |
| shared_expert | 13.781 | 4.5% |
| q_b | 13.552 | 4.5% |
| wo_b | 12.965 | 4.3% |
| hc_ffn_pre | 10.588 | 3.5% |
| attn_keys/topk | 10.049 | 3.3% |
| hc_attn_pre | 8.065 | 2.7% |
| router | 5.800 | 1.9% |
| attn_keys/candidates | 4.491 | 1.5% |
| attn_keys/index_q | 4.345 | 1.4% |
| attn_keys/qr | 3.016 | 1.0% |
| hc_post | 2.215 | 0.7% |
| combine | 1.911 | 0.6% |
| attn_keys/kv_window | 1.876 | 0.6% |
| attn_keys/compressor | 1.478 | 0.5% |
| attn_keys/index_keys | 0.264 | 0.1% |
| attn_keys/kv_append | 0.221 | 0.1% |
| attn_idx_rows | 0.106 | 0.0% |
| moe_plan | 0.081 | 0.0% |

host API calls: 629 calls, 303.542 ms; top: cuStreamSynchronize 300.363 ms x1, cuMemAllocAsync 2.359 ms x133, cudaLaunchKernel_v7000 0.592 ms x174, cuMemFreeAsync 0.104 ms x131, cuMemsetD8Async 0.059 ms x12, cuMemcpyDtoDAsync_v2 0.042 ms x4

### L21/prefill (2 instances): wall 248.808 ms, GPU busy 248.553 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 120.327 | 48.4% |
| sparse_attn | 34.985 | 14.1% |
| wo_a | 20.193 | 8.1% |
| q_b | 13.657 | 5.5% |
| shared_expert | 13.455 | 5.4% |
| wo_b | 12.947 | 5.2% |
| hc_ffn_pre | 10.196 | 4.1% |
| hc_attn_pre | 8.012 | 3.2% |
| router | 5.585 | 2.2% |
| attn_keys/qr | 3.018 | 1.2% |
| hc_post | 2.214 | 0.9% |
| attn_keys/kv_window | 1.881 | 0.8% |
| combine | 1.871 | 0.8% |
| attn_idx_rows | 0.105 | 0.0% |
| moe_plan | 0.078 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 248.406 ms; top: cuStreamSynchronize 246.631 ms x1, cuMemAllocAsync 1.038 ms x118, cudaLaunchKernel_v7000 0.524 ms x134, cuMemFreeAsync 0.087 ms x116, cuMemsetD8Async 0.058 ms x12, cuMemcpyDtoDAsync_v2 0.046 ms x3

### L2/decode (6 instances): wall 0.870 ms, GPU busy 0.615 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 25.8% |
| attn_keys/compressor | 0.049 | 7.9% |
| wo_a | 0.049 | 7.9% |
| wo_b | 0.044 | 7.2% |
| shared_expert | 0.044 | 7.1% |
| sparse_attn | 0.036 | 5.8% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| attn_keys/topk | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.4% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.578 ms; top: cuStreamSynchronize 0.314 ms x1, cudaLaunchKernel_v7000 0.174 ms x60, cuMemAllocAsync 0.033 ms x52, cuMemFreeAsync 0.023 ms x51, cuMemcpyDtoDAsync_v2 0.020 ms x6, cuKernelGetName 0.007 ms x60

### L20/decode (6 instances): wall 0.703 ms, GPU busy 0.658 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 24.5% |
| wo_a | 0.050 | 7.6% |
| wo_b | 0.045 | 6.8% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/topk | 0.040 | 6.1% |
| sparse_attn | 0.036 | 5.5% |
| attn_keys/index_q | 0.034 | 5.1% |
| q_b | 0.032 | 4.8% |
| router | 0.030 | 4.6% |
| attn_keys/candidates | 0.029 | 4.5% |
| attn_keys/kv_append | 0.027 | 4.1% |
| attn_keys/compressor | 0.023 | 3.5% |
| hc_ffn_pre | 0.021 | 3.2% |
| hc_attn_pre | 0.017 | 2.6% |
| attn_keys/kv_window | 0.014 | 2.1% |
| attn_keys/qr | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/index_keys | 0.011 | 1.6% |
| attn_keys/index_score | 0.010 | 1.6% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.629 ms; top: cuStreamSynchronize 0.413 ms x1, cudaLaunchKernel_v7000 0.144 ms x63, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.021 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.545 ms, GPU busy 0.508 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.7% |
| wo_a | 0.050 | 9.8% |
| wo_b | 0.044 | 8.7% |
| shared_expert | 0.044 | 8.7% |
| sparse_attn | 0.037 | 7.2% |
| q_b | 0.032 | 6.3% |
| router | 0.030 | 5.9% |
| attn_keys/kv_append | 0.023 | 4.6% |
| hc_ffn_pre | 0.021 | 4.2% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.4% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.491 ms; top: cuStreamSynchronize 0.324 ms x1, cudaLaunchKernel_v7000 0.106 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.005 ms x3

