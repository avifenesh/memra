### L2/prefill (2 instances): wall 299.353 ms, GPU busy 299.159 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 129.625 | 43.3% |
| sparse_attn | 34.990 | 11.7% |
| attn_keys/index_score | 22.145 | 7.4% |
| wo_a | 20.223 | 6.8% |
| shared_expert | 13.736 | 4.6% |
| q_b | 13.500 | 4.5% |
| wo_b | 12.988 | 4.3% |
| attn_keys/compressor | 11.690 | 3.9% |
| hc_ffn_pre | 10.377 | 3.5% |
| hc_attn_pre | 8.022 | 2.7% |
| router | 5.783 | 1.9% |
| attn_keys/index_q | 4.438 | 1.5% |
| attn_keys/qr | 3.022 | 1.0% |
| hc_post | 2.218 | 0.7% |
| attn_keys/topk | 2.161 | 0.7% |
| combine | 1.913 | 0.6% |
| attn_keys/kv_window | 1.878 | 0.6% |
| attn_keys/index_keys | 0.139 | 0.0% |
| attn_keys/kv_append | 0.129 | 0.0% |
| attn_idx_rows | 0.102 | 0.0% |
| moe_plan | 0.081 | 0.0% |

host API calls: 627 calls, 299.014 ms; top: cuStreamSynchronize 296.479 ms x1, cuMemAllocAsync 1.880 ms x134, cudaLaunchKernel_v7000 0.485 ms x172, cuMemFreeAsync 0.070 ms x132, cuMemsetD8Async 0.041 ms x12, cuMemcpyDtoDAsync_v2 0.038 ms x4

### L20/prefill (2 instances): wall 309.736 ms, GPU busy 309.511 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.850 | 39.7% |
| attn_keys/index_score | 44.092 | 14.2% |
| sparse_attn | 35.139 | 11.4% |
| wo_a | 20.455 | 6.6% |
| shared_expert | 13.798 | 4.5% |
| q_b | 13.556 | 4.4% |
| wo_b | 13.082 | 4.2% |
| hc_ffn_pre | 10.551 | 3.4% |
| hc_attn_pre | 8.045 | 2.6% |
| router | 5.826 | 1.9% |
| attn_keys/index_q | 4.446 | 1.4% |
| attn_keys/topk | 4.269 | 1.4% |
| attn_keys/qr | 3.008 | 1.0% |
| attn_keys/candidates | 2.221 | 0.7% |
| hc_post | 2.216 | 0.7% |
| combine | 1.914 | 0.6% |
| attn_keys/kv_window | 1.876 | 0.6% |
| attn_keys/compressor | 1.474 | 0.5% |
| attn_keys/index_keys | 0.264 | 0.1% |
| attn_keys/kv_append | 0.242 | 0.1% |
| attn_idx_rows | 0.106 | 0.0% |
| moe_plan | 0.082 | 0.0% |

host API calls: 629 calls, 309.386 ms; top: cuStreamSynchronize 306.633 ms x1, cuMemAllocAsync 2.077 ms x133, cudaLaunchKernel_v7000 0.496 ms x174, cuMemFreeAsync 0.079 ms x131, cuMemsetD8Async 0.045 ms x12, cuMemcpyDtoDAsync_v2 0.035 ms x4

### L21/prefill (2 instances): wall 248.460 ms, GPU busy 248.202 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 120.131 | 48.4% |
| sparse_attn | 34.887 | 14.1% |
| wo_a | 20.156 | 8.1% |
| q_b | 13.622 | 5.5% |
| shared_expert | 13.446 | 5.4% |
| wo_b | 12.916 | 5.2% |
| hc_ffn_pre | 10.221 | 4.1% |
| hc_attn_pre | 8.037 | 3.2% |
| router | 5.599 | 2.3% |
| attn_keys/qr | 3.012 | 1.2% |
| hc_post | 2.213 | 0.9% |
| attn_keys/kv_window | 1.875 | 0.8% |
| combine | 1.874 | 0.8% |
| attn_idx_rows | 0.104 | 0.0% |
| moe_plan | 0.079 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 247.594 ms; top: cuStreamSynchronize 243.009 ms x1, cuMemAllocAsync 2.882 ms x118, cudaLaunchKernel_v7000 1.241 ms x134, cuMemFreeAsync 0.201 ms x116, cuMemsetD8Async 0.118 ms x12, cuMemcpyDtoDAsync_v2 0.085 ms x3

### L2/decode (6 instances): wall 0.707 ms, GPU busy 0.633 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 25.1% |
| wo_a | 0.049 | 7.8% |
| attn_keys/compressor | 0.048 | 7.7% |
| shared_expert | 0.044 | 6.9% |
| wo_b | 0.044 | 6.9% |
| attn_keys/topk | 0.036 | 5.8% |
| sparse_attn | 0.036 | 5.7% |
| attn_keys/index_q | 0.034 | 5.4% |
| q_b | 0.032 | 5.1% |
| router | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.3% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/index_score | 0.016 | 2.5% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.2% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.620 ms; top: cuStreamSynchronize 0.380 ms x1, cudaLaunchKernel_v7000 0.158 ms x60, cuMemAllocAsync 0.031 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.727 ms, GPU busy 0.687 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 23.4% |
| attn_keys/topk | 0.062 | 9.0% |
| wo_a | 0.050 | 7.3% |
| wo_b | 0.044 | 6.4% |
| shared_expert | 0.044 | 6.4% |
| sparse_attn | 0.036 | 5.2% |
| attn_keys/index_q | 0.034 | 4.9% |
| attn_keys/candidates | 0.032 | 4.7% |
| q_b | 0.032 | 4.6% |
| router | 0.030 | 4.4% |
| attn_keys/kv_append | 0.027 | 4.0% |
| attn_keys/compressor | 0.023 | 3.4% |
| hc_ffn_pre | 0.021 | 3.1% |
| hc_attn_pre | 0.017 | 2.5% |
| attn_keys/index_score | 0.016 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.0% |
| attn_keys/qr | 0.013 | 1.8% |
| moe_plan | 0.012 | 1.7% |
| attn_keys/index_keys | 0.011 | 1.5% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.661 ms; top: cuStreamSynchronize 0.454 ms x1, cudaLaunchKernel_v7000 0.138 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.541 ms, GPU busy 0.508 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.6% |
| wo_a | 0.050 | 9.8% |
| shared_expert | 0.044 | 8.7% |
| wo_b | 0.044 | 8.7% |
| sparse_attn | 0.037 | 7.3% |
| q_b | 0.032 | 6.3% |
| router | 0.030 | 5.9% |
| attn_keys/kv_append | 0.023 | 4.6% |
| hc_ffn_pre | 0.021 | 4.1% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.4% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.491 ms; top: cuStreamSynchronize 0.330 ms x1, cudaLaunchKernel_v7000 0.103 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

