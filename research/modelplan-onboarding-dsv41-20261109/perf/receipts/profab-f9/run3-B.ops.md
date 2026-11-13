### L2/prefill (2 instances): wall 38.387 ms, GPU busy 38.298 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.517 | 48.3% |
| sparse_attn | 4.636 | 12.1% |
| wo_a | 2.945 | 7.7% |
| q_b | 1.832 | 4.8% |
| shared_expert | 1.783 | 4.7% |
| wo_b | 1.731 | 4.5% |
| attn_keys/compressor | 1.538 | 4.0% |
| hc_ffn_pre | 1.181 | 3.1% |
| hc_attn_pre | 0.889 | 2.3% |
| router | 0.767 | 2.0% |
| attn_keys/index_q | 0.591 | 1.5% |
| moe_plan | 0.396 | 1.0% |
| attn_keys/qr | 0.382 | 1.0% |
| attn_keys/index_score | 0.313 | 0.8% |
| hc_post | 0.237 | 0.6% |
| attn_keys/kv_window | 0.231 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.063 | 0.2% |
| attn_keys/index_keys | 0.031 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 38.233 ms; top: cuStreamSynchronize 37.581 ms x1, cuMemAllocAsync 0.360 ms x53, cudaLaunchKernel_v7000 0.206 ms x64, cuMemFreeAsync 0.028 ms x51, cuMemcpyDtoDAsync_v2 0.027 ms x4, cuMemsetD8Async 0.017 ms x3

### L20/prefill (2 instances): wall 36.566 ms, GPU busy 36.447 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.353 | 47.6% |
| sparse_attn | 4.731 | 13.0% |
| wo_a | 2.956 | 8.1% |
| q_b | 1.859 | 5.1% |
| shared_expert | 1.802 | 4.9% |
| wo_b | 1.743 | 4.8% |
| hc_ffn_pre | 1.188 | 3.3% |
| hc_attn_pre | 0.900 | 2.5% |
| router | 0.772 | 2.1% |
| attn_keys/index_score | 0.653 | 1.8% |
| attn_keys/index_q | 0.593 | 1.6% |
| moe_plan | 0.406 | 1.1% |
| attn_keys/qr | 0.383 | 1.1% |
| hc_post | 0.241 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.6% |
| attn_keys/compressor | 0.219 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.089 | 0.2% |
| attn_keys/candidates | 0.041 | 0.1% |
| attn_keys/kv_append | 0.036 | 0.1% |
| attn_keys/index_keys | 0.034 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 36.406 ms; top: cuStreamSynchronize 35.625 ms x1, cuMemAllocAsync 0.487 ms x52, cudaLaunchKernel_v7000 0.207 ms x63, cuMemFreeAsync 0.034 ms x50, cuMemcpyDtoDAsync_v2 0.024 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 35.284 ms, GPU busy 35.173 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.673 | 50.2% |
| sparse_attn | 4.781 | 13.6% |
| wo_a | 2.967 | 8.4% |
| q_b | 1.851 | 5.3% |
| shared_expert | 1.809 | 5.1% |
| wo_b | 1.741 | 5.0% |
| hc_ffn_pre | 1.189 | 3.4% |
| hc_attn_pre | 0.901 | 2.6% |
| router | 0.774 | 2.2% |
| moe_plan | 0.408 | 1.2% |
| attn_keys/qr | 0.383 | 1.1% |
| hc_post | 0.241 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| combine | 0.208 | 0.6% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 35.146 ms; top: cuStreamSynchronize 34.647 ms x1, cuMemAllocAsync 0.254 ms x46, cudaLaunchKernel_v7000 0.168 ms x47, cuMemFreeAsync 0.028 ms x44, cuMemcpyDtoDAsync_v2 0.020 ms x3, cuMemsetD8Async 0.018 ms x3

### L2/decode (6 instances): wall 0.693 ms, GPU busy 0.629 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 25.4% |
| wo_b | 0.063 | 10.0% |
| wo_a | 0.049 | 7.8% |
| attn_keys/compressor | 0.048 | 7.6% |
| shared_expert | 0.045 | 7.1% |
| sparse_attn | 0.040 | 6.4% |
| router | 0.037 | 5.9% |
| attn_keys/index_q | 0.034 | 5.3% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.2% |
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

host API calls: 233 calls, 0.608 ms; top: cuStreamSynchronize 0.379 ms x1, cudaLaunchKernel_v7000 0.151 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.007 ms x60

### L20/decode (6 instances): wall 0.655 ms, GPU busy 0.618 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 25.9% |
| wo_b | 0.066 | 10.7% |
| wo_a | 0.049 | 7.9% |
| shared_expert | 0.044 | 7.2% |
| sparse_attn | 0.040 | 6.5% |
| router | 0.037 | 6.0% |
| attn_keys/index_q | 0.034 | 5.4% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.4% |
| hc_attn_pre | 0.024 | 3.8% |
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

host API calls: 236 calls, 0.593 ms; top: cuStreamSynchronize 0.392 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.563 ms, GPU busy 0.530 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 29.9% |
| wo_b | 0.065 | 12.3% |
| wo_a | 0.048 | 9.1% |
| shared_expert | 0.045 | 8.4% |
| sparse_attn | 0.042 | 7.8% |
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

host API calls: 191 calls, 0.516 ms; top: cuStreamSynchronize 0.361 ms x1, cudaLaunchKernel_v7000 0.101 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

