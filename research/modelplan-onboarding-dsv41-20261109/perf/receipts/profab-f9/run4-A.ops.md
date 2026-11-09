### L2/prefill (2 instances): wall 45.764 ms, GPU busy 45.665 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.750 | 52.0% |
| sparse_attn | 5.032 | 11.0% |
| wo_a | 2.695 | 5.9% |
| q_b | 1.992 | 4.4% |
| shared_expert | 1.967 | 4.3% |
| wo_b | 1.897 | 4.2% |
| hc_ffn_pre | 1.732 | 3.8% |
| attn_keys/compressor | 1.666 | 3.6% |
| hc_attn_pre | 1.169 | 2.6% |
| router | 0.826 | 1.8% |
| attn_keys/index_q | 0.659 | 1.4% |
| hc_post | 0.498 | 1.1% |
| attn_keys/qr | 0.434 | 0.9% |
| moe_plan | 0.397 | 0.9% |
| attn_keys/index_score | 0.357 | 0.8% |
| attn_keys/kv_window | 0.261 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.064 | 0.1% |
| attn_keys/index_keys | 0.036 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 45.603 ms; top: cuStreamSynchronize 44.929 ms x1, cuMemAllocAsync 0.376 ms x53, cudaLaunchKernel_v7000 0.211 ms x64, cuMemFreeAsync 0.030 ms x51, cuMemcpyDtoDAsync_v2 0.029 ms x4, cuMemsetD8Async 0.017 ms x3

### L20/prefill (2 instances): wall 43.474 ms, GPU busy 43.341 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.171 | 51.2% |
| sparse_attn | 5.168 | 11.9% |
| wo_a | 2.712 | 6.3% |
| q_b | 2.001 | 4.6% |
| shared_expert | 1.979 | 4.6% |
| wo_b | 1.906 | 4.4% |
| hc_ffn_pre | 1.747 | 4.0% |
| hc_attn_pre | 1.176 | 2.7% |
| router | 0.829 | 1.9% |
| attn_keys/index_score | 0.734 | 1.7% |
| attn_keys/index_q | 0.655 | 1.5% |
| hc_post | 0.507 | 1.2% |
| attn_keys/qr | 0.436 | 1.0% |
| moe_plan | 0.409 | 0.9% |
| attn_keys/kv_window | 0.264 | 0.6% |
| attn_keys/compressor | 0.234 | 0.5% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.100 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.021 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 43.319 ms; top: cuStreamSynchronize 42.552 ms x1, cuMemAllocAsync 0.477 ms x52, cudaLaunchKernel_v7000 0.203 ms x63, cuMemFreeAsync 0.034 ms x50, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 41.909 ms, GPU busy 41.806 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.419 | 53.6% |
| sparse_attn | 5.186 | 12.4% |
| wo_a | 2.713 | 6.5% |
| q_b | 2.010 | 4.8% |
| shared_expert | 1.990 | 4.8% |
| wo_b | 1.902 | 4.6% |
| hc_ffn_pre | 1.746 | 4.2% |
| hc_attn_pre | 1.173 | 2.8% |
| router | 0.829 | 2.0% |
| hc_post | 0.510 | 1.2% |
| attn_keys/qr | 0.436 | 1.0% |
| moe_plan | 0.408 | 1.0% |
| attn_keys/kv_window | 0.260 | 0.6% |
| combine | 0.209 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 41.779 ms; top: cuStreamSynchronize 41.297 ms x1, cuMemAllocAsync 0.246 ms x46, cudaLaunchKernel_v7000 0.161 ms x47, cuMemFreeAsync 0.028 ms x44, cuMemcpyDtoDAsync_v2 0.022 ms x3, cuMemsetD8Async 0.017 ms x3

### L2/decode (6 instances): wall 0.725 ms, GPU busy 0.663 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 23.9% |
| wo_b | 0.062 | 9.4% |
| wo_a | 0.048 | 7.3% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.0% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.4% |
| router | 0.035 | 5.2% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/kv_append | 0.005 | 0.8% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.645 ms; top: cuStreamSynchronize 0.416 ms x1, cudaLaunchKernel_v7000 0.150 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.697 ms, GPU busy 0.661 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 24.0% |
| wo_b | 0.066 | 9.9% |
| wo_a | 0.047 | 7.2% |
| sparse_attn | 0.047 | 7.0% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.042 | 6.4% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.034 | 5.2% |
| attn_keys/compressor | 0.032 | 4.8% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 2.0% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.5% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.636 ms; top: cuStreamSynchronize 0.439 ms x1, cudaLaunchKernel_v7000 0.132 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.580 ms, GPU busy 0.548 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 28.6% |
| wo_b | 0.065 | 11.9% |
| sparse_attn | 0.048 | 8.7% |
| wo_a | 0.047 | 8.6% |
| shared_expert | 0.044 | 8.1% |
| hc_ffn_pre | 0.036 | 6.6% |
| router | 0.034 | 6.3% |
| q_b | 0.032 | 5.8% |
| hc_attn_pre | 0.029 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.007 | 1.4% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.532 ms; top: cuStreamSynchronize 0.379 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

