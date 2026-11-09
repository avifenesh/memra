### L2/prefill (2 instances): wall 45.400 ms, GPU busy 45.297 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.560 | 52.0% |
| sparse_attn | 4.962 | 11.0% |
| wo_a | 2.679 | 5.9% |
| q_b | 1.979 | 4.4% |
| shared_expert | 1.949 | 4.3% |
| wo_b | 1.884 | 4.2% |
| hc_ffn_pre | 1.721 | 3.8% |
| attn_keys/compressor | 1.660 | 3.7% |
| hc_attn_pre | 1.164 | 2.6% |
| router | 0.818 | 1.8% |
| attn_keys/index_q | 0.654 | 1.4% |
| hc_post | 0.492 | 1.1% |
| attn_keys/qr | 0.430 | 0.9% |
| moe_plan | 0.398 | 0.9% |
| attn_keys/index_score | 0.356 | 0.8% |
| attn_keys/kv_window | 0.257 | 0.6% |
| combine | 0.208 | 0.5% |
| attn_keys/topk | 0.063 | 0.1% |
| attn_keys/index_keys | 0.035 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 45.234 ms; top: cuStreamSynchronize 44.544 ms x1, cuMemAllocAsync 0.380 ms x53, cudaLaunchKernel_v7000 0.225 ms x64, cuMemFreeAsync 0.032 ms x51, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.018 ms x3

### L20/prefill (2 instances): wall 43.053 ms, GPU busy 42.930 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 21.985 | 51.2% |
| sparse_attn | 5.090 | 11.9% |
| wo_a | 2.683 | 6.3% |
| q_b | 1.987 | 4.6% |
| shared_expert | 1.961 | 4.6% |
| wo_b | 1.891 | 4.4% |
| hc_ffn_pre | 1.728 | 4.0% |
| hc_attn_pre | 1.172 | 2.7% |
| router | 0.820 | 1.9% |
| attn_keys/index_score | 0.728 | 1.7% |
| attn_keys/index_q | 0.651 | 1.5% |
| hc_post | 0.500 | 1.2% |
| attn_keys/qr | 0.430 | 1.0% |
| moe_plan | 0.411 | 1.0% |
| attn_keys/kv_window | 0.259 | 0.6% |
| attn_keys/compressor | 0.232 | 0.5% |
| combine | 0.204 | 0.5% |
| attn_keys/topk | 0.098 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.020 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 42.887 ms; top: cuStreamSynchronize 42.092 ms x1, cuMemAllocAsync 0.492 ms x52, cudaLaunchKernel_v7000 0.215 ms x63, cuMemFreeAsync 0.034 ms x50, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 41.532 ms, GPU busy 41.424 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.220 | 53.6% |
| sparse_attn | 5.110 | 12.3% |
| wo_a | 2.687 | 6.5% |
| q_b | 2.001 | 4.8% |
| shared_expert | 1.978 | 4.8% |
| wo_b | 1.891 | 4.6% |
| hc_ffn_pre | 1.726 | 4.2% |
| hc_attn_pre | 1.170 | 2.8% |
| router | 0.821 | 2.0% |
| hc_post | 0.502 | 1.2% |
| attn_keys/qr | 0.431 | 1.0% |
| moe_plan | 0.410 | 1.0% |
| attn_keys/kv_window | 0.257 | 0.6% |
| combine | 0.205 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 41.395 ms; top: cuStreamSynchronize 40.888 ms x1, cuMemAllocAsync 0.250 ms x46, cudaLaunchKernel_v7000 0.172 ms x47, cuMemFreeAsync 0.031 ms x44, cuMemcpyDtoDAsync_v2 0.027 ms x3, cuMemsetD8Async 0.019 ms x3

### L2/decode (6 instances): wall 0.722 ms, GPU busy 0.662 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 24.0% |
| wo_b | 0.063 | 9.5% |
| wo_a | 0.048 | 7.3% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.046 | 7.0% |
| shared_expert | 0.044 | 6.6% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.4% |
| router | 0.035 | 5.2% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.2% |
| attn_keys/topk | 0.014 | 2.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| hc_post | 0.008 | 1.2% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/kv_append | 0.005 | 0.8% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.640 ms; top: cuStreamSynchronize 0.411 ms x1, cudaLaunchKernel_v7000 0.151 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.695 ms, GPU busy 0.659 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 23.8% |
| wo_b | 0.065 | 9.9% |
| wo_a | 0.047 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.042 | 6.4% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.034 | 5.2% |
| attn_keys/compressor | 0.032 | 4.8% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.008 | 1.2% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.631 ms; top: cuStreamSynchronize 0.431 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.578 ms, GPU busy 0.548 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 28.5% |
| wo_b | 0.066 | 12.0% |
| sparse_attn | 0.047 | 8.6% |
| wo_a | 0.047 | 8.6% |
| shared_expert | 0.044 | 8.0% |
| hc_ffn_pre | 0.036 | 6.5% |
| router | 0.034 | 6.3% |
| q_b | 0.032 | 5.8% |
| hc_attn_pre | 0.028 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.012 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.008 | 1.5% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.530 ms; top: cuStreamSynchronize 0.374 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

