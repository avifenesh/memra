### L2/prefill (2 instances): wall 45.222 ms, GPU busy 45.132 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.480 | 52.0% |
| sparse_attn | 4.924 | 10.9% |
| wo_a | 2.670 | 5.9% |
| q_b | 1.975 | 4.4% |
| shared_expert | 1.944 | 4.3% |
| wo_b | 1.878 | 4.2% |
| hc_ffn_pre | 1.719 | 3.8% |
| attn_keys/compressor | 1.653 | 3.7% |
| hc_attn_pre | 1.163 | 2.6% |
| router | 0.815 | 1.8% |
| attn_keys/index_q | 0.655 | 1.5% |
| hc_post | 0.490 | 1.1% |
| attn_keys/qr | 0.429 | 1.0% |
| moe_plan | 0.397 | 0.9% |
| attn_keys/index_score | 0.355 | 0.8% |
| attn_keys/kv_window | 0.256 | 0.6% |
| combine | 0.206 | 0.5% |
| attn_keys/topk | 0.062 | 0.1% |
| attn_keys/index_keys | 0.035 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 45.065 ms; top: cuStreamSynchronize 44.434 ms x1, cuMemAllocAsync 0.356 ms x53, cudaLaunchKernel_v7000 0.202 ms x64, cuMemFreeAsync 0.028 ms x51, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.013 ms x3

### L20/prefill (2 instances): wall 43.024 ms, GPU busy 42.916 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 21.972 | 51.2% |
| sparse_attn | 5.093 | 11.9% |
| wo_a | 2.685 | 6.3% |
| q_b | 1.987 | 4.6% |
| shared_expert | 1.958 | 4.6% |
| wo_b | 1.888 | 4.4% |
| hc_ffn_pre | 1.728 | 4.0% |
| hc_attn_pre | 1.168 | 2.7% |
| router | 0.821 | 1.9% |
| attn_keys/index_score | 0.726 | 1.7% |
| attn_keys/index_q | 0.650 | 1.5% |
| hc_post | 0.499 | 1.2% |
| attn_keys/qr | 0.431 | 1.0% |
| moe_plan | 0.410 | 1.0% |
| attn_keys/kv_window | 0.261 | 0.6% |
| attn_keys/compressor | 0.231 | 0.5% |
| combine | 0.205 | 0.5% |
| attn_keys/topk | 0.099 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.020 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 42.870 ms; top: cuStreamSynchronize 42.126 ms x1, cuMemAllocAsync 0.474 ms x52, cudaLaunchKernel_v7000 0.195 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.020 ms x4, cuMemsetD8Async 0.014 ms x3

### L21/prefill (2 instances): wall 41.530 ms, GPU busy 41.436 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.239 | 53.7% |
| sparse_attn | 5.111 | 12.3% |
| wo_a | 2.688 | 6.5% |
| q_b | 1.993 | 4.8% |
| shared_expert | 1.977 | 4.8% |
| wo_b | 1.889 | 4.6% |
| hc_ffn_pre | 1.730 | 4.2% |
| hc_attn_pre | 1.165 | 2.8% |
| router | 0.822 | 2.0% |
| hc_post | 0.505 | 1.2% |
| attn_keys/qr | 0.431 | 1.0% |
| moe_plan | 0.408 | 1.0% |
| attn_keys/kv_window | 0.259 | 0.6% |
| combine | 0.206 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 41.398 ms; top: cuStreamSynchronize 40.939 ms x1, cuMemAllocAsync 0.240 ms x46, cudaLaunchKernel_v7000 0.154 ms x47, cuMemFreeAsync 0.027 ms x44, cuMemcpyDtoDAsync_v2 0.017 ms x3, cuMemsetD8Async 0.014 ms x3

### L2/decode (6 instances): wall 0.720 ms, GPU busy 0.660 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 24.0% |
| wo_b | 0.062 | 9.4% |
| wo_a | 0.048 | 7.2% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.042 | 6.4% |
| hc_ffn_pre | 0.036 | 5.4% |
| router | 0.035 | 5.2% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.2% |
| attn_keys/topk | 0.014 | 2.1% |
| attn_keys/kv_window | 0.013 | 2.0% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/kv_append | 0.005 | 0.8% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.639 ms; top: cuStreamSynchronize 0.412 ms x1, cudaLaunchKernel_v7000 0.150 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.695 ms, GPU busy 0.659 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 23.9% |
| wo_b | 0.065 | 9.9% |
| wo_a | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.042 | 6.4% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.034 | 5.2% |
| attn_keys/compressor | 0.032 | 4.9% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.5% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.632 ms; top: cuStreamSynchronize 0.431 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.578 ms, GPU busy 0.547 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 28.5% |
| wo_b | 0.065 | 11.9% |
| sparse_attn | 0.048 | 8.7% |
| wo_a | 0.047 | 8.6% |
| shared_expert | 0.044 | 8.1% |
| hc_ffn_pre | 0.036 | 6.5% |
| router | 0.034 | 6.3% |
| q_b | 0.032 | 5.8% |
| hc_attn_pre | 0.028 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.012 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.007 | 1.4% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.529 ms; top: cuStreamSynchronize 0.375 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

