### L2/prefill (2 instances): wall 45.516 ms, GPU busy 45.428 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.620 | 52.0% |
| sparse_attn | 4.996 | 11.0% |
| wo_a | 2.682 | 5.9% |
| q_b | 1.985 | 4.4% |
| shared_expert | 1.952 | 4.3% |
| wo_b | 1.887 | 4.2% |
| hc_ffn_pre | 1.726 | 3.8% |
| attn_keys/compressor | 1.661 | 3.7% |
| hc_attn_pre | 1.162 | 2.6% |
| router | 0.821 | 1.8% |
| attn_keys/index_q | 0.659 | 1.4% |
| hc_post | 0.499 | 1.1% |
| attn_keys/qr | 0.431 | 0.9% |
| moe_plan | 0.398 | 0.9% |
| attn_keys/index_score | 0.356 | 0.8% |
| attn_keys/kv_window | 0.257 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.063 | 0.1% |
| attn_keys/index_keys | 0.036 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 45.359 ms; top: cuStreamSynchronize 44.697 ms x1, cuMemAllocAsync 0.369 ms x53, cudaLaunchKernel_v7000 0.208 ms x64, cuMemFreeAsync 0.030 ms x51, cuMemcpyDtoDAsync_v2 0.027 ms x4, cuMemsetD8Async 0.017 ms x3

### L20/prefill (2 instances): wall 43.206 ms, GPU busy 43.101 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.070 | 51.2% |
| sparse_attn | 5.123 | 11.9% |
| wo_a | 2.698 | 6.3% |
| q_b | 1.992 | 4.6% |
| shared_expert | 1.967 | 4.6% |
| wo_b | 1.895 | 4.4% |
| hc_ffn_pre | 1.734 | 4.0% |
| hc_attn_pre | 1.170 | 2.7% |
| router | 0.825 | 1.9% |
| attn_keys/index_score | 0.733 | 1.7% |
| attn_keys/index_q | 0.650 | 1.5% |
| hc_post | 0.501 | 1.2% |
| attn_keys/qr | 0.433 | 1.0% |
| moe_plan | 0.410 | 1.0% |
| attn_keys/kv_window | 0.258 | 0.6% |
| attn_keys/compressor | 0.233 | 0.5% |
| combine | 0.205 | 0.5% |
| attn_keys/topk | 0.099 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.020 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 43.054 ms; top: cuStreamSynchronize 42.301 ms x1, cuMemAllocAsync 0.467 ms x52, cudaLaunchKernel_v7000 0.199 ms x63, cuMemFreeAsync 0.034 ms x50, cuMemcpyDtoDAsync_v2 0.026 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 41.714 ms, GPU busy 41.605 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.322 | 53.7% |
| sparse_attn | 5.147 | 12.4% |
| wo_a | 2.704 | 6.5% |
| q_b | 1.998 | 4.8% |
| shared_expert | 1.981 | 4.8% |
| wo_b | 1.895 | 4.6% |
| hc_ffn_pre | 1.736 | 4.2% |
| hc_attn_pre | 1.168 | 2.8% |
| router | 0.825 | 2.0% |
| hc_post | 0.507 | 1.2% |
| attn_keys/qr | 0.433 | 1.0% |
| moe_plan | 0.410 | 1.0% |
| attn_keys/kv_window | 0.259 | 0.6% |
| combine | 0.206 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 41.577 ms; top: cuStreamSynchronize 41.082 ms x1, cuMemAllocAsync 0.254 ms x46, cudaLaunchKernel_v7000 0.163 ms x47, cuMemFreeAsync 0.029 ms x44, cuMemcpyDtoDAsync_v2 0.020 ms x3, cuMemsetD8Async 0.020 ms x3

### L2/decode (6 instances): wall 0.721 ms, GPU busy 0.662 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 24.0% |
| wo_b | 0.062 | 9.3% |
| wo_a | 0.048 | 7.3% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.0% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.043 | 6.5% |
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
| attn_keys/index_keys | 0.005 | 0.8% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.641 ms; top: cuStreamSynchronize 0.414 ms x1, cudaLaunchKernel_v7000 0.151 ms x60, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.696 ms, GPU busy 0.659 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 23.8% |
| wo_b | 0.066 | 10.0% |
| wo_a | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.035 | 5.2% |
| attn_keys/compressor | 0.032 | 4.9% |
| q_b | 0.031 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 2.0% |
| moe_plan | 0.013 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.5% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.632 ms; top: cuStreamSynchronize 0.432 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.579 ms, GPU busy 0.548 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 28.6% |
| wo_b | 0.066 | 12.0% |
| sparse_attn | 0.047 | 8.7% |
| wo_a | 0.047 | 8.6% |
| shared_expert | 0.044 | 8.1% |
| hc_ffn_pre | 0.036 | 6.5% |
| router | 0.034 | 6.2% |
| q_b | 0.032 | 5.8% |
| hc_attn_pre | 0.028 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.012 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.007 | 1.4% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.531 ms; top: cuStreamSynchronize 0.373 ms x1, cudaLaunchKernel_v7000 0.104 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

