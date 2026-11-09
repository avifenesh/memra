### L2/prefill (2 instances): wall 37.174 ms, GPU busy 37.072 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.923 | 48.3% |
| sparse_attn | 5.078 | 13.7% |
| wo_a | 2.708 | 7.3% |
| q_b | 1.714 | 4.6% |
| shared_expert | 1.666 | 4.5% |
| wo_b | 1.613 | 4.4% |
| attn_keys/compressor | 1.540 | 4.2% |
| hc_ffn_pre | 1.174 | 3.2% |
| hc_attn_pre | 0.885 | 2.4% |
| router | 0.738 | 2.0% |
| attn_keys/index_q | 0.617 | 1.7% |
| attn_keys/qr | 0.376 | 1.0% |
| hc_post | 0.239 | 0.6% |
| attn_keys/kv_window | 0.233 | 0.6% |
| combine | 0.209 | 0.6% |
| attn_keys/index_score | 0.208 | 0.6% |
| attn_keys/topk | 0.063 | 0.2% |
| attn_keys/index_keys | 0.041 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 37.003 ms; top: cuStreamSynchronize 36.322 ms x1, cuMemAllocAsync 0.385 ms x53, cudaLaunchKernel_v7000 0.213 ms x64, cuMemFreeAsync 0.028 ms x51, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.021 ms x3

### L20/prefill (2 instances): wall 35.247 ms, GPU busy 35.125 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.763 | 47.7% |
| sparse_attn | 5.188 | 14.8% |
| wo_a | 2.720 | 7.7% |
| q_b | 1.737 | 4.9% |
| shared_expert | 1.680 | 4.8% |
| wo_b | 1.623 | 4.6% |
| hc_ffn_pre | 1.185 | 3.4% |
| hc_attn_pre | 0.889 | 2.5% |
| router | 0.744 | 2.1% |
| attn_keys/index_q | 0.612 | 1.7% |
| attn_keys/index_score | 0.441 | 1.3% |
| attn_keys/qr | 0.376 | 1.1% |
| attn_keys/compressor | 0.249 | 0.7% |
| hc_post | 0.241 | 0.7% |
| attn_keys/kv_window | 0.235 | 0.7% |
| combine | 0.207 | 0.6% |
| attn_keys/topk | 0.087 | 0.2% |
| attn_keys/index_keys | 0.053 | 0.2% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.031 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 35.084 ms; top: cuStreamSynchronize 34.324 ms x1, cuMemAllocAsync 0.478 ms x52, cudaLaunchKernel_v7000 0.202 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.021 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 34.049 ms, GPU busy 33.940 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.039 | 50.2% |
| sparse_attn | 5.250 | 15.5% |
| wo_a | 2.724 | 8.0% |
| q_b | 1.716 | 5.1% |
| shared_expert | 1.688 | 5.0% |
| wo_b | 1.623 | 4.8% |
| hc_ffn_pre | 1.184 | 3.5% |
| hc_attn_pre | 0.885 | 2.6% |
| router | 0.744 | 2.2% |
| attn_keys/qr | 0.375 | 1.1% |
| hc_post | 0.239 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| combine | 0.210 | 0.6% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 33.909 ms; top: cuStreamSynchronize 33.429 ms x1, cuMemAllocAsync 0.246 ms x46, cudaLaunchKernel_v7000 0.162 ms x47, cuMemFreeAsync 0.028 ms x44, cuMemcpyDtoDAsync_v2 0.019 ms x3, cuMemsetD8Async 0.018 ms x3

### L2/decode (6 instances): wall 0.680 ms, GPU busy 0.617 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.162 | 26.2% |
| wo_b | 0.063 | 10.2% |
| wo_a | 0.049 | 7.9% |
| attn_keys/compressor | 0.049 | 7.9% |
| shared_expert | 0.045 | 7.3% |
| sparse_attn | 0.043 | 7.0% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.1% |
| router | 0.029 | 4.7% |
| hc_ffn_pre | 0.020 | 3.3% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.1% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/kv_append | 0.006 | 0.9% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.596 ms; top: cuStreamSynchronize 0.369 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.641 ms, GPU busy 0.605 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.6% |
| wo_b | 0.066 | 10.9% |
| wo_a | 0.049 | 8.0% |
| shared_expert | 0.044 | 7.3% |
| sparse_attn | 0.043 | 7.1% |
| attn_keys/index_q | 0.034 | 5.6% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.8% |
| attn_keys/compressor | 0.023 | 3.9% |
| hc_ffn_pre | 0.020 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.014 | 2.2% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.005 | 0.7% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_keys/candidates | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.579 ms; top: cuStreamSynchronize 0.380 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.549 ms, GPU busy 0.518 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.1% |
| wo_b | 0.066 | 12.7% |
| wo_a | 0.048 | 9.3% |
| sparse_attn | 0.045 | 8.7% |
| shared_expert | 0.045 | 8.6% |
| q_b | 0.032 | 6.2% |
| router | 0.029 | 5.6% |
| hc_ffn_pre | 0.020 | 3.8% |
| hc_attn_pre | 0.017 | 3.3% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.014 | 2.7% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.502 ms; top: cuStreamSynchronize 0.348 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

