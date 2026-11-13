### L2/prefill (2 instances): wall 35.067 ms, GPU busy 34.971 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.301 | 49.5% |
| sparse_attn | 4.145 | 11.9% |
| wo_a | 2.450 | 7.0% |
| q_b | 1.672 | 4.8% |
| shared_expert | 1.618 | 4.6% |
| wo_b | 1.577 | 4.5% |
| attn_keys/compressor | 1.491 | 4.3% |
| hc_ffn_pre | 1.162 | 3.3% |
| hc_attn_pre | 0.882 | 2.5% |
| router | 0.716 | 2.0% |
| attn_keys/index_q | 0.600 | 1.7% |
| attn_keys/qr | 0.364 | 1.0% |
| hc_post | 0.235 | 0.7% |
| attn_keys/kv_window | 0.225 | 0.6% |
| combine | 0.204 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.061 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 34.895 ms; top: cuStreamSynchronize 34.231 ms x1, cuMemAllocAsync 0.360 ms x53, cudaLaunchKernel_v7000 0.215 ms x64, cuMemFreeAsync 0.033 ms x51, cuMemcpyDtoDAsync_v2 0.026 ms x4, cuMemsetD8Async 0.019 ms x3

### L20/prefill (2 instances): wall 33.371 ms, GPU busy 33.240 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.295 | 49.0% |
| sparse_attn | 4.273 | 12.9% |
| wo_a | 2.474 | 7.4% |
| q_b | 1.698 | 5.1% |
| shared_expert | 1.643 | 4.9% |
| wo_b | 1.596 | 4.8% |
| hc_ffn_pre | 1.181 | 3.6% |
| hc_attn_pre | 0.891 | 2.7% |
| router | 0.723 | 2.2% |
| attn_keys/index_q | 0.597 | 1.8% |
| attn_keys/index_score | 0.392 | 1.2% |
| attn_keys/qr | 0.367 | 1.1% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.230 | 0.7% |
| attn_keys/compressor | 0.209 | 0.6% |
| combine | 0.203 | 0.6% |
| attn_keys/topk | 0.087 | 0.3% |
| attn_keys/index_keys | 0.049 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.030 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.184 ms; top: cuStreamSynchronize 32.334 ms x1, cuMemAllocAsync 0.514 ms x52, cudaLaunchKernel_v7000 0.234 ms x63, cuMemFreeAsync 0.040 ms x50, cuMemcpyDtoDAsync_v2 0.030 ms x4, cuMemsetD8Async 0.020 ms x3

### L21/prefill (2 instances): wall 32.266 ms, GPU busy 32.129 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.556 | 51.5% |
| sparse_attn | 4.318 | 13.4% |
| wo_a | 2.477 | 7.7% |
| q_b | 1.676 | 5.2% |
| shared_expert | 1.649 | 5.1% |
| wo_b | 1.596 | 5.0% |
| hc_ffn_pre | 1.175 | 3.7% |
| hc_attn_pre | 0.893 | 2.8% |
| router | 0.725 | 2.3% |
| attn_keys/qr | 0.366 | 1.1% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.228 | 0.7% |
| combine | 0.204 | 0.6% |
| moe_plan | 0.016 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.079 ms; top: cuStreamSynchronize 31.391 ms x1, cuMemAllocAsync 0.359 ms x46, cudaLaunchKernel_v7000 0.224 ms x47, cuMemFreeAsync 0.040 ms x44, cuMemsetD8Async 0.028 ms x3, cuMemcpyDtoDAsync_v2 0.027 ms x3

### L2/decode (6 instances): wall 0.700 ms, GPU busy 0.585 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 26.7% |
| wo_a | 0.049 | 8.5% |
| attn_keys/compressor | 0.048 | 8.3% |
| wo_b | 0.044 | 7.6% |
| shared_expert | 0.044 | 7.6% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.8% |
| q_b | 0.032 | 5.4% |
| router | 0.029 | 4.9% |
| hc_ffn_pre | 0.020 | 3.5% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.566 ms; top: cuStreamSynchronize 0.284 ms x1, cudaLaunchKernel_v7000 0.186 ms x60, cuMemAllocAsync 0.036 ms x52, cuMemcpyDtoDAsync_v2 0.023 ms x6, cuMemFreeAsync 0.023 ms x51, cuKernelGetName 0.007 ms x60

### L20/decode (6 instances): wall 0.611 ms, GPU busy 0.569 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.155 | 27.3% |
| wo_a | 0.049 | 8.5% |
| shared_expert | 0.044 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.035 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.5% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.023 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.5% |
| attn_keys/qr | 0.014 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.9% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/index_score | 0.003 | 0.6% |
| attn_keys/candidates | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.533 ms; top: cuStreamSynchronize 0.308 ms x1, cudaLaunchKernel_v7000 0.151 ms x63, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.021 ms x50, cuMemcpyDtoDAsync_v2 0.015 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.520 ms, GPU busy 0.479 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.155 | 32.4% |
| wo_a | 0.048 | 10.0% |
| shared_expert | 0.044 | 9.3% |
| wo_b | 0.044 | 9.1% |
| sparse_attn | 0.036 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.029 | 6.0% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.6% |
| attn_keys/kv_window | 0.014 | 3.0% |
| attn_keys/qr | 0.014 | 2.9% |
| moe_plan | 0.012 | 2.5% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.465 ms; top: cuStreamSynchronize 0.295 ms x1, cudaLaunchKernel_v7000 0.112 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.005 ms x3

