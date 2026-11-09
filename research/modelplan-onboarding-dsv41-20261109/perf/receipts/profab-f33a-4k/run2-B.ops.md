### L2/prefill (2 instances): wall 35.476 ms, GPU busy 35.335 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.494 | 49.5% |
| sparse_attn | 4.197 | 11.9% |
| wo_a | 2.475 | 7.0% |
| q_b | 1.687 | 4.8% |
| shared_expert | 1.636 | 4.6% |
| wo_b | 1.590 | 4.5% |
| attn_keys/compressor | 1.506 | 4.3% |
| hc_ffn_pre | 1.173 | 3.3% |
| hc_attn_pre | 0.884 | 2.5% |
| router | 0.720 | 2.0% |
| attn_keys/index_q | 0.598 | 1.7% |
| attn_keys/qr | 0.367 | 1.0% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.228 | 0.6% |
| combine | 0.205 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.068 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.270 ms; top: cuStreamSynchronize 34.485 ms x1, cuMemAllocAsync 0.429 ms x53, cudaLaunchKernel_v7000 0.258 ms x64, cuMemFreeAsync 0.037 ms x51, cuMemcpyDtoDAsync_v2 0.028 ms x4, cuMemsetD8Async 0.019 ms x3

### L20/prefill (2 instances): wall 33.535 ms, GPU busy 33.351 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.343 | 49.0% |
| sparse_attn | 4.292 | 12.9% |
| wo_a | 2.481 | 7.4% |
| q_b | 1.705 | 5.1% |
| shared_expert | 1.648 | 4.9% |
| wo_b | 1.597 | 4.8% |
| hc_ffn_pre | 1.183 | 3.5% |
| hc_attn_pre | 0.892 | 2.7% |
| router | 0.726 | 2.2% |
| attn_keys/index_q | 0.596 | 1.8% |
| attn_keys/index_score | 0.393 | 1.2% |
| attn_keys/qr | 0.369 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.230 | 0.7% |
| attn_keys/compressor | 0.210 | 0.6% |
| combine | 0.205 | 0.6% |
| attn_keys/topk | 0.103 | 0.3% |
| attn_keys/index_keys | 0.049 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.277 ms; top: cuStreamSynchronize 32.178 ms x1, cuMemAllocAsync 0.660 ms x52, cudaLaunchKernel_v7000 0.309 ms x63, cuMemFreeAsync 0.055 ms x50, cuMemcpyDtoDAsync_v2 0.035 ms x4, cuMemsetD8Async 0.026 ms x3

### L21/prefill (2 instances): wall 32.384 ms, GPU busy 32.254 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.618 | 51.5% |
| sparse_attn | 4.339 | 13.5% |
| wo_a | 2.488 | 7.7% |
| q_b | 1.686 | 5.2% |
| shared_expert | 1.653 | 5.1% |
| wo_b | 1.600 | 5.0% |
| hc_ffn_pre | 1.176 | 3.6% |
| hc_attn_pre | 0.898 | 2.8% |
| router | 0.727 | 2.3% |
| attn_keys/qr | 0.367 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.229 | 0.7% |
| combine | 0.205 | 0.6% |
| moe_plan | 0.016 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.223 ms; top: cuStreamSynchronize 31.664 ms x1, cuMemAllocAsync 0.271 ms x46, cudaLaunchKernel_v7000 0.202 ms x47, cuMemFreeAsync 0.034 ms x44, cuMemcpyDtoDAsync_v2 0.023 ms x3, cuMemsetD8Async 0.021 ms x3

### L2/decode (6 instances): wall 0.753 ms, GPU busy 0.586 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 26.7% |
| wo_a | 0.049 | 8.4% |
| attn_keys/compressor | 0.048 | 8.2% |
| wo_b | 0.045 | 7.7% |
| shared_expert | 0.044 | 7.6% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.5% |
| router | 0.028 | 4.8% |
| hc_ffn_pre | 0.020 | 3.5% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/kv_append | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.572 ms; top: cuStreamSynchronize 0.252 ms x1, cudaLaunchKernel_v7000 0.210 ms x60, cuMemAllocAsync 0.039 ms x52, cuMemFreeAsync 0.029 ms x51, cuMemcpyDtoDAsync_v2 0.026 ms x6, cuKernelGetName 0.009 ms x60

### L20/decode (6 instances): wall 0.615 ms, GPU busy 0.571 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 27.3% |
| wo_a | 0.049 | 8.5% |
| shared_expert | 0.044 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 6.0% |
| q_b | 0.032 | 5.5% |
| router | 0.028 | 5.0% |
| attn_keys/compressor | 0.024 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/candidates | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.526 ms; top: cuStreamSynchronize 0.270 ms x1, cudaLaunchKernel_v7000 0.170 ms x63, cuMemAllocAsync 0.033 ms x52, cuMemFreeAsync 0.026 ms x50, cuMemcpyDtoDAsync_v2 0.015 ms x4, cuKernelGetName 0.007 ms x63

### L21/decode (6 instances): wall 0.525 ms, GPU busy 0.481 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 32.3% |
| wo_a | 0.049 | 10.1% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.044 | 9.2% |
| sparse_attn | 0.036 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.028 | 5.9% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 2.9% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.5% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.455 ms; top: cuStreamSynchronize 0.252 ms x1, cudaLaunchKernel_v7000 0.131 ms x47, cuMemAllocAsync 0.027 ms x46, cuMemFreeAsync 0.023 ms x44, cuMemcpyDtoDAsync_v2 0.012 ms x3, cuMemsetD8Async 0.006 ms x3

