### L2/prefill (2 instances): wall 35.858 ms, GPU busy 35.735 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.712 | 49.6% |
| sparse_attn | 4.248 | 11.9% |
| wo_a | 2.498 | 7.0% |
| q_b | 1.702 | 4.8% |
| shared_expert | 1.654 | 4.6% |
| wo_b | 1.612 | 4.5% |
| attn_keys/compressor | 1.526 | 4.3% |
| hc_ffn_pre | 1.181 | 3.3% |
| hc_attn_pre | 0.894 | 2.5% |
| router | 0.727 | 2.0% |
| attn_keys/index_q | 0.604 | 1.7% |
| attn_keys/qr | 0.371 | 1.0% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.231 | 0.6% |
| combine | 0.207 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.062 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.673 ms; top: cuStreamSynchronize 34.934 ms x1, cuMemAllocAsync 0.402 ms x53, cudaLaunchKernel_v7000 0.241 ms x64, cuMemFreeAsync 0.033 ms x51, cuMemcpyDtoDAsync_v2 0.029 ms x4, cuMemsetD8Async 0.022 ms x3

### L20/prefill (2 instances): wall 33.679 ms, GPU busy 33.535 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.465 | 49.1% |
| sparse_attn | 4.297 | 12.8% |
| wo_a | 2.494 | 7.4% |
| q_b | 1.707 | 5.1% |
| shared_expert | 1.659 | 4.9% |
| wo_b | 1.612 | 4.8% |
| hc_ffn_pre | 1.192 | 3.6% |
| hc_attn_pre | 0.894 | 2.7% |
| router | 0.732 | 2.2% |
| attn_keys/index_q | 0.601 | 1.8% |
| attn_keys/index_score | 0.394 | 1.2% |
| attn_keys/qr | 0.371 | 1.1% |
| hc_post | 0.237 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| attn_keys/compressor | 0.211 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.088 | 0.3% |
| attn_keys/index_keys | 0.050 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.031 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.448 ms; top: cuStreamSynchronize 32.365 ms x1, cuMemAllocAsync 0.673 ms x52, cudaLaunchKernel_v7000 0.298 ms x63, cuMemFreeAsync 0.045 ms x50, cuMemcpyDtoDAsync_v2 0.034 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 32.558 ms, GPU busy 32.437 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.722 | 51.6% |
| sparse_attn | 4.347 | 13.4% |
| wo_a | 2.499 | 7.7% |
| q_b | 1.697 | 5.2% |
| shared_expert | 1.667 | 5.1% |
| wo_b | 1.612 | 5.0% |
| hc_ffn_pre | 1.188 | 3.7% |
| hc_attn_pre | 0.898 | 2.8% |
| router | 0.732 | 2.3% |
| attn_keys/qr | 0.369 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.230 | 0.7% |
| combine | 0.206 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.397 ms; top: cuStreamSynchronize 31.865 ms x1, cuMemAllocAsync 0.260 ms x46, cudaLaunchKernel_v7000 0.186 ms x47, cuMemFreeAsync 0.033 ms x44, cuMemsetD8Async 0.022 ms x3, cuMemcpyDtoDAsync_v2 0.022 ms x3

### L2/decode (6 instances): wall 0.682 ms, GPU busy 0.586 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 26.9% |
| wo_a | 0.049 | 8.4% |
| attn_keys/compressor | 0.049 | 8.3% |
| shared_expert | 0.045 | 7.6% |
| wo_b | 0.044 | 7.5% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.8% |
| q_b | 0.032 | 5.4% |
| router | 0.029 | 4.9% |
| hc_ffn_pre | 0.021 | 3.5% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.8% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.573 ms; top: cuStreamSynchronize 0.313 ms x1, cudaLaunchKernel_v7000 0.172 ms x60, cuMemAllocAsync 0.033 ms x52, cuMemFreeAsync 0.022 ms x51, cuMemcpyDtoDAsync_v2 0.020 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.612 ms, GPU busy 0.570 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 27.3% |
| wo_a | 0.049 | 8.6% |
| shared_expert | 0.044 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.5% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.023 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.014 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/index_score | 0.003 | 0.6% |
| attn_keys/candidates | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.543 ms; top: cuStreamSynchronize 0.322 ms x1, cudaLaunchKernel_v7000 0.147 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.021 ms x50, cuMemcpyDtoDAsync_v2 0.015 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.523 ms, GPU busy 0.483 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 32.3% |
| wo_a | 0.049 | 10.1% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.044 | 9.2% |
| sparse_attn | 0.036 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.029 | 6.0% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 2.9% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.4% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.470 ms; top: cuStreamSynchronize 0.300 ms x1, cudaLaunchKernel_v7000 0.110 ms x47, cuMemAllocAsync 0.022 ms x46, cuMemFreeAsync 0.018 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.004 ms x3

