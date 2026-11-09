### L2/prefill (2 instances): wall 35.850 ms, GPU busy 35.764 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.725 | 49.6% |
| sparse_attn | 4.244 | 11.9% |
| wo_a | 2.500 | 7.0% |
| q_b | 1.705 | 4.8% |
| shared_expert | 1.653 | 4.6% |
| wo_b | 1.610 | 4.5% |
| attn_keys/compressor | 1.530 | 4.3% |
| hc_ffn_pre | 1.184 | 3.3% |
| hc_attn_pre | 0.896 | 2.5% |
| router | 0.731 | 2.0% |
| attn_keys/index_q | 0.602 | 1.7% |
| attn_keys/qr | 0.371 | 1.0% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.230 | 0.6% |
| combine | 0.207 | 0.6% |
| attn_keys/index_score | 0.187 | 0.5% |
| attn_keys/topk | 0.070 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.679 ms; top: cuStreamSynchronize 35.002 ms x1, cuMemAllocAsync 0.380 ms x53, cudaLaunchKernel_v7000 0.216 ms x64, cuMemFreeAsync 0.031 ms x51, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.015 ms x3

### L20/prefill (2 instances): wall 108.742 ms, GPU busy 33.579 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.482 | 49.1% |
| sparse_attn | 4.307 | 12.8% |
| wo_a | 2.505 | 7.5% |
| q_b | 1.696 | 5.1% |
| shared_expert | 1.659 | 4.9% |
| wo_b | 1.610 | 4.8% |
| hc_ffn_pre | 1.190 | 3.5% |
| hc_attn_pre | 0.896 | 2.7% |
| router | 0.730 | 2.2% |
| attn_keys/index_q | 0.605 | 1.8% |
| attn_keys/index_score | 0.395 | 1.2% |
| attn_keys/qr | 0.372 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.7% |
| attn_keys/compressor | 0.211 | 0.6% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.103 | 0.3% |
| attn_keys/index_keys | 0.050 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 236 calls, 108.558 ms; top: cuMemAllocAsync 76.601 ms x52, cuStreamSynchronize 31.634 ms x1, cudaLaunchKernel_v7000 0.234 ms x63, cuMemFreeAsync 0.036 ms x50, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 32.590 ms, GPU busy 32.486 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.749 | 51.6% |
| sparse_attn | 4.361 | 13.4% |
| wo_a | 2.511 | 7.7% |
| q_b | 1.692 | 5.2% |
| shared_expert | 1.666 | 5.1% |
| wo_b | 1.614 | 5.0% |
| hc_ffn_pre | 1.185 | 3.6% |
| hc_attn_pre | 0.900 | 2.8% |
| router | 0.731 | 2.3% |
| attn_keys/qr | 0.370 | 1.1% |
| hc_post | 0.237 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.7% |
| combine | 0.206 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.448 ms; top: cuStreamSynchronize 28.867 ms x1, cuMemAllocAsync 3.332 ms x46, cudaLaunchKernel_v7000 0.172 ms x47, cuMemFreeAsync 0.030 ms x44, cuMemcpyDtoDAsync_v2 0.020 ms x3, cuMemsetD8Async 0.018 ms x3

### L2/decode (6 instances): wall 0.660 ms, GPU busy 0.588 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 26.8% |
| wo_a | 0.049 | 8.4% |
| attn_keys/compressor | 0.049 | 8.3% |
| wo_b | 0.045 | 7.7% |
| shared_expert | 0.045 | 7.6% |
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
| attn_keys/index_keys | 0.005 | 0.9% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.569 ms; top: cuStreamSynchronize 0.321 ms x1, cudaLaunchKernel_v7000 0.163 ms x60, cuMemAllocAsync 0.032 ms x52, cuMemFreeAsync 0.023 ms x51, cuMemcpyDtoDAsync_v2 0.019 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.612 ms, GPU busy 0.572 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 27.5% |
| wo_a | 0.048 | 8.5% |
| shared_expert | 0.045 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.5% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.023 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 2.9% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.6% |
| combine | 0.004 | 0.8% |
| attn_keys/candidates | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.543 ms; top: cuStreamSynchronize 0.324 ms x1, cudaLaunchKernel_v7000 0.146 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.022 ms x50, cuMemcpyDtoDAsync_v2 0.014 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.522 ms, GPU busy 0.483 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 32.4% |
| wo_a | 0.049 | 10.1% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.044 | 9.2% |
| sparse_attn | 0.036 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.029 | 5.9% |
| hc_ffn_pre | 0.020 | 4.2% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 3.0% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.5% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.469 ms; top: cuStreamSynchronize 0.300 ms x1, cudaLaunchKernel_v7000 0.109 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.018 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.004 ms x3

