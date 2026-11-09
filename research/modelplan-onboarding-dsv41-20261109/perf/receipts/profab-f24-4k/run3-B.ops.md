### L2/prefill (2 instances): wall 36.035 ms, GPU busy 35.948 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.825 | 49.6% |
| sparse_attn | 4.258 | 11.8% |
| wo_a | 2.522 | 7.0% |
| q_b | 1.713 | 4.8% |
| shared_expert | 1.659 | 4.6% |
| wo_b | 1.620 | 4.5% |
| attn_keys/compressor | 1.537 | 4.3% |
| hc_ffn_pre | 1.190 | 3.3% |
| hc_attn_pre | 0.898 | 2.5% |
| router | 0.733 | 2.0% |
| attn_keys/index_q | 0.610 | 1.7% |
| attn_keys/qr | 0.372 | 1.0% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.6% |
| combine | 0.208 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.062 | 0.2% |
| attn_keys/index_keys | 0.040 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.871 ms; top: cuStreamSynchronize 35.197 ms x1, cuMemAllocAsync 0.382 ms x53, cudaLaunchKernel_v7000 0.211 ms x64, cuMemFreeAsync 0.029 ms x51, cuMemcpyDtoDAsync_v2 0.024 ms x4, cuMemsetD8Async 0.018 ms x3

### L20/prefill (2 instances): wall 33.987 ms, GPU busy 33.858 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.639 | 49.1% |
| sparse_attn | 4.337 | 12.8% |
| wo_a | 2.520 | 7.4% |
| q_b | 1.723 | 5.1% |
| shared_expert | 1.670 | 4.9% |
| wo_b | 1.625 | 4.8% |
| hc_ffn_pre | 1.201 | 3.5% |
| hc_attn_pre | 0.902 | 2.7% |
| router | 0.740 | 2.2% |
| attn_keys/index_q | 0.605 | 1.8% |
| attn_keys/index_score | 0.397 | 1.2% |
| attn_keys/qr | 0.373 | 1.1% |
| hc_post | 0.239 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| attn_keys/compressor | 0.212 | 0.6% |
| combine | 0.207 | 0.6% |
| attn_keys/topk | 0.089 | 0.3% |
| attn_keys/index_keys | 0.049 | 0.1% |
| attn_keys/kv_append | 0.039 | 0.1% |
| attn_keys/candidates | 0.031 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.825 ms; top: cuStreamSynchronize 33.043 ms x1, cuMemAllocAsync 0.483 ms x52, cudaLaunchKernel_v7000 0.210 ms x63, cuMemFreeAsync 0.033 ms x50, cuMemcpyDtoDAsync_v2 0.027 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 32.824 ms, GPU busy 32.715 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.884 | 51.6% |
| sparse_attn | 4.381 | 13.4% |
| wo_a | 2.523 | 7.7% |
| q_b | 1.706 | 5.2% |
| shared_expert | 1.675 | 5.1% |
| wo_b | 1.625 | 5.0% |
| hc_ffn_pre | 1.196 | 3.7% |
| hc_attn_pre | 0.905 | 2.8% |
| router | 0.737 | 2.3% |
| attn_keys/qr | 0.372 | 1.1% |
| hc_post | 0.241 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.7% |
| combine | 0.207 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.685 ms; top: cuStreamSynchronize 32.199 ms x1, cuMemAllocAsync 0.240 ms x46, cudaLaunchKernel_v7000 0.161 ms x47, cuMemFreeAsync 0.030 ms x44, cuMemcpyDtoDAsync_v2 0.028 ms x3, cuMemsetD8Async 0.019 ms x3

### L2/decode (6 instances): wall 0.656 ms, GPU busy 0.588 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 26.9% |
| wo_a | 0.049 | 8.3% |
| attn_keys/compressor | 0.049 | 8.3% |
| shared_expert | 0.045 | 7.6% |
| wo_b | 0.044 | 7.5% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.8% |
| q_b | 0.032 | 5.4% |
| router | 0.029 | 4.9% |
| hc_ffn_pre | 0.021 | 3.5% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/kv_append | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.568 ms; top: cuStreamSynchronize 0.335 ms x1, cudaLaunchKernel_v7000 0.154 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.609 ms, GPU busy 0.573 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 27.5% |
| wo_a | 0.049 | 8.5% |
| shared_expert | 0.045 | 7.8% |
| wo_b | 0.044 | 7.7% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.5% |
| router | 0.029 | 5.0% |
| attn_keys/compressor | 0.024 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.014 | 2.4% |
| attn_keys/topk | 0.012 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.011 | 1.9% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.8% |
| attn_keys/index_score | 0.003 | 0.6% |
| attn_keys/candidates | 0.003 | 0.5% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.546 ms; top: cuStreamSynchronize 0.342 ms x1, cudaLaunchKernel_v7000 0.136 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.519 ms, GPU busy 0.486 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 32.5% |
| wo_a | 0.049 | 10.0% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.044 | 9.1% |
| sparse_attn | 0.037 | 7.6% |
| q_b | 0.032 | 6.6% |
| router | 0.029 | 5.9% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 2.9% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.5% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.471 ms; top: cuStreamSynchronize 0.315 ms x1, cudaLaunchKernel_v7000 0.102 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

