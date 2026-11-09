### L2/prefill (2 instances): wall 38.251 ms, GPU busy 38.163 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.473 | 48.4% |
| sparse_attn | 4.613 | 12.1% |
| wo_a | 2.923 | 7.7% |
| q_b | 1.831 | 4.8% |
| shared_expert | 1.776 | 4.7% |
| wo_b | 1.722 | 4.5% |
| attn_keys/compressor | 1.526 | 4.0% |
| hc_ffn_pre | 1.173 | 3.1% |
| hc_attn_pre | 0.890 | 2.3% |
| router | 0.762 | 2.0% |
| attn_keys/index_q | 0.587 | 1.5% |
| moe_plan | 0.396 | 1.0% |
| attn_keys/qr | 0.381 | 1.0% |
| attn_keys/index_score | 0.311 | 0.8% |
| hc_post | 0.238 | 0.6% |
| attn_keys/kv_window | 0.231 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.062 | 0.2% |
| attn_keys/index_keys | 0.030 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 38.092 ms; top: cuStreamSynchronize 37.453 ms x1, cuMemAllocAsync 0.362 ms x53, cudaLaunchKernel_v7000 0.201 ms x64, cuMemFreeAsync 0.029 ms x51, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.014 ms x3

### L20/prefill (2 instances): wall 36.113 ms, GPU busy 36.017 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.165 | 47.7% |
| sparse_attn | 4.648 | 12.9% |
| wo_a | 2.911 | 8.1% |
| q_b | 1.842 | 5.1% |
| shared_expert | 1.780 | 4.9% |
| wo_b | 1.724 | 4.8% |
| hc_ffn_pre | 1.174 | 3.3% |
| hc_attn_pre | 0.895 | 2.5% |
| router | 0.763 | 2.1% |
| attn_keys/index_score | 0.644 | 1.8% |
| attn_keys/index_q | 0.587 | 1.6% |
| moe_plan | 0.406 | 1.1% |
| attn_keys/qr | 0.380 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.232 | 0.6% |
| attn_keys/compressor | 0.216 | 0.6% |
| combine | 0.204 | 0.6% |
| attn_keys/topk | 0.088 | 0.2% |
| attn_keys/candidates | 0.040 | 0.1% |
| attn_keys/kv_append | 0.036 | 0.1% |
| attn_keys/index_keys | 0.034 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 35.959 ms; top: cuStreamSynchronize 35.218 ms x1, cuMemAllocAsync 0.474 ms x52, cudaLaunchKernel_v7000 0.192 ms x63, cuMemFreeAsync 0.031 ms x50, cuMemcpyDtoDAsync_v2 0.019 ms x4, cuMemsetD8Async 0.014 ms x3

### L21/prefill (2 instances): wall 34.822 ms, GPU busy 34.705 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.445 | 50.3% |
| sparse_attn | 4.686 | 13.5% |
| wo_a | 2.936 | 8.5% |
| q_b | 1.838 | 5.3% |
| shared_expert | 1.790 | 5.2% |
| wo_b | 1.715 | 4.9% |
| hc_ffn_pre | 1.167 | 3.4% |
| hc_attn_pre | 0.891 | 2.6% |
| router | 0.763 | 2.2% |
| moe_plan | 0.409 | 1.2% |
| attn_keys/qr | 0.377 | 1.1% |
| hc_post | 0.238 | 0.7% |
| attn_keys/kv_window | 0.231 | 0.7% |
| combine | 0.205 | 0.6% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 34.670 ms; top: cuStreamSynchronize 34.166 ms x1, cuMemAllocAsync 0.241 ms x46, cudaLaunchKernel_v7000 0.177 ms x47, cuMemFreeAsync 0.036 ms x44, cuMemsetD8Async 0.021 ms x3, cuMemcpyDtoDAsync_v2 0.021 ms x3

### L2/decode (6 instances): wall 0.690 ms, GPU busy 0.626 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 25.3% |
| wo_b | 0.062 | 9.9% |
| wo_a | 0.049 | 7.9% |
| attn_keys/compressor | 0.048 | 7.7% |
| shared_expert | 0.044 | 7.1% |
| sparse_attn | 0.040 | 6.4% |
| router | 0.037 | 5.9% |
| attn_keys/index_q | 0.034 | 5.4% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.3% |
| hc_attn_pre | 0.023 | 3.7% |
| attn_keys/kv_window | 0.013 | 2.1% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/kv_append | 0.006 | 0.9% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.605 ms; top: cuStreamSynchronize 0.375 ms x1, cudaLaunchKernel_v7000 0.152 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.653 ms, GPU busy 0.614 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 25.7% |
| wo_b | 0.065 | 10.6% |
| wo_a | 0.049 | 8.0% |
| shared_expert | 0.044 | 7.2% |
| sparse_attn | 0.040 | 6.5% |
| router | 0.037 | 6.0% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.1% |
| hc_ffn_pre | 0.027 | 4.3% |
| hc_attn_pre | 0.023 | 3.8% |
| attn_keys/compressor | 0.022 | 3.6% |
| attn_keys/kv_window | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/qr | 0.012 | 2.0% |
| attn_keys/topk | 0.012 | 2.0% |
| attn_keys/index_keys | 0.010 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.004 | 0.6% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.585 ms; top: cuStreamSynchronize 0.378 ms x1, cudaLaunchKernel_v7000 0.138 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.014 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.558 ms, GPU busy 0.527 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 29.9% |
| wo_b | 0.065 | 12.4% |
| wo_a | 0.048 | 9.1% |
| shared_expert | 0.044 | 8.3% |
| sparse_attn | 0.041 | 7.8% |
| router | 0.037 | 7.0% |
| q_b | 0.032 | 6.0% |
| hc_ffn_pre | 0.026 | 5.0% |
| hc_attn_pre | 0.024 | 4.5% |
| attn_keys/kv_window | 0.013 | 2.5% |
| attn_keys/qr | 0.013 | 2.4% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.8% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.509 ms; top: cuStreamSynchronize 0.354 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

