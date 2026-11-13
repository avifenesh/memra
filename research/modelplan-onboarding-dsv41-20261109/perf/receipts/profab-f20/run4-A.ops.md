### L2/prefill (2 instances): wall 37.921 ms, GPU busy 37.836 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.571 | 49.1% |
| sparse_attn | 4.644 | 12.3% |
| wo_a | 2.700 | 7.1% |
| q_b | 1.837 | 4.9% |
| shared_expert | 1.787 | 4.7% |
| wo_b | 1.748 | 4.6% |
| attn_keys/compressor | 1.541 | 4.1% |
| hc_ffn_pre | 1.192 | 3.1% |
| hc_attn_pre | 0.900 | 2.4% |
| router | 0.737 | 1.9% |
| attn_keys/index_q | 0.634 | 1.7% |
| attn_keys/qr | 0.397 | 1.0% |
| attn_keys/index_score | 0.307 | 0.8% |
| attn_keys/kv_window | 0.244 | 0.6% |
| hc_post | 0.239 | 0.6% |
| combine | 0.209 | 0.6% |
| attn_keys/topk | 0.061 | 0.2% |
| attn_keys/index_keys | 0.042 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 37.759 ms; top: cuStreamSynchronize 37.099 ms x1, cuMemAllocAsync 0.373 ms x53, cudaLaunchKernel_v7000 0.209 ms x64, cuMemFreeAsync 0.029 ms x51, cuMemcpyDtoDAsync_v2 0.022 ms x4, cuMemsetD8Async 0.018 ms x3

### L20/prefill (2 instances): wall 35.977 ms, GPU busy 35.849 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.326 | 48.3% |
| sparse_attn | 4.731 | 13.2% |
| wo_a | 2.715 | 7.6% |
| q_b | 1.853 | 5.2% |
| shared_expert | 1.796 | 5.0% |
| wo_b | 1.752 | 4.9% |
| hc_ffn_pre | 1.200 | 3.3% |
| hc_attn_pre | 0.902 | 2.5% |
| router | 0.743 | 2.1% |
| attn_keys/index_score | 0.641 | 1.8% |
| attn_keys/index_q | 0.629 | 1.8% |
| attn_keys/qr | 0.399 | 1.1% |
| attn_keys/compressor | 0.247 | 0.7% |
| attn_keys/kv_window | 0.245 | 0.7% |
| hc_post | 0.238 | 0.7% |
| combine | 0.206 | 0.6% |
| attn_keys/topk | 0.083 | 0.2% |
| attn_keys/index_keys | 0.052 | 0.1% |
| attn_keys/kv_append | 0.037 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 35.810 ms; top: cuStreamSynchronize 35.022 ms x1, cuMemAllocAsync 0.485 ms x52, cudaLaunchKernel_v7000 0.218 ms x63, cuMemFreeAsync 0.033 ms x50, cuMemcpyDtoDAsync_v2 0.024 ms x4, cuMemsetD8Async 0.018 ms x3

### L21/prefill (2 instances): wall 34.543 ms, GPU busy 34.449 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.604 | 51.1% |
| sparse_attn | 4.763 | 13.8% |
| wo_a | 2.717 | 7.9% |
| q_b | 1.843 | 5.3% |
| shared_expert | 1.806 | 5.2% |
| wo_b | 1.755 | 5.1% |
| hc_ffn_pre | 1.196 | 3.5% |
| hc_attn_pre | 0.907 | 2.6% |
| router | 0.742 | 2.2% |
| attn_keys/qr | 0.395 | 1.1% |
| attn_keys/kv_window | 0.243 | 0.7% |
| hc_post | 0.240 | 0.7% |
| combine | 0.207 | 0.6% |
| moe_plan | 0.017 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 34.411 ms; top: cuStreamSynchronize 33.946 ms x1, cuMemAllocAsync 0.232 ms x46, cudaLaunchKernel_v7000 0.163 ms x47, cuMemFreeAsync 0.027 ms x44, cuMemcpyDtoDAsync_v2 0.019 ms x3, cuMemsetD8Async 0.016 ms x3

### L2/decode (6 instances): wall 0.675 ms, GPU busy 0.613 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.3% |
| wo_b | 0.063 | 10.2% |
| wo_a | 0.049 | 8.0% |
| attn_keys/compressor | 0.049 | 7.9% |
| shared_expert | 0.045 | 7.3% |
| sparse_attn | 0.040 | 6.6% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| hc_ffn_pre | 0.020 | 3.2% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.2% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.593 ms; top: cuStreamSynchronize 0.369 ms x1, cudaLaunchKernel_v7000 0.148 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.019 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.636 ms, GPU busy 0.600 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 26.7% |
| wo_b | 0.065 | 10.9% |
| wo_a | 0.048 | 8.1% |
| shared_expert | 0.044 | 7.4% |
| sparse_attn | 0.040 | 6.7% |
| attn_keys/index_q | 0.034 | 5.6% |
| q_b | 0.032 | 5.3% |
| router | 0.029 | 4.8% |
| attn_keys/compressor | 0.023 | 3.9% |
| hc_ffn_pre | 0.020 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.014 | 2.3% |
| attn_keys/topk | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_keys | 0.011 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.4% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.575 ms; top: cuStreamSynchronize 0.377 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.546 ms, GPU busy 0.514 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 31.2% |
| wo_b | 0.066 | 12.8% |
| wo_a | 0.048 | 9.4% |
| shared_expert | 0.044 | 8.7% |
| sparse_attn | 0.042 | 8.2% |
| q_b | 0.032 | 6.2% |
| router | 0.029 | 5.6% |
| hc_ffn_pre | 0.020 | 3.9% |
| hc_attn_pre | 0.017 | 3.3% |
| attn_keys/kv_window | 0.015 | 2.9% |
| attn_keys/qr | 0.014 | 2.7% |
| moe_plan | 0.012 | 2.3% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.499 ms; top: cuStreamSynchronize 0.347 ms x1, cudaLaunchKernel_v7000 0.099 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

