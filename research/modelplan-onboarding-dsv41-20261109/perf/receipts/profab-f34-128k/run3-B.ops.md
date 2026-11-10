### L2/prefill (2 instances): wall 1572.687 ms, GPU busy 1572.276 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 613.278 | 39.0% |
| attn_keys/index_score | 270.960 | 17.2% |
| sparse_attn | 157.989 | 10.0% |
| wo_a | 101.520 | 6.5% |
| q_b | 61.062 | 3.9% |
| wo_b | 58.226 | 3.7% |
| shared_expert | 56.848 | 3.6% |
| attn_keys/compressor | 47.111 | 3.0% |
| hc_ffn_pre | 45.739 | 2.9% |
| attn_keys/topk | 44.960 | 2.9% |
| hc_attn_pre | 32.424 | 2.1% |
| router | 24.838 | 1.6% |
| attn_keys/index_q | 18.698 | 1.2% |
| attn_keys/qr | 12.103 | 0.8% |
| hc_post | 9.052 | 0.6% |
| combine | 8.051 | 0.5% |
| attn_keys/kv_window | 7.654 | 0.5% |
| attn_keys/kv_append | 0.511 | 0.0% |
| attn_keys/index_keys | 0.503 | 0.0% |
| attn_idx_rows | 0.403 | 0.0% |
| moe_plan | 0.346 | 0.0% |

host API calls: 2175 calls, 1571.869 ms; top: cuStreamSynchronize 1562.915 ms x1, cuMemAllocAsync 7.206 ms x458, cudaLaunchKernel_v7000 1.363 ms x604, cuMemFreeAsync 0.185 ms x456, cuMemsetD8Async 0.109 ms x48, cuKernelGetName 0.056 ms x604

### L20/prefill (2 instances): wall 2134.512 ms, GPU busy 2134.078 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 587.366 | 27.5% |
| attn_keys/index_score | 587.203 | 27.5% |
| attn_keys/topk | 255.749 | 12.0% |
| sparse_attn | 159.336 | 7.5% |
| wo_a | 102.250 | 4.8% |
| attn_keys/candidates | 97.332 | 4.6% |
| q_b | 61.334 | 2.9% |
| wo_b | 58.254 | 2.7% |
| shared_expert | 56.954 | 2.7% |
| hc_ffn_pre | 45.788 | 2.1% |
| hc_attn_pre | 32.881 | 1.5% |
| router | 24.937 | 1.2% |
| attn_keys/index_q | 18.736 | 0.9% |
| attn_keys/qr | 12.116 | 0.6% |
| hc_post | 9.052 | 0.4% |
| combine | 8.052 | 0.4% |
| attn_keys/kv_window | 7.668 | 0.4% |
| attn_keys/compressor | 6.092 | 0.3% |
| attn_keys/kv_append | 1.232 | 0.1% |
| attn_keys/index_keys | 0.988 | 0.0% |
| attn_idx_rows | 0.417 | 0.0% |
| moe_plan | 0.344 | 0.0% |

host API calls: 2201 calls, 2133.671 ms; top: cuStreamSynchronize 2115.575 ms x1, cuMemAllocAsync 16.289 ms x457, cudaLaunchKernel_v7000 1.399 ms x618, cuMemFreeAsync 0.208 ms x455, cuMemsetD8Async 0.103 ms x48, cuKernelGetName 0.057 ms x618

### L21/prefill (2 instances): wall 1345.908 ms, GPU busy 1152.571 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 599.520 | 52.0% |
| sparse_attn | 150.958 | 13.1% |
| wo_a | 91.119 | 7.9% |
| q_b | 58.938 | 5.1% |
| shared_expert | 56.698 | 4.9% |
| wo_b | 55.233 | 4.8% |
| hc_ffn_pre | 45.427 | 3.9% |
| hc_attn_pre | 32.533 | 2.8% |
| router | 24.578 | 2.1% |
| attn_keys/qr | 12.035 | 1.0% |
| hc_post | 9.013 | 0.8% |
| combine | 7.943 | 0.7% |
| attn_keys/kv_window | 7.619 | 0.7% |
| attn_idx_rows | 0.433 | 0.0% |
| moe_plan | 0.345 | 0.0% |
| attn_keys/kv_append | 0.179 | 0.0% |

host API calls: 1826 calls, 1345.227 ms; top: cuStreamSynchronize 1120.841 ms x1, cuMemAllocAsync 222.942 ms x406, cudaLaunchKernel_v7000 1.092 ms x482, cuMemFreeAsync 0.176 ms x404, cuMemsetD8Async 0.104 ms x48, cuKernelGetName 0.044 ms x482

### L2/decode (6 instances): wall 0.874 ms, GPU busy 0.768 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.164 | 21.3% |
| attn_keys/topk | 0.118 | 15.4% |
| wo_a | 0.051 | 6.6% |
| attn_keys/kv_append | 0.050 | 6.5% |
| attn_keys/compressor | 0.049 | 6.4% |
| wo_b | 0.046 | 6.0% |
| shared_expert | 0.044 | 5.8% |
| q_b | 0.037 | 4.9% |
| sparse_attn | 0.036 | 4.7% |
| attn_keys/index_q | 0.034 | 4.4% |
| router | 0.030 | 3.9% |
| hc_ffn_pre | 0.021 | 2.7% |
| attn_keys/index_score | 0.018 | 2.3% |
| hc_attn_pre | 0.017 | 2.2% |
| attn_keys/kv_window | 0.014 | 1.8% |
| attn_keys/qr | 0.013 | 1.6% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/index_keys | 0.005 | 0.7% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.778 ms; top: cuStreamSynchronize 0.434 ms x1, cudaLaunchKernel_v7000 0.265 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.022 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.007 ms x60

### L20/decode (6 instances): wall 1.091 ms, GPU busy 1.055 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.225 | 21.4% |
| experts | 0.169 | 16.0% |
| attn_keys/kv_append | 0.123 | 11.6% |
| attn_keys/candidates | 0.105 | 10.0% |
| wo_a | 0.050 | 4.7% |
| wo_b | 0.044 | 4.2% |
| shared_expert | 0.044 | 4.2% |
| q_b | 0.043 | 4.1% |
| sparse_attn | 0.037 | 3.5% |
| attn_keys/index_q | 0.034 | 3.2% |
| router | 0.030 | 2.9% |
| attn_keys/index_score | 0.030 | 2.8% |
| attn_keys/compressor | 0.024 | 2.2% |
| hc_ffn_pre | 0.021 | 2.0% |
| hc_attn_pre | 0.017 | 1.6% |
| attn_keys/kv_window | 0.014 | 1.3% |
| attn_keys/qr | 0.012 | 1.2% |
| moe_plan | 0.012 | 1.2% |
| attn_keys/index_keys | 0.011 | 1.0% |
| combine | 0.005 | 0.4% |
| hc_post | 0.003 | 0.3% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.026 ms; top: cuStreamSynchronize 0.830 ms x1, cudaLaunchKernel_v7000 0.132 ms x63, cuMemAllocAsync 0.023 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.662 ms, GPU busy 0.631 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.169 | 26.7% |
| attn_keys/kv_append | 0.120 | 19.0% |
| wo_a | 0.051 | 8.1% |
| wo_b | 0.045 | 7.1% |
| shared_expert | 0.044 | 7.0% |
| q_b | 0.044 | 6.9% |
| sparse_attn | 0.039 | 6.2% |
| router | 0.031 | 4.9% |
| hc_ffn_pre | 0.022 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.2% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| combine | 0.005 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 191 calls, 0.612 ms; top: cuStreamSynchronize 0.459 ms x1, cudaLaunchKernel_v7000 0.098 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

