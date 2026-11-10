### L2/prefill (2 instances): wall 1545.940 ms, GPU busy 1545.530 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 602.485 | 39.0% |
| attn_keys/index_score | 266.339 | 17.2% |
| sparse_attn | 155.643 | 10.1% |
| wo_a | 98.535 | 6.4% |
| q_b | 60.203 | 3.9% |
| wo_b | 57.299 | 3.7% |
| shared_expert | 56.186 | 3.6% |
| attn_keys/compressor | 46.512 | 3.0% |
| hc_ffn_pre | 44.971 | 2.9% |
| attn_keys/topk | 44.323 | 2.9% |
| hc_attn_pre | 32.194 | 2.1% |
| router | 24.216 | 1.6% |
| attn_keys/index_q | 18.459 | 1.2% |
| attn_keys/qr | 11.988 | 0.8% |
| hc_post | 9.017 | 0.6% |
| combine | 7.841 | 0.5% |
| attn_keys/kv_window | 7.560 | 0.5% |
| attn_keys/kv_append | 0.503 | 0.0% |
| attn_keys/index_keys | 0.499 | 0.0% |
| attn_idx_rows | 0.414 | 0.0% |
| moe_plan | 0.340 | 0.0% |

host API calls: 2175 calls, 1545.095 ms; top: cuStreamSynchronize 1536.411 ms x1, cuMemAllocAsync 6.914 ms x458, cudaLaunchKernel_v7000 1.372 ms x604, cuMemFreeAsync 0.193 ms x456, cuMemsetD8Async 0.107 ms x48, cuKernelGetName 0.057 ms x604

### L20/prefill (2 instances): wall 2100.945 ms, GPU busy 2100.515 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 578.902 | 27.6% |
| experts | 578.008 | 27.5% |
| attn_keys/topk | 250.134 | 11.9% |
| sparse_attn | 157.255 | 7.5% |
| wo_a | 99.596 | 4.7% |
| attn_keys/candidates | 96.691 | 4.6% |
| q_b | 60.596 | 2.9% |
| wo_b | 57.518 | 2.7% |
| shared_expert | 56.257 | 2.7% |
| hc_ffn_pre | 45.227 | 2.2% |
| hc_attn_pre | 32.217 | 1.5% |
| router | 24.397 | 1.2% |
| attn_keys/index_q | 18.519 | 0.9% |
| attn_keys/qr | 11.969 | 0.6% |
| hc_post | 8.944 | 0.4% |
| combine | 7.841 | 0.4% |
| attn_keys/kv_window | 7.575 | 0.4% |
| attn_keys/compressor | 5.937 | 0.3% |
| attn_keys/kv_append | 1.188 | 0.1% |
| attn_keys/index_keys | 0.978 | 0.0% |
| attn_idx_rows | 0.426 | 0.0% |
| moe_plan | 0.340 | 0.0% |

host API calls: 2201 calls, 2100.087 ms; top: cuStreamSynchronize 2085.851 ms x1, cuMemAllocAsync 12.441 ms x457, cudaLaunchKernel_v7000 1.386 ms x618, cuMemFreeAsync 0.207 ms x455, cuMemsetD8Async 0.104 ms x48, cuKernelGetName 0.057 ms x618

### L21/prefill (2 instances): wall 1157.122 ms, GPU busy 1156.748 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 600.188 | 51.9% |
| sparse_attn | 151.994 | 13.1% |
| wo_a | 92.780 | 8.0% |
| q_b | 59.367 | 5.1% |
| shared_expert | 56.632 | 4.9% |
| wo_b | 55.719 | 4.8% |
| hc_ffn_pre | 45.393 | 3.9% |
| hc_attn_pre | 32.500 | 2.8% |
| router | 24.587 | 2.1% |
| attn_keys/qr | 12.053 | 1.0% |
| hc_post | 9.006 | 0.8% |
| combine | 7.942 | 0.7% |
| attn_keys/kv_window | 7.630 | 0.7% |
| attn_idx_rows | 0.432 | 0.0% |
| moe_plan | 0.344 | 0.0% |
| attn_keys/kv_append | 0.179 | 0.0% |

host API calls: 1826 calls, 1156.444 ms; top: cuStreamSynchronize 1150.031 ms x1, cuMemAllocAsync 4.986 ms x406, cudaLaunchKernel_v7000 1.075 ms x482, cuMemFreeAsync 0.180 ms x404, cuMemsetD8Async 0.103 ms x48, cuKernelGetName 0.044 ms x482

### L2/decode (6 instances): wall 0.961 ms, GPU busy 0.772 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.166 | 21.5% |
| attn_keys/topk | 0.118 | 15.3% |
| wo_a | 0.050 | 6.5% |
| attn_keys/kv_append | 0.050 | 6.5% |
| attn_keys/compressor | 0.049 | 6.4% |
| wo_b | 0.046 | 6.0% |
| shared_expert | 0.045 | 5.8% |
| q_b | 0.037 | 4.8% |
| sparse_attn | 0.036 | 4.7% |
| attn_keys/index_q | 0.034 | 4.4% |
| router | 0.030 | 3.9% |
| hc_ffn_pre | 0.021 | 2.7% |
| attn_keys/index_score | 0.017 | 2.3% |
| hc_attn_pre | 0.017 | 2.2% |
| attn_keys/kv_window | 0.014 | 1.8% |
| attn_keys/qr | 0.013 | 1.7% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/index_keys | 0.006 | 0.7% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.845 ms; top: cuStreamSynchronize 0.455 ms x1, cudaLaunchKernel_v7000 0.171 ms x60, cuMemcpyDtoDAsync_v2 0.150 ms x6, cuMemAllocAsync 0.032 ms x52, cuMemFreeAsync 0.022 ms x51, cuKernelGetName 0.008 ms x60

### L20/decode (6 instances): wall 1.097 ms, GPU busy 1.060 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.226 | 21.3% |
| experts | 0.170 | 16.1% |
| attn_keys/kv_append | 0.124 | 11.7% |
| attn_keys/candidates | 0.106 | 10.0% |
| wo_a | 0.050 | 4.7% |
| wo_b | 0.045 | 4.2% |
| shared_expert | 0.044 | 4.2% |
| q_b | 0.043 | 4.1% |
| sparse_attn | 0.037 | 3.5% |
| attn_keys/index_q | 0.034 | 3.2% |
| router | 0.030 | 2.8% |
| attn_keys/index_score | 0.030 | 2.8% |
| attn_keys/compressor | 0.024 | 2.2% |
| hc_ffn_pre | 0.022 | 2.0% |
| hc_attn_pre | 0.017 | 1.6% |
| attn_keys/kv_window | 0.014 | 1.3% |
| attn_keys/qr | 0.012 | 1.2% |
| moe_plan | 0.012 | 1.1% |
| attn_keys/index_keys | 0.011 | 1.0% |
| combine | 0.005 | 0.4% |
| hc_post | 0.003 | 0.3% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.028 ms; top: cuStreamSynchronize 0.828 ms x1, cudaLaunchKernel_v7000 0.135 ms x63, cuMemAllocAsync 0.024 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.663 ms, GPU busy 0.631 ms

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
| hc_ffn_pre | 0.022 | 3.5% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.2% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| combine | 0.005 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 191 calls, 0.612 ms; top: cuStreamSynchronize 0.455 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.022 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

