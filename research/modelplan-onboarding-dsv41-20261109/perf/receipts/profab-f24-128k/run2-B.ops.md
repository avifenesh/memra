### L2/prefill (2 instances): wall 1614.021 ms, GPU busy 1613.618 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 604.080 | 37.4% |
| attn_keys/index_score | 268.741 | 16.7% |
| sparse_attn | 156.389 | 9.7% |
| attn_keys/topk | 105.122 | 6.5% |
| wo_a | 99.752 | 6.2% |
| q_b | 60.433 | 3.7% |
| wo_b | 57.632 | 3.6% |
| shared_expert | 56.300 | 3.5% |
| attn_keys/compressor | 46.678 | 2.9% |
| hc_ffn_pre | 45.257 | 2.8% |
| hc_attn_pre | 32.265 | 2.0% |
| router | 24.369 | 1.5% |
| attn_keys/index_q | 18.380 | 1.1% |
| attn_keys/qr | 12.000 | 0.7% |
| hc_post | 8.997 | 0.6% |
| combine | 7.891 | 0.5% |
| attn_keys/kv_window | 7.583 | 0.5% |
| attn_keys/index_keys | 0.499 | 0.0% |
| attn_keys/kv_append | 0.496 | 0.0% |
| attn_idx_rows | 0.409 | 0.0% |
| moe_plan | 0.345 | 0.0% |

host API calls: 2175 calls, 1613.219 ms; top: cuStreamSynchronize 1604.831 ms x1, cuMemAllocAsync 6.655 ms x458, cudaLaunchKernel_v7000 1.349 ms x604, cuMemFreeAsync 0.185 ms x456, cuMemsetD8Async 0.107 ms x48, cuKernelGetName 0.056 ms x604

### L20/prefill (2 instances): wall 2220.653 ms, GPU busy 2220.187 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 588.877 | 26.5% |
| experts | 588.322 | 26.5% |
| attn_keys/topk | 257.011 | 11.6% |
| attn_keys/candidates | 178.644 | 8.0% |
| sparse_attn | 159.694 | 7.2% |
| wo_a | 102.660 | 4.6% |
| q_b | 61.440 | 2.8% |
| wo_b | 58.410 | 2.6% |
| shared_expert | 57.055 | 2.6% |
| hc_ffn_pre | 45.709 | 2.1% |
| hc_attn_pre | 32.734 | 1.5% |
| router | 24.924 | 1.1% |
| attn_keys/index_q | 18.796 | 0.8% |
| attn_keys/qr | 12.113 | 0.5% |
| hc_post | 9.027 | 0.4% |
| combine | 8.061 | 0.4% |
| attn_keys/kv_window | 7.667 | 0.3% |
| attn_keys/compressor | 6.093 | 0.3% |
| attn_keys/kv_append | 1.197 | 0.1% |
| attn_keys/index_keys | 0.991 | 0.0% |
| attn_idx_rows | 0.416 | 0.0% |
| moe_plan | 0.346 | 0.0% |

host API calls: 2201 calls, 2219.815 ms; top: cuStreamSynchronize 2207.842 ms x1, cuMemAllocAsync 10.150 ms x457, cudaLaunchKernel_v7000 1.409 ms x618, cuMemFreeAsync 0.208 ms x455, cuMemsetD8Async 0.104 ms x48, cuKernelGetName 0.057 ms x618

### L21/prefill (2 instances): wall 1160.822 ms, GPU busy 1160.421 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 601.818 | 51.9% |
| sparse_attn | 152.497 | 13.1% |
| wo_a | 93.391 | 8.0% |
| q_b | 59.577 | 5.1% |
| shared_expert | 56.822 | 4.9% |
| wo_b | 55.864 | 4.8% |
| hc_ffn_pre | 45.536 | 3.9% |
| hc_attn_pre | 32.536 | 2.8% |
| router | 24.723 | 2.1% |
| attn_keys/qr | 12.075 | 1.0% |
| hc_post | 8.970 | 0.8% |
| combine | 8.010 | 0.7% |
| attn_keys/kv_window | 7.647 | 0.7% |
| attn_idx_rows | 0.430 | 0.0% |
| moe_plan | 0.345 | 0.0% |
| attn_keys/kv_append | 0.180 | 0.0% |

host API calls: 1826 calls, 1160.143 ms; top: cuStreamSynchronize 1153.964 ms x1, cuMemAllocAsync 4.744 ms x406, cudaLaunchKernel_v7000 1.080 ms x482, cuMemFreeAsync 0.174 ms x404, cuMemsetD8Async 0.106 ms x48, cuKernelGetName 0.046 ms x482

### L2/decode (6 instances): wall 0.893 ms, GPU busy 0.769 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.165 | 21.4% |
| attn_keys/topk | 0.118 | 15.3% |
| wo_a | 0.050 | 6.6% |
| attn_keys/kv_append | 0.050 | 6.5% |
| attn_keys/compressor | 0.049 | 6.4% |
| wo_b | 0.046 | 6.0% |
| shared_expert | 0.044 | 5.8% |
| q_b | 0.037 | 4.8% |
| sparse_attn | 0.036 | 4.7% |
| attn_keys/index_q | 0.034 | 4.4% |
| router | 0.030 | 3.9% |
| hc_ffn_pre | 0.021 | 2.8% |
| attn_keys/index_score | 0.017 | 2.3% |
| hc_attn_pre | 0.017 | 2.2% |
| attn_keys/kv_window | 0.014 | 1.8% |
| attn_keys/qr | 0.013 | 1.6% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/index_keys | 0.005 | 0.7% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.788 ms; top: cuStreamSynchronize 0.553 ms x1, cudaLaunchKernel_v7000 0.156 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.125 ms, GPU busy 1.088 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.225 | 20.7% |
| experts | 0.169 | 15.6% |
| attn_keys/candidates | 0.135 | 12.4% |
| attn_keys/kv_append | 0.124 | 11.4% |
| wo_a | 0.050 | 4.6% |
| wo_b | 0.044 | 4.1% |
| shared_expert | 0.044 | 4.0% |
| q_b | 0.044 | 4.0% |
| sparse_attn | 0.037 | 3.4% |
| attn_keys/index_q | 0.034 | 3.1% |
| router | 0.030 | 2.8% |
| attn_keys/index_score | 0.030 | 2.7% |
| attn_keys/compressor | 0.024 | 2.2% |
| hc_ffn_pre | 0.021 | 2.0% |
| hc_attn_pre | 0.017 | 1.6% |
| attn_keys/kv_window | 0.014 | 1.3% |
| attn_keys/qr | 0.013 | 1.2% |
| moe_plan | 0.012 | 1.1% |
| attn_keys/index_keys | 0.011 | 1.0% |
| combine | 0.005 | 0.4% |
| hc_post | 0.003 | 0.3% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.061 ms; top: cuStreamSynchronize 0.861 ms x1, cudaLaunchKernel_v7000 0.136 ms x63, cuMemAllocAsync 0.023 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.663 ms, GPU busy 0.631 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.169 | 26.8% |
| attn_keys/kv_append | 0.120 | 19.0% |
| wo_a | 0.052 | 8.2% |
| wo_b | 0.045 | 7.1% |
| shared_expert | 0.044 | 7.0% |
| q_b | 0.043 | 6.9% |
| sparse_attn | 0.039 | 6.2% |
| router | 0.031 | 4.9% |
| hc_ffn_pre | 0.022 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.2% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| combine | 0.005 | 0.7% |
| attn_idx_rows | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |

host API calls: 191 calls, 0.615 ms; top: cuStreamSynchronize 0.459 ms x1, cudaLaunchKernel_v7000 0.101 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.004 ms x3

