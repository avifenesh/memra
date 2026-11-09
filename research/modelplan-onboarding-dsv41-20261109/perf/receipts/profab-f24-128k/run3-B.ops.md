### L2/prefill (2 instances): wall 1614.031 ms, GPU busy 1613.650 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 604.120 | 37.4% |
| attn_keys/index_score | 268.873 | 16.7% |
| sparse_attn | 156.436 | 9.7% |
| attn_keys/topk | 105.126 | 6.5% |
| wo_a | 99.766 | 6.2% |
| q_b | 60.416 | 3.7% |
| wo_b | 57.625 | 3.6% |
| shared_expert | 56.274 | 3.5% |
| attn_keys/compressor | 46.663 | 2.9% |
| hc_ffn_pre | 45.152 | 2.8% |
| hc_attn_pre | 32.294 | 2.0% |
| router | 24.363 | 1.5% |
| attn_keys/index_q | 18.319 | 1.1% |
| attn_keys/qr | 11.994 | 0.7% |
| hc_post | 9.012 | 0.6% |
| combine | 7.884 | 0.5% |
| attn_keys/kv_window | 7.584 | 0.5% |
| attn_keys/index_keys | 0.501 | 0.0% |
| attn_keys/kv_append | 0.495 | 0.0% |
| attn_idx_rows | 0.408 | 0.0% |
| moe_plan | 0.344 | 0.0% |

host API calls: 2175 calls, 1613.230 ms; top: cuStreamSynchronize 1604.912 ms x1, cuMemAllocAsync 6.628 ms x458, cudaLaunchKernel_v7000 1.317 ms x604, cuMemFreeAsync 0.182 ms x456, cuMemsetD8Async 0.106 ms x48, cuKernelGetName 0.057 ms x604

### L20/prefill (2 instances): wall 2219.380 ms, GPU busy 2218.977 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 588.479 | 26.5% |
| experts | 588.068 | 26.5% |
| attn_keys/topk | 256.181 | 11.5% |
| attn_keys/candidates | 178.774 | 8.1% |
| sparse_attn | 159.665 | 7.2% |
| wo_a | 102.723 | 4.6% |
| q_b | 61.420 | 2.8% |
| wo_b | 58.438 | 2.6% |
| shared_expert | 57.065 | 2.6% |
| hc_ffn_pre | 45.718 | 2.1% |
| hc_attn_pre | 32.857 | 1.5% |
| router | 24.944 | 1.1% |
| attn_keys/index_q | 18.797 | 0.8% |
| attn_keys/qr | 12.120 | 0.5% |
| hc_post | 8.959 | 0.4% |
| combine | 8.065 | 0.4% |
| attn_keys/kv_window | 7.676 | 0.3% |
| attn_keys/compressor | 6.083 | 0.3% |
| attn_keys/kv_append | 1.191 | 0.1% |
| attn_keys/index_keys | 0.992 | 0.0% |
| attn_idx_rows | 0.415 | 0.0% |
| moe_plan | 0.346 | 0.0% |

host API calls: 2201 calls, 2218.545 ms; top: cuStreamSynchronize 2207.029 ms x1, cuMemAllocAsync 9.759 ms x457, cudaLaunchKernel_v7000 1.371 ms x618, cuMemFreeAsync 0.195 ms x455, cuMemsetD8Async 0.102 ms x48, cuKernelGetName 0.057 ms x618

### L21/prefill (2 instances): wall 1164.749 ms, GPU busy 1164.382 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 603.931 | 51.9% |
| sparse_attn | 153.229 | 13.2% |
| wo_a | 93.568 | 8.0% |
| q_b | 59.753 | 5.1% |
| shared_expert | 56.972 | 4.9% |
| wo_b | 55.930 | 4.8% |
| hc_ffn_pre | 45.616 | 3.9% |
| hc_attn_pre | 32.744 | 2.8% |
| router | 24.838 | 2.1% |
| attn_keys/qr | 12.111 | 1.0% |
| hc_post | 9.020 | 0.8% |
| combine | 8.037 | 0.7% |
| attn_keys/kv_window | 7.676 | 0.7% |
| attn_idx_rows | 0.432 | 0.0% |
| moe_plan | 0.346 | 0.0% |
| attn_keys/kv_append | 0.179 | 0.0% |

host API calls: 1826 calls, 1164.072 ms; top: cuStreamSynchronize 1157.633 ms x1, cuMemAllocAsync 5.000 ms x406, cudaLaunchKernel_v7000 1.087 ms x482, cuMemFreeAsync 0.179 ms x404, cuMemsetD8Async 0.104 ms x48, cuKernelGetName 0.044 ms x482

### L2/decode (6 instances): wall 0.837 ms, GPU busy 0.771 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.166 | 21.5% |
| attn_keys/topk | 0.118 | 15.3% |
| wo_a | 0.051 | 6.6% |
| attn_keys/kv_append | 0.050 | 6.5% |
| attn_keys/compressor | 0.049 | 6.4% |
| wo_b | 0.046 | 5.9% |
| shared_expert | 0.045 | 5.8% |
| q_b | 0.037 | 4.8% |
| sparse_attn | 0.036 | 4.7% |
| attn_keys/index_q | 0.034 | 4.4% |
| router | 0.030 | 3.9% |
| hc_ffn_pre | 0.021 | 2.7% |
| attn_keys/index_score | 0.017 | 2.2% |
| hc_attn_pre | 0.017 | 2.2% |
| attn_keys/kv_window | 0.014 | 1.8% |
| attn_keys/qr | 0.013 | 1.7% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/index_keys | 0.005 | 0.7% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 233 calls, 0.751 ms; top: cuStreamSynchronize 0.527 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.125 ms, GPU busy 1.089 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.225 | 20.6% |
| experts | 0.170 | 15.6% |
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
| hc_ffn_pre | 0.022 | 2.0% |
| hc_attn_pre | 0.017 | 1.6% |
| attn_keys/kv_window | 0.014 | 1.3% |
| attn_keys/qr | 0.013 | 1.2% |
| moe_plan | 0.012 | 1.1% |
| attn_keys/index_keys | 0.011 | 1.0% |
| combine | 0.005 | 0.4% |
| hc_post | 0.003 | 0.3% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.062 ms; top: cuStreamSynchronize 0.865 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.023 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.664 ms, GPU busy 0.634 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.170 | 26.9% |
| attn_keys/kv_append | 0.120 | 18.9% |
| wo_a | 0.051 | 8.1% |
| wo_b | 0.045 | 7.0% |
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

host API calls: 191 calls, 0.617 ms; top: cuStreamSynchronize 0.463 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuKernelGetName 0.004 ms x47

