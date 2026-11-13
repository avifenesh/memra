### L2/prefill (2 instances): wall 2065.237 ms, GPU busy 2064.825 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 813.931 | 39.4% |
| attn_keys/index_score | 442.309 | 21.4% |
| sparse_attn | 184.399 | 8.9% |
| wo_a | 101.187 | 4.9% |
| q_b | 70.222 | 3.4% |
| hc_ffn_pre | 68.850 | 3.3% |
| wo_b | 67.534 | 3.3% |
| shared_expert | 67.319 | 3.3% |
| attn_keys/topk | 50.946 | 2.5% |
| attn_keys/compressor | 50.350 | 2.4% |
| hc_attn_pre | 40.714 | 2.0% |
| router | 27.221 | 1.3% |
| hc_post | 22.754 | 1.1% |
| attn_keys/index_q | 19.990 | 1.0% |
| attn_keys/qr | 13.299 | 0.6% |
| attn_keys/kv_window | 8.137 | 0.4% |
| combine | 7.765 | 0.4% |
| moe_plan | 6.658 | 0.3% |
| attn_keys/kv_append | 0.501 | 0.0% |
| attn_idx_rows | 0.414 | 0.0% |
| attn_keys/index_keys | 0.326 | 0.0% |

host API calls: 2175 calls, 2064.449 ms; top: cuStreamSynchronize 2056.208 ms x1, cuMemAllocAsync 6.390 ms x458, cudaLaunchKernel_v7000 1.453 ms x604, cuMemFreeAsync 0.191 ms x456, cuMemsetD8Async 0.116 ms x48, cuKernelGetName 0.058 ms x604

### L20/prefill (2 instances): wall 2621.223 ms, GPU busy 2620.804 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/index_score | 879.103 | 33.5% |
| experts | 788.820 | 30.1% |
| sparse_attn | 187.845 | 7.2% |
| attn_keys/topk | 124.702 | 4.8% |
| wo_a | 104.790 | 4.0% |
| attn_keys/candidates | 99.859 | 3.8% |
| q_b | 71.016 | 2.7% |
| hc_ffn_pre | 70.613 | 2.7% |
| wo_b | 68.574 | 2.6% |
| shared_expert | 67.869 | 2.6% |
| hc_attn_pre | 41.084 | 1.6% |
| router | 27.913 | 1.1% |
| hc_post | 23.712 | 0.9% |
| attn_keys/index_q | 20.224 | 0.8% |
| attn_keys/qr | 13.438 | 0.5% |
| attn_keys/kv_window | 8.222 | 0.3% |
| combine | 7.972 | 0.3% |
| moe_plan | 6.865 | 0.3% |
| attn_keys/compressor | 5.914 | 0.2% |
| attn_keys/kv_append | 1.204 | 0.0% |
| attn_keys/index_keys | 0.646 | 0.0% |
| attn_idx_rows | 0.418 | 0.0% |

host API calls: 2201 calls, 2620.437 ms; top: cuStreamSynchronize 2608.751 ms x1, cuMemAllocAsync 9.547 ms x457, cudaLaunchKernel_v7000 1.738 ms x618, cuMemFreeAsync 0.204 ms x455, cuMemsetD8Async 0.105 ms x48, cuKernelGetName 0.058 ms x618

### L21/prefill (2 instances): wall 1496.214 ms, GPU busy 1495.839 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 812.450 | 54.3% |
| sparse_attn | 182.013 | 12.2% |
| wo_a | 97.967 | 6.5% |
| hc_ffn_pre | 70.317 | 4.7% |
| q_b | 69.328 | 4.6% |
| shared_expert | 67.928 | 4.5% |
| wo_b | 66.229 | 4.4% |
| hc_attn_pre | 41.061 | 2.7% |
| router | 27.855 | 1.9% |
| hc_post | 23.699 | 1.6% |
| attn_keys/qr | 13.428 | 0.9% |
| attn_keys/kv_window | 8.204 | 0.5% |
| combine | 7.928 | 0.5% |
| moe_plan | 6.823 | 0.5% |
| attn_idx_rows | 0.430 | 0.0% |
| attn_keys/kv_append | 0.181 | 0.0% |

host API calls: 1826 calls, 1495.477 ms; top: cuStreamSynchronize 1487.878 ms x1, cuMemAllocAsync 6.016 ms x406, cudaLaunchKernel_v7000 1.191 ms x482, cuMemFreeAsync 0.206 ms x404, cuMemsetD8Async 0.110 ms x48, cuKernelGetName 0.050 ms x482

### L2/decode (6 instances): wall 0.935 ms, GPU busy 0.871 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.164 | 18.9% |
| attn_keys/topk | 0.146 | 16.8% |
| wo_b | 0.065 | 7.5% |
| wo_a | 0.051 | 5.8% |
| attn_keys/kv_append | 0.050 | 5.8% |
| attn_keys/compressor | 0.048 | 5.5% |
| sparse_attn | 0.047 | 5.4% |
| shared_expert | 0.044 | 5.0% |
| attn_keys/index_q | 0.043 | 4.9% |
| hc_ffn_pre | 0.037 | 4.3% |
| q_b | 0.037 | 4.2% |
| router | 0.036 | 4.1% |
| hc_attn_pre | 0.028 | 3.2% |
| attn_keys/index_score | 0.016 | 1.9% |
| attn_keys/kv_window | 0.013 | 1.5% |
| moe_plan | 0.012 | 1.4% |
| attn_keys/qr | 0.012 | 1.4% |
| hc_post | 0.008 | 0.9% |
| attn_keys/index_keys | 0.005 | 0.6% |
| combine | 0.004 | 0.5% |
| attn_idx_rows | 0.003 | 0.3% |

host API calls: 233 calls, 0.852 ms; top: cuStreamSynchronize 0.623 ms x1, cudaLaunchKernel_v7000 0.152 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 1.238 ms, GPU busy 1.201 ms

| op | GPU ms | share of busy |
|---|---|---|
| attn_keys/topk | 0.280 | 23.3% |
| experts | 0.167 | 13.9% |
| attn_keys/kv_append | 0.124 | 10.3% |
| attn_keys/candidates | 0.114 | 9.5% |
| wo_b | 0.062 | 5.1% |
| wo_a | 0.051 | 4.3% |
| sparse_attn | 0.048 | 4.0% |
| q_b | 0.044 | 3.7% |
| shared_expert | 0.043 | 3.6% |
| attn_keys/index_q | 0.042 | 3.5% |
| hc_ffn_pre | 0.038 | 3.1% |
| router | 0.036 | 3.0% |
| attn_keys/compressor | 0.032 | 2.7% |
| attn_keys/index_score | 0.030 | 2.5% |
| hc_attn_pre | 0.028 | 2.3% |
| attn_keys/kv_window | 0.013 | 1.1% |
| moe_plan | 0.013 | 1.0% |
| attn_keys/qr | 0.012 | 1.0% |
| attn_keys/index_keys | 0.010 | 0.8% |
| hc_post | 0.007 | 0.6% |
| combine | 0.005 | 0.4% |
| attn_idx_rows | 0.003 | 0.2% |

host API calls: 236 calls, 1.167 ms; top: cuStreamSynchronize 0.961 ms x1, cudaLaunchKernel_v7000 0.138 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.725 ms, GPU busy 0.693 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.167 | 24.1% |
| attn_keys/kv_append | 0.120 | 17.3% |
| wo_b | 0.063 | 9.1% |
| wo_a | 0.053 | 7.7% |
| sparse_attn | 0.050 | 7.2% |
| q_b | 0.044 | 6.3% |
| shared_expert | 0.043 | 6.3% |
| hc_ffn_pre | 0.037 | 5.3% |
| router | 0.035 | 5.1% |
| hc_attn_pre | 0.029 | 4.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.7% |
| hc_post | 0.008 | 1.1% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.003 | 0.4% |

host API calls: 191 calls, 0.670 ms; top: cuStreamSynchronize 0.504 ms x1, cudaLaunchKernel_v7000 0.104 ms x47, cuMemAllocAsync 0.025 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.011 ms x3, cuMemsetD8Async 0.005 ms x3

