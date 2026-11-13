### L2/prefill (2 instances): wall 298.210 ms, GPU busy 298.018 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 131.128 | 44.0% |
| sparse_attn | 35.414 | 11.9% |
| wo_a | 20.443 | 6.9% |
| attn_keys/index_score | 15.192 | 5.1% |
| shared_expert | 13.910 | 4.7% |
| q_b | 13.664 | 4.6% |
| wo_b | 13.075 | 4.4% |
| attn_keys/compressor | 11.837 | 4.0% |
| hc_ffn_pre | 10.406 | 3.5% |
| hc_attn_pre | 8.106 | 2.7% |
| router | 5.854 | 2.0% |
| attn_keys/topk | 4.974 | 1.7% |
| attn_keys/index_q | 4.433 | 1.5% |
| attn_keys/qr | 3.053 | 1.0% |
| hc_post | 2.235 | 0.8% |
| combine | 1.945 | 0.7% |
| attn_keys/kv_window | 1.900 | 0.6% |
| attn_keys/index_keys | 0.141 | 0.0% |
| attn_keys/kv_append | 0.123 | 0.0% |
| attn_idx_rows | 0.102 | 0.0% |
| moe_plan | 0.082 | 0.0% |

host API calls: 627 calls, 297.866 ms; top: cuStreamSynchronize 295.252 ms x1, cuMemAllocAsync 1.925 ms x134, cudaLaunchKernel_v7000 0.511 ms x172, cuMemFreeAsync 0.075 ms x132, cuMemsetD8Async 0.048 ms x12, cuMemcpyDtoDAsync_v2 0.033 ms x4

### L20/prefill (2 instances): wall 309.868 ms, GPU busy 309.691 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 124.906 | 40.3% |
| sparse_attn | 35.753 | 11.5% |
| attn_keys/index_score | 31.846 | 10.3% |
| wo_a | 20.728 | 6.7% |
| shared_expert | 14.031 | 4.5% |
| q_b | 13.789 | 4.5% |
| wo_b | 13.210 | 4.3% |
| hc_ffn_pre | 10.748 | 3.5% |
| attn_keys/topk | 10.199 | 3.3% |
| hc_attn_pre | 8.188 | 2.6% |
| router | 5.968 | 1.9% |
| attn_keys/candidates | 4.562 | 1.5% |
| attn_keys/index_q | 4.400 | 1.4% |
| attn_keys/qr | 3.063 | 1.0% |
| hc_post | 2.239 | 0.7% |
| combine | 1.956 | 0.6% |
| attn_keys/kv_window | 1.907 | 0.6% |
| attn_keys/compressor | 1.512 | 0.5% |
| attn_keys/index_keys | 0.269 | 0.1% |
| attn_keys/kv_append | 0.226 | 0.1% |
| attn_idx_rows | 0.106 | 0.0% |
| moe_plan | 0.085 | 0.0% |

host API calls: 629 calls, 309.563 ms; top: cuStreamSynchronize 307.017 ms x1, cuMemAllocAsync 1.927 ms x133, cudaLaunchKernel_v7000 0.453 ms x174, cuMemFreeAsync 0.074 ms x131, cuMemsetD8Async 0.040 ms x12, cuMemcpyDtoDAsync_v2 0.032 ms x4

### L21/prefill (2 instances): wall 252.917 ms, GPU busy 252.746 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.686 | 48.5% |
| sparse_attn | 35.435 | 14.0% |
| wo_a | 20.512 | 8.1% |
| q_b | 13.857 | 5.5% |
| shared_expert | 13.671 | 5.4% |
| wo_b | 13.074 | 5.2% |
| hc_ffn_pre | 10.322 | 4.1% |
| hc_attn_pre | 8.170 | 3.2% |
| router | 5.711 | 2.3% |
| attn_keys/qr | 3.059 | 1.2% |
| hc_post | 2.213 | 0.9% |
| combine | 1.916 | 0.8% |
| attn_keys/kv_window | 1.906 | 0.8% |
| attn_idx_rows | 0.104 | 0.0% |
| moe_plan | 0.080 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 252.670 ms; top: cuStreamSynchronize 251.231 ms x1, cuMemAllocAsync 0.941 ms x118, cudaLaunchKernel_v7000 0.353 ms x134, cuMemFreeAsync 0.061 ms x116, cuMemsetD8Async 0.037 ms x12, cuMemcpyDtoDAsync_v2 0.032 ms x3

### L2/decode (6 instances): wall 0.685 ms, GPU busy 0.618 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 25.8% |
| attn_keys/compressor | 0.049 | 7.9% |
| wo_a | 0.049 | 7.9% |
| wo_b | 0.044 | 7.2% |
| shared_expert | 0.044 | 7.1% |
| sparse_attn | 0.036 | 5.8% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| attn_keys/topk | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.4% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.598 ms; top: cuStreamSynchronize 0.368 ms x1, cudaLaunchKernel_v7000 0.150 ms x60, cuMemAllocAsync 0.030 ms x52, cuMemFreeAsync 0.021 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.700 ms, GPU busy 0.661 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.163 | 24.7% |
| wo_a | 0.051 | 7.7% |
| shared_expert | 0.044 | 6.7% |
| wo_b | 0.044 | 6.7% |
| attn_keys/topk | 0.040 | 6.1% |
| sparse_attn | 0.036 | 5.4% |
| attn_keys/index_q | 0.034 | 5.1% |
| q_b | 0.032 | 4.8% |
| router | 0.030 | 4.6% |
| attn_keys/candidates | 0.029 | 4.4% |
| attn_keys/kv_append | 0.027 | 4.1% |
| attn_keys/compressor | 0.024 | 3.6% |
| hc_ffn_pre | 0.021 | 3.2% |
| hc_attn_pre | 0.017 | 2.6% |
| attn_keys/kv_window | 0.014 | 2.1% |
| attn_keys/qr | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/index_keys | 0.011 | 1.6% |
| attn_keys/index_score | 0.010 | 1.6% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.633 ms; top: cuStreamSynchronize 0.428 ms x1, cudaLaunchKernel_v7000 0.137 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.549 ms, GPU busy 0.516 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.164 | 31.8% |
| wo_a | 0.050 | 9.8% |
| shared_expert | 0.045 | 8.7% |
| wo_b | 0.044 | 8.6% |
| sparse_attn | 0.038 | 7.4% |
| q_b | 0.032 | 6.2% |
| router | 0.031 | 5.9% |
| attn_keys/kv_append | 0.023 | 4.5% |
| hc_ffn_pre | 0.021 | 4.1% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.7% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.3% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.500 ms; top: cuStreamSynchronize 0.342 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

