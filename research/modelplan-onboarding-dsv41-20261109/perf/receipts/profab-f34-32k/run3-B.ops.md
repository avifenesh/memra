### L2/prefill (2 instances): wall 290.191 ms, GPU busy 290.018 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 127.980 | 44.1% |
| sparse_attn | 34.904 | 12.0% |
| wo_a | 20.141 | 6.9% |
| attn_keys/index_score | 15.002 | 5.2% |
| shared_expert | 13.626 | 4.7% |
| q_b | 13.456 | 4.6% |
| wo_b | 12.904 | 4.4% |
| attn_keys/compressor | 11.697 | 4.0% |
| hc_ffn_pre | 10.223 | 3.5% |
| hc_attn_pre | 8.032 | 2.8% |
| router | 5.713 | 2.0% |
| attn_keys/index_q | 4.450 | 1.5% |
| attn_keys/qr | 3.029 | 1.0% |
| attn_keys/topk | 2.423 | 0.8% |
| hc_post | 2.215 | 0.8% |
| combine | 1.892 | 0.7% |
| attn_keys/kv_window | 1.880 | 0.6% |
| attn_keys/index_keys | 0.139 | 0.0% |
| attn_keys/kv_append | 0.131 | 0.0% |
| attn_idx_rows | 0.102 | 0.0% |
| moe_plan | 0.079 | 0.0% |

host API calls: 627 calls, 289.879 ms; top: cuStreamSynchronize 287.490 ms x1, cuMemAllocAsync 1.794 ms x134, cudaLaunchKernel_v7000 0.446 ms x172, cuMemFreeAsync 0.066 ms x132, cuMemsetD8Async 0.036 ms x12, cuMemcpyDtoDAsync_v2 0.026 ms x4

### L20/prefill (2 instances): wall 296.785 ms, GPU busy 296.589 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.373 | 41.3% |
| sparse_attn | 35.201 | 11.9% |
| attn_keys/index_score | 31.451 | 10.6% |
| wo_a | 20.350 | 6.9% |
| shared_expert | 13.796 | 4.7% |
| q_b | 13.611 | 4.6% |
| wo_b | 13.044 | 4.4% |
| hc_ffn_pre | 10.447 | 3.5% |
| hc_attn_pre | 8.079 | 2.7% |
| router | 5.778 | 1.9% |
| attn_keys/index_q | 4.456 | 1.5% |
| attn_keys/topk | 4.331 | 1.5% |
| attn_keys/qr | 3.030 | 1.0% |
| attn_keys/candidates | 2.440 | 0.8% |
| hc_post | 2.224 | 0.8% |
| combine | 1.908 | 0.6% |
| attn_keys/kv_window | 1.885 | 0.6% |
| attn_keys/compressor | 1.491 | 0.5% |
| attn_keys/index_keys | 0.262 | 0.1% |
| attn_keys/kv_append | 0.245 | 0.1% |
| attn_idx_rows | 0.106 | 0.0% |
| moe_plan | 0.083 | 0.0% |

host API calls: 629 calls, 296.466 ms; top: cuStreamSynchronize 293.894 ms x1, cuMemAllocAsync 1.955 ms x133, cudaLaunchKernel_v7000 0.457 ms x174, cuMemFreeAsync 0.074 ms x131, cuMemsetD8Async 0.039 ms x12, cuMemcpyDtoDAsync_v2 0.027 ms x4

### L21/prefill (2 instances): wall 249.444 ms, GPU busy 249.272 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 120.539 | 48.4% |
| sparse_attn | 35.094 | 14.1% |
| wo_a | 20.334 | 8.2% |
| q_b | 13.718 | 5.5% |
| shared_expert | 13.508 | 5.4% |
| wo_b | 12.988 | 5.2% |
| hc_ffn_pre | 10.257 | 4.1% |
| hc_attn_pre | 7.994 | 3.2% |
| router | 5.616 | 2.3% |
| attn_keys/qr | 3.032 | 1.2% |
| hc_post | 2.210 | 0.9% |
| attn_keys/kv_window | 1.885 | 0.8% |
| combine | 1.883 | 0.8% |
| attn_idx_rows | 0.105 | 0.0% |
| moe_plan | 0.079 | 0.0% |
| attn_keys/kv_append | 0.029 | 0.0% |

host API calls: 518 calls, 249.187 ms; top: cuStreamSynchronize 247.792 ms x1, cuMemAllocAsync 0.903 ms x118, cudaLaunchKernel_v7000 0.352 ms x134, cuMemFreeAsync 0.057 ms x116, cuMemsetD8Async 0.038 ms x12, cuMemcpyDtoDAsync_v2 0.029 ms x3

### L2/decode (6 instances): wall 0.695 ms, GPU busy 0.618 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 25.8% |
| wo_a | 0.049 | 8.0% |
| attn_keys/compressor | 0.049 | 7.9% |
| wo_b | 0.044 | 7.2% |
| shared_expert | 0.044 | 7.2% |
| sparse_attn | 0.036 | 5.8% |
| attn_keys/index_q | 0.034 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.7% |
| attn_keys/topk | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.4% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.3% |
| attn_keys/qr | 0.013 | 2.1% |
| moe_plan | 0.012 | 2.0% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.602 ms; top: cuStreamSynchronize 0.351 ms x1, cudaLaunchKernel_v7000 0.151 ms x60, cuMemcpyDtoDAsync_v2 0.037 ms x6, cuMemAllocAsync 0.031 ms x52, cuMemFreeAsync 0.021 ms x51, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.699 ms, GPU busy 0.662 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.162 | 24.4% |
| wo_a | 0.050 | 7.6% |
| wo_b | 0.045 | 6.8% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/topk | 0.040 | 6.1% |
| sparse_attn | 0.036 | 5.4% |
| attn_keys/index_q | 0.034 | 5.1% |
| q_b | 0.032 | 4.8% |
| attn_keys/candidates | 0.032 | 4.8% |
| router | 0.030 | 4.5% |
| attn_keys/kv_append | 0.027 | 4.1% |
| attn_keys/compressor | 0.023 | 3.5% |
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

host API calls: 236 calls, 0.635 ms; top: cuStreamSynchronize 0.434 ms x1, cudaLaunchKernel_v7000 0.133 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.545 ms, GPU busy 0.511 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.162 | 31.7% |
| wo_a | 0.050 | 9.8% |
| shared_expert | 0.045 | 8.7% |
| wo_b | 0.044 | 8.6% |
| sparse_attn | 0.037 | 7.3% |
| q_b | 0.032 | 6.3% |
| router | 0.030 | 5.9% |
| attn_keys/kv_append | 0.024 | 4.6% |
| hc_ffn_pre | 0.021 | 4.1% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.4% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.496 ms; top: cuStreamSynchronize 0.337 ms x1, cudaLaunchKernel_v7000 0.101 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

