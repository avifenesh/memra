### L2/prefill (2 instances): wall 44.680 ms, GPU busy 44.603 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.191 | 52.0% |
| sparse_attn | 4.856 | 10.9% |
| wo_a | 2.640 | 5.9% |
| q_b | 1.955 | 4.4% |
| shared_expert | 1.926 | 4.3% |
| wo_b | 1.859 | 4.2% |
| hc_ffn_pre | 1.697 | 3.8% |
| attn_keys/compressor | 1.635 | 3.7% |
| hc_attn_pre | 1.154 | 2.6% |
| router | 0.805 | 1.8% |
| attn_keys/index_q | 0.648 | 1.5% |
| hc_post | 0.485 | 1.1% |
| attn_keys/qr | 0.423 | 0.9% |
| moe_plan | 0.397 | 0.9% |
| attn_keys/index_score | 0.353 | 0.8% |
| attn_keys/kv_window | 0.251 | 0.6% |
| combine | 0.204 | 0.5% |
| attn_keys/topk | 0.061 | 0.1% |
| attn_keys/index_keys | 0.035 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 44.527 ms; top: cuStreamSynchronize 43.863 ms x1, cuMemAllocAsync 0.380 ms x53, cudaLaunchKernel_v7000 0.204 ms x64, cuMemFreeAsync 0.031 ms x51, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.015 ms x3

### L20/prefill (2 instances): wall 42.496 ms, GPU busy 42.406 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 21.753 | 51.3% |
| sparse_attn | 4.973 | 11.7% |
| wo_a | 2.659 | 6.3% |
| q_b | 1.972 | 4.7% |
| shared_expert | 1.938 | 4.6% |
| wo_b | 1.870 | 4.4% |
| hc_ffn_pre | 1.710 | 4.0% |
| hc_attn_pre | 1.161 | 2.7% |
| router | 0.809 | 1.9% |
| attn_keys/index_score | 0.713 | 1.7% |
| attn_keys/index_q | 0.638 | 1.5% |
| hc_post | 0.490 | 1.2% |
| attn_keys/qr | 0.425 | 1.0% |
| moe_plan | 0.410 | 1.0% |
| attn_keys/kv_window | 0.257 | 0.6% |
| attn_keys/compressor | 0.229 | 0.5% |
| combine | 0.203 | 0.5% |
| attn_keys/topk | 0.095 | 0.2% |
| attn_keys/index_keys | 0.038 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.019 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 42.344 ms; top: cuStreamSynchronize 41.582 ms x1, cuMemAllocAsync 0.485 ms x52, cudaLaunchKernel_v7000 0.200 ms x63, cuMemFreeAsync 0.032 ms x50, cuMemcpyDtoDAsync_v2 0.020 ms x4, cuMemsetD8Async 0.016 ms x3

### L21/prefill (2 instances): wall 40.998 ms, GPU busy 40.921 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 21.990 | 53.7% |
| sparse_attn | 5.004 | 12.2% |
| wo_a | 2.659 | 6.5% |
| q_b | 1.976 | 4.8% |
| shared_expert | 1.949 | 4.8% |
| wo_b | 1.871 | 4.6% |
| hc_ffn_pre | 1.708 | 4.2% |
| hc_attn_pre | 1.158 | 2.8% |
| router | 0.808 | 2.0% |
| hc_post | 0.493 | 1.2% |
| attn_keys/qr | 0.425 | 1.0% |
| moe_plan | 0.408 | 1.0% |
| attn_keys/kv_window | 0.254 | 0.6% |
| combine | 0.204 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 40.872 ms; top: cuStreamSynchronize 40.415 ms x1, cuMemAllocAsync 0.235 ms x46, cudaLaunchKernel_v7000 0.156 ms x47, cuMemFreeAsync 0.027 ms x44, cuMemcpyDtoDAsync_v2 0.017 ms x3, cuMemsetD8Async 0.014 ms x3

### L2/decode (6 instances): wall 0.719 ms, GPU busy 0.659 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 23.8% |
| wo_b | 0.062 | 9.5% |
| wo_a | 0.048 | 7.3% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.043 | 6.6% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.4% |
| router | 0.035 | 5.3% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/kv_append | 0.005 | 0.8% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.633 ms; top: cuStreamSynchronize 0.390 ms x1, cudaLaunchKernel_v7000 0.160 ms x60, cuMemAllocAsync 0.031 ms x52, cuMemFreeAsync 0.022 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.007 ms x60

### L20/decode (6 instances): wall 0.694 ms, GPU busy 0.658 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 23.9% |
| wo_b | 0.065 | 9.9% |
| wo_a | 0.047 | 7.2% |
| sparse_attn | 0.046 | 7.1% |
| shared_expert | 0.044 | 6.6% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.034 | 5.2% |
| attn_keys/compressor | 0.032 | 4.9% |
| q_b | 0.031 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.5% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.629 ms; top: cuStreamSynchronize 0.414 ms x1, cudaLaunchKernel_v7000 0.143 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.021 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.577 ms, GPU busy 0.545 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 28.7% |
| wo_b | 0.065 | 12.0% |
| wo_a | 0.047 | 8.6% |
| sparse_attn | 0.047 | 8.6% |
| shared_expert | 0.043 | 8.0% |
| hc_ffn_pre | 0.036 | 6.6% |
| router | 0.034 | 6.3% |
| q_b | 0.031 | 5.8% |
| hc_attn_pre | 0.028 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.012 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.007 | 1.4% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.526 ms; top: cuStreamSynchronize 0.359 ms x1, cudaLaunchKernel_v7000 0.108 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.018 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

