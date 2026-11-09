### L2/prefill (2 instances): wall 44.822 ms, GPU busy 44.709 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 23.252 | 52.0% |
| sparse_attn | 4.872 | 10.9% |
| wo_a | 2.647 | 5.9% |
| q_b | 1.961 | 4.4% |
| shared_expert | 1.923 | 4.3% |
| wo_b | 1.863 | 4.2% |
| hc_ffn_pre | 1.707 | 3.8% |
| attn_keys/compressor | 1.641 | 3.7% |
| hc_attn_pre | 1.153 | 2.6% |
| router | 0.806 | 1.8% |
| attn_keys/index_q | 0.648 | 1.4% |
| hc_post | 0.483 | 1.1% |
| attn_keys/qr | 0.423 | 0.9% |
| moe_plan | 0.397 | 0.9% |
| attn_keys/index_score | 0.352 | 0.8% |
| attn_keys/kv_window | 0.253 | 0.6% |
| combine | 0.205 | 0.5% |
| attn_keys/topk | 0.062 | 0.1% |
| attn_keys/index_keys | 0.034 | 0.1% |
| attn_keys/kv_append | 0.019 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 44.655 ms; top: cuStreamSynchronize 43.964 ms x1, cuMemAllocAsync 0.376 ms x53, cudaLaunchKernel_v7000 0.221 ms x64, cuMemcpyDtoDAsync_v2 0.033 ms x4, cuMemFreeAsync 0.031 ms x51, cuMemsetD8Async 0.020 ms x3

### L20/prefill (2 instances): wall 43.487 ms, GPU busy 43.365 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.195 | 51.2% |
| sparse_attn | 5.175 | 11.9% |
| wo_a | 2.713 | 6.3% |
| q_b | 1.999 | 4.6% |
| shared_expert | 1.980 | 4.6% |
| wo_b | 1.905 | 4.4% |
| hc_ffn_pre | 1.744 | 4.0% |
| hc_attn_pre | 1.177 | 2.7% |
| router | 0.828 | 1.9% |
| attn_keys/index_score | 0.740 | 1.7% |
| attn_keys/index_q | 0.652 | 1.5% |
| hc_post | 0.506 | 1.2% |
| attn_keys/qr | 0.435 | 1.0% |
| moe_plan | 0.411 | 0.9% |
| attn_keys/kv_window | 0.260 | 0.6% |
| attn_keys/compressor | 0.232 | 0.5% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.101 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.035 | 0.1% |
| attn_keys/candidates | 0.021 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 43.332 ms; top: cuStreamSynchronize 42.547 ms x1, cuMemAllocAsync 0.487 ms x52, cudaLaunchKernel_v7000 0.204 ms x63, cuMemFreeAsync 0.035 ms x50, cuMemcpyDtoDAsync_v2 0.029 ms x4, cuMemsetD8Async 0.019 ms x3

### L21/prefill (2 instances): wall 41.939 ms, GPU busy 41.843 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 22.442 | 53.6% |
| sparse_attn | 5.199 | 12.4% |
| wo_a | 2.718 | 6.5% |
| q_b | 2.011 | 4.8% |
| shared_expert | 1.988 | 4.8% |
| wo_b | 1.907 | 4.6% |
| hc_ffn_pre | 1.745 | 4.2% |
| hc_attn_pre | 1.176 | 2.8% |
| router | 0.830 | 2.0% |
| hc_post | 0.501 | 1.2% |
| attn_keys/qr | 0.436 | 1.0% |
| moe_plan | 0.411 | 1.0% |
| attn_keys/kv_window | 0.260 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 41.804 ms; top: cuStreamSynchronize 41.324 ms x1, cuMemAllocAsync 0.244 ms x46, cudaLaunchKernel_v7000 0.160 ms x47, cuMemFreeAsync 0.028 ms x44, cuMemcpyDtoDAsync_v2 0.021 ms x3, cuMemsetD8Async 0.018 ms x3

### L2/decode (6 instances): wall 0.723 ms, GPU busy 0.664 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 23.9% |
| wo_b | 0.062 | 9.4% |
| wo_a | 0.049 | 7.4% |
| attn_keys/compressor | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.0% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.043 | 6.4% |
| hc_ffn_pre | 0.036 | 5.4% |
| router | 0.035 | 5.2% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.2% |
| attn_keys/topk | 0.014 | 2.0% |
| attn_keys/kv_window | 0.013 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| attn_keys/index_keys | 0.005 | 0.8% |
| attn_keys/kv_append | 0.005 | 0.8% |
| combine | 0.004 | 0.7% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 233 calls, 0.643 ms; top: cuStreamSynchronize 0.413 ms x1, cudaLaunchKernel_v7000 0.153 ms x60, cuMemAllocAsync 0.029 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.697 ms, GPU busy 0.660 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 23.9% |
| wo_b | 0.065 | 9.8% |
| wo_a | 0.048 | 7.2% |
| sparse_attn | 0.047 | 7.1% |
| shared_expert | 0.044 | 6.7% |
| attn_keys/index_q | 0.043 | 6.5% |
| hc_ffn_pre | 0.036 | 5.5% |
| router | 0.035 | 5.2% |
| attn_keys/compressor | 0.032 | 4.9% |
| q_b | 0.032 | 4.8% |
| hc_attn_pre | 0.028 | 4.3% |
| attn_keys/topk | 0.014 | 2.2% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/qr | 0.012 | 1.9% |
| attn_keys/index_keys | 0.010 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.3% |
| hc_post | 0.007 | 1.1% |
| attn_keys/index_score | 0.006 | 1.0% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.635 ms; top: cuStreamSynchronize 0.430 ms x1, cudaLaunchKernel_v7000 0.136 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.013 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.581 ms, GPU busy 0.549 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.158 | 28.7% |
| wo_b | 0.065 | 11.9% |
| sparse_attn | 0.047 | 8.6% |
| wo_a | 0.047 | 8.6% |
| shared_expert | 0.044 | 8.1% |
| hc_ffn_pre | 0.036 | 6.5% |
| router | 0.034 | 6.3% |
| q_b | 0.032 | 5.8% |
| hc_attn_pre | 0.028 | 5.2% |
| attn_keys/kv_window | 0.013 | 2.4% |
| attn_keys/qr | 0.012 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| hc_post | 0.007 | 1.4% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.532 ms; top: cuStreamSynchronize 0.374 ms x1, cudaLaunchKernel_v7000 0.103 ms x47, cuMemAllocAsync 0.021 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

