### L2/prefill (2 instances): wall 34.903 ms, GPU busy 34.794 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.160 | 49.3% |
| sparse_attn | 4.069 | 11.7% |
| (unranged) | 2.471 | 7.1% |
| wo_a | 2.429 | 7.0% |
| q_b | 1.644 | 4.7% |
| wo_b | 1.565 | 4.5% |
| shared_expert | 1.525 | 4.4% |
| compressor | 1.485 | 4.3% |
| router | 0.712 | 2.0% |
| index_q | 0.607 | 1.7% |
| qr | 0.361 | 1.0% |
| kv_window | 0.224 | 0.6% |
| combine | 0.205 | 0.6% |
| index_score | 0.186 | 0.5% |
| topk | 0.068 | 0.2% |
| index_keys | 0.038 | 0.1% |
| kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 202 calls, 34.683 ms; top: cuStreamSynchronize 33.912 ms x2, cuMemAllocAsync 0.390 ms x34, cudaLaunchKernel_v7000 0.251 ms x63, cuMemcpyDtoDAsync_v2 0.064 ms x5, cuMemFreeAsync 0.029 ms x32, cuMemsetD8Async 0.026 ms x3

### L20/prefill (2 instances): wall 33.137 ms, GPU busy 33.033 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.132 | 48.8% |
| sparse_attn | 4.228 | 12.8% |
| (unranged) | 2.504 | 7.6% |
| wo_a | 2.448 | 7.4% |
| q_b | 1.660 | 5.0% |
| wo_b | 1.578 | 4.8% |
| shared_expert | 1.539 | 4.7% |
| router | 0.720 | 2.2% |
| index_q | 0.590 | 1.8% |
| index_score | 0.391 | 1.2% |
| qr | 0.363 | 1.1% |
| kv_window | 0.225 | 0.7% |
| compressor | 0.212 | 0.6% |
| combine | 0.205 | 0.6% |
| topk | 0.101 | 0.3% |
| index_keys | 0.048 | 0.1% |
| kv_append | 0.036 | 0.1% |
| candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 197 calls, 32.965 ms; top: cuStreamSynchronize 32.323 ms x2, cuMemAllocAsync 0.331 ms x32, cudaLaunchKernel_v7000 0.204 ms x62, cuMemcpyDtoDAsync_v2 0.055 ms x5, cuMemsetD8Async 0.021 ms x3, cuMemFreeAsync 0.020 ms x31

### L21/prefill (2 instances): wall 31.984 ms, GPU busy 31.881 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.381 | 51.4% |
| sparse_attn | 4.243 | 13.3% |
| (unranged) | 2.499 | 7.8% |
| wo_a | 2.448 | 7.7% |
| q_b | 1.643 | 5.2% |
| wo_b | 1.577 | 4.9% |
| shared_expert | 1.549 | 4.9% |
| router | 0.719 | 2.3% |
| qr | 0.362 | 1.1% |
| kv_window | 0.226 | 0.7% |
| combine | 0.204 | 0.6% |
| moe_plan | 0.016 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |
| kv_append | 0.004 | 0.0% |

host API calls: 154 calls, 31.834 ms; top: cuStreamSynchronize 31.359 ms x2, cuMemAllocAsync 0.216 ms x27, cudaLaunchKernel_v7000 0.160 ms x46, cuMemcpyDtoDAsync_v2 0.051 ms x4, cuMemsetD8Async 0.023 ms x3, cuMemFreeAsync 0.018 ms x26

### L2/decode (6 instances): wall 0.615 ms, GPU busy 0.609 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.126 | 20.8% |
| wo_b | 0.069 | 11.3% |
| shared_expert | 0.061 | 10.1% |
| wo_a | 0.049 | 8.1% |
| compressor | 0.049 | 8.0% |
| (unranged) | 0.040 | 6.5% |
| sparse_attn | 0.035 | 5.8% |
| index_q | 0.033 | 5.5% |
| q_b | 0.032 | 5.2% |
| router | 0.029 | 4.8% |
| kv_window | 0.020 | 3.4% |
| qr | 0.018 | 2.9% |
| moe_plan | 0.014 | 2.3% |
| topk | 0.013 | 2.2% |
| kv_append | 0.006 | 0.9% |
| index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 144 calls, 0.534 ms; top: cuStreamSynchronize 0.314 ms x2, cudaLaunchKernel_v7000 0.167 ms x59, cuMemcpyDtoDAsync_v2 0.023 ms x7, cuMemAllocAsync 0.009 ms x8, cuKernelGetName 0.007 ms x59, cuMemsetD8Async 0.006 ms x3

### L20/decode (6 instances): wall 0.568 ms, GPU busy 0.595 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.128 | 21.4% |
| wo_b | 0.069 | 11.5% |
| shared_expert | 0.060 | 10.2% |
| wo_a | 0.049 | 8.3% |
| (unranged) | 0.040 | 6.8% |
| sparse_attn | 0.035 | 5.9% |
| index_q | 0.033 | 5.5% |
| q_b | 0.032 | 5.3% |
| router | 0.030 | 5.0% |
| compressor | 0.023 | 3.9% |
| kv_window | 0.020 | 3.4% |
| qr | 0.018 | 3.0% |
| moe_plan | 0.013 | 2.2% |
| topk | 0.013 | 2.1% |
| index_keys | 0.011 | 1.8% |
| kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.7% |
| candidates | 0.004 | 0.6% |
| index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 142 calls, 0.512 ms; top: cuStreamSynchronize 0.328 ms x2, cudaLaunchKernel_v7000 0.151 ms x62, cuMemcpyDtoDAsync_v2 0.018 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.005 ms x3, cuMemAllocAsync 0.003 ms x2

### L21/decode (6 instances): wall 0.477 ms, GPU busy 0.508 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.128 | 25.2% |
| wo_b | 0.068 | 13.5% |
| shared_expert | 0.059 | 11.7% |
| wo_a | 0.049 | 9.7% |
| (unranged) | 0.041 | 8.2% |
| sparse_attn | 0.036 | 7.0% |
| q_b | 0.031 | 6.2% |
| router | 0.029 | 5.7% |
| kv_window | 0.021 | 4.1% |
| qr | 0.019 | 3.7% |
| moe_plan | 0.014 | 2.7% |
| kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.9% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 107 calls, 0.437 ms; top: cuStreamSynchronize 0.299 ms x2, cudaLaunchKernel_v7000 0.113 ms x46, cuMemcpyDtoDAsync_v2 0.014 ms x4, cuMemsetD8Async 0.005 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

