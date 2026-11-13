### L2/prefill (5 instances): wall 1.414 ms, GPU busy 1.456 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.854 | 58.7% |
| (unranged) | 0.108 | 7.4% |
| shared_expert | 0.080 | 5.5% |
| compressor | 0.067 | 4.6% |
| index_q | 0.065 | 4.5% |
| wo_b | 0.058 | 4.0% |
| wo_a | 0.053 | 3.7% |
| router | 0.039 | 2.7% |
| q_b | 0.038 | 2.6% |
| kv_window | 0.018 | 1.3% |
| qr | 0.017 | 1.1% |
| moe_plan | 0.014 | 1.0% |
| index_keys | 0.012 | 0.8% |
| index_score | 0.008 | 0.5% |
| sparse_attn | 0.008 | 0.5% |
| combine | 0.008 | 0.5% |
| kv_append | 0.005 | 0.3% |
| topk | 0.003 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 140 calls, 1.354 ms; top: cuStreamSynchronize 1.184 ms x2, cudaLaunchKernel_v7000 0.143 ms x63, cuMemcpyDtoDAsync_v2 0.016 ms x5, cuKernelGetName 0.005 ms x63, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L20/prefill (5 instances): wall 1.315 ms, GPU busy 1.363 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.773 | 56.7% |
| (unranged) | 0.108 | 7.9% |
| shared_expert | 0.082 | 6.0% |
| index_q | 0.065 | 4.8% |
| wo_b | 0.056 | 4.1% |
| wo_a | 0.053 | 3.9% |
| compressor | 0.051 | 3.8% |
| router | 0.039 | 2.9% |
| q_b | 0.038 | 2.8% |
| kv_window | 0.019 | 1.4% |
| qr | 0.017 | 1.3% |
| moe_plan | 0.014 | 1.0% |
| index_keys | 0.012 | 0.9% |
| index_score | 0.009 | 0.6% |
| combine | 0.007 | 0.5% |
| sparse_attn | 0.007 | 0.5% |
| kv_append | 0.006 | 0.4% |
| topk | 0.003 | 0.2% |
| candidates | 0.002 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 138 calls, 1.262 ms; top: cuStreamSynchronize 1.104 ms x2, cudaLaunchKernel_v7000 0.132 ms x62, cuMemcpyDtoDAsync_v2 0.014 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L21/prefill (5 instances): wall 1.094 ms, GPU busy 1.144 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.706 | 61.7% |
| (unranged) | 0.107 | 9.4% |
| shared_expert | 0.080 | 7.0% |
| wo_b | 0.056 | 4.9% |
| wo_a | 0.051 | 4.5% |
| router | 0.039 | 3.4% |
| q_b | 0.038 | 3.3% |
| kv_window | 0.019 | 1.6% |
| qr | 0.017 | 1.5% |
| moe_plan | 0.014 | 1.2% |
| combine | 0.007 | 0.7% |
| sparse_attn | 0.007 | 0.6% |
| attn_idx_rows | 0.001 | 0.1% |
| kv_append | 0.001 | 0.1% |

host API calls: 105 calls, 1.053 ms; top: cuStreamSynchronize 0.934 ms x2, cudaLaunchKernel_v7000 0.099 ms x46, cuMemcpyDtoDAsync_v2 0.011 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

### L2/decode (6 instances): wall 0.519 ms, GPU busy 0.523 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.123 | 23.5% |
| wo_a | 0.049 | 9.4% |
| compressor | 0.048 | 9.3% |
| shared_expert | 0.043 | 8.2% |
| router | 0.043 | 8.2% |
| wo_b | 0.042 | 8.0% |
| (unranged) | 0.040 | 7.7% |
| index_q | 0.034 | 6.5% |
| q_b | 0.031 | 6.0% |
| kv_window | 0.014 | 2.6% |
| moe_plan | 0.013 | 2.4% |
| qr | 0.013 | 2.4% |
| sparse_attn | 0.012 | 2.4% |
| index_keys | 0.006 | 1.1% |
| combine | 0.004 | 0.8% |
| kv_append | 0.003 | 0.6% |
| index_score | 0.002 | 0.4% |
| topk | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.2% |

host API calls: 142 calls, 0.462 ms; top: cuStreamSynchronize 0.296 ms x2, cudaLaunchKernel_v7000 0.132 ms x59, cuMemcpyDtoDAsync_v2 0.018 ms x7, cuMemAllocAsync 0.006 ms x8, cuKernelGetName 0.005 ms x59, cuMemsetD8Async 0.004 ms x3

### L20/decode (6 instances): wall 0.497 ms, GPU busy 0.509 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.123 | 24.1% |
| wo_a | 0.050 | 9.8% |
| wo_b | 0.043 | 8.5% |
| shared_expert | 0.043 | 8.4% |
| router | 0.042 | 8.3% |
| (unranged) | 0.040 | 7.9% |
| index_q | 0.034 | 6.6% |
| q_b | 0.031 | 6.2% |
| compressor | 0.023 | 4.5% |
| kv_window | 0.013 | 2.6% |
| moe_plan | 0.013 | 2.5% |
| sparse_attn | 0.012 | 2.4% |
| qr | 0.012 | 2.4% |
| index_keys | 0.011 | 2.1% |
| kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| candidates | 0.003 | 0.6% |
| index_score | 0.002 | 0.4% |
| topk | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.3% |

host API calls: 141 calls, 0.447 ms; top: cuStreamSynchronize 0.289 ms x2, cudaLaunchKernel_v7000 0.133 ms x62, cuMemcpyDtoDAsync_v2 0.013 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuMemAllocAsync 0.002 ms x3

### L21/decode (6 instances): wall 0.419 ms, GPU busy 0.438 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.122 | 27.9% |
| wo_a | 0.050 | 11.4% |
| wo_b | 0.045 | 10.4% |
| router | 0.045 | 10.2% |
| shared_expert | 0.044 | 9.9% |
| (unranged) | 0.042 | 9.5% |
| q_b | 0.031 | 7.2% |
| kv_window | 0.014 | 3.1% |
| qr | 0.013 | 2.9% |
| moe_plan | 0.013 | 2.9% |
| sparse_attn | 0.013 | 2.9% |
| combine | 0.005 | 1.0% |
| attn_idx_rows | 0.001 | 0.3% |
| kv_append | 0.001 | 0.2% |

host API calls: 105 calls, 0.384 ms; top: cuStreamSynchronize 0.265 ms x2, cudaLaunchKernel_v7000 0.099 ms x46, cuMemcpyDtoDAsync_v2 0.010 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

