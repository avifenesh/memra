### L2/prefill (5 instances): wall 1.218 ms, GPU busy 1.263 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.714 | 56.5% |
| shared_expert | 0.083 | 6.6% |
| compressor | 0.067 | 5.3% |
| index_q | 0.065 | 5.1% |
| (unranged) | 0.056 | 4.5% |
| wo_a | 0.054 | 4.2% |
| wo_b | 0.053 | 4.2% |
| router | 0.041 | 3.2% |
| q_b | 0.038 | 3.0% |
| kv_window | 0.019 | 1.5% |
| qr | 0.017 | 1.3% |
| moe_plan | 0.014 | 1.1% |
| index_keys | 0.012 | 0.9% |
| index_score | 0.008 | 0.6% |
| combine | 0.007 | 0.6% |
| sparse_attn | 0.007 | 0.6% |
| kv_append | 0.005 | 0.4% |
| topk | 0.003 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 140 calls, 1.162 ms; top: cuStreamSynchronize 0.995 ms x2, cudaLaunchKernel_v7000 0.139 ms x63, cuMemcpyDtoDAsync_v2 0.016 ms x5, cuKernelGetName 0.005 ms x63, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L20/prefill (5 instances): wall 1.137 ms, GPU busy 1.175 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.637 | 54.2% |
| shared_expert | 0.081 | 6.9% |
| index_q | 0.065 | 5.5% |
| (unranged) | 0.056 | 4.8% |
| wo_a | 0.054 | 4.6% |
| wo_b | 0.052 | 4.4% |
| compressor | 0.051 | 4.4% |
| router | 0.041 | 3.5% |
| q_b | 0.038 | 3.2% |
| kv_window | 0.019 | 1.6% |
| qr | 0.017 | 1.5% |
| moe_plan | 0.017 | 1.4% |
| index_keys | 0.012 | 1.0% |
| index_score | 0.009 | 0.7% |
| sparse_attn | 0.007 | 0.6% |
| combine | 0.007 | 0.6% |
| kv_append | 0.006 | 0.5% |
| topk | 0.003 | 0.2% |
| candidates | 0.002 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 138 calls, 1.086 ms; top: cuStreamSynchronize 0.926 ms x2, cudaLaunchKernel_v7000 0.135 ms x62, cuMemcpyDtoDAsync_v2 0.014 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L21/prefill (5 instances): wall 0.926 ms, GPU busy 0.975 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.584 | 59.9% |
| shared_expert | 0.081 | 8.3% |
| (unranged) | 0.056 | 5.7% |
| wo_a | 0.053 | 5.5% |
| wo_b | 0.052 | 5.4% |
| router | 0.042 | 4.3% |
| q_b | 0.039 | 4.0% |
| kv_window | 0.019 | 2.0% |
| qr | 0.017 | 1.7% |
| moe_plan | 0.016 | 1.6% |
| combine | 0.007 | 0.8% |
| sparse_attn | 0.007 | 0.7% |
| attn_idx_rows | 0.001 | 0.1% |
| kv_append | 0.001 | 0.1% |

host API calls: 105 calls, 0.886 ms; top: cuStreamSynchronize 0.766 ms x2, cudaLaunchKernel_v7000 0.099 ms x46, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuMemsetD8Async 0.005 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

### L2/decode (6 instances): wall 0.505 ms, GPU busy 0.510 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.108 | 21.1% |
| wo_a | 0.049 | 9.7% |
| compressor | 0.049 | 9.6% |
| shared_expert | 0.043 | 8.5% |
| router | 0.043 | 8.4% |
| wo_b | 0.042 | 8.2% |
| (unranged) | 0.040 | 7.9% |
| index_q | 0.034 | 6.7% |
| q_b | 0.032 | 6.2% |
| kv_window | 0.014 | 2.7% |
| moe_plan | 0.013 | 2.6% |
| sparse_attn | 0.012 | 2.4% |
| qr | 0.012 | 2.4% |
| index_keys | 0.006 | 1.1% |
| combine | 0.004 | 0.9% |
| kv_append | 0.003 | 0.6% |
| topk | 0.002 | 0.4% |
| index_score | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.3% |

host API calls: 142 calls, 0.447 ms; top: cuStreamSynchronize 0.287 ms x2, cudaLaunchKernel_v7000 0.125 ms x59, cuMemcpyDtoDAsync_v2 0.018 ms x7, cuMemAllocAsync 0.006 ms x8, cuKernelGetName 0.005 ms x59, cuMemsetD8Async 0.004 ms x3

### L20/decode (6 instances): wall 0.482 ms, GPU busy 0.495 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.108 | 21.7% |
| wo_a | 0.050 | 10.1% |
| shared_expert | 0.043 | 8.8% |
| router | 0.042 | 8.5% |
| wo_b | 0.042 | 8.5% |
| (unranged) | 0.040 | 8.2% |
| index_q | 0.034 | 6.8% |
| q_b | 0.032 | 6.4% |
| compressor | 0.023 | 4.7% |
| kv_window | 0.014 | 2.8% |
| moe_plan | 0.013 | 2.6% |
| qr | 0.013 | 2.5% |
| sparse_attn | 0.012 | 2.5% |
| index_keys | 0.011 | 2.2% |
| kv_append | 0.005 | 1.1% |
| combine | 0.004 | 0.9% |
| candidates | 0.003 | 0.6% |
| index_score | 0.002 | 0.4% |
| topk | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.3% |

host API calls: 141 calls, 0.434 ms; top: cuStreamSynchronize 0.278 ms x2, cudaLaunchKernel_v7000 0.131 ms x62, cuMemcpyDtoDAsync_v2 0.013 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuMemAllocAsync 0.002 ms x3

### L21/decode (6 instances): wall 0.407 ms, GPU busy 0.426 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.108 | 25.5% |
| wo_a | 0.051 | 11.9% |
| wo_b | 0.045 | 10.6% |
| router | 0.044 | 10.4% |
| shared_expert | 0.044 | 10.3% |
| (unranged) | 0.042 | 9.9% |
| q_b | 0.032 | 7.5% |
| kv_window | 0.014 | 3.2% |
| moe_plan | 0.013 | 3.1% |
| sparse_attn | 0.013 | 3.1% |
| qr | 0.013 | 3.0% |
| combine | 0.005 | 1.1% |
| attn_idx_rows | 0.001 | 0.3% |
| kv_append | 0.001 | 0.2% |

host API calls: 105 calls, 0.372 ms; top: cuStreamSynchronize 0.256 ms x2, cudaLaunchKernel_v7000 0.097 ms x46, cuMemcpyDtoDAsync_v2 0.010 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

