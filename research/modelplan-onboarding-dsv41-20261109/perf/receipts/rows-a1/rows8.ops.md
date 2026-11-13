### L2/prefill (5 instances): wall 1.397 ms, GPU busy 1.442 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.886 | 61.5% |
| shared_expert | 0.084 | 5.8% |
| compressor | 0.067 | 4.6% |
| index_q | 0.065 | 4.5% |
| wo_b | 0.057 | 4.0% |
| (unranged) | 0.057 | 3.9% |
| wo_a | 0.053 | 3.7% |
| router | 0.041 | 2.8% |
| q_b | 0.038 | 2.6% |
| kv_window | 0.019 | 1.3% |
| qr | 0.017 | 1.2% |
| moe_plan | 0.015 | 1.0% |
| index_keys | 0.012 | 0.8% |
| index_score | 0.008 | 0.5% |
| combine | 0.008 | 0.5% |
| sparse_attn | 0.007 | 0.5% |
| kv_append | 0.005 | 0.4% |
| topk | 0.003 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 140 calls, 1.335 ms; top: cuStreamSynchronize 1.154 ms x2, cudaLaunchKernel_v7000 0.152 ms x63, cuMemcpyDtoDAsync_v2 0.017 ms x5, cuKernelGetName 0.006 ms x63, cuMemsetD8Async 0.005 ms x3, cuStreamWaitEvent 0.001 ms x2

### L20/prefill (5 instances): wall 1.304 ms, GPU busy 1.343 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.800 | 59.5% |
| shared_expert | 0.082 | 6.1% |
| index_q | 0.065 | 4.8% |
| wo_b | 0.057 | 4.2% |
| (unranged) | 0.057 | 4.2% |
| wo_a | 0.054 | 4.0% |
| compressor | 0.052 | 3.9% |
| router | 0.041 | 3.1% |
| q_b | 0.038 | 2.8% |
| kv_window | 0.019 | 1.4% |
| qr | 0.018 | 1.3% |
| moe_plan | 0.015 | 1.1% |
| index_keys | 0.012 | 0.9% |
| index_score | 0.009 | 0.7% |
| combine | 0.008 | 0.6% |
| sparse_attn | 0.007 | 0.5% |
| kv_append | 0.006 | 0.4% |
| topk | 0.003 | 0.2% |
| candidates | 0.002 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 138 calls, 1.248 ms; top: cuStreamSynchronize 1.077 ms x2, cudaLaunchKernel_v7000 0.143 ms x62, cuMemcpyDtoDAsync_v2 0.015 ms x5, cuKernelGetName 0.006 ms x62, cuMemsetD8Async 0.005 ms x3, cuStreamWaitEvent 0.001 ms x2

### L21/prefill (5 instances): wall 1.086 ms, GPU busy 1.137 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.737 | 64.8% |
| shared_expert | 0.084 | 7.4% |
| wo_b | 0.058 | 5.1% |
| (unranged) | 0.057 | 5.0% |
| wo_a | 0.053 | 4.7% |
| router | 0.041 | 3.6% |
| q_b | 0.039 | 3.4% |
| kv_window | 0.019 | 1.7% |
| qr | 0.017 | 1.5% |
| moe_plan | 0.015 | 1.3% |
| combine | 0.008 | 0.7% |
| sparse_attn | 0.007 | 0.6% |
| attn_idx_rows | 0.001 | 0.1% |
| kv_append | 0.001 | 0.1% |

host API calls: 105 calls, 1.043 ms; top: cuStreamSynchronize 0.914 ms x2, cudaLaunchKernel_v7000 0.106 ms x46, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuMemsetD8Async 0.005 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

### L2/decode (6 instances): wall 0.525 ms, GPU busy 0.529 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.126 | 23.8% |
| wo_a | 0.050 | 9.5% |
| compressor | 0.049 | 9.3% |
| shared_expert | 0.043 | 8.1% |
| wo_b | 0.043 | 8.1% |
| router | 0.042 | 7.9% |
| (unranged) | 0.041 | 7.7% |
| index_q | 0.034 | 6.4% |
| q_b | 0.032 | 6.0% |
| kv_window | 0.014 | 2.6% |
| moe_plan | 0.013 | 2.5% |
| qr | 0.013 | 2.4% |
| sparse_attn | 0.012 | 2.3% |
| index_keys | 0.006 | 1.0% |
| combine | 0.005 | 0.9% |
| kv_append | 0.003 | 0.6% |
| topk | 0.002 | 0.4% |
| index_score | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.2% |

host API calls: 142 calls, 0.462 ms; top: cuStreamSynchronize 0.291 ms x2, cudaLaunchKernel_v7000 0.135 ms x59, cuMemcpyDtoDAsync_v2 0.019 ms x7, cuMemAllocAsync 0.006 ms x8, cuKernelGetName 0.006 ms x59, cuMemsetD8Async 0.004 ms x3

### L20/decode (6 instances): wall 0.504 ms, GPU busy 0.517 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.125 | 24.1% |
| wo_a | 0.050 | 9.7% |
| router | 0.044 | 8.6% |
| wo_b | 0.044 | 8.5% |
| shared_expert | 0.043 | 8.4% |
| (unranged) | 0.041 | 7.9% |
| index_q | 0.034 | 6.5% |
| q_b | 0.032 | 6.1% |
| compressor | 0.024 | 4.6% |
| kv_window | 0.014 | 2.7% |
| moe_plan | 0.013 | 2.5% |
| qr | 0.013 | 2.5% |
| sparse_attn | 0.012 | 2.4% |
| index_keys | 0.011 | 2.1% |
| kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| candidates | 0.003 | 0.6% |
| index_score | 0.002 | 0.4% |
| topk | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.3% |

host API calls: 141 calls, 0.453 ms; top: cuStreamSynchronize 0.288 ms x2, cudaLaunchKernel_v7000 0.139 ms x62, cuMemcpyDtoDAsync_v2 0.014 ms x5, cuKernelGetName 0.006 ms x62, cuMemsetD8Async 0.004 ms x3, cuMemAllocAsync 0.002 ms x3

### L21/decode (6 instances): wall 0.433 ms, GPU busy 0.451 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.127 | 28.1% |
| wo_a | 0.051 | 11.2% |
| router | 0.048 | 10.6% |
| wo_b | 0.046 | 10.2% |
| shared_expert | 0.044 | 9.8% |
| (unranged) | 0.043 | 9.5% |
| q_b | 0.032 | 7.0% |
| kv_window | 0.014 | 3.1% |
| sparse_attn | 0.013 | 3.0% |
| moe_plan | 0.013 | 2.9% |
| qr | 0.013 | 2.9% |
| combine | 0.005 | 1.1% |
| attn_idx_rows | 0.001 | 0.3% |
| kv_append | 0.001 | 0.2% |

host API calls: 105 calls, 0.395 ms; top: cuStreamSynchronize 0.271 ms x2, cudaLaunchKernel_v7000 0.104 ms x46, cuMemcpyDtoDAsync_v2 0.011 ms x4, cuKernelGetName 0.004 ms x46, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

