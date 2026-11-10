### L2/prefill (5 instances): wall 1.211 ms, GPU busy 1.252 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.703 | 56.1% |
| shared_expert | 0.082 | 6.5% |
| compressor | 0.067 | 5.3% |
| index_q | 0.065 | 5.2% |
| wo_b | 0.056 | 4.5% |
| (unranged) | 0.056 | 4.4% |
| wo_a | 0.053 | 4.3% |
| router | 0.041 | 3.2% |
| q_b | 0.038 | 3.0% |
| kv_window | 0.018 | 1.5% |
| qr | 0.016 | 1.3% |
| moe_plan | 0.014 | 1.1% |
| index_keys | 0.012 | 0.9% |
| index_score | 0.008 | 0.6% |
| combine | 0.007 | 0.6% |
| sparse_attn | 0.007 | 0.6% |
| kv_append | 0.005 | 0.4% |
| topk | 0.003 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 140 calls, 1.154 ms; top: cuStreamSynchronize 0.986 ms x2, cudaLaunchKernel_v7000 0.141 ms x63, cuMemcpyDtoDAsync_v2 0.016 ms x5, cuKernelGetName 0.005 ms x63, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L20/prefill (5 instances): wall 1.131 ms, GPU busy 1.169 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.630 | 53.9% |
| shared_expert | 0.080 | 6.8% |
| index_q | 0.065 | 5.6% |
| wo_b | 0.056 | 4.8% |
| (unranged) | 0.055 | 4.7% |
| wo_a | 0.053 | 4.5% |
| compressor | 0.051 | 4.4% |
| router | 0.043 | 3.7% |
| q_b | 0.038 | 3.2% |
| kv_window | 0.019 | 1.6% |
| qr | 0.017 | 1.5% |
| moe_plan | 0.015 | 1.3% |
| index_keys | 0.012 | 1.0% |
| index_score | 0.009 | 0.7% |
| combine | 0.007 | 0.6% |
| sparse_attn | 0.007 | 0.6% |
| kv_append | 0.006 | 0.5% |
| topk | 0.003 | 0.2% |
| candidates | 0.002 | 0.2% |
| attn_idx_rows | 0.001 | 0.1% |

host API calls: 138 calls, 1.080 ms; top: cuStreamSynchronize 0.921 ms x2, cudaLaunchKernel_v7000 0.134 ms x62, cuMemcpyDtoDAsync_v2 0.014 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuStreamWaitEvent 0.001 ms x2

### L21/prefill (5 instances): wall 0.922 ms, GPU busy 0.971 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.577 | 59.5% |
| shared_expert | 0.080 | 8.2% |
| wo_b | 0.057 | 5.8% |
| (unranged) | 0.055 | 5.6% |
| wo_a | 0.052 | 5.4% |
| router | 0.043 | 4.5% |
| q_b | 0.038 | 3.9% |
| kv_window | 0.019 | 1.9% |
| qr | 0.017 | 1.7% |
| moe_plan | 0.017 | 1.7% |
| combine | 0.007 | 0.7% |
| sparse_attn | 0.007 | 0.7% |
| attn_idx_rows | 0.001 | 0.1% |
| kv_append | 0.001 | 0.1% |

host API calls: 105 calls, 0.883 ms; top: cuStreamSynchronize 0.763 ms x2, cudaLaunchKernel_v7000 0.099 ms x46, cuMemcpyDtoDAsync_v2 0.011 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

### L2/decode (6 instances): wall 0.503 ms, GPU busy 0.508 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.106 | 21.0% |
| wo_a | 0.049 | 9.6% |
| compressor | 0.049 | 9.6% |
| router | 0.044 | 8.6% |
| shared_expert | 0.043 | 8.5% |
| wo_b | 0.043 | 8.4% |
| (unranged) | 0.040 | 7.9% |
| index_q | 0.034 | 6.6% |
| q_b | 0.032 | 6.2% |
| kv_window | 0.014 | 2.7% |
| moe_plan | 0.013 | 2.5% |
| qr | 0.012 | 2.4% |
| sparse_attn | 0.012 | 2.4% |
| index_keys | 0.005 | 1.1% |
| combine | 0.004 | 0.9% |
| kv_append | 0.003 | 0.6% |
| topk | 0.002 | 0.4% |
| index_score | 0.002 | 0.4% |
| attn_idx_rows | 0.001 | 0.3% |

host API calls: 142 calls, 0.446 ms; top: cuStreamSynchronize 0.288 ms x2, cudaLaunchKernel_v7000 0.125 ms x59, cuMemcpyDtoDAsync_v2 0.017 ms x7, cuMemAllocAsync 0.006 ms x8, cuKernelGetName 0.005 ms x59, cuMemsetD8Async 0.004 ms x3

### L20/decode (6 instances): wall 0.481 ms, GPU busy 0.494 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.106 | 21.5% |
| wo_a | 0.050 | 10.0% |
| wo_b | 0.044 | 8.8% |
| router | 0.043 | 8.7% |
| shared_expert | 0.043 | 8.7% |
| (unranged) | 0.040 | 8.2% |
| index_q | 0.033 | 6.7% |
| q_b | 0.031 | 6.4% |
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

host API calls: 141 calls, 0.434 ms; top: cuStreamSynchronize 0.280 ms x2, cudaLaunchKernel_v7000 0.130 ms x62, cuMemcpyDtoDAsync_v2 0.013 ms x5, cuKernelGetName 0.005 ms x62, cuMemsetD8Async 0.004 ms x3, cuMemAllocAsync 0.002 ms x3

### L21/decode (6 instances): wall 0.404 ms, GPU busy 0.422 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.107 | 25.4% |
| wo_a | 0.051 | 12.0% |
| wo_b | 0.046 | 10.8% |
| shared_expert | 0.043 | 10.2% |
| router | 0.043 | 10.1% |
| (unranged) | 0.042 | 9.9% |
| q_b | 0.032 | 7.5% |
| kv_window | 0.014 | 3.2% |
| moe_plan | 0.013 | 3.1% |
| qr | 0.013 | 3.1% |
| sparse_attn | 0.013 | 3.1% |
| combine | 0.005 | 1.1% |
| attn_idx_rows | 0.001 | 0.3% |
| kv_append | 0.001 | 0.2% |

host API calls: 105 calls, 0.370 ms; top: cuStreamSynchronize 0.254 ms x2, cudaLaunchKernel_v7000 0.097 ms x46, cuMemcpyDtoDAsync_v2 0.011 ms x4, cuMemsetD8Async 0.004 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.001 ms x2

