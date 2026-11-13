### L2/prefill (2 instances): wall 34.863 ms, GPU busy 34.770 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.156 | 49.3% |
| sparse_attn | 4.067 | 11.7% |
| (unranged) | 2.469 | 7.1% |
| wo_a | 2.428 | 7.0% |
| q_b | 1.641 | 4.7% |
| wo_b | 1.565 | 4.5% |
| shared_expert | 1.524 | 4.4% |
| compressor | 1.478 | 4.2% |
| router | 0.711 | 2.0% |
| index_q | 0.606 | 1.7% |
| qr | 0.361 | 1.0% |
| kv_window | 0.223 | 0.6% |
| combine | 0.205 | 0.6% |
| index_score | 0.185 | 0.5% |
| topk | 0.068 | 0.2% |
| index_keys | 0.038 | 0.1% |
| kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 202 calls, 34.681 ms; top: cuStreamSynchronize 33.988 ms x2, cuMemAllocAsync 0.364 ms x34, cudaLaunchKernel_v7000 0.217 ms x63, cuMemcpyDtoDAsync_v2 0.052 ms x5, cuMemFreeAsync 0.025 ms x32, cuMemsetD8Async 0.023 ms x3

### L20/prefill (2 instances): wall 33.075 ms, GPU busy 32.955 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.110 | 48.9% |
| sparse_attn | 4.208 | 12.8% |
| (unranged) | 2.495 | 7.6% |
| wo_a | 2.444 | 7.4% |
| q_b | 1.646 | 5.0% |
| wo_b | 1.578 | 4.8% |
| shared_expert | 1.537 | 4.7% |
| router | 0.719 | 2.2% |
| index_q | 0.589 | 1.8% |
| index_score | 0.390 | 1.2% |
| qr | 0.362 | 1.1% |
| kv_window | 0.224 | 0.7% |
| compressor | 0.211 | 0.6% |
| combine | 0.205 | 0.6% |
| topk | 0.101 | 0.3% |
| index_keys | 0.047 | 0.1% |
| kv_append | 0.037 | 0.1% |
| candidates | 0.028 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 197 calls, 32.884 ms; top: cuStreamSynchronize 32.170 ms x2, cuMemAllocAsync 0.370 ms x32, cudaLaunchKernel_v7000 0.225 ms x62, cuMemcpyDtoDAsync_v2 0.059 ms x5, cuMemFreeAsync 0.024 ms x31, cuMemsetD8Async 0.024 ms x3

### L21/prefill (2 instances): wall 31.911 ms, GPU busy 31.827 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.353 | 51.4% |
| sparse_attn | 4.232 | 13.3% |
| (unranged) | 2.495 | 7.8% |
| wo_a | 2.449 | 7.7% |
| q_b | 1.642 | 5.2% |
| wo_b | 1.576 | 5.0% |
| shared_expert | 1.540 | 4.8% |
| router | 0.718 | 2.3% |
| qr | 0.362 | 1.1% |
| kv_window | 0.225 | 0.7% |
| combine | 0.205 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |
| kv_append | 0.004 | 0.0% |

host API calls: 154 calls, 31.761 ms; top: cuStreamSynchronize 29.765 ms x2, cuMemAllocAsync 1.734 ms x27, cudaLaunchKernel_v7000 0.166 ms x46, cuMemcpyDtoDAsync_v2 0.046 ms x4, cuMemsetD8Async 0.022 ms x3, cuMemFreeAsync 0.020 ms x26

### L2/decode (6 instances): wall 0.632 ms, GPU busy 0.606 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 26.6% |
| compressor | 0.050 | 8.3% |
| wo_a | 0.050 | 8.2% |
| wo_b | 0.045 | 7.4% |
| shared_expert | 0.044 | 7.3% |
| router | 0.043 | 7.0% |
| (unranged) | 0.040 | 6.6% |
| index_q | 0.033 | 5.5% |
| sparse_attn | 0.033 | 5.4% |
| q_b | 0.033 | 5.4% |
| kv_window | 0.015 | 2.5% |
| topk | 0.013 | 2.2% |
| moe_plan | 0.013 | 2.2% |
| qr | 0.013 | 2.2% |
| kv_append | 0.005 | 0.9% |
| index_keys | 0.005 | 0.9% |
| combine | 0.005 | 0.8% |
| index_score | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 149 calls, 0.534 ms; top: cuStreamSynchronize 0.302 ms x2, cudaLaunchKernel_v7000 0.176 ms x59, cuMemcpyDtoDAsync_v2 0.023 ms x7, cuMemAllocAsync 0.011 ms x9, cuMemsetD8Async 0.007 ms x3, cuKernelGetName 0.006 ms x59

### L20/decode (6 instances): wall 0.575 ms, GPU busy 0.591 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.161 | 27.3% |
| wo_a | 0.050 | 8.4% |
| shared_expert | 0.044 | 7.5% |
| wo_b | 0.043 | 7.3% |
| router | 0.041 | 7.0% |
| (unranged) | 0.040 | 6.8% |
| index_q | 0.033 | 5.6% |
| q_b | 0.033 | 5.6% |
| sparse_attn | 0.033 | 5.6% |
| compressor | 0.025 | 4.2% |
| kv_window | 0.015 | 2.5% |
| moe_plan | 0.013 | 2.2% |
| qr | 0.013 | 2.2% |
| topk | 0.013 | 2.2% |
| index_keys | 0.011 | 1.8% |
| kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.8% |
| candidates | 0.004 | 0.6% |
| index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 146 calls, 0.508 ms; top: cuStreamSynchronize 0.317 ms x2, cudaLaunchKernel_v7000 0.155 ms x62, cuMemcpyDtoDAsync_v2 0.017 ms x5, cuKernelGetName 0.006 ms x62, cuMemsetD8Async 0.005 ms x3, cuMemAllocAsync 0.003 ms x2

### L21/decode (6 instances): wall 0.486 ms, GPU busy 0.507 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.163 | 32.2% |
| wo_a | 0.050 | 9.8% |
| shared_expert | 0.045 | 8.8% |
| wo_b | 0.044 | 8.7% |
| router | 0.042 | 8.2% |
| (unranged) | 0.041 | 8.2% |
| sparse_attn | 0.035 | 6.9% |
| q_b | 0.033 | 6.5% |
| kv_window | 0.015 | 3.0% |
| moe_plan | 0.013 | 2.6% |
| qr | 0.013 | 2.6% |
| kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 111 calls, 0.441 ms; top: cuStreamSynchronize 0.302 ms x2, cudaLaunchKernel_v7000 0.112 ms x46, cuMemcpyDtoDAsync_v2 0.014 ms x4, cuMemsetD8Async 0.005 ms x3, cuKernelGetName 0.004 ms x46, cuStreamWaitEvent 0.002 ms x4

