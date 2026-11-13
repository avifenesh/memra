### L2/prefill (2 instances): wall 297.447 ms, GPU busy 297.254 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 128.654 | 43.3% |
| sparse_attn | 34.880 | 11.7% |
| attn_keys/index_score | 21.988 | 7.4% |
| wo_a | 20.095 | 6.8% |
| shared_expert | 13.680 | 4.6% |
| q_b | 13.420 | 4.5% |
| wo_b | 12.908 | 4.3% |
| attn_keys/compressor | 11.622 | 3.9% |
| hc_ffn_pre | 10.290 | 3.5% |
| hc_attn_pre | 7.981 | 2.7% |
| router | 5.754 | 1.9% |
| attn_keys/index_q | 4.412 | 1.5% |
| attn_keys/qr | 3.007 | 1.0% |
| hc_post | 2.216 | 0.7% |
| attn_keys/topk | 2.141 | 0.7% |
| combine | 1.893 | 0.6% |
| attn_keys/kv_window | 1.868 | 0.6% |
| attn_keys/index_keys | 0.138 | 0.0% |
| attn_keys/kv_append | 0.127 | 0.0% |
| attn_idx_rows | 0.102 | 0.0% |
| moe_plan | 0.079 | 0.0% |

host API calls: 627 calls, 297.086 ms; top: cuStreamSynchronize 294.158 ms x1, cuMemAllocAsync 2.217 ms x134, cudaLaunchKernel_v7000 0.521 ms x172, cuMemFreeAsync 0.080 ms x132, cuMemcpyDtoDAsync_v2 0.044 ms x4, cuMemsetD8Async 0.043 ms x12

### L20/prefill (2 instances): wall 308.538 ms, GPU busy 308.304 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 122.245 | 39.7% |
| attn_keys/index_score | 43.971 | 14.3% |
| sparse_attn | 34.972 | 11.3% |
| wo_a | 20.361 | 6.6% |
| shared_expert | 13.749 | 4.5% |
| q_b | 13.510 | 4.4% |
| wo_b | 13.020 | 4.2% |
| hc_ffn_pre | 10.527 | 3.4% |
| hc_attn_pre | 8.039 | 2.6% |
| router | 5.809 | 1.9% |
| attn_keys/index_q | 4.434 | 1.4% |
| attn_keys/topk | 4.261 | 1.4% |
| attn_keys/qr | 3.014 | 1.0% |
| hc_post | 2.223 | 0.7% |
| attn_keys/candidates | 2.220 | 0.7% |
| combine | 1.905 | 0.6% |
| attn_keys/kv_window | 1.875 | 0.6% |
| attn_keys/compressor | 1.478 | 0.5% |
| attn_keys/index_keys | 0.260 | 0.1% |
| attn_keys/kv_append | 0.242 | 0.1% |
| attn_idx_rows | 0.106 | 0.0% |
| moe_plan | 0.082 | 0.0% |

host API calls: 629 calls, 308.103 ms; top: cuStreamSynchronize 304.873 ms x1, cuMemAllocAsync 2.382 ms x133, cudaLaunchKernel_v7000 0.609 ms x174, cuMemFreeAsync 0.107 ms x131, cuMemsetD8Async 0.059 ms x12, cuMemcpyDtoDAsync_v2 0.047 ms x4

### L21/prefill (2 instances): wall 247.672 ms, GPU busy 247.488 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 119.609 | 48.3% |
| sparse_attn | 34.880 | 14.1% |
| wo_a | 20.149 | 8.1% |
| q_b | 13.639 | 5.5% |
| shared_expert | 13.399 | 5.4% |
| wo_b | 12.898 | 5.2% |
| hc_ffn_pre | 10.173 | 4.1% |
| hc_attn_pre | 8.002 | 3.2% |
| router | 5.559 | 2.2% |
| attn_keys/qr | 3.015 | 1.2% |
| hc_post | 2.218 | 0.9% |
| attn_keys/kv_window | 1.873 | 0.8% |
| combine | 1.864 | 0.8% |
| attn_idx_rows | 0.104 | 0.0% |
| moe_plan | 0.078 | 0.0% |
| attn_keys/kv_append | 0.029 | 0.0% |

host API calls: 518 calls, 247.383 ms; top: cuStreamSynchronize 245.741 ms x1, cuMemAllocAsync 1.088 ms x118, cudaLaunchKernel_v7000 0.396 ms x134, cuMemFreeAsync 0.064 ms x116, cuMemsetD8Async 0.039 ms x12, cuMemcpyDtoDAsync_v2 0.037 ms x3

### L2/decode (6 instances): wall 0.703 ms, GPU busy 0.632 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 24.9% |
| wo_a | 0.049 | 7.8% |
| attn_keys/compressor | 0.049 | 7.7% |
| wo_b | 0.044 | 6.9% |
| shared_expert | 0.044 | 6.9% |
| attn_keys/topk | 0.037 | 5.8% |
| sparse_attn | 0.036 | 5.7% |
| attn_keys/index_q | 0.034 | 5.4% |
| q_b | 0.032 | 5.1% |
| router | 0.029 | 4.6% |
| hc_ffn_pre | 0.021 | 3.3% |
| hc_attn_pre | 0.017 | 2.7% |
| attn_keys/index_score | 0.016 | 2.5% |
| attn_keys/kv_append | 0.015 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.2% |
| attn_keys/qr | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.9% |
| attn_keys/index_keys | 0.006 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.610 ms; top: cuStreamSynchronize 0.377 ms x1, cudaLaunchKernel_v7000 0.153 ms x60, cuMemAllocAsync 0.030 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.018 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.724 ms, GPU busy 0.686 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 23.3% |
| attn_keys/topk | 0.062 | 9.0% |
| wo_a | 0.050 | 7.3% |
| wo_b | 0.045 | 6.5% |
| shared_expert | 0.044 | 6.4% |
| sparse_attn | 0.036 | 5.3% |
| attn_keys/index_q | 0.034 | 4.9% |
| attn_keys/candidates | 0.032 | 4.7% |
| q_b | 0.032 | 4.6% |
| router | 0.030 | 4.4% |
| attn_keys/kv_append | 0.027 | 3.9% |
| attn_keys/compressor | 0.023 | 3.4% |
| hc_ffn_pre | 0.021 | 3.1% |
| hc_attn_pre | 0.017 | 2.5% |
| attn_keys/index_score | 0.016 | 2.4% |
| attn_keys/kv_window | 0.014 | 2.0% |
| attn_keys/qr | 0.012 | 1.8% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/index_keys | 0.011 | 1.5% |
| combine | 0.004 | 0.6% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.657 ms; top: cuStreamSynchronize 0.454 ms x1, cudaLaunchKernel_v7000 0.135 ms x63, cuMemAllocAsync 0.026 ms x52, cuMemFreeAsync 0.020 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.541 ms, GPU busy 0.507 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 31.6% |
| wo_a | 0.050 | 9.9% |
| wo_b | 0.044 | 8.7% |
| shared_expert | 0.044 | 8.7% |
| sparse_attn | 0.037 | 7.3% |
| q_b | 0.032 | 6.3% |
| router | 0.030 | 5.9% |
| attn_keys/kv_append | 0.023 | 4.6% |
| hc_ffn_pre | 0.021 | 4.1% |
| hc_attn_pre | 0.017 | 3.4% |
| attn_keys/kv_window | 0.014 | 2.8% |
| attn_keys/qr | 0.013 | 2.5% |
| moe_plan | 0.012 | 2.4% |
| combine | 0.004 | 0.9% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.490 ms; top: cuStreamSynchronize 0.332 ms x1, cudaLaunchKernel_v7000 0.101 ms x47, cuMemAllocAsync 0.022 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

