### L2/prefill (2 instances): wall 35.641 ms, GPU busy 35.478 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.577 | 49.5% |
| sparse_attn | 4.214 | 11.9% |
| wo_a | 2.481 | 7.0% |
| q_b | 1.693 | 4.8% |
| shared_expert | 1.638 | 4.6% |
| wo_b | 1.596 | 4.5% |
| attn_keys/compressor | 1.513 | 4.3% |
| hc_ffn_pre | 1.174 | 3.3% |
| hc_attn_pre | 0.889 | 2.5% |
| router | 0.723 | 2.0% |
| attn_keys/index_q | 0.602 | 1.7% |
| attn_keys/qr | 0.368 | 1.0% |
| hc_post | 0.235 | 0.7% |
| attn_keys/kv_window | 0.230 | 0.6% |
| combine | 0.205 | 0.6% |
| attn_keys/index_score | 0.186 | 0.5% |
| attn_keys/topk | 0.069 | 0.2% |
| attn_keys/index_keys | 0.039 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| moe_plan | 0.016 | 0.0% |
| attn_idx_rows | 0.010 | 0.0% |

host API calls: 240 calls, 35.387 ms; top: cuStreamSynchronize 34.418 ms x1, cuMemAllocAsync 0.512 ms x53, cudaLaunchKernel_v7000 0.324 ms x64, cuMemFreeAsync 0.049 ms x51, cuMemcpyDtoDAsync_v2 0.038 ms x4, cuMemsetD8Async 0.027 ms x3

### L20/prefill (2 instances): wall 33.760 ms, GPU busy 33.633 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.498 | 49.1% |
| sparse_attn | 4.324 | 12.9% |
| wo_a | 2.507 | 7.5% |
| q_b | 1.716 | 5.1% |
| shared_expert | 1.657 | 4.9% |
| wo_b | 1.609 | 4.8% |
| hc_ffn_pre | 1.190 | 3.5% |
| hc_attn_pre | 0.900 | 2.7% |
| router | 0.732 | 2.2% |
| attn_keys/index_q | 0.602 | 1.8% |
| attn_keys/index_score | 0.397 | 1.2% |
| attn_keys/qr | 0.372 | 1.1% |
| hc_post | 0.236 | 0.7% |
| attn_keys/kv_window | 0.233 | 0.7% |
| attn_keys/compressor | 0.210 | 0.6% |
| combine | 0.205 | 0.6% |
| attn_keys/topk | 0.103 | 0.3% |
| attn_keys/index_keys | 0.049 | 0.1% |
| attn_keys/kv_append | 0.038 | 0.1% |
| attn_keys/candidates | 0.029 | 0.1% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 33.580 ms; top: cuStreamSynchronize 32.725 ms x1, cuMemAllocAsync 0.522 ms x52, cudaLaunchKernel_v7000 0.235 ms x63, cuMemFreeAsync 0.038 ms x50, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.024 ms x3

### L21/prefill (2 instances): wall 32.644 ms, GPU busy 32.508 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 16.752 | 51.5% |
| sparse_attn | 4.382 | 13.5% |
| wo_a | 2.506 | 7.7% |
| q_b | 1.693 | 5.2% |
| shared_expert | 1.667 | 5.1% |
| wo_b | 1.612 | 5.0% |
| hc_ffn_pre | 1.187 | 3.7% |
| hc_attn_pre | 0.900 | 2.8% |
| router | 0.733 | 2.3% |
| attn_keys/qr | 0.371 | 1.1% |
| hc_post | 0.239 | 0.7% |
| attn_keys/kv_window | 0.231 | 0.7% |
| combine | 0.206 | 0.6% |
| moe_plan | 0.017 | 0.1% |
| attn_idx_rows | 0.010 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 32.479 ms; top: cuStreamSynchronize 31.887 ms x1, cuMemAllocAsync 0.286 ms x46, cudaLaunchKernel_v7000 0.204 ms x47, cuMemFreeAsync 0.037 ms x44, cuMemcpyDtoDAsync_v2 0.034 ms x3, cuMemsetD8Async 0.021 ms x3

### L2/decode (6 instances): wall 1.600 ms, GPU busy 0.586 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.157 | 26.8% |
| wo_a | 0.049 | 8.3% |
| attn_keys/compressor | 0.048 | 8.2% |
| wo_b | 0.045 | 7.7% |
| shared_expert | 0.044 | 7.6% |
| sparse_attn | 0.036 | 6.1% |
| attn_keys/index_q | 0.034 | 5.9% |
| q_b | 0.032 | 5.4% |
| router | 0.028 | 4.8% |
| hc_ffn_pre | 0.020 | 3.5% |
| hc_attn_pre | 0.017 | 2.8% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.014 | 2.3% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.006 | 1.0% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.7% |
| hc_post | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.555 ms; top: cuStreamSynchronize 0.273 ms x1, cudaLaunchKernel_v7000 0.185 ms x60, cuMemAllocAsync 0.036 ms x52, cuMemFreeAsync 0.024 ms x51, cuMemcpyDtoDAsync_v2 0.022 ms x6, cuKernelGetName 0.008 ms x60

### L20/decode (6 instances): wall 0.614 ms, GPU busy 0.572 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 27.3% |
| wo_a | 0.048 | 8.5% |
| shared_expert | 0.045 | 7.8% |
| wo_b | 0.045 | 7.8% |
| sparse_attn | 0.036 | 6.2% |
| attn_keys/index_q | 0.034 | 6.0% |
| q_b | 0.031 | 5.5% |
| router | 0.028 | 5.0% |
| attn_keys/compressor | 0.024 | 4.1% |
| hc_ffn_pre | 0.021 | 3.6% |
| hc_attn_pre | 0.017 | 3.0% |
| attn_keys/kv_window | 0.014 | 2.4% |
| attn_keys/qr | 0.013 | 2.4% |
| attn_keys/topk | 0.013 | 2.2% |
| moe_plan | 0.012 | 2.1% |
| attn_keys/index_keys | 0.010 | 1.8% |
| attn_keys/kv_append | 0.009 | 1.5% |
| combine | 0.004 | 0.8% |
| attn_keys/candidates | 0.004 | 0.7% |
| attn_keys/index_score | 0.003 | 0.6% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.540 ms; top: cuStreamSynchronize 0.316 ms x1, cudaLaunchKernel_v7000 0.150 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.022 ms x50, cuMemcpyDtoDAsync_v2 0.015 ms x4, cuKernelGetName 0.006 ms x63

### L21/decode (6 instances): wall 0.527 ms, GPU busy 0.481 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.156 | 32.4% |
| wo_a | 0.048 | 10.1% |
| shared_expert | 0.045 | 9.3% |
| wo_b | 0.044 | 9.2% |
| sparse_attn | 0.036 | 7.5% |
| q_b | 0.032 | 6.6% |
| router | 0.028 | 5.9% |
| hc_ffn_pre | 0.021 | 4.3% |
| hc_attn_pre | 0.017 | 3.5% |
| attn_keys/kv_window | 0.014 | 2.9% |
| attn_keys/qr | 0.014 | 2.8% |
| moe_plan | 0.012 | 2.5% |
| attn_keys/kv_append | 0.005 | 1.0% |
| combine | 0.005 | 0.9% |
| hc_post | 0.003 | 0.6% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.470 ms; top: cuStreamSynchronize 0.295 ms x1, cudaLaunchKernel_v7000 0.112 ms x47, cuMemAllocAsync 0.024 ms x46, cuMemFreeAsync 0.018 ms x44, cuMemcpyDtoDAsync_v2 0.012 ms x3, cuMemsetD8Async 0.005 ms x3

