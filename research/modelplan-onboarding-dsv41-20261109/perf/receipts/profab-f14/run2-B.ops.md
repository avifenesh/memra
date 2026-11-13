### L2/prefill (2 instances): wall 38.094 ms, GPU busy 38.006 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 18.403 | 48.4% |
| sparse_attn | 4.544 | 12.0% |
| wo_a | 2.915 | 7.7% |
| q_b | 1.827 | 4.8% |
| shared_expert | 1.767 | 4.7% |
| wo_b | 1.713 | 4.5% |
| attn_keys/compressor | 1.524 | 4.0% |
| hc_ffn_pre | 1.178 | 3.1% |
| hc_attn_pre | 0.896 | 2.4% |
| router | 0.747 | 2.0% |
| attn_keys/index_q | 0.584 | 1.5% |
| moe_plan | 0.396 | 1.0% |
| attn_keys/qr | 0.395 | 1.0% |
| attn_keys/index_score | 0.307 | 0.8% |
| attn_keys/kv_window | 0.241 | 0.6% |
| hc_post | 0.237 | 0.6% |
| combine | 0.207 | 0.5% |
| attn_keys/topk | 0.060 | 0.2% |
| attn_keys/index_keys | 0.036 | 0.1% |
| attn_keys/kv_append | 0.020 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 240 calls, 37.922 ms; top: cuStreamSynchronize 36.561 ms x1, cuMemAllocAsync 1.059 ms x53, cudaLaunchKernel_v7000 0.217 ms x64, cuMemFreeAsync 0.032 ms x51, cuMemcpyDtoDAsync_v2 0.025 ms x4, cuMemsetD8Async 0.017 ms x3

### L20/prefill (2 instances): wall 36.125 ms, GPU busy 36.004 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.113 | 47.5% |
| sparse_attn | 4.665 | 13.0% |
| wo_a | 2.923 | 8.1% |
| q_b | 1.840 | 5.1% |
| shared_expert | 1.776 | 4.9% |
| wo_b | 1.717 | 4.8% |
| hc_ffn_pre | 1.184 | 3.3% |
| hc_attn_pre | 0.897 | 2.5% |
| router | 0.748 | 2.1% |
| attn_keys/index_score | 0.641 | 1.8% |
| attn_keys/index_q | 0.583 | 1.6% |
| moe_plan | 0.407 | 1.1% |
| attn_keys/qr | 0.394 | 1.1% |
| attn_keys/kv_window | 0.240 | 0.7% |
| hc_post | 0.239 | 0.7% |
| attn_keys/compressor | 0.227 | 0.6% |
| combine | 0.204 | 0.6% |
| attn_keys/topk | 0.083 | 0.2% |
| attn_keys/index_keys | 0.047 | 0.1% |
| attn_keys/kv_append | 0.037 | 0.1% |
| attn_keys/candidates | 0.028 | 0.1% |
| attn_idx_rows | 0.009 | 0.0% |

host API calls: 236 calls, 35.955 ms; top: cuStreamSynchronize 35.178 ms x1, cuMemAllocAsync 0.493 ms x52, cudaLaunchKernel_v7000 0.199 ms x63, cuMemFreeAsync 0.036 ms x50, cuMemcpyDtoDAsync_v2 0.023 ms x4, cuMemsetD8Async 0.016 ms x3

### L21/prefill (2 instances): wall 34.809 ms, GPU busy 34.710 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 17.406 | 50.1% |
| sparse_attn | 4.715 | 13.6% |
| wo_a | 2.925 | 8.4% |
| q_b | 1.829 | 5.3% |
| shared_expert | 1.785 | 5.1% |
| wo_b | 1.717 | 4.9% |
| hc_ffn_pre | 1.184 | 3.4% |
| hc_attn_pre | 0.899 | 2.6% |
| router | 0.751 | 2.2% |
| moe_plan | 0.407 | 1.2% |
| attn_keys/qr | 0.394 | 1.1% |
| attn_keys/kv_window | 0.241 | 0.7% |
| hc_post | 0.238 | 0.7% |
| combine | 0.206 | 0.6% |
| attn_idx_rows | 0.009 | 0.0% |
| attn_keys/kv_append | 0.004 | 0.0% |

host API calls: 191 calls, 34.674 ms; top: cuStreamSynchronize 34.199 ms x1, cuMemAllocAsync 0.244 ms x46, cudaLaunchKernel_v7000 0.158 ms x47, cuMemFreeAsync 0.030 ms x44, cuMemcpyDtoDAsync_v2 0.020 ms x3, cuMemsetD8Async 0.015 ms x3

### L2/decode (6 instances): wall 0.726 ms, GPU busy 0.666 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.181 | 27.1% |
| wo_a | 0.050 | 7.5% |
| attn_keys/compressor | 0.049 | 7.3% |
| shared_expert | 0.049 | 7.3% |
| sparse_attn | 0.040 | 6.1% |
| hc_ffn_pre | 0.040 | 6.0% |
| q_b | 0.038 | 5.8% |
| router | 0.036 | 5.5% |
| wo_b | 0.036 | 5.4% |
| hc_attn_pre | 0.035 | 5.3% |
| attn_keys/index_q | 0.035 | 5.3% |
| attn_keys/kv_window | 0.015 | 2.3% |
| attn_keys/qr | 0.013 | 2.0% |
| attn_keys/topk | 0.013 | 2.0% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/index_keys | 0.006 | 0.8% |
| attn_keys/kv_append | 0.006 | 0.8% |
| combine | 0.004 | 0.6% |
| attn_keys/index_score | 0.003 | 0.4% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.643 ms; top: cuStreamSynchronize 0.415 ms x1, cudaLaunchKernel_v7000 0.152 ms x60, cuMemAllocAsync 0.028 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.685 ms, GPU busy 0.650 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.179 | 27.6% |
| wo_a | 0.049 | 7.5% |
| shared_expert | 0.048 | 7.4% |
| sparse_attn | 0.040 | 6.2% |
| hc_ffn_pre | 0.040 | 6.2% |
| q_b | 0.038 | 5.8% |
| router | 0.037 | 5.6% |
| wo_b | 0.037 | 5.6% |
| hc_attn_pre | 0.036 | 5.5% |
| attn_keys/index_q | 0.035 | 5.4% |
| attn_keys/compressor | 0.023 | 3.6% |
| attn_keys/kv_window | 0.015 | 2.4% |
| attn_keys/qr | 0.013 | 2.1% |
| moe_plan | 0.013 | 1.9% |
| attn_keys/topk | 0.012 | 1.9% |
| attn_keys/index_keys | 0.011 | 1.6% |
| attn_keys/kv_append | 0.009 | 1.4% |
| combine | 0.004 | 0.7% |
| attn_keys/candidates | 0.003 | 0.5% |
| attn_keys/index_score | 0.003 | 0.5% |
| hc_post | 0.003 | 0.4% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 236 calls, 0.623 ms; top: cuStreamSynchronize 0.429 ms x1, cudaLaunchKernel_v7000 0.129 ms x63, cuMemAllocAsync 0.025 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.591 ms, GPU busy 0.560 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.179 | 32.0% |
| wo_a | 0.048 | 8.7% |
| shared_expert | 0.048 | 8.6% |
| sparse_attn | 0.040 | 7.2% |
| hc_ffn_pre | 0.040 | 7.1% |
| q_b | 0.039 | 6.9% |
| wo_b | 0.037 | 6.6% |
| router | 0.036 | 6.5% |
| hc_attn_pre | 0.035 | 6.3% |
| attn_keys/kv_window | 0.016 | 2.8% |
| attn_keys/qr | 0.013 | 2.4% |
| moe_plan | 0.012 | 2.2% |
| attn_keys/kv_append | 0.005 | 0.9% |
| combine | 0.004 | 0.8% |
| hc_post | 0.003 | 0.5% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.543 ms; top: cuStreamSynchronize 0.394 ms x1, cudaLaunchKernel_v7000 0.097 ms x47, cuMemAllocAsync 0.020 ms x46, cuMemFreeAsync 0.016 ms x44, cuMemcpyDtoDAsync_v2 0.009 ms x3, cuMemsetD8Async 0.004 ms x3

