### L2/prefill (2 instances): wall 384.444 ms, GPU busy 384.236 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 177.387 | 46.2% |
| sparse_attn | 42.012 | 10.9% |
| attn_keys/index_score | 27.009 | 7.0% |
| wo_a | 21.932 | 5.7% |
| shared_expert | 16.674 | 4.3% |
| q_b | 15.947 | 4.2% |
| hc_ffn_pre | 15.638 | 4.1% |
| wo_b | 15.494 | 4.0% |
| attn_keys/compressor | 12.844 | 3.3% |
| hc_attn_pre | 10.192 | 2.7% |
| router | 6.664 | 1.7% |
| hc_post | 5.698 | 1.5% |
| attn_keys/index_q | 4.765 | 1.2% |
| attn_keys/qr | 3.405 | 0.9% |
| attn_keys/topk | 2.617 | 0.7% |
| attn_keys/kv_window | 2.051 | 0.5% |
| combine | 1.937 | 0.5% |
| moe_plan | 1.651 | 0.4% |
| attn_keys/kv_append | 0.126 | 0.0% |
| attn_idx_rows | 0.103 | 0.0% |
| attn_keys/index_keys | 0.092 | 0.0% |

host API calls: 627 calls, 384.115 ms; top: cuStreamSynchronize 381.402 ms x1, cuMemAllocAsync 2.032 ms x134, cudaLaunchKernel_v7000 0.509 ms x172, cuMemFreeAsync 0.072 ms x132, cuMemsetD8Async 0.043 ms x12, cuMemcpyDtoDAsync_v2 0.036 ms x4

### L20/prefill (2 instances): wall 397.242 ms, GPU busy 397.015 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 168.021 | 42.3% |
| attn_keys/index_score | 52.469 | 13.2% |
| sparse_attn | 42.691 | 10.8% |
| wo_a | 22.411 | 5.6% |
| shared_expert | 16.804 | 4.2% |
| hc_ffn_pre | 16.484 | 4.2% |
| q_b | 16.073 | 4.0% |
| wo_b | 15.760 | 4.0% |
| hc_attn_pre | 10.225 | 2.6% |
| router | 6.768 | 1.7% |
| hc_post | 5.859 | 1.5% |
| attn_keys/topk | 5.268 | 1.3% |
| attn_keys/index_q | 4.758 | 1.2% |
| attn_keys/qr | 3.418 | 0.9% |
| attn_keys/candidates | 2.250 | 0.6% |
| attn_keys/kv_window | 2.057 | 0.5% |
| combine | 1.950 | 0.5% |
| moe_plan | 1.734 | 0.4% |
| attn_keys/compressor | 1.496 | 0.4% |
| attn_keys/kv_append | 0.238 | 0.1% |
| attn_keys/index_keys | 0.174 | 0.0% |
| attn_idx_rows | 0.106 | 0.0% |

host API calls: 629 calls, 396.917 ms; top: cuStreamSynchronize 394.163 ms x1, cuMemAllocAsync 2.080 ms x133, cudaLaunchKernel_v7000 0.494 ms x174, cuMemFreeAsync 0.077 ms x131, cuMemsetD8Async 0.045 ms x12, cuMemcpyDtoDAsync_v2 0.034 ms x4

### L21/prefill (2 instances): wall 324.624 ms, GPU busy 324.479 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 166.512 | 51.3% |
| sparse_attn | 41.860 | 12.9% |
| wo_a | 21.829 | 6.7% |
| shared_expert | 16.588 | 5.1% |
| q_b | 16.021 | 4.9% |
| wo_b | 15.361 | 4.7% |
| hc_ffn_pre | 14.654 | 4.5% |
| hc_attn_pre | 10.204 | 3.1% |
| router | 6.543 | 2.0% |
| hc_post | 5.685 | 1.8% |
| attn_keys/qr | 3.416 | 1.1% |
| attn_keys/kv_window | 2.060 | 0.6% |
| combine | 1.916 | 0.6% |
| moe_plan | 1.694 | 0.5% |
| attn_idx_rows | 0.105 | 0.0% |
| attn_keys/kv_append | 0.030 | 0.0% |

host API calls: 518 calls, 324.380 ms; top: cuStreamSynchronize 322.909 ms x1, cuMemAllocAsync 0.982 ms x118, cudaLaunchKernel_v7000 0.350 ms x134, cuMemFreeAsync 0.058 ms x116, cuMemsetD8Async 0.036 ms x12, cuMemcpyDtoDAsync_v2 0.030 ms x3

### L2/decode (6 instances): wall 0.771 ms, GPU busy 0.696 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 22.8% |
| wo_b | 0.062 | 9.0% |
| wo_a | 0.050 | 7.2% |
| attn_keys/compressor | 0.048 | 6.9% |
| sparse_attn | 0.047 | 6.7% |
| shared_expert | 0.043 | 6.2% |
| attn_keys/index_q | 0.043 | 6.2% |
| hc_ffn_pre | 0.036 | 5.2% |
| router | 0.035 | 5.0% |
| attn_keys/topk | 0.034 | 4.9% |
| q_b | 0.032 | 4.6% |
| hc_attn_pre | 0.028 | 4.1% |
| attn_keys/kv_append | 0.015 | 2.1% |
| attn_keys/kv_window | 0.013 | 1.9% |
| moe_plan | 0.012 | 1.8% |
| attn_keys/qr | 0.012 | 1.8% |
| attn_keys/index_score | 0.007 | 1.0% |
| hc_post | 0.007 | 1.0% |
| attn_keys/index_keys | 0.005 | 0.8% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.3% |

host API calls: 233 calls, 0.687 ms; top: cuStreamSynchronize 0.459 ms x1, cudaLaunchKernel_v7000 0.149 ms x60, cuMemAllocAsync 0.031 ms x52, cuMemFreeAsync 0.020 ms x51, cuMemcpyDtoDAsync_v2 0.017 ms x6, cuKernelGetName 0.006 ms x60

### L20/decode (6 instances): wall 0.789 ms, GPU busy 0.753 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.160 | 21.3% |
| wo_b | 0.065 | 8.6% |
| attn_keys/topk | 0.055 | 7.3% |
| wo_a | 0.050 | 6.7% |
| sparse_attn | 0.047 | 6.2% |
| shared_expert | 0.043 | 5.8% |
| attn_keys/index_q | 0.042 | 5.6% |
| hc_ffn_pre | 0.037 | 4.9% |
| router | 0.035 | 4.7% |
| q_b | 0.032 | 4.3% |
| attn_keys/compressor | 0.032 | 4.2% |
| attn_keys/candidates | 0.029 | 3.9% |
| hc_attn_pre | 0.028 | 3.7% |
| attn_keys/kv_append | 0.027 | 3.6% |
| attn_keys/kv_window | 0.013 | 1.7% |
| moe_plan | 0.012 | 1.6% |
| attn_keys/qr | 0.012 | 1.6% |
| attn_keys/index_keys | 0.010 | 1.3% |
| attn_keys/index_score | 0.009 | 1.2% |
| hc_post | 0.007 | 0.9% |
| combine | 0.004 | 0.6% |
| attn_idx_rows | 0.002 | 0.2% |

host API calls: 236 calls, 0.724 ms; top: cuStreamSynchronize 0.522 ms x1, cudaLaunchKernel_v7000 0.134 ms x63, cuMemAllocAsync 0.027 ms x52, cuMemFreeAsync 0.019 ms x50, cuMemcpyDtoDAsync_v2 0.012 ms x4, cuKernelGetName 0.005 ms x63

### L21/decode (6 instances): wall 0.603 ms, GPU busy 0.571 ms

| op | GPU ms | share of busy |
|---|---|---|
| experts | 0.159 | 27.9% |
| wo_b | 0.064 | 11.2% |
| wo_a | 0.051 | 8.9% |
| sparse_attn | 0.048 | 8.4% |
| shared_expert | 0.043 | 7.6% |
| hc_ffn_pre | 0.036 | 6.3% |
| router | 0.036 | 6.2% |
| q_b | 0.032 | 5.6% |
| hc_attn_pre | 0.028 | 5.0% |
| attn_keys/kv_append | 0.023 | 4.0% |
| attn_keys/kv_window | 0.013 | 2.3% |
| moe_plan | 0.012 | 2.2% |
| attn_keys/qr | 0.012 | 2.1% |
| hc_post | 0.007 | 1.3% |
| combine | 0.004 | 0.8% |
| attn_idx_rows | 0.003 | 0.5% |

host API calls: 191 calls, 0.554 ms; top: cuStreamSynchronize 0.396 ms x1, cudaLaunchKernel_v7000 0.100 ms x47, cuMemAllocAsync 0.023 ms x46, cuMemFreeAsync 0.017 ms x44, cuMemcpyDtoDAsync_v2 0.010 ms x3, cuMemsetD8Async 0.004 ms x3

