prefill of 4096 tokens: 588.8 ms wall, 29.44 ms a layer over 20 layers

stream 0/14: busy 25.39 ms a layer; idle a layer: waiting for the host 3.53, before a copy 0.10, before a kernel 0.10
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 10.394 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.666 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.285 | 0.112 | 49.2 |
| sparse_attn | 1.0 | 2.130 | 0.000 |  |
| act_quant | 9.2 | 1.438 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.077 | 0.019 | 17.9 |
| gemm_ffma | 1.3 | 0.988 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.987 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.844 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.3 | 0.269 | 0.000 |  |
| hc_pre8 | 2.0 | 0.187 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.166 | 0.000 |  |
| swiglu | 2.0 | 0.108 | 0.000 |  |
| moe_combine | 1.0 | 0.099 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 769.0 |

stream 1/24: busy 24.42 ms a layer; idle a layer: waiting for the host 3.60, before a copy 1.01, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 9.763 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.540 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.285 | 0.112 | 49.2 |
| sparse_attn | 1.0 | 2.061 | 0.000 |  |
| act_quant | 9.2 | 1.398 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.076 | 0.019 | 17.9 |
| gemm_ffma | 1.3 | 0.961 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.940 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.816 | 0.000 |  |
| hc_post8 | 2.0 | 0.443 | 0.000 |  |
| blk_pack | 14.3 | 0.269 | 0.000 |  |
| hc_pre8 | 2.0 | 0.185 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.159 | 0.000 |  |
| swiglu | 2.0 | 0.107 | 0.000 |  |
| moe_combine | 1.0 | 0.099 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 769.4 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.2 | 29.721 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 14.788 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 2.083 |
| cudaLaunchKernel_v7000 | 124.7 | 1.354 |
| cuMemHostAlloc | 0.2 | 1.343 |
| cuMemAllocAsync | 56.5 | 1.009 |
| cuMemcpyDtoDAsync_v2 | 20.4 | 0.052 |
| cuStreamWaitEvent | 28.6 | 0.051 |
| cuMemsetD8Async | 9.7 | 0.033 |
| cuMemFreeAsync | 51.1 | 0.032 |
