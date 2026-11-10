prefill of 4096 tokens: 893.1 ms wall, 44.65 ms a layer over 20 layers

stream 0/14: busy 25.22 ms a layer; idle a layer: waiting for the host 19.04, before a copy 0.09, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 10.300 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.664 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.272 | 0.112 | 49.5 |
| sparse_attn | 1.0 | 2.112 | 0.000 |  |
| act_quant | 9.2 | 1.431 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.057 | 0.019 | 18.3 |
| gemm_ffma | 1.3 | 0.983 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.980 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.840 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.0 | 0.266 | 0.000 |  |
| hc_pre8 | 2.0 | 0.189 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.165 | 0.000 |  |
| swiglu | 2.0 | 0.108 | 0.000 |  |
| moe_combine | 1.0 | 0.099 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 767.7 |

stream 1/24: busy 24.58 ms a layer; idle a layer: waiting for the host 19.07, before a copy 0.71, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 10.017 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.552 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.273 | 0.112 | 49.5 |
| sparse_attn | 1.0 | 2.019 | 0.000 |  |
| act_quant | 9.2 | 1.380 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.055 | 0.019 | 18.3 |
| gemm_bf16 | 1.3 | 0.960 | 0.000 |  |
| gemm_ffma | 1.3 | 0.947 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.812 | 0.000 |  |
| hc_post8 | 2.0 | 0.443 | 0.000 |  |
| blk_pack | 14.0 | 0.260 | 0.000 |  |
| hc_pre8 | 2.0 | 0.187 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.156 | 0.000 |  |
| swiglu | 2.0 | 0.105 | 0.000 |  |
| moe_combine | 1.0 | 0.097 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.095 | 0.074 | 770.9 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 0.2 | 14.036 |
| cuStreamSynchronize | 0.2 | 7.820 |
| cuMemAllocAsync | 56.7 | 2.207 |
| cuMemHostAlloc | 0.2 | 2.075 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 2.045 |
| cudaLaunchKernel_v7000 | 124.1 | 0.595 |
| cuMemcpyDtoDAsync_v2 | 20.4 | 0.054 |
| cuStreamWaitEvent | 28.6 | 0.051 |
| cuMemFreeAsync | 51.3 | 0.034 |
| cuMemsetD8Async | 9.7 | 0.034 |
