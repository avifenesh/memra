prefill of 32768 tokens: 3476.4 ms wall, 173.82 ms a layer over 20 layers

stream 0/14: busy 171.19 ms a layer; idle a layer: waiting for the host 1.45, before a copy 0.99, before a kernel 0.18
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 51.450 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 29.719 | 0.000 |  |
| sparse_attn | 4.0 | 18.345 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.344 | 0.908 | 49.5 |
| act_quant | 18.7 | 14.973 | 0.000 |  |
| gemm_bf16 | 4.5 | 9.011 | 0.000 |  |
| hc_mixes_part | 2.0 | 7.041 | 0.000 |  |
| gemm_ffma | 1.3 | 6.874 | 0.000 |  |
| hc_post8 | 2.0 | 4.365 | 0.000 |  |
| hc_pre8 | 2.0 | 2.353 | 0.000 |  |
| blk_pack | 26.3 | 2.306 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.787 | 0.000 |  |
| swiglu | 2.0 | 1.169 | 0.000 |  |
| moe_combine | 1.0 | 1.035 | 0.000 |  |
| index_score | 0.3 | 0.663 | 0.000 |  |
| engram_gate | 0.4 | 0.404 | 0.000 |  |

stream 1/24: busy 164.02 ms a layer; idle a layer: waiting for the host 2.58, before a copy 7.07, before a kernel 0.15
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 49.158 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.661 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.346 | 0.908 | 49.5 |
| sparse_attn | 4.0 | 17.467 | 0.000 |  |
| act_quant | 18.7 | 14.325 | 0.000 |  |
| gemm_bf16 | 4.5 | 7.873 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.698 | 0.000 |  |
| gemm_ffma | 1.3 | 6.570 | 0.000 |  |
| hc_post8 | 2.0 | 4.227 | 0.000 |  |
| blk_pack | 26.3 | 2.288 | 0.000 |  |
| hc_pre8 | 2.0 | 2.249 | 0.000 |  |
| index_score | 0.3 | 1.731 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.709 | 0.000 |  |
| swiglu | 2.0 | 1.063 | 0.000 |  |
| moe_combine | 1.0 | 0.954 | 0.000 |  |
| engram_gate | 0.4 | 0.375 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 139.150 |
| cudaLaunchKernel_v7000 | 213.2 | 53.106 |
| cuMemcpyDtoHAsync_v2 | 0.1 | 49.961 |
| cuStreamWaitEvent | 52.6 | 33.620 |
| cuMemFreeHost | 0.2 | 9.564 |
| cuMemAllocAsync | 67.3 | 3.053 |
| cuMemHostAlloc | 0.2 | 1.854 |
| cuMemcpyDtoDAsync_v2 | 32.4 | 0.653 |
| cuMemsetD8Async | 9.6 | 0.126 |
| cuMemFreeAsync | 65.5 | 0.049 |
