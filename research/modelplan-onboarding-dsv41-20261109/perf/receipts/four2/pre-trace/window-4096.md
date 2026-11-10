prefill of 4096 tokens: 7448.7 ms wall, 186.22 ms a layer over 40 layers

stream 0/16: busy 122.93 ms a layer; idle a layer: waiting for the host 5.73, before a copy 1.65, before a kernel 55.68
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 107.518 | 0.214 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 4.901 | 0.000 |  |
| copy peer <1MB | 6.0 | 2.285 | 0.005 | 2.2 |
| gemm_f8_pipe | 7.2 | 1.904 | 0.000 |  |
| act_quant | 10.2 | 1.288 | 0.000 |  |
| sparse_attn | 1.0 | 1.054 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.905 | 0.010 | 10.7 |
| gemm_ffma | 1.1 | 0.828 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.472 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.406 | 0.000 |  |
| blk_pack | 24.0 | 0.302 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.149 | 0.000 |  |
| swiglu | 2.0 | 0.053 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.0 |

stream 1/28: busy 178.06 ms a layer; idle a layer: waiting for the host 5.72, before a copy 0.48, before a kernel 1.71
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 162.183 | 0.214 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.877 | 0.000 |  |
| copy peer <1MB | 6.0 | 2.849 | 0.005 | 1.7 |
| gemm_f8_pipe | 7.2 | 1.877 | 0.000 |  |
| act_quant | 10.2 | 1.282 | 0.000 |  |
| sparse_attn | 1.0 | 1.016 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.904 | 0.010 | 10.7 |
| gemm_ffma | 1.1 | 0.819 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.468 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.393 | 0.000 |  |
| blk_pack | 24.0 | 0.311 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.148 | 0.000 |  |
| swiglu | 2.0 | 0.053 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.6 |

stream 2/40: busy 176.86 ms a layer; idle a layer: waiting for the host 5.75, before a copy 0.59, before a kernel 2.79
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 161.519 | 0.214 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.693 | 0.000 |  |
| copy peer <1MB | 6.0 | 2.638 | 0.005 | 1.9 |
| gemm_f8_pipe | 7.2 | 1.834 | 0.000 |  |
| act_quant | 10.2 | 1.258 | 0.000 |  |
| sparse_attn | 1.0 | 0.985 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.904 | 0.010 | 10.7 |
| gemm_ffma | 1.1 | 0.802 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.455 | 0.000 |  |
| hc_post8 | 2.0 | 0.441 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.382 | 0.000 |  |
| blk_pack | 24.0 | 0.311 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.145 | 0.000 |  |
| swiglu | 2.0 | 0.052 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.5 |

stream 3/52: busy 176.37 ms a layer; idle a layer: waiting for the host 5.76, before a copy 0.94, before a kernel 2.92
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 161.232 | 0.214 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.544 | 0.000 |  |
| copy peer <1MB | 6.0 | 2.733 | 0.005 | 1.8 |
| gemm_f8_pipe | 7.2 | 1.783 | 0.000 |  |
| act_quant | 10.2 | 1.232 | 0.000 |  |
| sparse_attn | 1.0 | 0.957 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.904 | 0.010 | 10.7 |
| gemm_ffma | 1.1 | 0.778 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.440 | 0.000 |  |
| hc_post8 | 2.0 | 0.438 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.372 | 0.000 |  |
| blk_pack | 24.0 | 0.311 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.142 | 0.000 |  |
| swiglu | 2.0 | 0.051 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.3 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cudaLaunchKernel_v7000 | 294.5 | 189.208 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 159.644 |
| cuStreamSynchronize | 0.2 | 148.624 |
| cuStreamWaitEvent | 148.8 | 62.562 |
| cuMemsetD8Async | 17.9 | 60.120 |
| cuMemcpyDtoDAsync_v2 | 82.6 | 31.541 |
| cuMemAllocAsync | 172.4 | 12.350 |
| cuMemHostAlloc | 0.2 | 8.415 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 3.553 |
| cuMemFreeAsync | 164.7 | 0.146 |
