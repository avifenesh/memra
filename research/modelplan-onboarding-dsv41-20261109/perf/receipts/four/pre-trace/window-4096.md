prefill of 4096 tokens: 7793.0 ms wall, 194.82 ms a layer over 40 layers

stream 0/16: busy 128.10 ms a layer; idle a layer: waiting for the host 6.65, before a copy 0.08, before a kernel 59.78
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 113.870 | 0.214 | 1.9 |
| gemm_f8_pipe_tiles | 3.0 | 5.081 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 1.892 | 0.000 |  |
| hc_mixes_part | 2.0 | 1.554 | 0.000 |  |
| act_quant | 10.2 | 1.366 | 0.000 |  |
| sparse_attn | 1.0 | 1.059 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.906 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.830 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.471 | 0.000 |  |
| hc_post8 | 2.0 | 0.446 | 0.000 |  |
| hc_pre8 | 2.0 | 0.185 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.164 | 0.000 |  |
| swiglu | 2.0 | 0.055 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 808.0 |
| moe_combine | 1.0 | 0.042 | 0.000 |  |
| engram_gate | 0.1 | 0.021 | 0.000 |  |

stream 1/28: busy 185.97 ms a layer; idle a layer: waiting for the host 6.65, before a copy 0.20, before a kernel 1.76
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 171.888 | 0.214 | 1.2 |
| gemm_f8_pipe_tiles | 3.0 | 5.052 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 1.865 | 0.000 |  |
| hc_mixes_part | 2.0 | 1.533 | 0.000 |  |
| act_quant | 10.2 | 1.352 | 0.000 |  |
| sparse_attn | 1.0 | 1.020 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.906 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.819 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.468 | 0.000 |  |
| hc_post8 | 2.0 | 0.449 | 0.000 |  |
| hc_pre8 | 2.0 | 0.183 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.159 | 0.000 |  |
| swiglu | 2.0 | 0.054 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 812.1 |
| moe_combine | 1.0 | 0.042 | 0.000 |  |
| engram_gate | 0.1 | 0.021 | 0.000 |  |

stream 2/40: busy 185.71 ms a layer; idle a layer: waiting for the host 6.68, before a copy 0.58, before a kernel 1.62
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 172.020 | 0.214 | 1.2 |
| gemm_f8_pipe_tiles | 3.0 | 4.852 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 1.808 | 0.000 |  |
| hc_mixes_part | 2.0 | 1.489 | 0.000 |  |
| act_quant | 10.2 | 1.322 | 0.000 |  |
| sparse_attn | 1.0 | 0.988 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.906 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.797 | 0.000 |  |
| hc_post8 | 2.0 | 0.455 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.454 | 0.000 |  |
| hc_pre8 | 2.0 | 0.183 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.154 | 0.000 |  |
| swiglu | 2.0 | 0.053 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 811.3 |
| moe_combine | 1.0 | 0.041 | 0.000 |  |
| index_score | 0.2 | 0.023 | 0.000 |  |

stream 3/52: busy 183.44 ms a layer; idle a layer: waiting for the host 6.69, before a copy 0.91, before a kernel 3.56
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 170.098 | 0.214 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.708 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 1.757 | 0.000 |  |
| hc_mixes_part | 2.0 | 1.424 | 0.000 |  |
| act_quant | 10.2 | 1.295 | 0.000 |  |
| sparse_attn | 1.0 | 0.962 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.906 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.773 | 0.000 |  |
| hc_post8 | 2.0 | 0.455 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.440 | 0.000 |  |
| hc_pre8 | 2.0 | 0.182 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.150 | 0.000 |  |
| swiglu | 2.0 | 0.052 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 812.9 |
| moe_combine | 1.0 | 0.041 | 0.000 |  |
| index_score | 0.2 | 0.031 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 0.2 | 333.047 |
| cuStreamSynchronize | 0.2 | 220.895 |
| cudaLaunchKernel_v7000 | 194.5 | 124.174 |
| cuMemcpyDtoDAsync_v2 | 14.6 | 17.633 |
| cuMemAllocAsync | 108.1 | 16.287 |
| cuMemsetD8Async | 17.9 | 11.509 |
| cuMemHostAlloc | 0.2 | 7.912 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 3.562 |
| cuStreamWaitEvent | 100.8 | 1.227 |
| cuMemcpy2DAsync_v2 | 48.4 | 0.116 |
