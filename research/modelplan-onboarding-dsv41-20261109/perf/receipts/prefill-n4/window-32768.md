prefill of 32768 tokens: 37451.4 ms wall, 1872.57 ms a layer over 20 layers

stream 0/16: busy 1188.40 ms a layer; idle a layer: waiting for the host 42.25, before a copy 1.62, before a kernel 640.30
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.5 | 1079.469 | 2.126 | 2.0 |
| gemm_f8_pipe_tiles | 12.0 | 33.470 | 0.000 |  |
| gemm_f8_pipe | 22.6 | 24.937 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.728 | 0.000 |  |
| sparse_attn | 4.0 | 8.294 | 0.000 |  |
| act_quant | 31.6 | 7.075 | 0.000 |  |
| gemm_ffma | 4.3 | 6.991 | 0.000 |  |
| hc_post8 | 2.0 | 4.141 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.561 | 0.000 |  |
| hc_pre8 | 2.0 | 2.211 | 0.000 |  |
| moe_combine | 4.0 | 1.679 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.642 | 0.000 |  |
| swiglu | 8.0 | 0.615 | 0.000 |  |
| engram_gate | 0.4 | 0.370 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.9 |
| index_score | 0.1 | 0.164 | 0.000 |  |

stream 1/28: busy 1801.49 ms a layer; idle a layer: waiting for the host 42.95, before a copy 6.61, before a kernel 21.51
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.5 | 1698.190 | 2.216 | 1.3 |
| gemm_f8_pipe_tiles | 12.0 | 29.489 | 0.000 |  |
| gemm_f8_pipe | 22.6 | 24.477 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.432 | 0.000 |  |
| sparse_attn | 4.0 | 7.926 | 0.000 |  |
| gemm_ffma | 4.3 | 6.879 | 0.000 |  |
| act_quant | 31.6 | 6.669 | 0.000 |  |
| hc_post8 | 2.0 | 4.111 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.515 | 0.000 |  |
| hc_pre8 | 2.0 | 2.177 | 0.000 |  |
| moe_combine | 4.0 | 1.677 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.596 | 0.000 |  |
| swiglu | 8.0 | 0.533 | 0.000 |  |
| index_score | 0.1 | 0.413 | 0.000 |  |
| engram_gate | 0.4 | 0.368 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.8 |

stream 2/40: busy 1761.59 ms a layer; idle a layer: waiting for the host 43.49, before a copy 7.02, before a kernel 60.47
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.5 | 1659.303 | 2.185 | 1.3 |
| gemm_f8_pipe_tiles | 12.0 | 29.668 | 0.000 |  |
| gemm_f8_pipe | 22.6 | 23.817 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.154 | 0.000 |  |
| sparse_attn | 4.0 | 7.767 | 0.000 |  |
| gemm_ffma | 4.3 | 6.699 | 0.000 |  |
| act_quant | 31.6 | 6.660 | 0.000 |  |
| hc_post8 | 2.0 | 4.096 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.413 | 0.000 |  |
| hc_pre8 | 2.0 | 2.159 | 0.000 |  |
| moe_combine | 4.0 | 1.670 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.572 | 0.000 |  |
| index_score | 0.1 | 0.658 | 0.000 |  |
| swiglu | 8.0 | 0.558 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.8 |
| engram_gate | 0.4 | 0.364 | 0.000 |  |

stream 3/52: busy 1789.94 ms a layer; idle a layer: waiting for the host 44.05, before a copy 9.14, before a kernel 29.44
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.5 | 1689.437 | 2.160 | 1.3 |
| gemm_f8_pipe_tiles | 12.0 | 29.632 | 0.000 |  |
| gemm_f8_pipe | 22.6 | 23.082 | 0.000 |  |
| hc_mixes_part | 2.0 | 11.731 | 0.000 |  |
| sparse_attn | 4.0 | 7.485 | 0.000 |  |
| act_quant | 31.6 | 6.578 | 0.000 |  |
| gemm_ffma | 4.3 | 6.472 | 0.000 |  |
| hc_post8 | 2.0 | 4.077 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.297 | 0.000 |  |
| hc_pre8 | 2.0 | 2.127 | 0.000 |  |
| moe_combine | 4.0 | 1.663 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.529 | 0.000 |  |
| index_score | 0.1 | 0.874 | 0.000 |  |
| swiglu | 8.0 | 0.572 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.8 |
| engram_gate | 0.4 | 0.360 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 4.0 | 1789.109 |
| cuStreamSynchronize | 1.6 | 66.446 |
| cuMemHostAlloc | 0.4 | 37.479 |
| cuMemFreeHost | 0.4 | 28.000 |
| cuMemAllocAsync | 295.6 | 8.347 |
| cudaLaunchKernel_v7000 | 537.0 | 1.827 |
| cuStreamWaitEvent | 294.8 | 0.871 |
| cuMemsetD8Async | 55.2 | 0.512 |
| cuMemFreeAsync | 293.0 | 0.484 |
| cuMemcpyDtoDAsync_v2 | 62.0 | 0.291 |
