prefill of 32768 tokens: 3318.6 ms wall, 165.93 ms a layer over 20 layers (from its first mHC pass to the end of the prompt's own cell (NVTX))

stream 0/11: busy 17.76 ms a layer; idle a layer: waiting for the host 8.85, before a copy 139.01, before a kernel 0.00
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 13.2 | 17.755 | 0.908 | 51.1 |

stream 0/14: busy 145.33 ms a layer; idle a layer: waiting for the host 1.50, before a copy 0.00, before a kernel 19.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 49.497 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 28.669 | 0.000 |  |
| sparse_attn | 4.0 | 17.629 | 0.000 |  |
| gemm_bf16 | 4.5 | 11.955 | 0.000 |  |
| act_quant | 18.7 | 9.319 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.817 | 0.000 |  |
| gemm_ffma | 1.3 | 6.619 | 0.000 |  |
| hc_post8 | 2.0 | 4.243 | 0.000 |  |
| hc_pre8 | 2.0 | 2.271 | 0.000 |  |
| blk_pack | 26.3 | 2.200 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.729 | 0.000 |  |
| swiglu | 2.0 | 1.073 | 0.000 |  |
| moe_combine | 1.0 | 0.971 | 0.000 |  |
| index_score | 0.3 | 0.631 | 0.000 |  |
| engram_gate | 0.4 | 0.384 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.9 |

stream 1/20: busy 17.66 ms a layer; idle a layer: waiting for the host 8.85, before a copy 139.09, before a kernel 0.00
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 13.2 | 17.663 | 0.908 | 51.4 |

stream 1/24: busy 142.49 ms a layer; idle a layer: waiting for the host 1.67, before a copy 0.00, before a kernel 21.77
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 48.030 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.924 | 0.000 |  |
| sparse_attn | 4.0 | 17.158 | 0.000 |  |
| gemm_bf16 | 4.5 | 11.642 | 0.000 |  |
| act_quant | 18.7 | 9.071 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.593 | 0.000 |  |
| gemm_ffma | 1.3 | 6.425 | 0.000 |  |
| hc_post8 | 2.0 | 4.178 | 0.000 |  |
| hc_pre8 | 2.0 | 2.227 | 0.000 |  |
| blk_pack | 26.3 | 2.170 | 0.000 |  |
| index_score | 0.3 | 1.706 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.686 | 0.000 |  |
| swiglu | 2.0 | 1.057 | 0.000 |  |
| moe_combine | 1.0 | 0.935 | 0.000 |  |
| engram_gate | 0.4 | 0.376 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.9 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 145.661 |
| cudaLaunchKernel_v7000 | 203.2 | 53.217 |
| cuMemcpyDtoHAsync_v2 | 0.1 | 47.589 |
| cuStreamWaitEvent | 50.0 | 30.423 |
| cuMemcpyDtoDAsync_v2 | 5.9 | 0.575 |
| cuMemsetD8Async | 9.0 | 0.110 |
| cuMemcpyPeerAsync | 25.0 | 0.106 |
| cuMemAllocAsync | 62.3 | 0.057 |
| cuMemcpyHtoDAsync_v2 | 1.6 | 0.041 |
| cuMemFreeAsync | 61.7 | 0.041 |
