prefill of 4096 tokens: 473.3 ms wall, 23.66 ms a layer over 20 layers (from its first mHC pass to the end of the prompt's own cell (NVTX))

stream 0/11: busy 2.21 ms a layer; idle a layer: waiting for the host 0.85, before a copy 20.54, before a kernel 0.00
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 6.2 | 2.186 | 0.112 | 51.4 |
| copy peer <1MB | 1.0 | 0.020 | 0.001 | 52.2 |

stream 0/14: busy 21.25 ms a layer; idle a layer: waiting for the host 0.01, before a copy 0.00, before a kernel 2.40
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 9.835 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.631 | 0.000 |  |
| sparse_attn | 1.0 | 2.091 | 0.000 |  |
| gemm_bf16 | 1.3 | 1.438 | 0.000 |  |
| gemm_ffma | 1.3 | 0.982 | 0.000 |  |
| act_quant | 9.2 | 0.944 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.834 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.3 | 0.250 | 0.000 |  |
| hc_pre8 | 2.0 | 0.195 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.163 | 0.000 |  |
| swiglu | 2.0 | 0.108 | 0.000 |  |
| moe_combine | 1.0 | 0.099 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.044 | 0.003 | 56.8 |
| engram_gate | 0.1 | 0.043 | 0.000 |  |
| rope | 3.5 | 0.031 | 0.000 |  |

stream 1/20: busy 2.18 ms a layer; idle a layer: waiting for the host 0.85, before a copy 20.56, before a kernel 0.00
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 6.2 | 2.161 | 0.112 | 52.0 |
| copy peer <1MB | 1.0 | 0.020 | 0.001 | 51.7 |

stream 1/24: busy 20.48 ms a layer; idle a layer: waiting for the host 0.03, before a copy 0.00, before a kernel 3.14
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 9.427 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.511 | 0.000 |  |
| sparse_attn | 1.0 | 2.009 | 0.000 |  |
| gemm_bf16 | 1.3 | 1.394 | 0.000 |  |
| gemm_ffma | 1.3 | 0.944 | 0.000 |  |
| act_quant | 9.2 | 0.912 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.803 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.3 | 0.242 | 0.000 |  |
| hc_pre8 | 2.0 | 0.194 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.155 | 0.000 |  |
| swiglu | 2.0 | 0.105 | 0.000 |  |
| moe_combine | 1.0 | 0.096 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.044 | 0.003 | 56.8 |
| engram_gate | 0.1 | 0.042 | 0.000 |  |
| rope | 3.5 | 0.030 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.2 | 30.613 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 14.088 |
| cudaLaunchKernel_v7000 | 124.5 | 1.407 |
| cuMemcpyPeerAsync | 14.2 | 0.060 |
| cuStreamWaitEvent | 28.5 | 0.049 |
| cuMemAllocAsync | 53.8 | 0.043 |
| cuMemFreeAsync | 48.5 | 0.030 |
| cuMemsetD8Async | 9.4 | 0.024 |
| cuMemcpyDtoDAsync_v2 | 6.1 | 0.018 |
| cuKernelGetName | 124.5 | 0.012 |
