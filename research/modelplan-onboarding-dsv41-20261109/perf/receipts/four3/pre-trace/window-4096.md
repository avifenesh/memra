prefill of 4096 tokens: 4291.9 ms wall, 107.30 ms a layer over 40 layers

stream 0/16: busy 95.82 ms a layer; idle a layer: waiting for the host 6.68, before a copy 0.91, before a kernel 3.66
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 80.857 | 0.164 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 4.825 | 0.000 |  |
| copy peer <1MB | 9.0 | 2.203 | 0.006 | 2.9 |
| gemm_f8_pipe | 7.2 | 1.894 | 0.000 |  |
| act_quant | 9.2 | 1.077 | 0.000 |  |
| sparse_attn | 1.0 | 1.039 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.908 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.823 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.468 | 0.000 |  |
| hc_post8 | 2.0 | 0.441 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.397 | 0.000 |  |
| blk_pack | 28.0 | 0.258 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.148 | 0.000 |  |
| swiglu | 2.0 | 0.053 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 808.8 |

stream 1/28: busy 95.25 ms a layer; idle a layer: waiting for the host 6.65, before a copy 0.37, before a kernel 4.77
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 80.243 | 0.164 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 4.832 | 0.000 |  |
| copy peer <1MB | 9.0 | 2.263 | 0.006 | 2.9 |
| gemm_f8_pipe | 7.2 | 1.896 | 0.000 |  |
| act_quant | 9.2 | 1.078 | 0.000 |  |
| sparse_attn | 1.0 | 1.012 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.908 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.816 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.464 | 0.000 |  |
| hc_post8 | 2.0 | 0.441 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.398 | 0.000 |  |
| blk_pack | 28.0 | 0.264 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.148 | 0.000 |  |
| swiglu | 2.0 | 0.053 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.2 |

stream 2/40: busy 98.57 ms a layer; idle a layer: waiting for the host 6.70, before a copy 0.22, before a kernel 1.57
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 83.841 | 0.164 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 4.612 | 0.000 |  |
| copy peer <1MB | 9.0 | 2.419 | 0.006 | 2.7 |
| gemm_f8_pipe | 7.2 | 1.815 | 0.000 |  |
| act_quant | 9.2 | 1.042 | 0.000 |  |
| sparse_attn | 1.0 | 0.975 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.907 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.793 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.450 | 0.000 |  |
| hc_post8 | 2.0 | 0.440 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.379 | 0.000 |  |
| blk_pack | 28.0 | 0.259 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.144 | 0.000 |  |
| swiglu | 2.0 | 0.051 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.8 |

stream 3/52: busy 83.28 ms a layer; idle a layer: waiting for the host 6.69, before a copy 0.69, before a kernel 16.41
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 12.6 | 68.773 | 0.164 | 2.4 |
| gemm_f8_pipe_tiles | 3.0 | 4.653 | 0.000 |  |
| copy peer <1MB | 9.0 | 2.228 | 0.006 | 2.9 |
| gemm_f8_pipe | 7.2 | 1.816 | 0.000 |  |
| act_quant | 9.2 | 1.028 | 0.000 |  |
| sparse_attn | 1.0 | 0.952 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 0.907 | 0.010 | 10.6 |
| gemm_ffma | 1.1 | 0.776 | 0.000 |  |
| hc_post8 | 2.0 | 0.438 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.438 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.374 | 0.000 |  |
| blk_pack | 28.0 | 0.255 | 0.000 |  |
| hc_pre8 | 2.0 | 0.212 | 0.000 |  |
| rmsnorm32 | 4.2 | 0.142 | 0.000 |  |
| copy dev>dev >=1MB | 2.1 | 0.051 | 0.041 | 810.3 |
| swiglu | 2.0 | 0.050 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamWaitEvent | 172.8 | 95.469 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 81.082 |
| cuStreamSynchronize | 0.2 | 66.385 |
| cudaLaunchKernel_v7000 | 302.5 | 64.299 |
| cuMemsetD8Async | 17.9 | 17.160 |
| cuMemAllocAsync | 172.6 | 9.434 |
| cuMemHostAlloc | 0.2 | 5.111 |
| cuMemcpyDtoDAsync_v2 | 98.6 | 4.574 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 3.565 |
| cuMemFreeAsync | 164.7 | 0.176 |
