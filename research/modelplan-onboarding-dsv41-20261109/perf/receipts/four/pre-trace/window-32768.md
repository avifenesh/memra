prefill of 32768 tokens: 60354.2 ms wall, 1508.86 ms a layer over 40 layers

stream 0/16: busy 1002.93 ms a layer; idle a layer: waiting for the host 20.04, before a copy 1.09, before a kernel 484.80
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 30.6 | 909.646 | 1.709 | 1.9 |
| gemm_f8_pipe_tiles | 3.0 | 24.402 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.672 | 0.000 |  |
| act_quant | 19.4 | 13.424 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.719 | 0.000 |  |
| sparse_attn | 4.0 | 8.738 | 0.000 |  |
| gemm_ffma | 1.1 | 5.321 | 0.000 |  |
| hc_post8 | 2.0 | 4.100 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.563 | 0.000 |  |
| hc_pre8 | 2.0 | 2.184 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.634 | 0.000 |  |
| swiglu | 2.0 | 0.521 | 0.000 |  |
| moe_combine | 1.0 | 0.461 | 0.000 |  |
| index_score | 0.2 | 0.450 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.185 | 0.010 | 56.0 |
| engram_gate | 0.2 | 0.184 | 0.000 |  |

stream 1/28: busy 1471.97 ms a layer; idle a layer: waiting for the host 20.10, before a copy 2.26, before a kernel 14.53
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 30.6 | 1379.894 | 1.709 | 1.2 |
| gemm_f8_pipe_tiles | 3.0 | 23.914 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.471 | 0.000 |  |
| act_quant | 19.4 | 13.156 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.421 | 0.000 |  |
| sparse_attn | 4.0 | 8.361 | 0.000 |  |
| gemm_ffma | 1.1 | 5.239 | 0.000 |  |
| hc_post8 | 2.0 | 4.088 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.528 | 0.000 |  |
| hc_pre8 | 2.0 | 2.152 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.593 | 0.000 |  |
| index_score | 0.2 | 1.088 | 0.000 |  |
| swiglu | 2.0 | 0.520 | 0.000 |  |
| moe_combine | 1.0 | 0.458 | 0.000 |  |
| engram_gate | 0.2 | 0.183 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.8 |

stream 2/40: busy 1484.36 ms a layer; idle a layer: waiting for the host 20.23, before a copy 3.83, before a kernel 0.44
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 30.6 | 1393.897 | 1.709 | 1.2 |
| gemm_f8_pipe_tiles | 3.0 | 23.258 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.020 | 0.000 |  |
| act_quant | 19.4 | 12.909 | 0.000 |  |
| hc_mixes_part | 2.0 | 12.125 | 0.000 |  |
| sparse_attn | 4.0 | 8.150 | 0.000 |  |
| gemm_ffma | 1.1 | 5.095 | 0.000 |  |
| hc_post8 | 2.0 | 4.077 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.415 | 0.000 |  |
| hc_pre8 | 2.0 | 2.134 | 0.000 |  |
| index_score | 0.2 | 1.669 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.561 | 0.000 |  |
| swiglu | 2.0 | 0.515 | 0.000 |  |
| moe_combine | 1.0 | 0.454 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.8 |
| engram_gate | 0.2 | 0.181 | 0.000 |  |

stream 3/52: busy 1464.35 ms a layer; idle a layer: waiting for the host 20.31, before a copy 6.06, before a kernel 18.14
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 30.6 | 1375.872 | 1.709 | 1.2 |
| gemm_f8_pipe_tiles | 3.0 | 22.566 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 13.617 | 0.000 |  |
| act_quant | 19.4 | 12.599 | 0.000 |  |
| hc_mixes_part | 2.0 | 11.721 | 0.000 |  |
| sparse_attn | 4.0 | 7.878 | 0.000 |  |
| gemm_ffma | 1.1 | 4.935 | 0.000 |  |
| hc_post8 | 2.0 | 4.070 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.304 | 0.000 |  |
| index_score | 0.2 | 2.128 | 0.000 |  |
| hc_pre8 | 2.0 | 2.112 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.525 | 0.000 |  |
| swiglu | 2.0 | 0.508 | 0.000 |  |
| moe_combine | 1.0 | 0.451 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.9 |
| engram_gate | 0.2 | 0.179 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cudaLaunchKernel_v7000 | 316.9 | 1774.461 |
| cuStreamSynchronize | 0.8 | 1628.731 |
| cuStreamWaitEvent | 244.8 | 677.556 |
| cuMemFreeHost | 0.2 | 68.036 |
| cuMemcpyDtoDAsync_v2 | 14.6 | 57.904 |
| cuMemAllocAsync | 129.8 | 55.935 |
| cuMemHostAlloc | 0.2 | 23.848 |
| cuMemsetD8Async | 17.8 | 19.501 |
| cuMemcpy2DAsync_v2 | 120.0 | 0.286 |
| cuMemFreeAsync | 128.2 | 0.128 |
