prefill of 32768 tokens: 58220.2 ms wall, 1455.51 ms a layer over 40 layers

stream 0/16: busy 970.99 ms a layer; idle a layer: waiting for the host 19.43, before a copy 4.60, before a kernel 460.49
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.6 | 883.951 | 1.748 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 24.391 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 15.062 | 0.000 |  |
| act_quant | 19.4 | 13.383 | 0.000 |  |
| sparse_attn | 4.0 | 8.722 | 0.000 |  |
| gemm_ffma | 1.1 | 5.339 | 0.000 |  |
| hc_post8 | 2.0 | 4.121 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.570 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.218 | 0.000 |  |
| blk_pack | 48.0 | 2.902 | 0.000 |  |
| hc_pre8 | 2.0 | 2.188 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.634 | 0.000 |  |
| swiglu | 2.0 | 0.520 | 0.000 |  |
| moe_combine | 1.0 | 0.461 | 0.000 |  |
| index_score | 0.2 | 0.450 | 0.000 |  |
| engram_gate | 0.2 | 0.184 | 0.000 |  |

stream 1/28: busy 1417.67 ms a layer; idle a layer: waiting for the host 19.51, before a copy 2.84, before a kernel 15.49
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.6 | 1331.489 | 1.748 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 23.938 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.850 | 0.000 |  |
| act_quant | 19.4 | 13.136 | 0.000 |  |
| sparse_attn | 4.0 | 8.360 | 0.000 |  |
| gemm_ffma | 1.1 | 5.262 | 0.000 |  |
| hc_post8 | 2.0 | 4.107 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.538 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.149 | 0.000 |  |
| blk_pack | 48.0 | 2.923 | 0.000 |  |
| hc_pre8 | 2.0 | 2.179 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.596 | 0.000 |  |
| index_score | 0.2 | 1.088 | 0.000 |  |
| swiglu | 2.0 | 0.520 | 0.000 |  |
| moe_combine | 1.0 | 0.460 | 0.000 |  |
| engram_gate | 0.2 | 0.183 | 0.000 |  |

stream 2/40: busy 1430.75 ms a layer; idle a layer: waiting for the host 19.64, before a copy 3.43, before a kernel 1.69
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.6 | 1345.898 | 1.748 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 23.313 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.389 | 0.000 |  |
| act_quant | 19.4 | 12.913 | 0.000 |  |
| sparse_attn | 4.0 | 8.157 | 0.000 |  |
| gemm_ffma | 1.1 | 5.121 | 0.000 |  |
| hc_post8 | 2.0 | 4.098 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.428 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.057 | 0.000 |  |
| blk_pack | 48.0 | 2.922 | 0.000 |  |
| hc_pre8 | 2.0 | 2.167 | 0.000 |  |
| index_score | 0.2 | 1.672 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.569 | 0.000 |  |
| swiglu | 2.0 | 0.516 | 0.000 |  |
| moe_combine | 1.0 | 0.455 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.9 |

stream 3/52: busy 1419.98 ms a layer; idle a layer: waiting for the host 19.73, before a copy 5.53, before a kernel 10.26
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 36.6 | 1336.853 | 1.748 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 22.607 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 13.974 | 0.000 |  |
| act_quant | 19.4 | 12.599 | 0.000 |  |
| sparse_attn | 4.0 | 7.886 | 0.000 |  |
| gemm_ffma | 1.1 | 4.957 | 0.000 |  |
| hc_post8 | 2.0 | 4.085 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.314 | 0.000 |  |
| hc_mixes_part | 2.0 | 2.944 | 0.000 |  |
| blk_pack | 48.0 | 2.933 | 0.000 |  |
| hc_pre8 | 2.0 | 2.147 | 0.000 |  |
| index_score | 0.2 | 2.130 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.531 | 0.000 |  |
| swiglu | 2.0 | 0.508 | 0.000 |  |
| moe_combine | 1.0 | 0.452 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.8 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cudaLaunchKernel_v7000 | 512.9 | 2741.198 |
| cuStreamWaitEvent | 292.8 | 735.654 |
| cuStreamSynchronize | 0.8 | 716.386 |
| cuMemsetD8Async | 17.8 | 262.746 |
| cuMemcpyDtoHAsync_v2 | 0.0 | 179.635 |
| cuMemFreeHost | 0.2 | 63.934 |
| cuMemAllocAsync | 322.1 | 59.637 |
| cuMemcpyDtoDAsync_v2 | 154.6 | 26.103 |
| cuMemHostAlloc | 0.2 | 20.662 |
| cuMemFreeAsync | 320.2 | 0.373 |
