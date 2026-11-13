prefill of 32768 tokens: 5296.1 ms wall, 264.81 ms a layer over 20 layers

stream 0/14: busy 166.35 ms a layer; idle a layer: waiting for the host 97.24, before a copy 1.03, before a kernel 0.18
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 50.173 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 28.168 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.381 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 17.963 | 0.000 |  |
| act_quant | 18.7 | 14.618 | 0.000 |  |
| gemm_bf16 | 4.5 | 8.497 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.889 | 0.000 |  |
| gemm_ffma | 1.3 | 6.634 | 0.000 |  |
| hc_post8 | 2.0 | 4.316 | 0.000 |  |
| hc_pre8 | 2.0 | 2.300 | 0.000 |  |
| blk_pack | 26.0 | 2.261 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.745 | 0.000 |  |
| swiglu | 2.0 | 1.110 | 0.000 |  |
| moe_combine | 1.0 | 0.988 | 0.000 |  |
| index_score | 0.3 | 0.610 | 0.000 |  |
| engram_gate | 0.4 | 0.376 | 0.000 |  |

stream 1/24: busy 161.11 ms a layer; idle a layer: waiting for the host 97.34, before a copy 6.20, before a kernel 0.15
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 47.838 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.212 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.385 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 17.075 | 0.000 |  |
| act_quant | 18.7 | 13.983 | 0.000 |  |
| gemm_bf16 | 4.5 | 8.171 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.510 | 0.000 |  |
| gemm_ffma | 1.3 | 6.316 | 0.000 |  |
| hc_post8 | 2.0 | 4.188 | 0.000 |  |
| hc_pre8 | 2.0 | 2.217 | 0.000 |  |
| blk_pack | 26.0 | 2.208 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.671 | 0.000 |  |
| index_score | 0.3 | 1.660 | 0.000 |  |
| swiglu | 2.0 | 1.065 | 0.000 |  |
| moe_combine | 1.0 | 0.941 | 0.000 |  |
| engram_gate | 0.4 | 0.369 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 120.802 |
| cudaLaunchKernel_v7000 | 212.6 | 62.572 |
| cuMemcpyDtoHAsync_v2 | 0.1 | 48.213 |
| cuStreamWaitEvent | 52.6 | 18.913 |
| cuMemFreeHost | 0.2 | 6.971 |
| cuMemAllocAsync | 67.3 | 6.012 |
| cuMemHostAlloc | 0.2 | 1.990 |
| cuMemcpyDtoDAsync_v2 | 32.4 | 1.849 |
| cuMemsetD8Async | 9.6 | 0.634 |
| cuMemFreeAsync | 65.5 | 0.047 |
