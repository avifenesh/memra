prefill of 32768 tokens: 32427.8 ms wall, 810.69 ms a layer over 40 layers

stream 0/16: busy 774.66 ms a layer; idle a layer: waiting for the host 20.35, before a copy 3.45, before a kernel 12.24
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 39.6 | 691.208 | 1.364 | 2.0 |
| gemm_f8_pipe_tiles | 3.0 | 23.973 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.835 | 0.000 |  |
| act_quant | 18.4 | 11.380 | 0.000 |  |
| sparse_attn | 4.0 | 8.614 | 0.000 |  |
| gemm_ffma | 1.1 | 5.262 | 0.000 |  |
| hc_post8 | 2.0 | 4.113 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.523 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.171 | 0.000 |  |
| blk_pack | 52.0 | 2.296 | 0.000 |  |
| hc_pre8 | 2.0 | 2.183 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.615 | 0.000 |  |
| swiglu | 2.0 | 0.518 | 0.000 |  |
| moe_combine | 1.0 | 0.460 | 0.000 |  |
| index_score | 0.2 | 0.443 | 0.000 |  |
| engram_gate | 0.2 | 0.183 | 0.000 |  |

stream 1/28: busy 725.28 ms a layer; idle a layer: waiting for the host 20.25, before a copy 1.00, before a kernel 64.16
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 39.6 | 642.131 | 1.364 | 2.1 |
| gemm_f8_pipe_tiles | 3.0 | 23.637 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.858 | 0.000 |  |
| act_quant | 18.4 | 11.201 | 0.000 |  |
| sparse_attn | 4.0 | 8.345 | 0.000 |  |
| gemm_ffma | 1.1 | 5.198 | 0.000 |  |
| hc_post8 | 2.0 | 4.097 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.511 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.115 | 0.000 |  |
| blk_pack | 52.0 | 2.307 | 0.000 |  |
| hc_pre8 | 2.0 | 2.173 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.587 | 0.000 |  |
| index_score | 0.2 | 1.078 | 0.000 |  |
| swiglu | 2.0 | 0.517 | 0.000 |  |
| moe_combine | 1.0 | 0.458 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.8 |

stream 2/40: busy 786.21 ms a layer; idle a layer: waiting for the host 20.56, before a copy 0.69, before a kernel 3.24
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 39.6 | 704.773 | 1.364 | 1.9 |
| gemm_f8_pipe_tiles | 3.0 | 22.976 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.213 | 0.000 |  |
| act_quant | 18.4 | 10.954 | 0.000 |  |
| sparse_attn | 4.0 | 8.046 | 0.000 |  |
| gemm_ffma | 1.1 | 5.053 | 0.000 |  |
| hc_post8 | 2.0 | 4.097 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.381 | 0.000 |  |
| hc_mixes_part | 2.0 | 3.012 | 0.000 |  |
| blk_pack | 52.0 | 2.312 | 0.000 |  |
| hc_pre8 | 2.0 | 2.159 | 0.000 |  |
| index_score | 0.2 | 1.650 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.551 | 0.000 |  |
| swiglu | 2.0 | 0.513 | 0.000 |  |
| moe_combine | 1.0 | 0.453 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.8 |

stream 3/52: busy 630.37 ms a layer; idle a layer: waiting for the host 20.37, before a copy 4.94, before a kernel 155.01
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 39.6 | 550.041 | 1.364 | 2.5 |
| gemm_f8_pipe_tiles | 3.0 | 22.344 | 0.000 |  |
| gemm_f8_pipe | 13.4 | 14.121 | 0.000 |  |
| act_quant | 18.4 | 10.742 | 0.000 |  |
| sparse_attn | 4.0 | 7.825 | 0.000 |  |
| gemm_ffma | 1.1 | 4.922 | 0.000 |  |
| hc_post8 | 2.0 | 4.080 | 0.000 |  |
| gemm_bf16 | 4.3 | 3.325 | 0.000 |  |
| hc_mixes_part | 2.0 | 2.914 | 0.000 |  |
| blk_pack | 52.0 | 2.264 | 0.000 |  |
| hc_pre8 | 2.0 | 2.142 | 0.000 |  |
| index_score | 0.2 | 2.115 | 0.000 |  |
| rmsnorm32 | 4.2 | 1.521 | 0.000 |  |
| swiglu | 2.0 | 0.506 | 0.000 |  |
| moe_combine | 1.0 | 0.451 | 0.000 |  |
| copy host>dev >=1MB | 0.4 | 0.183 | 0.010 | 56.9 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cudaLaunchKernel_v7000 | 520.9 | 1000.581 |
| cuStreamWaitEvent | 316.8 | 831.664 |
| cuStreamSynchronize | 0.8 | 386.482 |
| cuMemAllocAsync | 338.1 | 38.182 |
| cuMemcpyDtoDAsync_v2 | 170.6 | 27.040 |
| cuMemsetD8Async | 17.8 | 26.222 |
| cuMemFreeHost | 0.2 | 19.482 |
| cuMemHostAlloc | 0.2 | 13.539 |
| cuMemFreeAsync | 336.2 | 0.452 |
| cuEventRecord | 106.4 | 0.061 |
