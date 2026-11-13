prefill of 32768 tokens: 3425.2 ms wall, 171.26 ms a layer over 20 layers

stream 0/14: busy 168.39 ms a layer; idle a layer: waiting for the host 1.61, before a copy 1.07, before a kernel 0.19
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 50.644 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 29.001 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.369 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 18.055 | 0.000 |  |
| act_quant | 18.7 | 14.756 | 0.000 |  |
| gemm_bf16 | 4.5 | 8.641 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.956 | 0.000 |  |
| gemm_ffma | 1.3 | 6.779 | 0.000 |  |
| hc_post8 | 2.0 | 4.317 | 0.000 |  |
| hc_pre8 | 2.0 | 2.320 | 0.000 |  |
| blk_pack | 26.3 | 2.291 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.760 | 0.000 |  |
| swiglu | 2.0 | 1.131 | 0.000 |  |
| moe_combine | 1.0 | 0.999 | 0.000 |  |
| index_score | 0.3 | 0.643 | 0.000 |  |
| engram_gate | 0.4 | 0.390 | 0.000 |  |

stream 1/24: busy 163.88 ms a layer; idle a layer: waiting for the host 2.38, before a copy 4.84, before a kernel 0.16
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 49.079 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.612 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.371 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 17.444 | 0.000 |  |
| act_quant | 18.7 | 14.314 | 0.000 |  |
| gemm_bf16 | 4.5 | 7.866 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.697 | 0.000 |  |
| gemm_ffma | 1.3 | 6.561 | 0.000 |  |
| hc_post8 | 2.0 | 4.229 | 0.000 |  |
| blk_pack | 26.3 | 2.292 | 0.000 |  |
| hc_pre8 | 2.0 | 2.253 | 0.000 |  |
| index_score | 0.3 | 1.733 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.706 | 0.000 |  |
| swiglu | 2.0 | 1.064 | 0.000 |  |
| moe_combine | 1.0 | 0.957 | 0.000 |  |
| engram_gate | 0.4 | 0.375 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 137.010 |
| cudaLaunchKernel_v7000 | 213.2 | 52.409 |
| cuStreamWaitEvent | 52.6 | 33.218 |
| cuMemFreeHost | 0.2 | 9.504 |
| cuMemAllocAsync | 67.3 | 3.025 |
| cuMemHostAlloc | 0.2 | 1.916 |
| cuMemcpyDtoDAsync_v2 | 32.4 | 0.642 |
| cuMemsetD8Async | 9.6 | 0.124 |
| cuMemFreeAsync | 65.5 | 0.048 |
| cuMemcpyHtoDAsync_v2 | 1.6 | 0.042 |
