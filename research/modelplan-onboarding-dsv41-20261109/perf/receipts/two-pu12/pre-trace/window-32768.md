prefill of 32768 tokens: 5068.1 ms wall, 253.41 ms a layer over 20 layers

stream 0/14: busy 163.98 ms a layer; idle a layer: waiting for the host 87.99, before a copy 1.25, before a kernel 0.18
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 49.454 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.561 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.367 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 17.676 | 0.000 |  |
| act_quant | 18.7 | 14.457 | 0.000 |  |
| gemm_bf16 | 4.5 | 8.253 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.768 | 0.000 |  |
| gemm_ffma | 1.3 | 6.537 | 0.000 |  |
| hc_post8 | 2.0 | 4.278 | 0.000 |  |
| hc_pre8 | 2.0 | 2.283 | 0.000 |  |
| blk_pack | 26.3 | 2.278 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.722 | 0.000 |  |
| swiglu | 2.0 | 1.094 | 0.000 |  |
| moe_combine | 1.0 | 0.975 | 0.000 |  |
| index_score | 0.3 | 0.602 | 0.000 |  |
| engram_gate | 0.4 | 0.373 | 0.000 |  |

stream 1/24: busy 162.28 ms a layer; idle a layer: waiting for the host 88.19, before a copy 2.78, before a kernel 0.15
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 48.594 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 27.115 | 0.000 |  |
| copy peer >=1MB | 13.2 | 18.372 | 0.908 | 49.4 |
| sparse_attn | 4.0 | 17.320 | 0.000 |  |
| act_quant | 18.7 | 14.202 | 0.000 |  |
| gemm_bf16 | 4.5 | 7.800 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.623 | 0.000 |  |
| gemm_ffma | 1.3 | 6.451 | 0.000 |  |
| hc_post8 | 2.0 | 4.217 | 0.000 |  |
| blk_pack | 26.3 | 2.285 | 0.000 |  |
| hc_pre8 | 2.0 | 2.241 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.691 | 0.000 |  |
| index_score | 0.3 | 1.680 | 0.000 |  |
| swiglu | 2.0 | 1.059 | 0.000 |  |
| moe_combine | 1.0 | 0.950 | 0.000 |  |
| engram_gate | 0.4 | 0.369 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 118.242 |
| cudaLaunchKernel_v7000 | 213.2 | 50.791 |
| cuStreamWaitEvent | 52.6 | 32.125 |
| cuMemFreeHost | 0.2 | 10.301 |
| cuMemAllocAsync | 67.3 | 4.930 |
| cuMemHostAlloc | 0.2 | 1.559 |
| cuMemcpyDtoDAsync_v2 | 32.4 | 0.607 |
| cuMemsetD8Async | 9.6 | 0.127 |
| cuMemFreeAsync | 65.5 | 0.053 |
| cuMemcpyHtoDAsync_v2 | 1.6 | 0.043 |
