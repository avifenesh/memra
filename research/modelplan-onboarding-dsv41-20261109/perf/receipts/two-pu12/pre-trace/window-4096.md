prefill of 4096 tokens: 810.5 ms wall, 40.52 ms a layer over 20 layers

stream 0/14: busy 24.90 ms a layer; idle a layer: waiting for the host 15.20, before a copy 0.12, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 10.095 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.606 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.287 | 0.112 | 49.2 |
| sparse_attn | 1.0 | 2.098 | 0.000 |  |
| act_quant | 9.2 | 1.418 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.057 | 0.019 | 18.2 |
| gemm_ffma | 1.3 | 0.971 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.963 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.833 | 0.000 |  |
| hc_post8 | 2.0 | 0.441 | 0.000 |  |
| blk_pack | 14.3 | 0.268 | 0.000 |  |
| hc_pre8 | 2.0 | 0.186 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.163 | 0.000 |  |
| swiglu | 2.0 | 0.107 | 0.000 |  |
| moe_combine | 1.0 | 0.099 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.097 | 0.074 | 760.9 |

stream 1/24: busy 24.26 ms a layer; idle a layer: waiting for the host 15.25, before a copy 0.72, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 9.673 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.521 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.288 | 0.112 | 49.1 |
| sparse_attn | 1.0 | 2.047 | 0.000 |  |
| act_quant | 9.2 | 1.392 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.055 | 0.019 | 18.3 |
| gemm_ffma | 1.3 | 0.956 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.935 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.811 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.3 | 0.268 | 0.000 |  |
| hc_pre8 | 2.0 | 0.185 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.158 | 0.000 |  |
| swiglu | 2.0 | 0.107 | 0.000 |  |
| moe_combine | 1.0 | 0.098 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.095 | 0.074 | 770.5 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 0.2 | 14.524 |
| cuStreamSynchronize | 0.2 | 6.749 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 2.046 |
| cudaLaunchKernel_v7000 | 124.7 | 1.343 |
| cuMemHostAlloc | 0.2 | 1.050 |
| cuMemAllocAsync | 56.5 | 1.011 |
| cuMemcpyDtoDAsync_v2 | 20.4 | 0.052 |
| cuStreamWaitEvent | 28.6 | 0.049 |
| cuMemFreeAsync | 51.1 | 0.034 |
| cuMemsetD8Async | 9.7 | 0.032 |
