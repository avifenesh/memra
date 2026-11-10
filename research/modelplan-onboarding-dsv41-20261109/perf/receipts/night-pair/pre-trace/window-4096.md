prefill of 4096 tokens: 603.1 ms wall, 30.15 ms a layer over 20 layers

stream 0/14: busy 25.94 ms a layer; idle a layer: waiting for the host 3.72, before a copy 0.09, before a kernel 0.10
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 10.780 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.734 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.277 | 0.112 | 49.4 |
| sparse_attn | 1.0 | 2.163 | 0.000 |  |
| act_quant | 9.2 | 1.457 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.057 | 0.019 | 18.3 |
| gemm_bf16 | 1.3 | 1.023 | 0.000 |  |
| gemm_ffma | 1.3 | 1.002 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.855 | 0.000 |  |
| hc_post8 | 2.0 | 0.442 | 0.000 |  |
| blk_pack | 14.3 | 0.271 | 0.000 |  |
| hc_pre8 | 2.0 | 0.188 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.169 | 0.000 |  |
| swiglu | 2.0 | 0.110 | 0.000 |  |
| moe_combine | 1.0 | 0.101 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 766.4 |

stream 1/24: busy 24.36 ms a layer; idle a layer: waiting for the host 3.82, before a copy 1.59, before a kernel 0.09
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 9.735 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.538 | 0.000 |  |
| copy peer >=1MB | 6.2 | 2.277 | 0.112 | 49.4 |
| sparse_attn | 1.0 | 2.059 | 0.000 |  |
| act_quant | 9.2 | 1.396 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.056 | 0.019 | 18.3 |
| gemm_ffma | 1.3 | 0.963 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.940 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.816 | 0.000 |  |
| hc_post8 | 2.0 | 0.443 | 0.000 |  |
| blk_pack | 14.3 | 0.268 | 0.000 |  |
| hc_pre8 | 2.0 | 0.185 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.158 | 0.000 |  |
| swiglu | 2.0 | 0.107 | 0.000 |  |
| moe_combine | 1.0 | 0.098 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 769.5 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.2 | 30.543 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 15.215 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 2.042 |
| cudaLaunchKernel_v7000 | 124.7 | 1.377 |
| cuMemAllocAsync | 56.5 | 1.369 |
| cuMemHostAlloc | 0.2 | 1.203 |
| cuMemcpyDtoDAsync_v2 | 20.4 | 0.051 |
| cuStreamWaitEvent | 28.6 | 0.049 |
| cuMemsetD8Async | 9.7 | 0.033 |
| cuMemFreeAsync | 51.1 | 0.032 |
