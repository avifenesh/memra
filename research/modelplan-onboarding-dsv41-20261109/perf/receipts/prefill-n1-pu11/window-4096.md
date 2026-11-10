prefill of 4096 tokens: 590.8 ms wall, 73.86 ms a layer over 8 layers

stream 0/13: busy 39.92 ms a layer; idle a layer: waiting for the host 33.44, before a copy 0.10, before a kernel 0.04
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 3.0 | 20.147 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 6.009 | 0.000 |  |
| sparse_attn | 1.0 | 3.668 | 0.000 |  |
| act_quant | 10.2 | 2.043 | 0.000 |  |
| gemm_bf16 | 1.2 | 1.976 | 0.000 |  |
| hc_mixes_part | 2.0 | 1.644 | 0.000 |  |
| copy host>dev >=1MB | 0.2 | 1.619 | 0.045 | 27.9 |
| gemm_ffma | 1.2 | 0.934 | 0.000 |  |
| hc_post8 | 2.0 | 0.475 | 0.000 |  |
| copy dev>dev >=1MB | 2.4 | 0.233 | 0.174 | 744.9 |
| swiglu | 2.0 | 0.222 | 0.000 |  |
| moe_combine | 1.0 | 0.216 | 0.000 |  |
| hc_pre8 | 2.1 | 0.182 | 0.000 |  |
| rmsnorm32 | 4.4 | 0.172 | 0.000 |  |
| gemv_ffma | 0.1 | 0.143 | 0.000 |  |
| rope | 3.4 | 0.059 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 0.2 | 32.823 |
| cuMemcpyHtoDAsync_v2 | 0.5 | 1.582 |
| cuStreamSynchronize | 0.1 | 0.911 |
| cuMemHostAlloc | 0.2 | 0.458 |
| cuMemAllocAsync | 32.1 | 0.453 |
| cudaLaunchKernel_v7000 | 48.9 | 0.165 |
| cuMemsetD8Async | 5.1 | 0.017 |
| cuMemFreeAsync | 27.6 | 0.016 |
| cuMemcpyDtoDAsync_v2 | 2.9 | 0.012 |
| cuKernelGetName | 49.0 | 0.006 |
