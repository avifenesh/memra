prefill of 16 tokens: 336.8 ms wall, 16.84 ms a layer over 20 layers

stream 0/14: busy 16.22 ms a layer; idle a layer: waiting for the host 0.22, before a copy 0.16, before a kernel 0.23
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 24.0 | 6.144 | 0.000 |  |
| gemm_f8_pipe | 57.4 | 3.476 | 0.000 |  |
| gemm_ffma | 10.4 | 2.888 | 0.000 |  |
| gemm_bf16 | 9.2 | 1.065 | 0.000 |  |
| hc_mixes_part | 16.0 | 0.852 | 0.000 |  |
| pull | 49.2 | 0.368 | 0.000 |  |
| push | 49.2 | 0.341 | 0.000 |  |
| gemv_ffma | 0.4 | 0.183 | 0.000 |  |
| rmsnorm32 | 34.8 | 0.133 | 0.000 |  |
| act_quant | 74.0 | 0.103 | 0.000 |  |
| hc_finish | 16.0 | 0.076 | 0.000 |  |
| blk_pack | 16.0 | 0.060 | 0.000 |  |
| route | 8.0 | 0.054 | 0.000 |  |
| copy peer <1MB | 8.0 | 0.052 | 0.002 | 39.7 |
| sparse_attn | 8.0 | 0.051 | 0.000 |  |
| swiglu | 16.0 | 0.043 | 0.000 |  |

stream 1/24: busy 16.27 ms a layer; idle a layer: waiting for the host 0.22, before a copy 0.08, before a kernel 0.27
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 24.0 | 6.140 | 0.000 |  |
| gemm_f8_pipe | 57.4 | 3.547 | 0.000 |  |
| gemm_ffma | 10.4 | 2.945 | 0.000 |  |
| gemm_bf16 | 9.2 | 1.083 | 0.000 |  |
| hc_mixes_part | 16.0 | 0.862 | 0.000 |  |
| push | 49.2 | 0.351 | 0.000 |  |
| pull | 49.2 | 0.236 | 0.000 |  |
| gemv_ffma | 0.4 | 0.182 | 0.000 |  |
| rmsnorm32 | 34.8 | 0.136 | 0.000 |  |
| act_quant | 74.0 | 0.105 | 0.000 |  |
| hc_finish | 16.0 | 0.078 | 0.000 |  |
| blk_pack | 16.0 | 0.061 | 0.000 |  |
| route | 8.0 | 0.055 | 0.000 |  |
| sparse_attn | 8.0 | 0.053 | 0.000 |  |
| copy peer <1MB | 8.0 | 0.051 | 0.002 | 40.4 |
| swiglu | 16.0 | 0.043 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 1.6 | 20.374 |
| cuMemcpyDtoHAsync_v2 | 1.4 | 8.123 |
| cudaLaunchKernel_v7000 | 982.3 | 2.109 |
| cuMemHostAlloc | 0.2 | 0.337 |
| cuMemcpyDtoDAsync_v2 | 64.8 | 0.157 |
| cuMemsetD8Async | 52.1 | 0.093 |
| cuKernelGetName | 982.3 | 0.092 |
| cuStreamWaitEvent | 32.0 | 0.050 |
| cuMemcpyHtoDAsync_v2 | 4.6 | 0.029 |
| cuMemAllocAsync | 22.5 | 0.024 |
