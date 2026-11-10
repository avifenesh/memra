prefill of 32 tokens: 257.0 ms wall, 12.85 ms a layer over 20 layers

stream 0/14: busy 12.42 ms a layer; idle a layer: waiting for the host 0.08, before a copy 0.16, before a kernel 0.19
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 18.0 | 4.822 | 0.000 |  |
| gemm_f8_pipe | 43.2 | 2.627 | 0.000 |  |
| gemm_ffma | 7.8 | 2.169 | 0.000 |  |
| gemm_bf16 | 7.2 | 0.819 | 0.000 |  |
| hc_mixes_part | 12.0 | 0.640 | 0.000 |  |
| pull | 34.9 | 0.254 | 0.000 |  |
| push | 34.9 | 0.245 | 0.000 |  |
| gemv_ffma | 0.3 | 0.138 | 0.000 |  |
| rmsnorm32 | 26.1 | 0.100 | 0.000 |  |
| act_quant | 55.5 | 0.080 | 0.000 |  |
| blk_pack | 16.0 | 0.060 | 0.000 |  |
| hc_finish | 12.0 | 0.057 | 0.000 |  |
| copy peer <1MB | 8.0 | 0.052 | 0.002 | 39.7 |
| route | 6.0 | 0.040 | 0.000 |  |
| sparse_attn | 6.0 | 0.038 | 0.000 |  |
| swiglu | 12.0 | 0.033 | 0.000 |  |

stream 1/24: busy 12.47 ms a layer; idle a layer: waiting for the host 0.08, before a copy 0.07, before a kernel 0.22
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 18.0 | 4.821 | 0.000 |  |
| gemm_f8_pipe | 43.2 | 2.680 | 0.000 |  |
| gemm_ffma | 7.8 | 2.212 | 0.000 |  |
| gemm_bf16 | 7.2 | 0.833 | 0.000 |  |
| hc_mixes_part | 12.0 | 0.646 | 0.000 |  |
| push | 34.9 | 0.252 | 0.000 |  |
| pull | 34.9 | 0.171 | 0.000 |  |
| gemv_ffma | 0.3 | 0.137 | 0.000 |  |
| rmsnorm32 | 26.1 | 0.102 | 0.000 |  |
| act_quant | 55.5 | 0.082 | 0.000 |  |
| blk_pack | 16.0 | 0.061 | 0.000 |  |
| hc_finish | 12.0 | 0.059 | 0.000 |  |
| copy peer <1MB | 8.0 | 0.051 | 0.002 | 40.4 |
| route | 6.0 | 0.042 | 0.000 |  |
| sparse_attn | 6.0 | 0.040 | 0.000 |  |
| swiglu | 12.0 | 0.033 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 1.2 | 15.806 |
| cuMemcpyDtoHAsync_v2 | 1.2 | 7.074 |
| cudaLaunchKernel_v7000 | 740.8 | 1.557 |
| cuMemcpyDtoDAsync_v2 | 52.6 | 0.124 |
| cuMemsetD8Async | 45.0 | 0.077 |
| cuKernelGetName | 740.9 | 0.069 |
| cuStreamWaitEvent | 32.0 | 0.050 |
| cuMemcpyHtoDAsync_v2 | 3.6 | 0.020 |
| cuMemAllocAsync | 13.8 | 0.015 |
| cuEventRecord | 33.2 | 0.012 |
