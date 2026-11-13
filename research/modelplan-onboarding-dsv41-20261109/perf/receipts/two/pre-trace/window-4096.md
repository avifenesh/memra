prefill of 4096 tokens: 1032.3 ms wall, 51.61 ms a layer over 20 layers

stream 0/14: busy 36.67 ms a layer; idle a layer: waiting for the host 14.55, before a copy 0.04, before a kernel 0.14
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| blk_pack | 14.3 | 15.288 | 0.000 |  |
| gemm_f8_pipe_tiles | 3.0 | 9.474 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.490 | 0.000 |  |
| sparse_attn | 1.0 | 2.044 | 0.000 |  |
| act_quant | 9.2 | 1.380 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.033 | 0.019 | 18.7 |
| gemm_ffma | 1.3 | 0.957 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.920 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.821 | 0.000 |  |
| hc_post8 | 2.0 | 0.441 | 0.000 |  |
| hc_pre8 | 2.0 | 0.182 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.152 | 0.000 |  |
| swiglu | 2.0 | 0.106 | 0.000 |  |
| moe_combine | 1.0 | 0.098 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.097 | 0.074 | 758.5 |
| engram_gate | 0.1 | 0.043 | 0.000 |  |

stream 1/24: busy 35.80 ms a layer; idle a layer: waiting for the host 14.60, before a copy 0.04, before a kernel 0.96
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| blk_pack | 14.3 | 15.112 | 0.000 |  |
| gemm_f8_pipe_tiles | 3.0 | 9.156 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.367 | 0.000 |  |
| sparse_attn | 1.0 | 1.935 | 0.000 |  |
| act_quant | 9.2 | 1.338 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.032 | 0.019 | 18.7 |
| gemm_ffma | 1.3 | 0.926 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.894 | 0.000 |  |
| hc_mixes_part | 2.0 | 0.775 | 0.000 |  |
| hc_post8 | 2.0 | 0.443 | 0.000 |  |
| hc_pre8 | 2.0 | 0.182 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.149 | 0.000 |  |
| swiglu | 2.0 | 0.104 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 769.0 |
| moe_combine | 1.0 | 0.095 | 0.000 |  |
| engram_gate | 0.1 | 0.042 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.2 | 21.840 |
| cuMemcpyDtoHAsync_v2 | 0.2 | 20.892 |
| cudaLaunchKernel_v7000 | 124.7 | 2.431 |
| cuMemcpyHtoDAsync_v2 | 0.6 | 1.999 |
| cuMemHostAlloc | 0.2 | 1.695 |
| cuMemAllocAsync | 56.4 | 1.020 |
| cuStreamWaitEvent | 28.6 | 0.078 |
| cuMemFreeAsync | 51.0 | 0.033 |
| cuMemsetD8Async | 9.7 | 0.030 |
| cuMemcpyDtoDAsync_v2 | 6.1 | 0.020 |
