prefill of 4096 tokens: 4820.2 ms wall, 241.01 ms a layer over 20 layers

stream 0/16: busy 154.36 ms a layer; idle a layer: waiting for the host 12.94, before a copy 0.17, before a kernel 72.89
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 9.4 | 138.254 | 0.267 | 1.9 |
| gemm_f8_pipe_tiles | 3.0 | 4.992 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.233 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.813 | 0.019 | 10.6 |
| hc_mixes_part | 2.0 | 1.576 | 0.000 |  |
| sparse_attn | 1.0 | 0.998 | 0.000 |  |
| gemm_ffma | 1.3 | 0.949 | 0.000 |  |
| act_quant | 10.2 | 0.783 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.464 | 0.000 |  |
| hc_post8 | 2.0 | 0.453 | 0.000 |  |
| moe_combine | 1.0 | 0.195 | 0.000 |  |
| hc_pre8 | 2.0 | 0.176 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.166 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.095 | 0.074 | 773.2 |
| swiglu | 2.0 | 0.060 | 0.000 |  |
| engram_gate | 0.1 | 0.042 | 0.000 |  |

stream 1/28: busy 223.87 ms a layer; idle a layer: waiting for the host 13.01, before a copy 0.84, before a kernel 2.61
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 9.4 | 208.514 | 0.279 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.442 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.180 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.811 | 0.019 | 10.7 |
| hc_mixes_part | 2.0 | 1.547 | 0.000 |  |
| sparse_attn | 1.0 | 0.958 | 0.000 |  |
| gemm_ffma | 1.3 | 0.932 | 0.000 |  |
| act_quant | 10.2 | 0.744 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.459 | 0.000 |  |
| hc_post8 | 2.0 | 0.455 | 0.000 |  |
| moe_combine | 1.0 | 0.195 | 0.000 |  |
| hc_pre8 | 2.0 | 0.175 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.160 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.096 | 0.074 | 769.0 |
| swiglu | 2.0 | 0.052 | 0.000 |  |
| engram_gate | 0.1 | 0.042 | 0.000 |  |

stream 2/40: busy 225.01 ms a layer; idle a layer: waiting for the host 13.12, before a copy 0.84, before a kernel 1.39
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 9.4 | 209.639 | 0.271 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.620 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 3.094 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.810 | 0.019 | 10.7 |
| hc_mixes_part | 2.0 | 1.511 | 0.000 |  |
| sparse_attn | 1.0 | 0.934 | 0.000 |  |
| gemm_ffma | 1.3 | 0.912 | 0.000 |  |
| act_quant | 10.2 | 0.751 | 0.000 |  |
| hc_post8 | 2.0 | 0.457 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.448 | 0.000 |  |
| moe_combine | 1.0 | 0.195 | 0.000 |  |
| hc_pre8 | 2.0 | 0.174 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.157 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.095 | 0.074 | 774.6 |
| swiglu | 2.0 | 0.057 | 0.000 |  |
| engram_gate | 0.1 | 0.041 | 0.000 |  |

stream 3/52: busy 218.39 ms a layer; idle a layer: waiting for the host 13.19, before a copy 1.26, before a kernel 7.51
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| copy peer >=1MB | 9.4 | 203.360 | 0.269 | 1.3 |
| gemm_f8_pipe_tiles | 3.0 | 4.545 | 0.000 |  |
| gemm_f8_pipe | 7.2 | 2.995 | 0.000 |  |
| copy host>dev >=1MB | 0.1 | 1.810 | 0.019 | 10.7 |
| hc_mixes_part | 2.0 | 1.442 | 0.000 |  |
| sparse_attn | 1.0 | 0.903 | 0.000 |  |
| gemm_ffma | 1.3 | 0.883 | 0.000 |  |
| act_quant | 10.2 | 0.737 | 0.000 |  |
| hc_post8 | 2.0 | 0.456 | 0.000 |  |
| gemm_bf16 | 1.3 | 0.433 | 0.000 |  |
| moe_combine | 1.0 | 0.194 | 0.000 |  |
| hc_pre8 | 2.0 | 0.173 | 0.000 |  |
| rmsnorm32 | 4.3 | 0.151 | 0.000 |  |
| copy dev>dev >=1MB | 2.2 | 0.095 | 0.074 | 775.4 |
| swiglu | 2.0 | 0.058 | 0.000 |  |
| engram_gate | 0.1 | 0.040 | 0.000 |  |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemcpyDtoHAsync_v2 | 1.4 | 231.859 |
| cuMemHostAlloc | 0.4 | 25.711 |
| cuMemAllocAsync | 120.6 | 7.346 |
| cuMemcpyHtoDAsync_v2 | 1.2 | 7.109 |
| cuStreamSynchronize | 0.4 | 4.669 |
| cudaLaunchKernel_v7000 | 196.2 | 0.734 |
| cuStreamWaitEvent | 76.4 | 0.232 |
| cuMemFreeAsync | 109.8 | 0.142 |
| cuMemcpyDtoDAsync_v2 | 26.0 | 0.115 |
| cuMemcpy2DAsync_v2 | 24.8 | 0.079 |
