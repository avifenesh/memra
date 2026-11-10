prefill of 32768 tokens: 6948.7 ms wall, 347.43 ms a layer over 20 layers

stream 0/14: busy 262.34 ms a layer; idle a layer: waiting for the host 83.97, before a copy 0.00, before a kernel 1.12
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| blk_pack | 26.3 | 124.905 | 0.000 |  |
| gemm_f8_pipe_tiles | 3.0 | 47.291 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 26.779 | 0.000 |  |
| sparse_attn | 4.0 | 16.922 | 0.000 |  |
| act_quant | 18.7 | 13.893 | 0.000 |  |
| gemm_bf16 | 4.5 | 7.468 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.497 | 0.000 |  |
| gemm_ffma | 1.3 | 6.297 | 0.000 |  |
| hc_post8 | 2.0 | 4.182 | 0.000 |  |
| hc_pre8 | 2.0 | 2.202 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.661 | 0.000 |  |
| swiglu | 2.0 | 1.052 | 0.000 |  |
| moe_combine | 1.0 | 0.938 | 0.000 |  |
| index_score | 0.3 | 0.591 | 0.000 |  |
| engram_gate | 0.4 | 0.372 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.8 |

stream 1/24: busy 256.22 ms a layer; idle a layer: waiting for the host 84.45, before a copy 0.00, before a kernel 6.74
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| blk_pack | 26.3 | 123.271 | 0.000 |  |
| gemm_f8_pipe_tiles | 3.0 | 45.218 | 0.000 |  |
| gemm_f8_pipe | 13.7 | 25.713 | 0.000 |  |
| sparse_attn | 4.0 | 15.985 | 0.000 |  |
| act_quant | 18.7 | 13.399 | 0.000 |  |
| gemm_bf16 | 4.5 | 7.230 | 0.000 |  |
| hc_mixes_part | 2.0 | 6.216 | 0.000 |  |
| gemm_ffma | 1.3 | 6.122 | 0.000 |  |
| hc_post8 | 2.0 | 4.117 | 0.000 |  |
| hc_pre8 | 2.0 | 2.173 | 0.000 |  |
| index_score | 0.3 | 1.619 | 0.000 |  |
| rmsnorm32 | 4.3 | 1.610 | 0.000 |  |
| swiglu | 2.0 | 1.018 | 0.000 |  |
| moe_combine | 1.0 | 0.892 | 0.000 |  |
| engram_gate | 0.4 | 0.368 | 0.000 |  |
| copy host>dev >=1MB | 0.8 | 0.365 | 0.021 | 56.9 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuStreamSynchronize | 0.8 | 191.071 |
| cudaLaunchKernel_v7000 | 213.2 | 80.495 |
| cuMemcpyDtoHAsync_v2 | 0.1 | 77.259 |
| cuStreamWaitEvent | 52.6 | 56.607 |
| cuMemFreeHost | 0.2 | 18.051 |
| cuMemAllocAsync | 67.1 | 4.461 |
| cuMemHostAlloc | 0.2 | 2.260 |
| cuMemcpyDtoDAsync_v2 | 6.1 | 0.723 |
| cuMemFreeAsync | 65.3 | 0.047 |
| cuMemsetD8Async | 9.6 | 0.043 |
