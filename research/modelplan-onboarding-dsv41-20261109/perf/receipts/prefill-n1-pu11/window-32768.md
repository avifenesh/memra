prefill of 32768 tokens: 3347.9 ms wall, 418.49 ms a layer over 8 layers

stream 0/13: busy 291.25 ms a layer; idle a layer: waiting for the host 127.16, before a copy 0.00, before a kernel 0.08
| kernel or copy | a layer | ms a layer | GB a layer | GB/s |
|---|---|---|---|---|
| gemm_f8_pipe_tiles | 12.0 | 134.427 | 0.000 |  |
| gemm_f8_pipe | 23.0 | 49.550 | 0.000 |  |
| sparse_attn | 4.0 | 32.347 | 0.000 |  |
| act_quant | 32.0 | 18.742 | 0.000 |  |
| gemm_bf16 | 4.6 | 17.525 | 0.000 |  |
| hc_mixes_part | 2.0 | 13.781 | 0.000 |  |
| gemm_ffma | 4.2 | 7.482 | 0.000 |  |
| hc_post8 | 2.0 | 4.427 | 0.000 |  |
| hc_pre8 | 2.1 | 2.427 | 0.000 |  |
| index_score | 0.5 | 2.054 | 0.000 |  |
| swiglu | 8.0 | 2.044 | 0.000 |  |
| moe_combine | 4.0 | 1.976 | 0.000 |  |
| rmsnorm32 | 4.4 | 1.783 | 0.000 |  |
| rope | 9.8 | 0.585 | 0.000 |  |
| engram_gate | 0.5 | 0.464 | 0.000 |  |
| copy host>dev >=1MB | 1.0 | 0.456 | 0.026 | 56.9 |

| host CUDA call | calls a layer | ms a layer |
|---|---|---|
| cuMemFreeHost | 0.2 | 28.468 |
| cuStreamSynchronize | 0.5 | 7.452 |
| cudaLaunchKernel_v7000 | 137.0 | 5.230 |
| cuMemAllocAsync | 75.2 | 1.430 |
| cuMemHostAlloc | 0.2 | 0.902 |
| cuMemsetD8Async | 14.0 | 0.475 |
| cuMemFreeAsync | 74.1 | 0.042 |
| cuMemcpyHtoDAsync_v2 | 1.0 | 0.028 |
| cuKernelGetName | 137.0 | 0.015 |
| cuMemcpyDtoDAsync_v2 | 2.9 | 0.011 |
