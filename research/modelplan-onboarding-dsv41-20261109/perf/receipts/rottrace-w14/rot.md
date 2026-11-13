mixed: 6992 kernels, 263.9 ms in kernels, span 260.4 ms; memset: 323 (0.3 ms) memcpy: 21615 (3012.8 ms)
alone1: 1748 kernels, 47.7 ms in kernels, span 46.4 ms; memset: 83 (0.1 ms) memcpy: 21171 (3024.0 ms)
alone2: 1748 kernels, 56.7 ms in kernels, span 55.0 ms; memset: 83 (0.1 ms) memcpy: 21171 (3017.8 ms)
alone3: 1748 kernels, 71.1 ms in kernels, span 68.8 ms; memset: 83 (0.1 ms) memcpy: 21171 (3014.1 ms)
alone4: 1748 kernels, 82.3 ms in kernels, span 75.8 ms; memset: 83 (0.1 ms) memcpy: 21171 (3004.4 ms)
kernels: mixed 263.9 ms against the alone traces' 257.8 ms (1.024)
| kernel | launches mixed / alone | ms mixed | ms alone | ratio |
|---|---|---|---|---|
| gemv_f4s_tiles | 432 / 432 | 146.02 | 142.83 | 1.022 |
| gemv_f8_rows | 792 / 792 | 31.81 | 30.59 | 1.040 |
| gemv_ffma | 16 / 16 | 14.61 | 14.21 | 1.028 |
| gemm_f8_pipe | 264 / 264 | 17.12 | 16.80 | 1.019 |
| gemv_ffma_narrow_parts | 208 / 208 | 8.70 | 8.40 | 1.035 |
| gemv_bf16 | 120 / 120 | 7.42 | 7.30 | 1.016 |
| gemm_bf16 | 88 / 88 | 11.65 | 11.54 | 1.010 |
| hc_mixes_part1 | 288 / 288 | 7.39 | 7.29 | 1.014 |
| act_quant | 1488 / 1488 | 3.54 | 3.44 | 1.029 |
| rmsnorm32 | 656 / 656 | 3.14 | 3.07 | 1.021 |
| swiglu | 288 / 288 | 1.92 | 1.87 | 1.025 |
| moe_plan_small | 144 / 144 | 1.51 | 1.48 | 1.021 |
| sparse_attn | 144 / 144 | 1.17 | 1.14 | 1.025 |
| route | 144 / 144 | 1.13 | 1.11 | 1.019 |
| hc_post8 | 288 / 288 | 0.96 | 0.94 | 1.021 |
| rope | 528 / 528 | 1.08 | 1.06 | 1.016 |
| moe_combine | 144 / 144 | 1.10 | 1.09 | 1.013 |
| hc_finish | 288 / 288 | 1.73 | 1.74 | 0.993 |
| fp4_qdq | 96 / 96 | 0.51 | 0.51 | 1.017 |
| hc_pre8 | 304 / 304 | 0.51 | 0.52 | 0.984 |
| engram_gate | 16 / 16 | 0.27 | 0.26 | 1.022 |
| attn_idx | 144 / 144 | 0.17 | 0.16 | 1.017 |
| index_score | 32 / 32 | 0.27 | 0.27 | 1.009 |
| topk | 32 / 32 | 0.09 | 0.09 | 1.023 |
| host call | calls mixed / alone | ms mixed | ms alone |
|---|---|---|---|
| cuMemcpyHtoDAsync_v2 | 21087 / 84156 | 3652.12 | 14634.87 |
| cuMemAllocAsync | 21493 / 84668 | 401.16 | 1118.92 |
| cuMemFreeAsync | 21493 / 84668 | 16.95 | 66.23 |
| cuModuleLoadData | 8 / 32 | 8.73 | 35.41 |
| cuMemGetInfo_v2 | 11 / 44 | 7.09 | 28.07 |
| cuMemFreeHost | 2 / 8 | 0.37 | 21.05 |
| cuLibraryLoadData | 1 / 4 | 1.92 | 7.49 |
| cuMemcpyDtoHAsync_v2 | 32 / 32 | 194.02 | 188.47 |
| cudaLaunchKernel_v7000 | 6992 / 6992 | 26.97 | 21.67 |
| cuCtxSetCurrent | 10 / 40 | 1.38 | 5.82 |
| cuStreamSynchronize | 17 / 20 | 25.84 | 21.81 |
| cuMemHostAlloc | 2 / 8 | 0.79 | 3.25 |
