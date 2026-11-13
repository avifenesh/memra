decode window: 1536 marker launches = 24.00 steps of 4 ranks x 8 layers; measured over 23.00 steps, 242.535 ms -> 10.545 ms/step, 1318.1 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 386 | 158 | 3.500 | 437.5 | 33% |
| 0/15 | 390 | 150 | 3.633 | 454.1 | 34% |
| 0/17 | 388 | 150 | 3.552 | 444.0 | 34% |
| 0/19 | 384 | 150 | 3.446 | 430.7 | 33% |

some stream busy 7.305 ms/step (69% of wall); every stream idle 3.240 ms/step

stream 0/13: idle 7.044 ms/step = 880.5 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 10.85 | 295.5 | 27.2 |
| copy | act_quant | 3.05 | 56.6 | 18.5 |
| rmsnorm32 | gemv_ffma_narrow | 0.99 | 49.4 | 49.7 |
| act_quant | gemv_f8 | 7.23 | 43.1 | 6.0 |
| hc_post8 | hc_mixes_part1 | 1.74 | 33.2 | 19.0 |
| moe_combine | hc_post8 | 0.99 | 32.5 | 32.7 |
| gemv_f8 | copy | 1.99 | 32.3 | 16.2 |
| copy | hc_post8 | 1.00 | 28.6 | 28.6 |
| act_quant | gemv_f8_tiles | 1.60 | 25.2 | 15.8 |
| gemv_f8_tiles | gemv_f8_tiles | 0.80 | 24.7 | 30.9 |
| gemv_f8 | engram_gate | 0.12 | 24.0 | 192.2 |
| hc_post8 | copy | 0.12 | 18.6 | 148.9 |
| gemv_f8_tiles | swiglu | 0.80 | 18.0 | 22.5 |
| act_quant | gemv_bf16 | 1.00 | 17.9 | 17.9 |
| gemv_f8 | act_quant | 0.99 | 16.4 | 16.5 |
| copy | moe_combine | 0.99 | 15.2 | 15.3 |
| gemv_ffma | copy | 0.12 | 14.1 | 117.5 |
| swiglu | act_quant | 1.79 | 12.2 | 6.8 |
| rmsnorm32 | act_quant | 2.00 | 10.8 | 5.4 |
| memset | gemv_ffma | 0.12 | 9.9 | 82.9 |
| gemv_f8_tiles | act_quant | 0.80 | 9.8 | 12.3 |
| rmsnorm32 | memset | 0.12 | 9.6 | 80.7 |
| gemv_ffma_narrow | route | 0.99 | 9.0 | 9.0 |
| gemv_f8 | swiglu | 0.99 | 7.6 | 7.6 |
| engram_gate | hc_mixes_part1 | 0.12 | 7.1 | 56.9 |
| rope | act_quant | 2.00 | 6.2 | 3.1 |
| route | memset | 0.99 | 5.2 | 5.2 |
| moe_place | copy | 0.99 | 5.1 | 5.1 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 487.5 | 219.5 | 268.0 | 29% |
| 0/15 | 499.1 | 222.9 | 276.2 | 29% |
| 0/17 | 502.8 | 215.6 | 287.2 | 28% |
| 0/19 | 547.4 | 247.8 | 299.6 | 29% |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 1548 | 193.5 | 4.581 | 3.0 |
| cuMemcpy2DAsync_v2 | 377 | 47.2 | 3.715 | 9.8 |
| cuMemcpyDtoHAsync_v2 | 12 | 1.5 | 2.675 | 227.0 |
| cuMemAllocAsync | 1419 | 177.3 | 2.471 | 1.7 |
| cuStreamSynchronize | 366 | 45.7 | 2.470 | 6.8 |
| cuMemFreeAsync | 1418 | 177.2 | 1.345 | 0.9 |
| cuMemcpyDtoDAsync_v2 | 98 | 12.3 | 0.404 | 4.1 |
| cuMemsetD8Async | 107 | 13.4 | 0.296 | 2.8 |
| cuKernelGetName | 1548 | 193.5 | 0.169 | 0.1 |
| cuMemcpyHtoDAsync_v2 | 16 | 2.0 | 0.072 | 4.6 |
| cuEventRecord | 8 | 1.0 | 0.026 | 3.2 |
| cuEventSynchronize | 8 | 1.0 | 0.018 | 2.2 |
| cuStreamWaitEvent | 8 | 1.0 | 0.009 | 1.1 |

host threads, time inside CUDA calls per step (ms): 4.804, 3.404, 3.382, 3.355, 3.305

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 231 | 7.23 | 525.2 |
| v41_gemv_f8_tiles_kernel | 79 | 2.48 | 219.3 |
| v41_gemv_bf16_kernel | 38 | 1.19 | 170.1 |
| v41_gemv_ffma_narrow_kernel | 40 | 1.24 | 160.3 |
| v41_gemv_ffma_kernel | 4 | 0.12 | 139.8 |
| v41_rmsnorm32_kernel | 136 | 4.24 | 75.4 |
| v41_act_quant_kernel | 348 | 10.89 | 74.7 |
| v41_hc_mixes_part1_kernel | 64 | 1.99 | 52.6 |
| v41_sparse_attn_kernel | 32 | 1.00 | 49.0 |
| v41_hc_finish_kernel | 64 | 1.99 | 40.9 |
| v41_swiglu_kernel | 58 | 1.82 | 34.0 |
| v41_route_kernel | 32 | 0.99 | 25.0 |
| v41_hc_post8_kernel | 64 | 1.99 | 22.0 |
| v41_moe_scan_kernel | 32 | 0.99 | 20.5 |
| v41_moe_combine_kernel | 32 | 0.99 | 18.1 |
| v41_rope_kernel | 104 | 3.26 | 17.3 |
| v41_moe_place_kernel | 32 | 0.99 | 15.2 |
| v41_engram_gate_kernel | 4 | 0.12 | 11.1 |
| v41_hc_pre8_kernel | 68 | 2.11 | 10.7 |
| v41_moe_count_kernel | 32 | 0.99 | 5.9 |
| v41_fp4_qdq_kernel | 8 | 0.26 | 5.1 |
| v41_attn_idx_kernel | 32 | 1.00 | 4.8 |
| v41_index_score_kernel | 4 | 0.12 | 1.3 |
| v41_topk_kernel | 4 | 0.12 | 1.3 |
| v41_comp_pool2_kernel | 2 | 0.07 | 1.1 |
| v41_engram_dequant_kernel | 4 | 0.12 | 1.1 |
