decode window: 1536 marker launches = 24.00 steps of 4 ranks x 8 layers; measured over 23.00 steps, 177.591 ms -> 7.721 ms/step, 965.2 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 386 | 136 | 3.556 | 444.6 | 46% |
| 0/15 | 390 | 127 | 3.700 | 462.5 | 48% |
| 0/17 | 388 | 128 | 3.634 | 454.2 | 47% |
| 0/19 | 384 | 128 | 3.521 | 440.1 | 46% |

some stream busy 6.986 ms/step (90% of wall); every stream idle 0.735 ms/step

stream 0/13: idle 4.165 ms/step = 520.6 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 8.05 | 76.5 | 9.5 |
| act_quant | gemv_f8 | 7.23 | 45.2 | 6.2 |
| gemv_f8 | copy | 1.99 | 34.8 | 17.5 |
| gemv_bf16 | copy | 1.00 | 32.2 | 32.2 |
| copy | act_quant | 3.05 | 31.5 | 10.3 |
| rmsnorm32 | memset | 0.12 | 24.7 | 206.3 |
| act_quant | gemv_f8_tiles | 1.60 | 24.5 | 15.3 |
| gemv_f8_tiles | gemv_f8_tiles | 0.80 | 24.4 | 30.6 |
| gemv_ffma | copy | 0.12 | 19.1 | 159.5 |
| gemv_f8_tiles | swiglu | 0.80 | 18.2 | 22.8 |
| gemv_f8 | act_quant | 0.99 | 16.6 | 16.7 |
| gemv_f8_tiles | act_quant | 0.80 | 13.5 | 16.9 |
| gemv_f8 | engram_gate | 0.12 | 13.2 | 105.8 |
| swiglu | act_quant | 1.79 | 11.0 | 6.1 |
| act_quant | gemv_bf16 | 1.00 | 10.9 | 10.9 |
| hc_post8 | copy | 0.12 | 10.6 | 84.9 |
| hc_pre8 | rmsnorm32 | 2.11 | 8.9 | 4.2 |
| moe_place | copy | 0.99 | 8.8 | 8.8 |
| engram_gate | hc_mixes_part1 | 0.12 | 7.5 | 59.8 |
| copy | moe_combine | 0.99 | 7.2 | 7.2 |
| gemv_f8 | swiglu | 0.99 | 7.0 | 7.0 |
| copy | hc_post8 | 1.00 | 6.3 | 6.3 |
| copy | engram_dequant | 0.12 | 5.9 | 47.5 |
| gemv_ffma_narrow | route | 0.99 | 5.9 | 6.0 |
| sparse_attn | rope | 1.00 | 5.6 | 5.6 |
| memset | gemv_ffma | 0.12 | 5.2 | 43.5 |
| rope | act_quant | 2.00 | 4.6 | 2.3 |
| rmsnorm32 | gemv_ffma_narrow | 0.99 | 4.2 | 4.2 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 304.5 | 44.1 | 260.4 | 6% |
| 0/15 | 309.2 | 44.3 | 264.9 | 6% |
| 0/17 | 316.3 | 47.4 | 268.9 | 6% |
| 0/19 | 323.9 | 48.9 | 274.9 | 6% |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 12 | 1.5 | 4.636 | 393.4 |
| cudaLaunchKernel_v7000 | 1549 | 193.7 | 4.190 | 2.7 |
| cuMemAllocAsync | 1421 | 177.6 | 2.281 | 1.6 |
| cuStreamSynchronize | 4 | 0.5 | 2.114 | 528.5 |
| cuMemFreeAsync | 1419 | 177.4 | 1.513 | 1.1 |
| cuMemcpyDtoDAsync_v2 | 370 | 46.2 | 1.329 | 3.6 |
| cuStreamWaitEvent | 567 | 70.9 | 0.421 | 0.7 |
| cuMemsetD8Async | 107 | 13.4 | 0.278 | 2.6 |
| cuKernelGetName | 1549 | 193.7 | 0.144 | 0.1 |
| cuEventRecord | 200 | 25.0 | 0.120 | 0.6 |
| cuMemcpyHtoDAsync_v2 | 16 | 2.0 | 0.067 | 4.3 |
| cuMemcpy2DAsync_v2 | 15 | 1.9 | 0.062 | 4.0 |
| cuEventSynchronize | 8 | 1.0 | 0.012 | 1.5 |

host threads, time inside CUDA calls per step (ms): 3.975, 3.936, 3.813, 3.785, 1.659

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 231 | 7.23 | 513.4 |
| v41_gemv_f8_tiles_kernel | 79 | 2.48 | 214.0 |
| v41_gemv_bf16_kernel | 38 | 1.19 | 167.4 |
| v41_gemv_ffma_narrow_kernel | 40 | 1.24 | 159.5 |
| v41_gemv_ffma_kernel | 4 | 0.12 | 137.6 |
| v41_rmsnorm32_kernel | 136 | 4.24 | 93.0 |
| v41_act_quant_kernel | 348 | 10.89 | 89.0 |
| v41_hc_mixes_part1_kernel | 64 | 1.99 | 59.0 |
| v41_sparse_attn_kernel | 32 | 1.00 | 53.4 |
| v41_hc_finish_kernel | 64 | 1.99 | 42.9 |
| v41_swiglu_kernel | 58 | 1.82 | 34.0 |
| v41_hc_post8_kernel | 64 | 1.99 | 25.5 |
| v41_route_kernel | 32 | 0.99 | 25.3 |
| v41_moe_scan_kernel | 32 | 0.99 | 20.9 |
| v41_moe_combine_kernel | 32 | 0.99 | 20.3 |
| v41_rope_kernel | 104 | 3.26 | 17.6 |
| v41_moe_place_kernel | 32 | 0.99 | 15.7 |
| v41_hc_pre8_kernel | 68 | 2.11 | 12.2 |
| v41_engram_gate_kernel | 4 | 0.12 | 10.4 |
| v41_attn_idx_kernel | 32 | 1.00 | 6.0 |
| v41_fp4_qdq_kernel | 8 | 0.26 | 5.4 |
| v41_moe_count_kernel | 32 | 0.99 | 5.1 |
| v41_topk_kernel | 4 | 0.12 | 1.5 |
| v41_index_score_kernel | 4 | 0.12 | 1.5 |
| v41_comp_pool2_kernel | 2 | 0.07 | 1.2 |
| v41_engram_dequant_kernel | 4 | 0.12 | 1.1 |
