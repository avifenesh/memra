decode window: 384 marker launches = 24.00 steps of 1 ranks x 8 layers; measured over 23.00 steps, 118.546 ms -> 5.154 ms/step, 644.3 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 395 | 60 | 4.809 | 601.1 | 93% |

some stream busy 4.809 ms/step (93% of wall); every stream idle 0.345 ms/step

stream 0/13: idle 0.345 ms/step = 43.2 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.54 | 19.0 | 7.5 |
| act_quant | gemv_f8 | 7.23 | 2.3 | 0.3 |
| route | memset | 0.99 | 2.2 | 2.2 |
| engram_gate | hc_mixes_part1 | 0.12 | 1.5 | 12.0 |
| copy | hc_mixes_part1 | 0.12 | 1.0 | 8.2 |
| memset | moe_count | 0.99 | 1.0 | 1.0 |
| memset | memset | 1.99 | 0.9 | 0.5 |
| hc_pre8 | rmsnorm32 | 2.11 | 0.7 | 0.3 |
| hc_finish | hc_pre8 | 1.99 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 1.99 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| swiglu | act_quant | 1.99 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| act_quant | gemv_f8_tiles | 1.99 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| gemv_ffma | copy | 0.12 | 0.5 | 4.5 |
| hc_post8 | hc_mixes_part1 | 1.74 | 0.5 | 0.3 |
| copy | engram_dequant | 0.12 | 0.5 | 4.3 |
| copy | act_quant | 1.06 | 0.5 | 0.4 |
| hc_post8 | copy | 0.12 | 0.4 | 3.4 |
| memset | copy | 0.12 | 0.4 | 3.3 |
| copy | memset | 0.12 | 0.3 | 2.9 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.07 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 0.99 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| gemv_f8 | rope | 1.12 | 0.3 | 0.3 |
| gemv_f8 | swiglu | 0.99 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 18.9 | 1.9 | 16.9 | 1% |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 1 | 0.1 | 2.932 | 3065.1 |
| cudaLaunchKernel_v7000 | 396 | 49.5 | 0.837 | 2.1 |
| cuStreamSynchronize | 1 | 0.1 | 0.346 | 345.7 |
| cuMemFreeAsync | 365 | 45.7 | 0.133 | 0.4 |
| cuMemAllocAsync | 365 | 45.6 | 0.130 | 0.4 |
| cuMemcpyDtoDAsync_v2 | 25 | 3.1 | 0.063 | 2.6 |
| cuMemsetD8Async | 27 | 3.4 | 0.040 | 1.5 |
| cuKernelGetName | 396 | 49.5 | 0.033 | 0.1 |
| cuMemcpy2DAsync_v2 | 4 | 0.5 | 0.011 | 2.7 |
| cuMemcpyHtoDAsync_v2 | 4 | 0.5 | 0.009 | 2.3 |
| cuEventRecord | 2 | 0.2 | 0.003 | 1.7 |
| cuEventSynchronize | 2 | 0.2 | 0.002 | 0.9 |
| cuStreamWaitEvent | 2 | 0.2 | 0.002 | 0.8 |

host threads, time inside CUDA calls per step (ms): 4.541

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_tiles_kernel | 24 | 2.98 | 151.7 |
| v41_gemv_f8_kernel | 58 | 7.23 | 139.4 |
| v41_gemv_ffma_kernel | 1 | 0.12 | 109.7 |
| v41_gemv_bf16_kernel | 10 | 1.19 | 49.0 |
| v41_gemv_ffma_narrow_kernel | 10 | 1.24 | 28.1 |
| v41_rmsnorm32_kernel | 34 | 4.24 | 17.1 |
| v41_hc_mixes_part1_kernel | 16 | 1.99 | 13.3 |
| v41_act_quant_kernel | 90 | 11.22 | 13.3 |
| v41_sparse_attn_kernel | 8 | 1.00 | 12.4 |
| v41_hc_finish_kernel | 16 | 1.99 | 10.0 |
| v41_route_kernel | 8 | 0.99 | 6.8 |
| v41_swiglu_kernel | 16 | 1.99 | 6.5 |
| v41_rope_kernel | 26 | 3.26 | 6.0 |
| v41_hc_post8_kernel | 16 | 1.99 | 5.7 |
| v41_moe_scan_kernel | 8 | 0.99 | 5.3 |
| v41_moe_combine_kernel | 8 | 0.99 | 4.4 |
| v41_moe_place_kernel | 8 | 0.99 | 4.1 |
| v41_hc_pre8_kernel | 17 | 2.11 | 2.8 |
| v41_engram_gate_kernel | 1 | 0.12 | 1.9 |
| v41_moe_count_kernel | 8 | 0.99 | 1.4 |
| v41_attn_idx_kernel | 8 | 1.00 | 1.2 |
| v41_fp4_qdq_kernel | 2 | 0.26 | 1.2 |
| v41_index_score_kernel | 1 | 0.12 | 0.3 |
| v41_comp_pool2_kernel | 1 | 0.07 | 0.3 |
| v41_topk_kernel | 1 | 0.12 | 0.2 |
| v41_engram_dequant_kernel | 1 | 0.12 | 0.1 |
