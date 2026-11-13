decode window: 1920 marker launches = 24.00 steps of 2 ranks x 20 layers; measured over 23.00 steps, 237.771 ms -> 10.338 ms/step, 516.9 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 986 | 218 | 7.538 | 376.9 | 73% |
| 1/24 | 987 | 198 | 7.756 | 387.8 | 75% |

some stream busy 8.683 ms/step (84% of wall); every stream idle 1.655 ms/step

stream 0/14: idle 2.800 ms/step = 140.0 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| gemv_f8 | copy | 1.98 | 43.0 | 21.8 |
| copy | act_quant | 3.07 | 20.9 | 6.8 |
| copy | copy | 2.32 | 15.2 | 6.6 |
| hc_post8 | copy | 0.10 | 11.4 | 113.7 |
| copy | hc_post8 | 1.00 | 8.3 | 8.3 |
| gemv_bf16 | copy | 1.00 | 7.3 | 7.3 |
| copy | moe_combine | 0.98 | 4.9 | 5.0 |
| act_quant | gemv_f8_tiles | 1.93 | 4.4 | 2.3 |
| moe_place | copy | 1.00 | 3.7 | 3.7 |
| engram_gate | hc_mixes_part1 | 0.10 | 2.7 | 26.8 |
| act_quant | gemv_f8 | 7.24 | 2.5 | 0.3 |
| route | memset | 1.00 | 1.3 | 1.3 |
| rmsnorm32 | act_quant | 2.00 | 0.9 | 0.5 |
| memset | memset | 2.00 | 0.8 | 0.4 |
| copy | hc_mixes_part1 | 0.05 | 0.7 | 15.3 |
| memset | moe_count | 1.00 | 0.7 | 0.7 |
| hc_pre8 | rmsnorm32 | 2.05 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| swiglu | act_quant | 1.97 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.85 | 0.5 | 0.3 |
| copy | engram_dequant | 0.10 | 0.4 | 4.0 |
| rmsnorm32 | rope | 1.08 | 0.3 | 0.3 |
| gemv_f8 | rope | 1.15 | 0.3 | 0.3 |
| copy | gemv_ffma_narrow | 0.15 | 0.3 | 2.0 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 56.2 | 17.7 | 38.5 | 5% |
| 1/24 | 60.3 | 21.0 | 39.3 | 5% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/14 | 442.3 | 481.1 | 343.9 | 11.9 | 34.0 | 53.8 | 38.4 | 76.4 |
| 1/24 | 441.5 | 481.1 | 356.1 | 11.2 | 33.3 | 41.9 | 39.4 | 88.3 |

routed-expert tile kernels a layer, us: mean over the ranks 82.3, the busiest rank 105.5, the idlest 59.1, all ranks together 164.6

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 1973 | 98.6 | 4.687 | 2.4 |
| cuMemcpyDtoHAsync_v2 | 22 | 1.1 | 3.048 | 139.4 |
| cuMemFreeAsync | 1807 | 90.3 | 1.088 | 0.6 |
| cuMemAllocAsync | 1808 | 90.4 | 0.946 | 0.5 |
| cuMemcpyDtoDAsync_v2 | 250 | 12.5 | 0.681 | 2.7 |
| cuStreamSynchronize | 4 | 0.2 | 0.508 | 126.9 |
| cuStreamWaitEvent | 247 | 12.3 | 0.397 | 1.6 |
| cuMemsetD8Async | 126 | 6.3 | 0.206 | 1.6 |
| cuKernelGetName | 1973 | 98.6 | 0.173 | 0.1 |
| cuEventRecord | 248 | 12.4 | 0.109 | 0.4 |
| cuCtxSetCurrent | 321 | 16.0 | 0.038 | 0.1 |
| cuMemcpyHtoDAsync_v2 | 12 | 0.6 | 0.038 | 3.2 |
| cuMemcpy2DAsync_v2 | 8 | 0.4 | 0.023 | 3.0 |
| cuEventSynchronize | 8 | 0.4 | 0.012 | 1.5 |

host threads, time inside CUDA calls per step (ms): 4.699, 4.631, 2.625

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 290 | 7.24 | 195.9 |
| v41_gemv_f8_tiles_kernel | 117 | 2.92 | 164.3 |
| v41_gemv_bf16_kernel | 49 | 1.23 | 76.0 |
| v41_gemv_ffma_narrow_kernel | 52 | 1.30 | 55.7 |
| v41_gemv_ffma_kernel | 2 | 0.05 | 43.9 |
| v41_rmsnorm32_kernel | 168 | 4.20 | 31.5 |
| v41_act_quant_kernel | 447 | 11.19 | 24.2 |
| v41_hc_mixes_part1_kernel | 80 | 2.00 | 24.0 |
| v41_sparse_attn_kernel | 40 | 1.00 | 22.6 |
| v41_hc_finish_kernel | 80 | 2.00 | 18.0 |
| v41_route_kernel | 40 | 1.00 | 13.5 |
| v41_swiglu_kernel | 79 | 1.97 | 11.6 |
| v41_moe_scan_kernel | 40 | 1.00 | 10.5 |
| v41_moe_place_kernel | 40 | 1.00 | 8.2 |
| v41_hc_post8_kernel | 80 | 2.00 | 8.0 |
| v41_moe_combine_kernel | 40 | 1.00 | 8.0 |
| v41_rope_kernel | 132 | 3.31 | 7.9 |
| v41_hc_pre8_kernel | 82 | 2.05 | 4.4 |
| v41_engram_gate_kernel | 4 | 0.10 | 2.9 |
| v41_fp4_qdq_kernel | 12 | 0.31 | 2.8 |
| v41_moe_count_kernel | 40 | 1.00 | 2.6 |
| v41_attn_idx_kernel | 40 | 1.00 | 2.3 |
| v41_index_score_kernel | 6 | 0.15 | 0.7 |
| v41_comp_pool2_kernel | 3 | 0.08 | 0.6 |
| v41_topk_kernel | 6 | 0.15 | 0.6 |
| v41_engram_dequant_kernel | 4 | 0.10 | 0.2 |
