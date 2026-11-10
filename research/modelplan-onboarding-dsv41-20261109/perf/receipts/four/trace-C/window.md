decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 308.615 ms -> 13.418 ms/step, 335.5 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 2253 | 261 | 11.998 | 299.9 | 89% |
| 1/28 | 2253 | 261 | 11.989 | 299.7 | 89% |
| 2/40 | 2253 | 261 | 11.988 | 299.7 | 89% |
| 3/52 | 2253 | 261 | 11.992 | 299.8 | 89% |

some stream busy 12.835 ms/step (96% of wall); every stream idle 0.583 ms/step

stream 0/16: idle 1.420 ms/step = 35.5 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| gemv_bf16 | push | 1.00 | 6.5 | 6.5 |
| copy | copy | 2.08 | 6.4 | 3.1 |
| act_quant | gemv_f8 | 6.25 | 2.0 | 0.3 |
| route | memset | 1.00 | 1.8 | 1.8 |
| push | pull | 4.00 | 1.1 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 1.0 | 1.0 |
| memset | moe_count | 1.00 | 0.8 | 0.8 |
| memset | memset | 2.00 | 0.8 | 0.4 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| hc_pre8 | rmsnorm32 | 2.02 | 0.6 | 0.3 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| act_quant | gemv_f8t_tiles | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.6 | 11.5 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| copy | act_quant | 1.14 | 0.5 | 0.5 |
| hc_post8 | copy | 0.05 | 0.5 | 9.1 |
| swiglu | push | 1.00 | 0.4 | 0.4 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| attn_idx | sparse_attn | 1.00 | 0.3 | 0.3 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |
| copy | hc_mixes_part1 | 0.02 | 0.3 | 12.2 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 25.4 | 8.2 | 17.2 | 0% |
| 1/28 | 25.8 | 8.1 | 17.7 | 0% |
| 2/40 | 25.7 | 8.1 | 17.5 | 0% |
| 3/52 | 25.6 | 8.2 | 17.4 | 0% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 294.1 | 325.0 | 290.7 | 4.0 | 9.5 | 3.8 | 17.3 | 0.0 |
| 1/28 | 294.1 | 325.0 | 290.3 | 4.2 | 9.4 | 3.8 | 17.7 | 0.0 |
| 2/40 | 294.1 | 325.0 | 290.4 | 4.1 | 9.5 | 3.8 | 17.6 | 0.0 |
| 3/52 | 294.1 | 325.0 | 290.6 | 4.0 | 9.5 | 3.8 | 17.4 | 0.0 |

routed-expert tile kernels a layer, us: mean over the ranks 0.0, the busiest rank 0.0, the idlest 0.0, all ranks together 0.0

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 9018 | 225.4 | 22.093 | 2.4 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 10.061 | 1314.8 |
| cuStreamSynchronize | 8 | 0.2 | 5.311 | 663.8 |
| cuMemcpyDtoDAsync_v2 | 506 | 12.7 | 1.476 | 2.9 |
| cuKernelGetName | 9018 | 225.4 | 0.847 | 0.1 |
| cuMemsetD8Async | 492 | 12.3 | 0.732 | 1.5 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.125 | 5.3 |
| cuMemAllocAsync | 32 | 0.8 | 0.081 | 2.5 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.048 | 3.1 |
| cuEventSynchronize | 24 | 0.6 | 0.031 | 1.3 |
| cuMemFreeAsync | 8 | 0.2 | 0.016 | 2.0 |
| cuEventRecord | 8 | 0.2 | 0.007 | 0.9 |

host threads, time inside CUDA calls per step (ms): 10.306, 10.221, 10.171, 10.129

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 242.4 |
| v41_gemv_f8t_tiles_kernel | 479 | 3.00 | 163.6 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.3 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 98.1 |
| v41_push_kernel | 640 | 4.00 | 64.8 |
| v41_rmsnorm32_kernel | 664 | 4.15 | 62.8 |
| v41_pull_kernel | 640 | 4.00 | 60.2 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.5 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.7 |
| v41_act_quant_kernel | 1639 | 10.25 | 43.4 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.8 |
| v41_route_kernel | 160 | 1.00 | 23.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_moe_scan_kernel | 160 | 1.00 | 20.7 |
| v41_moe_place_kernel | 160 | 1.00 | 16.1 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.8 |
| v41_moe_combine_kernel | 160 | 1.00 | 14.2 |
| v41_rope_kernel | 533 | 3.33 | 11.7 |
| v41_swiglu_kernel | 320 | 2.00 | 10.4 |
| v41_hc_pre8_kernel | 324 | 2.02 | 8.6 |
| v41_fp4_qdq_kernel | 53 | 0.33 | 6.4 |
| v41_moe_count_kernel | 160 | 1.00 | 5.0 |
| v41_attn_idx_kernel | 160 | 1.00 | 3.7 |
| v41_engram_gate_kernel | 8 | 0.05 | 2.9 |
| v41_index_score_kernel | 32 | 0.20 | 2.2 |
| v41_topk_kernel | 32 | 0.20 | 1.7 |
| v41_comp_pool2_kernel | 6 | 0.04 | 0.6 |
| v41_candidates_kernel | 4 | 0.03 | 0.3 |
| v41_engram_dequant_kernel | 8 | 0.05 | 0.2 |
