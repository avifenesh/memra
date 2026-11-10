decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 313.363 ms -> 13.624 ms/step, 340.6 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1980 | 15 | 12.500 | 312.5 | 92% |
| 0/57 | 160 | 0 | 0.707 | 17.7 | 5% |
| 1/28 | 1980 | 15 | 12.489 | 312.2 | 92% |
| 1/54 | 160 | 0 | 0.716 | 17.9 | 5% |
| 2/40 | 1980 | 15 | 12.490 | 312.3 | 92% |
| 2/56 | 160 | 0 | 0.716 | 17.9 | 5% |
| 3/52 | 1980 | 15 | 12.491 | 312.3 | 92% |
| 3/55 | 160 | 0 | 0.709 | 17.7 | 5% |

some stream busy 13.156 ms/step (97% of wall); every stream idle 0.468 ms/step

stream 0/16: idle 1.124 ms/step = 28.1 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | act_quant | 1.14 | 6.4 | 5.6 |
| copy | copy | 2.08 | 5.4 | 2.6 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| act_quant | gemv_f8 | 5.25 | 1.5 | 0.3 |
| push | pull | 4.00 | 1.1 | 0.3 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.6 | 12.2 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre_norm | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| gemv_bf16 | push | 1.00 | 0.3 | 0.3 |
| hc_post8 | copy | 0.05 | 0.3 | 6.1 |
| act_quant | gemv_f8t_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| hc_pre_norm | act_quant | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| hc_pre_norm | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| act_quant | swiglu_tiles | 1.00 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 22.7 | 6.8 | 15.9 | 0% |
| 0/57 | 322.5 | 46.4 | 276.1 | 3% |
| 1/28 | 23.0 | 6.8 | 16.3 | 0% |
| 1/54 | 322.2 | 46.1 | 276.1 | 3% |
| 2/40 | 22.9 | 6.8 | 16.1 | 0% |
| 2/56 | 322.2 | 46.4 | 275.9 | 3% |
| 3/52 | 22.8 | 6.9 | 16.0 | 0% |
| 3/55 | 322.4 | 46.4 | 276.0 | 3% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 302.1 | 330.4 | 307.2 | 0.1 | 7.2 | 0.3 | 15.9 | 0.0 |
| 1/28 | 302.1 | 330.4 | 306.9 | 0.1 | 7.2 | 0.3 | 16.3 | 0.0 |
| 2/40 | 302.0 | 330.4 | 306.9 | 0.1 | 7.3 | 0.3 | 16.1 | 0.0 |
| 3/52 | 302.1 | 330.4 | 307.0 | 0.1 | 7.3 | 0.3 | 16.0 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 8564 | 214.1 | 19.584 | 2.3 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 14.588 | 1906.4 |
| cuStreamSynchronize | 8 | 0.2 | 7.605 | 950.6 |
| cuKernelGetName | 8564 | 214.1 | 0.711 | 0.1 |
| cuEventRecord | 328 | 8.2 | 0.195 | 0.6 |
| cuStreamWaitEvent | 320 | 8.0 | 0.137 | 0.4 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.085 | 3.6 |
| cuMemAllocAsync | 32 | 0.8 | 0.071 | 2.2 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.050 | 3.3 |
| cuMemsetD8Async | 12 | 0.3 | 0.048 | 4.1 |
| cuEventSynchronize | 24 | 0.6 | 0.026 | 1.1 |
| cuMemFreeAsync | 8 | 0.2 | 0.011 | 1.4 |

host threads, time inside CUDA calls per step (ms): 10.892, 10.868, 10.707, 10.644

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 251.0 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.5 |
| v41_hc_pre_norm_kernel | 320 | 2.00 | 144.6 |
| v41_swiglu_tiles_kernel | 160 | 1.00 | 134.5 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 106.5 |
| v41_push_kernel | 640 | 4.00 | 65.0 |
| v41_pull_kernel | 640 | 4.00 | 61.0 |
| v41_gemv_f8t_tiles_kernel | 160 | 1.00 | 51.2 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.7 |
| v41_act_quant_kernel | 1639 | 10.25 | 47.4 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.8 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.9 |
| v41_rmsnorm32_kernel | 344 | 2.15 | 26.0 |
| v41_moe_plan_small_kernel | 160 | 1.00 | 25.0 |
| v41_route_kernel | 160 | 1.00 | 23.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.8 |
| v41_copy_kernel | 506 | 3.16 | 15.4 |
| v41_moe_combine_kernel | 160 | 1.00 | 14.2 |
| v41_rope_kernel | 533 | 3.33 | 11.7 |
| v41_fp4_qdq_kernel | 53 | 0.33 | 6.4 |
| v41_swiglu_kernel | 160 | 1.00 | 5.0 |
| v41_attn_idx_kernel | 160 | 1.00 | 3.7 |
| v41_engram_gate_kernel | 8 | 0.05 | 2.9 |
| v41_index_score_kernel | 32 | 0.20 | 2.2 |
| v41_topk_kernel | 32 | 0.20 | 1.7 |
| v41_comp_pool2_kernel | 6 | 0.04 | 0.7 |
| v41_candidates_kernel | 4 | 0.03 | 0.3 |
| v41_engram_dequant_kernel | 8 | 0.05 | 0.2 |
| v41_hc_pre8_kernel | 4 | 0.02 | 0.1 |
