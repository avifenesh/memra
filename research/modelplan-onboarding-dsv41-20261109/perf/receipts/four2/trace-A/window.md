decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 292.451 ms -> 12.715 ms/step, 317.9 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1934 | 141 | 11.458 | 286.5 | 90% |
| 0/55 | 160 | 0 | 0.706 | 17.7 | 6% |
| 1/28 | 1934 | 141 | 11.456 | 286.4 | 90% |
| 1/57 | 160 | 0 | 0.715 | 17.9 | 6% |
| 2/40 | 1934 | 141 | 11.453 | 286.3 | 90% |
| 2/56 | 160 | 0 | 0.714 | 17.9 | 6% |
| 3/52 | 1934 | 141 | 11.451 | 286.3 | 90% |
| 3/54 | 160 | 0 | 0.710 | 17.8 | 6% |

some stream busy 12.155 ms/step (96% of wall); every stream idle 0.560 ms/step

stream 0/16: idle 1.257 ms/step = 31.4 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.08 | 6.0 | 2.9 |
| gemv_bf16 | push | 1.00 | 4.9 | 4.9 |
| copy | act_quant | 1.14 | 2.2 | 1.9 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 1.7 | 1.7 |
| act_quant | gemv_f8 | 5.25 | 1.7 | 0.3 |
| push | pull | 4.00 | 1.2 | 0.3 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.7 | 13.4 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| hc_pre8 | rmsnorm32 | 2.02 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| hc_post8 | copy | 0.05 | 0.3 | 6.0 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_f8t_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 24.5 | 8.7 | 15.8 | 0% |
| 0/55 | 299.8 | 53.1 | 246.7 | 4% |
| 1/28 | 24.7 | 8.6 | 16.1 | 0% |
| 1/57 | 299.6 | 52.8 | 246.8 | 4% |
| 2/40 | 24.7 | 8.7 | 16.0 | 0% |
| 2/56 | 299.6 | 53.0 | 246.6 | 4% |
| 3/52 | 24.5 | 8.7 | 15.9 | 0% |
| 3/54 | 299.7 | 53.1 | 246.6 | 4% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 276.7 | 307.7 | 278.6 | 2.6 | 9.7 | 1.3 | 15.8 | 0.0 |
| 1/28 | 276.6 | 307.7 | 278.3 | 2.8 | 9.5 | 1.3 | 16.1 | 0.0 |
| 2/40 | 276.7 | 307.7 | 278.4 | 2.7 | 9.7 | 1.3 | 16.0 | 0.0 |
| 3/52 | 276.6 | 307.7 | 278.4 | 2.6 | 9.8 | 1.3 | 15.9 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 8378 | 209.4 | 19.495 | 2.3 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 10.537 | 1377.0 |
| cuStreamSynchronize | 8 | 0.2 | 5.572 | 696.4 |
| cuMemcpyDtoDAsync_v2 | 506 | 12.7 | 1.429 | 2.8 |
| cuKernelGetName | 8378 | 209.4 | 1.020 | 0.1 |
| cuEventRecord | 328 | 8.2 | 0.202 | 0.6 |
| cuStreamWaitEvent | 320 | 8.0 | 0.143 | 0.4 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.097 | 4.1 |
| cuMemAllocAsync | 32 | 0.8 | 0.075 | 2.3 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.046 | 3.0 |
| cuMemsetD8Async | 12 | 0.3 | 0.045 | 3.8 |
| cuEventSynchronize | 24 | 0.6 | 0.028 | 1.2 |
| cuMemFreeAsync | 8 | 0.2 | 0.012 | 1.5 |

host threads, time inside CUDA calls per step (ms): 9.765, 9.688, 9.687, 9.559

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 251.0 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.6 |
| v41_swiglu_tiles_kernel | 160 | 1.00 | 136.0 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 106.6 |
| v41_push_kernel | 640 | 4.00 | 64.8 |
| v41_rmsnorm32_kernel | 664 | 4.15 | 62.9 |
| v41_pull_kernel | 640 | 4.00 | 59.8 |
| v41_gemv_f8t_tiles_kernel | 160 | 1.00 | 51.0 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.8 |
| v41_act_quant_kernel | 1639 | 10.25 | 47.6 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.8 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.9 |
| v41_moe_plan_small_kernel | 160 | 1.00 | 25.0 |
| v41_route_kernel | 160 | 1.00 | 23.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.8 |
| v41_moe_combine_kernel | 160 | 1.00 | 14.2 |
| v41_rope_kernel | 533 | 3.33 | 11.7 |
| v41_hc_pre8_kernel | 324 | 2.02 | 8.6 |
| v41_fp4_qdq_kernel | 53 | 0.33 | 6.4 |
| v41_swiglu_kernel | 160 | 1.00 | 5.1 |
| v41_attn_idx_kernel | 160 | 1.00 | 3.7 |
| v41_engram_gate_kernel | 8 | 0.05 | 2.9 |
| v41_index_score_kernel | 32 | 0.20 | 2.2 |
| v41_topk_kernel | 32 | 0.20 | 1.7 |
| v41_comp_pool2_kernel | 6 | 0.04 | 0.7 |
| v41_candidates_kernel | 4 | 0.03 | 0.3 |
| v41_engram_dequant_kernel | 8 | 0.05 | 0.2 |
