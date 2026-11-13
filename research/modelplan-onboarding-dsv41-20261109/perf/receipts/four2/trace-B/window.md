decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 311.260 ms -> 13.533 ms/step, 338.3 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1854 | 141 | 12.450 | 311.2 | 92% |
| 0/57 | 160 | 0 | 0.709 | 17.7 | 5% |
| 1/28 | 1854 | 141 | 12.443 | 311.1 | 92% |
| 1/54 | 160 | 0 | 0.717 | 17.9 | 5% |
| 2/40 | 1854 | 141 | 12.438 | 311.0 | 92% |
| 2/56 | 160 | 0 | 0.718 | 17.9 | 5% |
| 3/52 | 1854 | 141 | 12.447 | 311.2 | 92% |
| 3/55 | 160 | 0 | 0.710 | 17.8 | 5% |

some stream busy 13.116 ms/step (97% of wall); every stream idle 0.417 ms/step

stream 0/16: idle 1.083 ms/step = 27.1 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.08 | 5.6 | 2.7 |
| hc_pre_norm | gemv_ffma_narrow | 1.00 | 2.7 | 2.8 |
| gemv_f8 | push | 1.00 | 2.1 | 2.1 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| act_quant | gemv_f8 | 5.25 | 1.5 | 0.3 |
| push | pull | 4.00 | 1.2 | 0.3 |
| gemv_bf16 | push | 1.00 | 1.0 | 1.0 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.6 | 12.7 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre_norm | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| copy | act_quant | 1.14 | 0.4 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| attn_idx | sparse_attn | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_f8t_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| hc_pre_norm | act_quant | 1.00 | 0.3 | 0.3 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 20.7 | 5.7 | 15.0 | 0% |
| 0/57 | 320.1 | 46.4 | 273.7 | 4% |
| 1/28 | 20.9 | 5.6 | 15.3 | 0% |
| 1/54 | 319.9 | 46.0 | 273.9 | 3% |
| 2/40 | 21.0 | 5.8 | 15.2 | 0% |
| 2/56 | 319.9 | 46.3 | 273.6 | 3% |
| 3/52 | 20.8 | 5.7 | 15.1 | 0% |
| 3/55 | 320.1 | 46.3 | 273.8 | 3% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 300.8 | 328.1 | 303.3 | 2.6 | 6.1 | 1.2 | 15.0 | 0.0 |
| 1/28 | 300.8 | 328.1 | 303.0 | 2.7 | 5.9 | 1.3 | 15.3 | 0.0 |
| 2/40 | 300.8 | 328.1 | 303.0 | 2.7 | 6.1 | 1.3 | 15.2 | 0.0 |
| 3/52 | 300.8 | 328.1 | 303.3 | 2.6 | 6.0 | 1.3 | 15.1 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 8058 | 201.4 | 18.070 | 2.2 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 13.835 | 1808.0 |
| cuStreamSynchronize | 8 | 0.2 | 7.780 | 972.5 |
| cuMemcpyDtoDAsync_v2 | 506 | 12.7 | 1.373 | 2.7 |
| cuKernelGetName | 8058 | 201.4 | 0.760 | 0.1 |
| cuEventRecord | 328 | 8.2 | 0.201 | 0.6 |
| cuStreamWaitEvent | 320 | 8.0 | 0.132 | 0.4 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.071 | 3.0 |
| cuMemAllocAsync | 32 | 0.8 | 0.070 | 2.2 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.045 | 2.9 |
| cuMemsetD8Async | 12 | 0.3 | 0.044 | 3.7 |
| cuEventSynchronize | 24 | 0.6 | 0.026 | 1.1 |
| cuMemFreeAsync | 8 | 0.2 | 0.010 | 1.2 |

host threads, time inside CUDA calls per step (ms): 10.683, 10.638, 10.559, 10.538

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 251.0 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.5 |
| v41_hc_pre_norm_kernel | 320 | 2.00 | 144.7 |
| v41_swiglu_tiles_kernel | 160 | 1.00 | 134.4 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 106.5 |
| v41_push_kernel | 640 | 4.00 | 64.6 |
| v41_pull_kernel | 640 | 4.00 | 61.5 |
| v41_gemv_f8t_tiles_kernel | 160 | 1.00 | 51.3 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.7 |
| v41_act_quant_kernel | 1639 | 10.25 | 47.6 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.8 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.9 |
| v41_rmsnorm32_kernel | 344 | 2.15 | 26.0 |
| v41_moe_plan_small_kernel | 160 | 1.00 | 25.0 |
| v41_route_kernel | 160 | 1.00 | 23.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.8 |
| v41_moe_combine_kernel | 160 | 1.00 | 14.2 |
| v41_rope_kernel | 533 | 3.33 | 11.7 |
| v41_fp4_qdq_kernel | 53 | 0.33 | 6.4 |
| v41_swiglu_kernel | 160 | 1.00 | 5.1 |
| v41_attn_idx_kernel | 160 | 1.00 | 3.7 |
| v41_engram_gate_kernel | 8 | 0.05 | 2.9 |
| v41_index_score_kernel | 32 | 0.20 | 2.2 |
| v41_topk_kernel | 32 | 0.20 | 1.7 |
| v41_comp_pool2_kernel | 6 | 0.04 | 0.6 |
| v41_candidates_kernel | 4 | 0.03 | 0.3 |
| v41_engram_dequant_kernel | 8 | 0.05 | 0.2 |
| v41_hc_pre8_kernel | 4 | 0.02 | 0.1 |
