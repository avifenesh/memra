decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 293.486 ms -> 12.760 ms/step, 319.0 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1934 | 141 | 11.442 | 286.0 | 90% |
| 0/56 | 160 | 0 | 0.708 | 17.7 | 6% |
| 1/28 | 1934 | 141 | 11.433 | 285.8 | 90% |
| 1/55 | 160 | 0 | 0.714 | 17.8 | 6% |
| 2/40 | 1934 | 141 | 11.433 | 285.8 | 90% |
| 2/54 | 160 | 0 | 0.713 | 17.8 | 6% |
| 3/52 | 1934 | 141 | 11.437 | 285.9 | 90% |
| 3/57 | 160 | 0 | 0.708 | 17.7 | 6% |

some stream busy 12.134 ms/step (95% of wall); every stream idle 0.626 ms/step

stream 0/16: idle 1.318 ms/step = 33.0 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.08 | 6.6 | 3.2 |
| gemv_bf16 | push | 1.00 | 4.9 | 4.9 |
| copy | act_quant | 1.14 | 2.6 | 2.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 1.9 | 1.9 |
| act_quant | gemv_f8 | 5.25 | 1.7 | 0.3 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| push | pull | 4.00 | 1.2 | 0.3 |
| hc_post8 | copy | 0.05 | 0.8 | 15.6 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.6 | 12.2 |
| hc_pre8 | rmsnorm32 | 2.02 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| attn_idx | sparse_attn | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_f8t_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| act_quant | swiglu_tiles | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 24.9 | 9.2 | 15.8 | 1% |
| 0/56 | 300.9 | 57.2 | 243.6 | 4% |
| 1/28 | 23.3 | 7.1 | 16.2 | 0% |
| 1/55 | 300.7 | 56.7 | 244.0 | 4% |
| 2/40 | 25.2 | 9.2 | 16.0 | 1% |
| 2/54 | 300.7 | 56.9 | 243.8 | 4% |
| 3/52 | 25.1 | 9.2 | 15.9 | 1% |
| 3/57 | 300.9 | 57.0 | 243.9 | 4% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 276.4 | 308.6 | 278.2 | 2.6 | 10.9 | 1.3 | 15.8 | 0.0 |
| 1/28 | 276.4 | 308.6 | 277.8 | 2.8 | 10.9 | 1.3 | 16.2 | 0.0 |
| 2/40 | 276.4 | 308.6 | 278.0 | 2.6 | 11.0 | 1.3 | 16.1 | 0.0 |
| 3/52 | 276.4 | 308.6 | 278.1 | 2.6 | 10.9 | 1.3 | 15.9 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 8378 | 209.4 | 20.403 | 2.4 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 9.693 | 1266.8 |
| cuStreamSynchronize | 8 | 0.2 | 5.062 | 632.7 |
| cuMemcpyDtoDAsync_v2 | 506 | 12.7 | 1.482 | 2.9 |
| cuKernelGetName | 8378 | 209.4 | 0.764 | 0.1 |
| cuEventRecord | 328 | 8.2 | 0.274 | 0.8 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.175 | 7.4 |
| cuStreamWaitEvent | 320 | 8.0 | 0.146 | 0.5 |
| cuMemAllocAsync | 32 | 0.8 | 0.078 | 2.5 |
| cuMemsetD8Async | 12 | 0.3 | 0.050 | 4.2 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.048 | 3.1 |
| cuEventSynchronize | 24 | 0.6 | 0.030 | 1.2 |
| cuMemFreeAsync | 8 | 0.2 | 0.012 | 1.5 |

host threads, time inside CUDA calls per step (ms): 9.588, 9.587, 9.574, 9.468

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 250.7 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.4 |
| v41_swiglu_tiles_kernel | 160 | 1.00 | 136.0 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 106.5 |
| v41_push_kernel | 640 | 4.00 | 64.6 |
| v41_rmsnorm32_kernel | 664 | 4.15 | 62.8 |
| v41_pull_kernel | 640 | 4.00 | 58.9 |
| v41_gemv_f8t_tiles_kernel | 160 | 1.00 | 51.1 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.7 |
| v41_act_quant_kernel | 1639 | 10.25 | 47.6 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.7 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.9 |
| v41_moe_plan_small_kernel | 160 | 1.00 | 25.0 |
| v41_route_kernel | 160 | 1.00 | 23.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.7 |
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
