decode window: 1920 marker launches = 24.00 steps of 2 ranks x 20 layers; measured over 23.00 steps, 168.085 ms -> 7.308 ms/step, 365.4 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 870 | 80 | 6.712 | 335.6 | 92% |
| 0/26 | 160 | 0 | 0.913 | 45.6 | 12% |
| 1/24 | 870 | 80 | 6.701 | 335.1 | 92% |
| 1/27 | 160 | 0 | 0.935 | 46.7 | 13% |

some stream busy 7.035 ms/step (96% of wall); every stream idle 0.273 ms/step

stream 0/14: idle 0.595 ms/step = 29.8 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.37 | 9.3 | 3.9 |
| pull | hc_post8 | 2.00 | 3.3 | 1.6 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| engram_gate | hc_pre8 | 0.10 | 1.6 | 15.9 |
| act_quant | gemv_f8 | 4.25 | 1.3 | 0.3 |
| push | pull | 4.00 | 1.2 | 0.3 |
| hc_pre8 | rmsnorm32 | 2.05 | 0.7 | 0.4 |
| copy | hc_pre8 | 0.05 | 0.6 | 13.5 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_pre8 | 1.85 | 0.5 | 0.3 |
| hc_post8 | copy | 0.10 | 0.4 | 3.5 |
| copy | act_quant | 1.07 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.08 | 0.3 | 0.3 |
| gemv_f8 | rope | 1.15 | 0.3 | 0.3 |
| act_quant | gemv_f4s_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |
| act_quant | swiglu_tiles | 1.00 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |
| gemv_f8 | moe_combine | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 19.3 | 1.9 | 17.4 | 0% |
| 0/26 | 319.5 | 77.8 | 241.6 | 4% |
| 1/24 | 19.7 | 1.8 | 17.9 | 0% |
| 1/27 | 318.4 | 77.6 | 240.7 | 4% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/26 | 305.2 | 335.4 | 45.7 | 0.0 | 83.8 | 0.0 | 241.1 | 0.0 |
| 1/27 | 305.2 | 335.4 | 46.8 | 0.0 | 83.6 | 0.0 | 240.2 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 2061 | 103.0 | 4.400 | 2.1 |
| cuStreamSynchronize | 4 | 0.2 | 4.224 | 1056.1 |
| cuMemcpyDtoHAsync_v2 | 4 | 0.2 | 2.712 | 708.8 |
| cuMemcpyDtoDAsync_v2 | 131 | 6.6 | 0.337 | 2.6 |
| cuKernelGetName | 2061 | 103.0 | 0.189 | 0.1 |
| cuStreamWaitEvent | 240 | 12.0 | 0.092 | 0.4 |
| cuEventRecord | 244 | 12.2 | 0.078 | 0.3 |
| cuMemcpyHtoDAsync_v2 | 12 | 0.6 | 0.028 | 2.4 |
| cuMemcpy2DAsync_v2 | 8 | 0.4 | 0.021 | 2.7 |
| cuMemsetD8Async | 6 | 0.3 | 0.016 | 2.8 |
| cuMemAllocAsync | 9 | 0.4 | 0.011 | 1.3 |
| cuEventSynchronize | 12 | 0.6 | 0.007 | 0.5 |
| cuMemFreeAsync | 4 | 0.2 | 0.003 | 0.7 |

host threads, time inside CUDA calls per step (ms): 6.091, 6.027

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 290 | 7.24 | 175.1 |
| v41_swiglu_tiles_kernel | 40 | 1.00 | 97.7 |
| v41_gemv_bf16_kernel | 49 | 1.23 | 77.4 |
| v41_gemv_ffma_narrow_kernel | 52 | 1.30 | 66.5 |
| v41_gemv_ffma_kernel | 2 | 0.05 | 43.7 |
| v41_gemv_f4s_tiles_kernel | 40 | 1.00 | 37.0 |
| v41_rmsnorm32_kernel | 168 | 4.20 | 35.8 |
| v41_push_kernel | 160 | 4.00 | 31.9 |
| v41_hc_mixes_part1_kernel | 80 | 2.00 | 25.8 |
| v41_hc_finish_kernel | 80 | 2.00 | 25.2 |
| v41_sparse_attn_kernel | 40 | 1.00 | 23.1 |
| v41_pull_kernel | 160 | 4.00 | 21.3 |
| v41_act_quant_kernel | 370 | 9.24 | 20.8 |
| v41_route_kernel | 40 | 1.00 | 17.2 |
| v41_moe_plan_small_kernel | 40 | 1.00 | 13.1 |
| v41_hc_post8_kernel | 80 | 2.00 | 8.3 |
| v41_rope_kernel | 132 | 3.31 | 8.3 |
| v41_moe_combine_kernel | 40 | 1.00 | 8.1 |
| v41_hc_pre8_kernel | 82 | 2.05 | 4.8 |
| v41_swiglu_kernel | 40 | 1.00 | 3.7 |
| v41_engram_gate_kernel | 4 | 0.10 | 3.0 |
| v41_fp4_qdq_kernel | 12 | 0.31 | 2.9 |
| v41_attn_idx_kernel | 40 | 1.00 | 2.4 |
| v41_index_score_kernel | 6 | 0.15 | 0.7 |
| v41_comp_pool2_kernel | 3 | 0.08 | 0.6 |
| v41_topk_kernel | 6 | 0.15 | 0.6 |
| v41_engram_dequant_kernel | 4 | 0.10 | 0.2 |
