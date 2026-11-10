decode window: 1920 marker launches = 24.00 steps of 2 ranks x 20 layers; measured over 23.00 steps, 176.206 ms -> 7.661 ms/step, 383.1 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 970 | 80 | 7.115 | 355.7 | 93% |
| 0/27 | 80 | 0 | 0.417 | 20.8 | 5% |
| 1/24 | 970 | 80 | 7.106 | 355.3 | 93% |
| 1/26 | 80 | 0 | 0.419 | 20.9 | 5% |

some stream busy 7.431 ms/step (97% of wall); every stream idle 0.230 ms/step

stream 0/14: idle 0.546 ms/step = 27.3 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.37 | 9.1 | 3.8 |
| swiglu_tiles | push | 1.00 | 1.7 | 1.7 |
| act_quant | gemv_f8 | 5.25 | 1.6 | 0.3 |
| push | pull | 4.00 | 1.2 | 0.3 |
| engram_gate | hc_mixes_part1 | 0.10 | 1.0 | 9.5 |
| hc_pre8 | rmsnorm32 | 2.05 | 0.6 | 0.3 |
| pull | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| pull | hc_post8 | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.85 | 0.5 | 0.3 |
| hc_post8 | copy | 0.10 | 0.4 | 4.1 |
| copy | act_quant | 1.07 | 0.4 | 0.4 |
| rmsnorm32 | rope | 1.08 | 0.3 | 0.3 |
| gemv_f8 | rope | 1.15 | 0.3 | 0.3 |
| act_quant | gemv_f8t_tiles | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| moe_plan_small | act_quant | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| moe_combine | push | 1.00 | 0.3 | 0.3 |
| act_quant | swiglu_tiles | 1.00 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |
| gemv_f8 | moe_combine | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 17.0 | 1.0 | 16.0 | 0% |
| 0/27 | 361.2 | 92.5 | 268.7 | 6% |
| 1/24 | 17.3 | 1.0 | 16.2 | 0% |
| 1/26 | 361.1 | 92.3 | 268.8 | 6% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/14 | 321.5 | 352.6 | 330.8 | 3.2 | 1.0 | 1.8 | 16.0 | 0.0 |
| 1/24 | 321.1 | 352.6 | 330.5 | 3.3 | 1.1 | 1.8 | 16.3 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 2101 | 105.0 | 4.797 | 2.3 |
| cuStreamSynchronize | 4 | 0.2 | 4.424 | 1106.1 |
| cuMemcpyDtoHAsync_v2 | 4 | 0.2 | 2.800 | 731.9 |
| cuMemcpyDtoDAsync_v2 | 131 | 6.6 | 0.364 | 2.8 |
| cuKernelGetName | 2101 | 105.0 | 0.207 | 0.1 |
| cuStreamWaitEvent | 80 | 4.0 | 0.035 | 0.4 |
| cuMemcpyHtoDAsync_v2 | 12 | 0.6 | 0.030 | 2.6 |
| cuEventRecord | 84 | 4.2 | 0.030 | 0.4 |
| cuMemcpy2DAsync_v2 | 8 | 0.4 | 0.022 | 2.9 |
| cuMemsetD8Async | 6 | 0.3 | 0.017 | 2.9 |
| cuMemAllocAsync | 9 | 0.4 | 0.012 | 1.4 |
| cuEventSynchronize | 12 | 0.6 | 0.007 | 0.6 |
| cuMemFreeAsync | 4 | 0.2 | 0.003 | 0.8 |

host threads, time inside CUDA calls per step (ms): 6.393, 6.358

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 290 | 7.24 | 173.8 |
| v41_swiglu_tiles_kernel | 40 | 1.00 | 97.9 |
| v41_gemv_bf16_kernel | 49 | 1.23 | 77.7 |
| v41_gemv_ffma_narrow_kernel | 52 | 1.30 | 61.0 |
| v41_gemv_ffma_kernel | 2 | 0.05 | 44.8 |
| v41_gemv_f8t_tiles_kernel | 40 | 1.00 | 40.9 |
| v41_rmsnorm32_kernel | 168 | 4.20 | 32.2 |
| v41_push_kernel | 160 | 4.00 | 32.1 |
| v41_act_quant_kernel | 410 | 10.24 | 25.0 |
| v41_hc_mixes_part1_kernel | 80 | 2.00 | 24.6 |
| v41_sparse_attn_kernel | 40 | 1.00 | 23.2 |
| v41_pull_kernel | 160 | 4.00 | 21.0 |
| v41_hc_finish_kernel | 80 | 2.00 | 18.4 |
| v41_route_kernel | 40 | 1.00 | 15.6 |
| v41_moe_plan_small_kernel | 40 | 1.00 | 13.1 |
| v41_hc_post8_kernel | 80 | 2.00 | 8.5 |
| v41_rope_kernel | 132 | 3.31 | 8.2 |
| v41_moe_combine_kernel | 40 | 1.00 | 8.0 |
| v41_hc_pre8_kernel | 82 | 2.05 | 4.8 |
| v41_swiglu_kernel | 40 | 1.00 | 3.8 |
| v41_engram_gate_kernel | 4 | 0.10 | 3.0 |
| v41_fp4_qdq_kernel | 12 | 0.31 | 2.8 |
| v41_attn_idx_kernel | 40 | 1.00 | 2.5 |
| v41_index_score_kernel | 6 | 0.15 | 0.7 |
| v41_comp_pool2_kernel | 3 | 0.08 | 0.7 |
| v41_topk_kernel | 6 | 0.15 | 0.6 |
| v41_engram_dequant_kernel | 4 | 0.10 | 0.2 |
