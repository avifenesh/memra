decode window: 432 marker launches = 24.00 steps of 1 ranks x 9 layers; measured over 23.00 steps, 156.102 ms -> 6.787 ms/step, 754.1 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 392 | 69 | 6.364 | 707.1 | 94% |
| 0/15 | 54 | 0 | 0.467 | 51.9 | 7% |

some stream busy 6.400 ms/step (94% of wall); every stream idle 0.387 ms/step

stream 0/13: idle 0.423 ms/step = 47.0 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.63 | 18.9 | 7.2 |
| hc_post8 | copy | 0.11 | 4.0 | 36.3 |
| route | memset | 1.00 | 2.4 | 2.4 |
| gemv_f8t_tiles | moe_combine | 1.00 | 1.8 | 1.8 |
| act_quant | gemv_f8 | 4.33 | 1.5 | 0.4 |
| memset | memset | 1.99 | 1.2 | 0.6 |
| memset | moe_count | 1.00 | 1.0 | 1.0 |
| engram_gate | hc_mixes_part1 | 0.11 | 0.8 | 7.5 |
| hc_pre8 | rmsnorm32 | 2.10 | 0.8 | 0.4 |
| hc_finish | hc_pre8 | 2.00 | 0.8 | 0.4 |
| act_quant | gemv_f8t_tiles | 1.99 | 0.7 | 0.4 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.7 | 0.4 |
| rmsnorm32 | act_quant | 2.00 | 0.7 | 0.4 |
| rope | act_quant | 2.00 | 0.7 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.7 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.77 | 0.6 | 0.4 |
| copy | hc_mixes_part1 | 0.11 | 0.5 | 4.4 |
| gemv_f8 | rope | 1.22 | 0.4 | 0.3 |
| rmsnorm32 | rope | 1.11 | 0.4 | 0.3 |
| swiglu | act_quant | 1.00 | 0.4 | 0.4 |
| copy | act_quant | 1.12 | 0.4 | 0.3 |
| gemv_ffma | copy | 0.11 | 0.4 | 3.5 |
| gemv_f8t_tiles | swiglu | 1.00 | 0.4 | 0.4 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.4 | 0.4 |
| moe_combine | hc_post8 | 1.00 | 0.4 | 0.4 |
| moe_place | act_quant | 1.00 | 0.4 | 0.4 |
| moe_scan | moe_place | 1.00 | 0.4 | 0.4 |
| sparse_attn | rope | 1.00 | 0.4 | 0.4 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 19.0 | 0.8 | 18.2 | 1% |
| 0/15 | 697.7 | 217.7 | 480.0 | 4% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/13 | 571.4 | 593.5 | 525.1 | 40.6 | 5.0 | 5.6 | 17.9 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 2 | 0.2 | 4.634 | 2422.5 |
| cudaLaunchKernel_v7000 | 447 | 49.6 | 0.918 | 2.1 |
| cuStreamSynchronize | 1 | 0.1 | 0.320 | 320.5 |
| cuMemcpyDtoDAsync_v2 | 30 | 3.3 | 0.082 | 2.8 |
| cuMemsetD8Async | 30 | 3.3 | 0.043 | 1.4 |
| cuKernelGetName | 447 | 49.6 | 0.037 | 0.1 |
| cuMemcpyHtoDAsync_v2 | 4 | 0.4 | 0.020 | 5.2 |
| cuMemcpy2DAsync_v2 | 4 | 0.4 | 0.010 | 2.6 |
| cuStreamWaitEvent | 18 | 2.0 | 0.007 | 0.4 |
| cuEventRecord | 19 | 2.1 | 0.005 | 0.3 |
| cuMemAllocAsync | 9 | 1.0 | 0.005 | 0.5 |
| cuMemFreeAsync | 9 | 1.0 | 0.004 | 0.4 |
| cuEventSynchronize | 3 | 0.3 | 0.002 | 0.5 |

host threads, time inside CUDA calls per step (ms): 6.086

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 66 | 7.32 | 157.7 |
| v41_gemv_f8t_tiles_kernel | 27 | 2.99 | 137.7 |
| v41_gemv_ffma_kernel | 1 | 0.11 | 118.8 |
| v41_gemv_bf16_kernel | 12 | 1.33 | 56.1 |
| v41_gemv_ffma_narrow_kernel | 13 | 1.44 | 49.6 |
| v41_sparse_attn_kernel | 9 | 1.00 | 37.5 |
| v41_topk_kernel | 2 | 0.22 | 23.4 |
| v41_rmsnorm32_kernel | 39 | 4.31 | 20.3 |
| v41_act_quant_kernel | 93 | 10.31 | 16.6 |
| v41_hc_mixes_part1_kernel | 18 | 2.00 | 15.9 |
| v41_route_kernel | 9 | 1.00 | 13.0 |
| v41_hc_finish_kernel | 18 | 2.00 | 12.2 |
| v41_swiglu_kernel | 18 | 1.99 | 7.6 |
| v41_rope_kernel | 31 | 3.43 | 7.5 |
| v41_moe_scan_kernel | 9 | 1.00 | 7.1 |
| v41_hc_post8_kernel | 18 | 2.00 | 6.5 |
| v41_moe_combine_kernel | 9 | 1.00 | 5.5 |
| v41_moe_place_kernel | 9 | 1.00 | 5.4 |
| v41_index_score_kernel | 2 | 0.22 | 4.0 |
| v41_hc_pre8_kernel | 19 | 2.10 | 3.3 |
| v41_attn_idx_kernel | 9 | 1.00 | 2.5 |
| v41_fp4_qdq_kernel | 4 | 0.43 | 2.4 |
| v41_engram_gate_kernel | 1 | 0.11 | 2.0 |
| v41_moe_count_kernel | 9 | 1.00 | 1.9 |
| v41_comp_pool2_kernel | 1 | 0.11 | 0.5 |
| v41_engram_dequant_kernel | 1 | 0.11 | 0.1 |
