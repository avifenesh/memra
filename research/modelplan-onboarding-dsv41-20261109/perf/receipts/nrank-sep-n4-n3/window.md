decode window: 3840 marker launches = 24.00 steps of 4 ranks x 20 layers; measured over 23.00 steps, 301.382 ms -> 13.104 ms/step, 655.2 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 1009 | 320 | 7.547 | 377.4 | 58% |
| 0/26 | 1009 | 320 | 7.806 | 390.3 | 60% |
| 1/24 | 1009 | 320 | 7.596 | 379.8 | 58% |
| 1/28 | 1009 | 320 | 7.726 | 386.3 | 59% |

some stream busy 10.973 ms/step (84% of wall); every stream idle 2.131 ms/step

stream 0/14: idle 5.556 ms/step = 277.8 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 8.36 | 58.2 | 7.0 |
| gemv_f8 | copy | 2.00 | 41.4 | 20.7 |
| copy | act_quant | 2.07 | 38.0 | 18.3 |
| gemv_bf16 | copy | 1.00 | 13.1 | 13.1 |
| copy | moe_owner | 1.00 | 11.7 | 11.7 |
| hc_post8 | copy | 0.10 | 11.5 | 114.6 |
| engram_gate | hc_mixes_part1 | 0.10 | 11.3 | 112.7 |
| copy | hc_post8 | 1.00 | 10.7 | 10.7 |
| act_quant | gemv_f8 | 7.24 | 6.7 | 0.9 |
| gemv_f8 | act_quant | 1.00 | 6.0 | 6.0 |
| rope | act_quant | 2.00 | 5.4 | 2.7 |
| copy | engram_dequant | 0.10 | 5.2 | 52.0 |
| gemv_f8_tiles | swiglu | 1.00 | 4.2 | 4.2 |
| memset | moe_count | 1.00 | 4.0 | 4.0 |
| swiglu | act_quant | 2.00 | 3.6 | 1.8 |
| moe_scan | moe_place | 1.00 | 3.5 | 3.5 |
| gemv_f8_tiles | gemv_f8_tiles | 1.00 | 3.4 | 3.4 |
| rmsnorm32 | act_quant | 2.00 | 3.2 | 1.6 |
| gemv_f8 | swiglu | 1.00 | 3.1 | 3.1 |
| act_quant | gemv_f8_tiles | 2.00 | 3.0 | 1.5 |
| sparse_attn | rope | 1.00 | 2.3 | 2.3 |
| memset | gemv_ffma | 0.05 | 2.2 | 45.0 |
| gemv_f8 | engram_gate | 0.10 | 1.9 | 18.8 |
| gemv_ffma | copy | 0.05 | 1.8 | 37.9 |
| moe_place | act_quant | 1.00 | 1.8 | 1.8 |
| route | memset | 1.00 | 1.7 | 1.7 |
| act_quant | gemv_bf16 | 1.00 | 1.7 | 1.7 |
| gemv_f8_tiles | act_quant | 1.00 | 1.6 | 1.6 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 147.1 | 43.8 | 103.3 | 1% |
| 0/26 | 157.6 | 46.4 | 111.3 | 2% |
| 1/24 | 136.9 | 32.1 | 104.8 | 2% |
| 1/28 | 148.1 | 56.5 | 91.6 | 2% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/14 | 523.0 | 621.8 | 345.2 | 20.4 | 61.8 | 93.5 | 102.3 | 44.9 |
| 0/26 | 522.8 | 621.8 | 357.4 | 21.4 | 62.6 | 74.4 | 107.0 | 55.8 |
| 1/24 | 523.9 | 621.7 | 346.8 | 21.3 | 63.5 | 88.5 | 102.6 | 50.3 |
| 1/28 | 523.5 | 621.8 | 353.4 | 20.5 | 63.4 | 96.0 | 89.5 | 52.5 |

routed-expert tile kernels a layer, us: mean over the ranks 50.9, the busiest rank 85.6, the idlest 18.2, all ranks together 203.6

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 4042 | 202.1 | 11.549 | 2.9 |
| cuStreamSynchronize | 8 | 0.4 | 10.032 | 1254.0 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.4 | 5.733 | 749.2 |
| cuStreamWaitEvent | 1456 | 72.8 | 3.731 | 2.6 |
| cuMemcpyDtoDAsync_v2 | 982 | 49.1 | 3.138 | 3.2 |
| cuMemsetD8Async | 252 | 12.6 | 0.533 | 2.1 |
| cuKernelGetName | 4042 | 202.1 | 0.436 | 0.1 |
| cuEventRecord | 496 | 24.8 | 0.303 | 0.6 |
| cuMemcpyHtoDAsync_v2 | 24 | 1.2 | 0.103 | 4.4 |
| cuMemAllocAsync | 17 | 0.9 | 0.077 | 4.5 |
| cuMemcpy2DAsync_v2 | 15 | 0.8 | 0.060 | 3.9 |
| cuEventSynchronize | 16 | 0.8 | 0.030 | 1.9 |
| cuMemFreeAsync | 8 | 0.4 | 0.013 | 1.7 |

host threads, time inside CUDA calls per step (ms): 9.385, 9.311, 9.193, 7.850

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 579 | 7.24 | 414.6 |
| v41_gemv_f8_tiles_kernel | 239 | 2.99 | 203.0 |
| v41_gemv_ffma_narrow_kernel | 104 | 1.30 | 158.0 |
| v41_gemv_bf16_kernel | 98 | 1.23 | 148.8 |
| v41_rmsnorm32_kernel | 336 | 4.20 | 69.7 |
| v41_act_quant_kernel | 899 | 11.24 | 62.0 |
| v41_hc_mixes_part1_kernel | 160 | 2.00 | 52.7 |
| v41_gemv_ffma_kernel | 4 | 0.05 | 52.3 |
| v41_sparse_attn_kernel | 80 | 1.00 | 47.7 |
| v41_swiglu_kernel | 160 | 2.00 | 39.8 |
| v41_hc_finish_kernel | 160 | 2.00 | 37.1 |
| v41_route_kernel | 80 | 1.00 | 25.4 |
| v41_moe_scan_kernel | 80 | 1.00 | 21.3 |
| v41_moe_place_kernel | 80 | 1.00 | 20.4 |
| v41_hc_post8_kernel | 160 | 2.00 | 18.1 |
| v41_moe_combine_kernel | 80 | 1.00 | 17.0 |
| v41_rope_kernel | 265 | 3.31 | 14.4 |
| v41_hc_pre8_kernel | 164 | 2.05 | 9.8 |
| v41_moe_owner_kernel | 80 | 1.00 | 7.7 |
| v41_engram_gate_kernel | 8 | 0.10 | 7.1 |
| v41_fp4_qdq_kernel | 25 | 0.31 | 6.1 |
| v41_moe_count_kernel | 80 | 1.00 | 5.7 |
| v41_attn_idx_kernel | 80 | 1.00 | 4.9 |
| v41_topk_kernel | 12 | 0.15 | 1.4 |
| v41_index_score_kernel | 12 | 0.15 | 1.4 |
| v41_comp_pool2_kernel | 6 | 0.08 | 1.3 |
| v41_engram_dequant_kernel | 8 | 0.10 | 0.6 |
