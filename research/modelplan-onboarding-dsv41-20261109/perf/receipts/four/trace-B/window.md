decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 371.622 ms -> 16.157 ms/step, 403.9 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1933 | 741 | 11.439 | 286.0 | 71% |
| 1/28 | 1933 | 741 | 11.691 | 292.3 | 72% |
| 2/40 | 1933 | 741 | 11.547 | 288.7 | 71% |
| 3/52 | 1933 | 741 | 11.455 | 286.4 | 71% |

some stream busy 14.155 ms/step (88% of wall); every stream idle 2.002 ms/step

stream 0/16: idle 4.719 ms/step = 118.0 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | hc_post8 | 2.00 | 27.7 | 13.8 |
| copy | act_quant | 3.13 | 20.4 | 6.5 |
| copy | copy | 10.08 | 15.9 | 1.6 |
| swiglu | copy | 1.00 | 9.4 | 9.4 |
| gemv_bf16 | copy | 1.00 | 9.1 | 9.1 |
| moe_combine | copy | 1.00 | 8.2 | 8.2 |
| gemv_f8 | copy | 1.00 | 7.1 | 7.1 |
| hc_post8 | copy | 0.05 | 2.5 | 50.5 |
| act_quant | gemv_f8 | 6.25 | 2.0 | 0.3 |
| route | memset | 1.00 | 1.5 | 1.5 |
| memset | memset | 2.00 | 1.1 | 0.6 |
| memset | moe_count | 1.00 | 0.7 | 0.7 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.7 | 13.5 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| hc_pre8 | rmsnorm32 | 2.02 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| act_quant | gemv_f8t_tiles | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| attn_idx | sparse_attn | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 62.5 | 8.5 | 53.9 | 1% |
| 1/28 | 57.4 | 4.7 | 52.7 | 1% |
| 2/40 | 61.1 | 9.2 | 51.9 | 1% |
| 3/52 | 63.2 | 7.2 | 56.0 | 1% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 361.4 | 392.8 | 256.9 | 23.8 | 13.1 | 45.6 | 53.7 | 0.0 |
| 1/28 | 361.5 | 392.8 | 262.7 | 24.3 | 13.1 | 40.6 | 52.5 | 0.0 |
| 2/40 | 362.4 | 392.8 | 259.4 | 24.0 | 13.1 | 44.9 | 51.7 | 0.0 |
| 3/52 | 361.5 | 392.8 | 257.5 | 23.6 | 13.3 | 42.9 | 55.9 | 0.0 |

routed-expert tile kernels a layer, us: mean over the ranks 0.0, the busiest rank 0.0, the idlest 0.0, all ranks together 0.0

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 7738 | 193.4 | 18.986 | 2.5 |
| cuStreamWaitEvent | 3840 | 96.0 | 8.961 | 2.3 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 6.477 | 846.4 |
| cuMemcpyDtoDAsync_v2 | 1946 | 48.7 | 5.117 | 2.6 |
| cuStreamSynchronize | 8 | 0.2 | 3.504 | 438.0 |
| cuMemcpy2DAsync_v2 | 495 | 12.4 | 1.111 | 2.2 |
| cuEventRecord | 1288 | 32.2 | 0.849 | 0.7 |
| cuMemsetD8Async | 492 | 12.3 | 0.763 | 1.6 |
| cuKernelGetName | 7738 | 193.4 | 0.739 | 0.1 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.345 | 14.6 |
| cuMemAllocAsync | 32 | 0.8 | 0.098 | 3.1 |
| cuEventSynchronize | 24 | 0.6 | 0.033 | 1.4 |
| cuMemFreeAsync | 8 | 0.2 | 0.015 | 1.9 |

host threads, time inside CUDA calls per step (ms): 12.350, 11.962, 11.654, 11.033

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 242.3 |
| v41_gemv_f8t_tiles_kernel | 479 | 3.00 | 163.0 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.3 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 98.2 |
| v41_rmsnorm32_kernel | 664 | 4.15 | 62.7 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.6 |
| v41_sparse_attn_kernel | 160 | 1.00 | 44.7 |
| v41_act_quant_kernel | 1639 | 10.25 | 44.1 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.7 |
| v41_route_kernel | 160 | 1.00 | 23.5 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.1 |
| v41_moe_scan_kernel | 160 | 1.00 | 20.6 |
| v41_moe_place_kernel | 160 | 1.00 | 16.2 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.6 |
| v41_moe_combine_kernel | 160 | 1.00 | 14.2 |
| v41_rope_kernel | 533 | 3.33 | 11.8 |
| v41_swiglu_kernel | 320 | 2.00 | 10.3 |
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
