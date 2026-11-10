decode window: 432 marker launches = 24.00 steps of 1 ranks x 9 layers; measured over 23.00 steps, 173.044 ms -> 7.524 ms/step, 836.0 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 392 | 69 | 7.148 | 794.2 | 95% |
| 0/15 | 54 | 0 | 0.460 | 51.1 | 6% |

some stream busy 7.185 ms/step (96% of wall); every stream idle 0.338 ms/step

stream 0/13: idle 0.376 ms/step = 41.7 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.63 | 17.4 | 6.6 |
| route | memset | 1.00 | 2.3 | 2.3 |
| gemv_f8t_tiles | moe_combine | 1.00 | 1.8 | 1.8 |
| act_quant | gemv_f8 | 4.33 | 1.4 | 0.3 |
| memset | memset | 1.99 | 1.2 | 0.6 |
| memset | moe_count | 1.00 | 1.0 | 1.0 |
| hc_pre8 | rmsnorm32 | 2.10 | 0.8 | 0.4 |
| engram_gate | hc_mixes_part1 | 0.11 | 0.8 | 7.0 |
| hc_finish | hc_pre8 | 2.00 | 0.7 | 0.4 |
| act_quant | gemv_f8t_tiles | 1.99 | 0.7 | 0.4 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.7 | 0.4 |
| rmsnorm32 | act_quant | 2.00 | 0.7 | 0.4 |
| rope | act_quant | 2.00 | 0.7 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.7 | 0.3 |
| copy | engram_dequant | 0.11 | 0.6 | 5.8 |
| hc_post8 | hc_mixes_part1 | 1.77 | 0.6 | 0.4 |
| hc_post8 | copy | 0.11 | 0.4 | 3.9 |
| copy | hc_mixes_part1 | 0.11 | 0.4 | 4.1 |
| gemv_f8 | rope | 1.22 | 0.4 | 0.3 |
| gemv_ffma | copy | 0.11 | 0.4 | 3.8 |
| rmsnorm32 | rope | 1.11 | 0.4 | 0.3 |
| swiglu | act_quant | 1.00 | 0.4 | 0.4 |
| copy | act_quant | 1.12 | 0.4 | 0.3 |
| gemv_f8t_tiles | swiglu | 1.00 | 0.4 | 0.4 |
| moe_combine | hc_post8 | 1.00 | 0.4 | 0.4 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.4 | 0.4 |
| moe_place | act_quant | 1.00 | 0.4 | 0.4 |
| moe_scan | moe_place | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 18.9 | 0.7 | 18.2 | 0% |
| 0/15 | 780.5 | 310.1 | 470.3 | 5% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/13 | 565.3 | 691.0 | 513.3 | 154.3 | 0.8 | 5.2 | 17.9 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 2 | 0.2 | 4.522 | 2363.9 |
| cuStreamSynchronize | 1 | 0.1 | 1.322 | 1321.9 |
| cudaLaunchKernel_v7000 | 447 | 49.6 | 0.921 | 2.1 |
| cuMemcpyDtoDAsync_v2 | 30 | 3.3 | 0.082 | 2.7 |
| cuMemsetD8Async | 30 | 3.3 | 0.042 | 1.4 |
| cuKernelGetName | 447 | 49.6 | 0.036 | 0.1 |
| cuMemcpy2DAsync_v2 | 4 | 0.4 | 0.010 | 2.7 |
| cuMemcpyHtoDAsync_v2 | 4 | 0.4 | 0.008 | 2.1 |
| cuEventRecord | 20 | 2.2 | 0.007 | 0.3 |
| cuStreamWaitEvent | 20 | 2.2 | 0.007 | 0.3 |
| cuMemAllocAsync | 9 | 1.0 | 0.004 | 0.5 |
| cuMemFreeAsync | 9 | 1.0 | 0.004 | 0.4 |
| cuEventSynchronize | 2 | 0.2 | 0.001 | 0.5 |

host threads, time inside CUDA calls per step (ms): 6.967

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 66 | 7.32 | 154.6 |
| v41_gemv_f8t_tiles_kernel | 27 | 2.99 | 135.8 |
| v41_gemv_ffma_kernel | 1 | 0.11 | 116.1 |
| v41_gemv_bf16_kernel | 12 | 1.33 | 55.0 |
| v41_gemv_ffma_narrow_kernel | 13 | 1.44 | 48.5 |
| v41_sparse_attn_kernel | 9 | 1.00 | 36.9 |
| v41_topk_kernel | 2 | 0.22 | 22.9 |
| v41_rmsnorm32_kernel | 39 | 4.31 | 19.8 |
| v41_act_quant_kernel | 93 | 10.31 | 16.1 |
| v41_hc_mixes_part1_kernel | 18 | 2.00 | 15.4 |
| v41_route_kernel | 9 | 1.00 | 12.4 |
| v41_hc_finish_kernel | 18 | 2.00 | 11.9 |
| v41_swiglu_kernel | 18 | 1.99 | 7.5 |
| v41_rope_kernel | 31 | 3.43 | 7.4 |
| v41_moe_scan_kernel | 9 | 1.00 | 7.0 |
| v41_hc_post8_kernel | 18 | 2.00 | 6.4 |
| v41_moe_combine_kernel | 9 | 1.00 | 5.4 |
| v41_moe_place_kernel | 9 | 1.00 | 5.3 |
| v41_index_score_kernel | 2 | 0.22 | 3.9 |
| v41_hc_pre8_kernel | 19 | 2.10 | 3.2 |
| v41_attn_idx_kernel | 9 | 1.00 | 2.5 |
| v41_fp4_qdq_kernel | 4 | 0.43 | 2.3 |
| v41_moe_count_kernel | 9 | 1.00 | 1.8 |
| v41_engram_gate_kernel | 1 | 0.11 | 1.7 |
| v41_comp_pool2_kernel | 1 | 0.11 | 0.5 |
| v41_engram_dequant_kernel | 1 | 0.11 | 0.1 |
