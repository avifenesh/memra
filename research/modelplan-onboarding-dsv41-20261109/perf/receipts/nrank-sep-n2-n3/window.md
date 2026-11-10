decode window: 1920 marker launches = 24.00 steps of 2 ranks x 20 layers; measured over 23.00 steps, 225.730 ms -> 9.814 ms/step, 490.7 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 1009 | 200 | 7.657 | 382.9 | 78% |
| 1/24 | 1009 | 200 | 7.863 | 393.1 | 80% |

some stream busy 8.816 ms/step (90% of wall); every stream idle 0.998 ms/step

stream 0/14: idle 2.157 ms/step = 107.9 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| gemv_f8 | copy | 2.00 | 43.1 | 21.6 |
| copy | copy | 2.37 | 13.3 | 5.6 |
| gemv_bf16 | copy | 1.00 | 8.1 | 8.1 |
| copy | moe_owner | 1.00 | 7.9 | 7.9 |
| copy | hc_post8 | 1.00 | 7.6 | 7.6 |
| copy | act_quant | 2.07 | 7.3 | 3.5 |
| act_quant | gemv_f8 | 7.24 | 2.2 | 0.3 |
| engram_gate | hc_mixes_part1 | 0.10 | 1.5 | 15.4 |
| route | memset | 1.00 | 1.5 | 1.5 |
| memset | memset | 2.00 | 1.0 | 0.5 |
| memset | moe_count | 1.00 | 0.7 | 0.7 |
| hc_pre8 | rmsnorm32 | 2.05 | 0.6 | 0.3 |
| swiglu | act_quant | 2.00 | 0.6 | 0.3 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| act_quant | gemv_f8_tiles | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | copy | 0.10 | 0.6 | 5.6 |
| hc_post8 | hc_mixes_part1 | 1.85 | 0.5 | 0.3 |
| copy | hc_mixes_part1 | 0.05 | 0.5 | 9.4 |
| copy | engram_dequant | 0.10 | 0.4 | 3.9 |
| gemv_f8 | rope | 1.15 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.08 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| moe_owner | moe_combine | 1.00 | 0.3 | 0.3 |
| moe_combine | hc_post8 | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 39.3 | 1.7 | 37.5 | 0% |
| 1/24 | 37.8 | 1.9 | 35.9 | 0% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/14 | 431.8 | 456.0 | 348.9 | 12.8 | 2.0 | 55.6 | 37.2 | 76.8 |
| 1/24 | 431.4 | 455.8 | 359.9 | 12.6 | 2.4 | 45.6 | 35.8 | 88.1 |

routed-expert tile kernels a layer, us: mean over the ranks 82.5, the busiest rank 105.4, the idlest 59.5, all ranks together 164.9

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuStreamSynchronize | 4 | 0.2 | 6.631 | 1657.7 |
| cudaLaunchKernel_v7000 | 2021 | 101.0 | 4.486 | 2.2 |
| cuMemcpyDtoHAsync_v2 | 4 | 0.2 | 3.679 | 961.6 |
| cuMemcpyDtoDAsync_v2 | 251 | 12.6 | 0.631 | 2.5 |
| cuStreamWaitEvent | 248 | 12.4 | 0.367 | 1.5 |
| cuMemsetD8Async | 126 | 6.3 | 0.196 | 1.6 |
| cuKernelGetName | 2021 | 101.0 | 0.180 | 0.1 |
| cuEventRecord | 248 | 12.4 | 0.102 | 0.4 |
| cuMemcpyHtoDAsync_v2 | 12 | 0.6 | 0.035 | 2.9 |
| cuMemAllocAsync | 9 | 0.4 | 0.026 | 3.0 |
| cuMemcpy2DAsync_v2 | 8 | 0.4 | 0.023 | 3.0 |
| cuEventSynchronize | 8 | 0.4 | 0.008 | 1.0 |
| cuMemFreeAsync | 4 | 0.2 | 0.007 | 1.7 |

host threads, time inside CUDA calls per step (ms): 8.218, 8.153

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 290 | 7.24 | 195.4 |
| v41_gemv_f8_tiles_kernel | 120 | 2.99 | 164.5 |
| v41_gemv_bf16_kernel | 49 | 1.23 | 75.9 |
| v41_gemv_ffma_narrow_kernel | 52 | 1.30 | 55.6 |
| v41_gemv_ffma_kernel | 2 | 0.05 | 43.9 |
| v41_rmsnorm32_kernel | 168 | 4.20 | 31.5 |
| v41_act_quant_kernel | 450 | 11.24 | 24.6 |
| v41_hc_mixes_part1_kernel | 80 | 2.00 | 24.0 |
| v41_sparse_attn_kernel | 40 | 1.00 | 22.7 |
| v41_hc_finish_kernel | 80 | 2.00 | 17.8 |
| v41_swiglu_kernel | 80 | 2.00 | 16.2 |
| v41_route_kernel | 40 | 1.00 | 13.5 |
| v41_moe_scan_kernel | 40 | 1.00 | 10.6 |
| v41_moe_combine_kernel | 40 | 1.00 | 8.8 |
| v41_moe_place_kernel | 40 | 1.00 | 8.3 |
| v41_hc_post8_kernel | 80 | 2.00 | 8.1 |
| v41_rope_kernel | 132 | 3.31 | 7.9 |
| v41_hc_pre8_kernel | 82 | 2.05 | 4.6 |
| v41_engram_gate_kernel | 4 | 0.10 | 2.9 |
| v41_fp4_qdq_kernel | 12 | 0.31 | 2.8 |
| v41_moe_count_kernel | 40 | 1.00 | 2.6 |
| v41_moe_owner_kernel | 40 | 1.00 | 2.6 |
| v41_attn_idx_kernel | 40 | 1.00 | 2.3 |
| v41_index_score_kernel | 6 | 0.15 | 0.6 |
| v41_comp_pool2_kernel | 3 | 0.08 | 0.6 |
| v41_topk_kernel | 6 | 0.15 | 0.6 |
| v41_engram_dequant_kernel | 4 | 0.10 | 0.2 |
