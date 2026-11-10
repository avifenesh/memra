decode window: 7680 marker launches = 24.00 steps of 4 ranks x 40 layers; measured over 23.00 steps, 411.191 ms -> 17.878 ms/step, 446.9 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/16 | 1973 | 621 | 12.959 | 324.0 | 72% |
| 1/28 | 1973 | 621 | 13.207 | 330.2 | 74% |
| 2/40 | 1973 | 621 | 13.021 | 325.5 | 73% |
| 3/52 | 1973 | 621 | 12.982 | 324.5 | 73% |

some stream busy 16.364 ms/step (92% of wall); every stream idle 1.514 ms/step

stream 0/16: idle 4.918 ms/step = 123.0 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| gemv_f8 | copy | 2.00 | 34.9 | 17.4 |
| copy | copy | 8.08 | 21.3 | 2.6 |
| copy | moe_owner | 1.00 | 17.7 | 17.7 |
| copy | hc_post8 | 1.00 | 10.7 | 10.7 |
| copy | act_quant | 2.14 | 10.7 | 5.0 |
| gemv_bf16 | copy | 1.00 | 9.5 | 9.5 |
| act_quant | gemv_f8 | 6.25 | 1.9 | 0.3 |
| route | memset | 1.00 | 1.6 | 1.6 |
| memset | memset | 2.00 | 1.2 | 0.6 |
| engram_gate | hc_mixes_part1 | 0.05 | 0.7 | 14.4 |
| memset | moe_count | 1.00 | 0.7 | 0.7 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| hc_pre8 | rmsnorm32 | 2.02 | 0.6 | 0.3 |
| swiglu | act_quant | 2.00 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| act_quant | gemv_f8t_tiles | 2.00 | 0.6 | 0.3 |
| hc_finish | hc_pre8 | 2.00 | 0.6 | 0.3 |
| hc_mixes_part1 | hc_finish | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_mixes_part1 | 1.92 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rope | 1.20 | 0.3 | 0.3 |
| rmsnorm32 | rope | 1.06 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| moe_combine | hc_post8 | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| moe_owner | moe_combine | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/16 | 53.7 | 1.0 | 52.7 | 0% |
| 1/28 | 54.9 | 5.2 | 49.7 | 0% |
| 2/40 | 54.3 | 2.5 | 51.8 | 0% |
| 3/52 | 60.4 | 7.8 | 52.6 | 0% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/16 | 415.7 | 435.9 | 278.6 | 40.2 | 8.0 | 56.6 | 52.7 | 0.0 |
| 1/28 | 415.7 | 435.9 | 281.7 | 43.2 | 8.0 | 53.8 | 49.6 | 0.0 |
| 2/40 | 415.8 | 435.9 | 278.8 | 41.3 | 8.1 | 56.3 | 51.7 | 0.0 |
| 3/52 | 415.5 | 435.9 | 280.1 | 39.5 | 8.2 | 56.1 | 52.4 | 0.0 |

routed-expert tile kernels a layer, us: mean over the ranks 0.0, the busiest rank 0.0, the idlest 0.0, all ranks together 0.0

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 7898 | 197.4 | 19.279 | 2.4 |
| cuMemcpyDtoHAsync_v2 | 8 | 0.2 | 14.234 | 1860.2 |
| cuStreamSynchronize | 8 | 0.2 | 7.819 | 977.4 |
| cuStreamWaitEvent | 2880 | 72.0 | 6.501 | 2.3 |
| cuMemcpyDtoDAsync_v2 | 1946 | 48.7 | 4.558 | 2.3 |
| cuKernelGetName | 7898 | 197.4 | 0.922 | 0.1 |
| cuMemsetD8Async | 492 | 12.3 | 0.736 | 1.5 |
| cuEventRecord | 968 | 24.2 | 0.598 | 0.6 |
| cuMemcpyHtoDAsync_v2 | 24 | 0.6 | 0.072 | 3.1 |
| cuMemAllocAsync | 32 | 0.8 | 0.071 | 2.2 |
| cuMemcpy2DAsync_v2 | 15 | 0.4 | 0.047 | 3.1 |
| cuEventSynchronize | 24 | 0.6 | 0.024 | 1.0 |
| cuMemFreeAsync | 8 | 0.2 | 0.011 | 1.4 |

host threads, time inside CUDA calls per step (ms): 14.029, 13.896, 13.817, 13.130

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 1159 | 7.25 | 311.7 |
| v41_gemv_bf16_kernel | 206 | 1.29 | 152.9 |
| v41_gemv_f8t_tiles_kernel | 479 | 3.00 | 148.6 |
| v41_gemv_ffma_narrow_kernel | 184 | 1.15 | 99.0 |
| v41_rmsnorm32_kernel | 664 | 4.15 | 62.8 |
| v41_hc_mixes_part1_kernel | 320 | 2.00 | 47.7 |
| v41_sparse_attn_kernel | 160 | 1.00 | 45.0 |
| v41_act_quant_kernel | 1639 | 10.25 | 44.2 |
| v41_hc_finish_kernel | 320 | 2.00 | 35.7 |
| v41_swiglu_kernel | 320 | 2.00 | 26.7 |
| v41_route_kernel | 160 | 1.00 | 24.6 |
| v41_gemv_ffma_kernel | 4 | 0.02 | 22.0 |
| v41_moe_scan_kernel | 160 | 1.00 | 20.6 |
| v41_moe_place_kernel | 160 | 1.00 | 16.4 |
| v41_hc_post8_kernel | 320 | 2.00 | 15.8 |
| v41_moe_combine_kernel | 160 | 1.00 | 15.3 |
| v41_rope_kernel | 533 | 3.33 | 12.0 |
| v41_hc_pre8_kernel | 324 | 2.02 | 8.7 |
| v41_fp4_qdq_kernel | 53 | 0.33 | 6.4 |
| v41_moe_owner_kernel | 160 | 1.00 | 6.1 |
| v41_moe_count_kernel | 160 | 1.00 | 5.0 |
| v41_attn_idx_kernel | 160 | 1.00 | 4.1 |
| v41_engram_gate_kernel | 8 | 0.05 | 2.9 |
| v41_index_score_kernel | 32 | 0.20 | 2.2 |
| v41_topk_kernel | 32 | 0.20 | 1.7 |
| v41_comp_pool2_kernel | 6 | 0.04 | 0.6 |
| v41_candidates_kernel | 4 | 0.03 | 0.3 |
| v41_engram_dequant_kernel | 8 | 0.05 | 0.2 |
