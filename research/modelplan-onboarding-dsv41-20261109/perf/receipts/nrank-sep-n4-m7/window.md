decode window: 3840 marker launches = 24.00 steps of 4 ranks x 20 layers; measured over 23.00 steps, 332.958 ms -> 14.476 ms/step, 723.8 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/14 | 960 | 329 | 7.148 | 357.4 | 49% |
| 0/26 | 970 | 308 | 7.610 | 380.5 | 53% |
| 1/24 | 970 | 308 | 7.365 | 368.3 | 51% |
| 1/28 | 970 | 307 | 7.500 | 375.0 | 52% |

some stream busy 11.191 ms/step (77% of wall); every stream idle 3.285 ms/step

stream 0/14: idle 7.328 ms/step = 366.4 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | act_quant | 3.07 | 53.2 | 17.3 |
| gemv_f8 | copy | 2.00 | 52.4 | 26.2 |
| copy | copy | 7.83 | 49.2 | 6.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 22.2 | 22.2 |
| moe_place | copy | 1.00 | 19.4 | 19.5 |
| copy | hc_post8 | 1.00 | 18.8 | 18.8 |
| hc_post8 | copy | 0.10 | 16.0 | 160.0 |
| gemv_bf16 | copy | 1.00 | 13.0 | 13.0 |
| copy | moe_combine | 1.00 | 11.0 | 11.0 |
| act_quant | gemv_f8 | 7.24 | 10.7 | 1.5 |
| engram_gate | hc_mixes_part1 | 0.10 | 10.0 | 100.4 |
| gemv_f8 | act_quant | 1.00 | 8.6 | 8.6 |
| rmsnorm32 | act_quant | 2.00 | 7.7 | 3.8 |
| hc_post8 | hc_mixes_part1 | 1.85 | 7.2 | 3.9 |
| act_quant | gemv_f8_tiles | 1.50 | 6.7 | 4.5 |
| rope | act_quant | 2.00 | 5.6 | 2.8 |
| gemv_f8_tiles | swiglu | 0.75 | 5.5 | 7.3 |
| gemv_f8_tiles | gemv_f8_tiles | 0.75 | 4.8 | 6.3 |
| gemv_f8 | engram_gate | 0.10 | 4.4 | 44.4 |
| gemv_f8 | swiglu | 1.00 | 3.8 | 3.8 |
| gemv_ffma | copy | 0.05 | 2.8 | 58.1 |
| memset | memset | 2.00 | 2.6 | 1.3 |
| copy | engram_dequant | 0.10 | 2.5 | 25.3 |
| act_quant | gemv_bf16 | 1.00 | 2.5 | 2.5 |
| swiglu | act_quant | 1.75 | 2.2 | 1.3 |
| route | memset | 1.00 | 2.0 | 2.0 |
| engram_dequant | act_quant | 0.10 | 1.6 | 15.8 |
| memset | moe_count | 1.00 | 1.6 | 1.6 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/14 | 205.4 | 101.6 | 103.8 | 16% |
| 0/26 | 232.3 | 119.1 | 113.2 | 16% |
| 1/24 | 230.1 | 126.0 | 104.0 | 17% |
| 1/28 | 218.8 | 117.2 | 101.6 | 17% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/14 | 614.6 | 692.7 | 328.4 | 17.4 | 146.5 | 98.1 | 103.7 | 43.5 |
| 0/26 | 615.6 | 692.7 | 351.9 | 17.1 | 141.8 | 71.9 | 111.4 | 56.9 |
| 1/24 | 615.7 | 692.5 | 339.8 | 17.1 | 151.0 | 83.0 | 103.1 | 51.0 |
| 1/28 | 615.2 | 692.7 | 345.0 | 17.3 | 140.8 | 91.1 | 99.7 | 51.9 |

routed-expert tile kernels a layer, us: mean over the ranks 50.8, the busiest rank 87.5, the idlest 13.6, all ranks together 203.3

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cudaLaunchKernel_v7000 | 3870 | 193.5 | 10.862 | 2.8 |
| cuMemAllocAsync | 3526 | 176.3 | 5.126 | 1.5 |
| cuMemFreeAsync | 3525 | 176.3 | 3.579 | 1.0 |
| cuMemcpyDtoDAsync_v2 | 938 | 46.9 | 2.981 | 3.2 |
| cuStreamWaitEvent | 1411 | 70.6 | 2.724 | 1.9 |
| cuStreamSynchronize | 8 | 0.4 | 2.063 | 257.8 |
| cuMemcpyDtoHAsync_v2 | 24 | 1.2 | 1.809 | 76.1 |
| cuMemsetD8Async | 251 | 12.5 | 0.613 | 2.4 |
| cuKernelGetName | 3870 | 193.5 | 0.393 | 0.1 |
| cuEventRecord | 496 | 24.8 | 0.284 | 0.6 |
| cuCtxSetCurrent | 609 | 30.5 | 0.185 | 0.3 |
| cuMemcpyHtoDAsync_v2 | 24 | 1.2 | 0.096 | 4.0 |
| cuMemcpy2DAsync_v2 | 15 | 0.8 | 0.056 | 3.7 |
| cuEventSynchronize | 16 | 0.8 | 0.027 | 1.7 |

host threads, time inside CUDA calls per step (ms): 7.832, 7.776, 7.133, 6.686, 1.373

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 579 | 7.24 | 414.7 |
| v41_gemv_f8_tiles_kernel | 196 | 2.45 | 202.9 |
| v41_gemv_ffma_narrow_kernel | 104 | 1.30 | 159.4 |
| v41_gemv_bf16_kernel | 98 | 1.23 | 149.7 |
| v41_rmsnorm32_kernel | 336 | 4.20 | 66.7 |
| v41_act_quant_kernel | 870 | 10.88 | 59.3 |
| v41_gemv_ffma_kernel | 4 | 0.05 | 52.1 |
| v41_hc_mixes_part1_kernel | 160 | 2.00 | 50.2 |
| v41_sparse_attn_kernel | 80 | 1.00 | 45.5 |
| v41_hc_finish_kernel | 160 | 2.00 | 37.2 |
| v41_swiglu_kernel | 145 | 1.81 | 25.5 |
| v41_route_kernel | 80 | 1.00 | 24.8 |
| v41_moe_scan_kernel | 80 | 1.00 | 20.6 |
| v41_hc_post8_kernel | 160 | 2.00 | 18.0 |
| v41_moe_combine_kernel | 80 | 1.00 | 17.2 |
| v41_moe_place_kernel | 80 | 1.00 | 16.3 |
| v41_rope_kernel | 265 | 3.31 | 14.0 |
| v41_hc_pre8_kernel | 164 | 2.05 | 9.3 |
| v41_engram_gate_kernel | 8 | 0.10 | 7.2 |
| v41_fp4_qdq_kernel | 25 | 0.31 | 6.0 |
| v41_moe_count_kernel | 80 | 1.00 | 5.1 |
| v41_attn_idx_kernel | 80 | 1.00 | 4.4 |
| v41_topk_kernel | 12 | 0.15 | 1.3 |
| v41_comp_pool2_kernel | 6 | 0.08 | 1.3 |
| v41_index_score_kernel | 12 | 0.15 | 1.3 |
| v41_engram_dequant_kernel | 8 | 0.10 | 0.6 |
