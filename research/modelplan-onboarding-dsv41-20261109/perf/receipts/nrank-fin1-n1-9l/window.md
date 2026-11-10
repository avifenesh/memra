decode window: 432 marker launches = 24.00 steps of 1 ranks x 9 layers; measured over 23.00 steps, 108.613 ms -> 4.722 ms/step, 524.7 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 339 | 43 | 4.352 | 483.5 | 92% |
| 0/15 | 90 | 0 | 0.671 | 74.6 | 14% |

some stream busy 4.394 ms/step (93% of wall); every stream idle 0.329 ms/step

stream 0/13: idle 0.370 ms/step = 41.1 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 2.64 | 19.7 | 7.5 |
| moe_combine | hc_post8 | 1.00 | 1.8 | 1.8 |
| gemv_f4s_tiles | moe_combine | 1.00 | 1.7 | 1.7 |
| engram_gate | hc_pre8 | 0.11 | 1.6 | 14.2 |
| act_quant | gemv_f8 | 4.33 | 1.6 | 0.4 |
| gemv_f8 | hc_post8 | 1.00 | 1.4 | 1.4 |
| copy | hc_pre8 | 0.11 | 1.3 | 12.7 |
| moe_plan_small | act_quant | 1.00 | 1.0 | 1.0 |
| hc_pre8 | rmsnorm32 | 2.10 | 0.9 | 0.4 |
| rmsnorm32 | act_quant | 2.00 | 0.6 | 0.3 |
| act_quant | gemv_f4s_tiles | 1.99 | 0.6 | 0.3 |
| rope | act_quant | 2.00 | 0.6 | 0.3 |
| gemv_f8 | rmsnorm32 | 2.00 | 0.6 | 0.3 |
| hc_post8 | hc_pre8 | 1.77 | 0.5 | 0.3 |
| gemv_ffma | copy | 0.11 | 0.4 | 3.8 |
| copy | act_quant | 1.11 | 0.4 | 0.4 |
| hc_post8 | copy | 0.11 | 0.4 | 3.5 |
| rmsnorm32 | rope | 1.12 | 0.3 | 0.3 |
| gemv_f8 | rope | 1.22 | 0.3 | 0.3 |
| route | moe_plan_small | 1.00 | 0.3 | 0.3 |
| memset | copy | 0.11 | 0.3 | 3.0 |
| act_quant | gemv_bf16 | 1.00 | 0.3 | 0.3 |
| rmsnorm32 | gemv_ffma_narrow | 1.00 | 0.3 | 0.3 |
| sparse_attn | rope | 1.00 | 0.3 | 0.3 |
| swiglu | act_quant | 1.00 | 0.3 | 0.3 |
| gemv_f4s_tiles | swiglu | 1.00 | 0.3 | 0.3 |
| gemv_ffma_narrow | route | 1.00 | 0.3 | 0.3 |
| act_quant | copy | 1.00 | 0.3 | 0.3 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 19.5 | 2.4 | 17.1 | 1% |
| 0/15 | 448.8 | 171.0 | 277.8 | 4% |

| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | launched kernel not started | routed-expert tile kernels |
|---|---|---|---|---|---|---|---|---|
| 0/15 | 374.8 | 403.2 | 74.7 | 0.0 | 199.2 | 0.0 | 265.0 | 0.0 |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 2 | 0.2 | 2.709 | 1416.3 |
| cudaLaunchKernel_v7000 | 429 | 47.7 | 0.964 | 2.2 |
| cuStreamSynchronize | 1 | 0.1 | 0.269 | 268.8 |
| cuMemcpyDtoDAsync_v2 | 30 | 3.3 | 0.082 | 2.7 |
| cuKernelGetName | 429 | 47.7 | 0.039 | 0.1 |
| cuStreamWaitEvent | 54 | 6.0 | 0.021 | 0.4 |
| cuEventRecord | 55 | 6.1 | 0.016 | 0.3 |
| cuMemcpy2DAsync_v2 | 4 | 0.4 | 0.011 | 2.9 |
| cuMemcpyHtoDAsync_v2 | 4 | 0.4 | 0.009 | 2.3 |
| cuMemsetD8Async | 3 | 0.3 | 0.008 | 2.6 |
| cuMemAllocAsync | 4 | 0.5 | 0.003 | 0.8 |
| cuEventSynchronize | 3 | 0.3 | 0.001 | 0.5 |
| cuMemFreeAsync | 2 | 0.2 | 0.001 | 0.5 |

host threads, time inside CUDA calls per step (ms): 4.134

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 66 | 7.32 | 142.1 |
| v41_gemv_f4s_tiles_kernel | 27 | 2.99 | 100.8 |
| v41_gemv_ffma_kernel | 1 | 0.11 | 92.3 |
| v41_gemv_bf16_kernel | 12 | 1.34 | 51.2 |
| v41_gemv_ffma_narrow_kernel | 13 | 1.44 | 46.5 |
| v41_rmsnorm32_kernel | 39 | 4.33 | 19.0 |
| v41_hc_mixes_part1_kernel | 18 | 2.00 | 14.1 |
| v41_act_quant_kernel | 93 | 10.31 | 13.7 |
| v41_hc_finish_kernel | 18 | 2.00 | 13.6 |
| v41_sparse_attn_kernel | 9 | 1.00 | 12.1 |
| v41_moe_plan_small_kernel | 9 | 1.00 | 8.9 |
| v41_route_kernel | 9 | 1.00 | 7.1 |
| v41_swiglu_kernel | 18 | 1.99 | 6.2 |
| v41_rope_kernel | 31 | 3.45 | 6.2 |
| v41_hc_post8_kernel | 18 | 2.00 | 5.3 |
| v41_moe_combine_kernel | 9 | 1.00 | 4.4 |
| v41_hc_pre8_kernel | 19 | 2.10 | 2.8 |
| v41_fp4_qdq_kernel | 4 | 0.45 | 2.1 |
| v41_engram_gate_kernel | 1 | 0.11 | 1.7 |
| v41_attn_idx_kernel | 9 | 1.00 | 1.2 |
| v41_index_score_kernel | 2 | 0.22 | 0.5 |
| v41_comp_pool2_kernel | 1 | 0.12 | 0.5 |
| v41_topk_kernel | 2 | 0.22 | 0.5 |
| v41_engram_dequant_kernel | 1 | 0.11 | 0.1 |
