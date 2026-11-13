decode window: 1536 marker launches = 24.00 steps of 4 ranks x 8 layers; measured over 23.00 steps, 224.031 ms -> 9.740 ms/step, 1217.6 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 403 | 133 | 3.690 | 461.3 | 38% |
| 0/15 | 403 | 133 | 3.794 | 474.2 | 39% |
| 0/17 | 403 | 133 | 3.741 | 467.6 | 38% |
| 0/19 | 403 | 133 | 3.665 | 458.1 | 38% |

some stream busy 7.018 ms/step (72% of wall); every stream idle 2.723 ms/step

stream 0/13: idle 6.050 ms/step = 756.2 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 8.65 | 126.3 | 14.6 |
| gemv_bf16 | copy | 1.00 | 70.0 | 70.0 |
| copy | act_quant | 2.06 | 50.9 | 24.7 |
| gemv_f8 | copy | 1.99 | 42.9 | 21.5 |
| act_quant | gemv_f8 | 7.23 | 30.4 | 4.2 |
| gemv_ffma | copy | 0.12 | 28.8 | 240.9 |
| moe_place | act_quant | 0.99 | 28.1 | 28.2 |
| moe_combine | hc_post8 | 0.99 | 27.5 | 27.6 |
| rmsnorm32 | act_quant | 2.00 | 26.4 | 13.2 |
| gemv_f8 | engram_gate | 0.12 | 24.7 | 197.3 |
| act_quant | gemv_f8_tiles | 1.99 | 18.4 | 9.2 |
| rmsnorm32 | memset | 0.12 | 18.3 | 153.2 |
| swiglu | act_quant | 1.99 | 17.4 | 8.8 |
| memset | gemv_ffma | 0.12 | 17.0 | 142.1 |
| copy | moe_owner | 0.99 | 16.0 | 16.0 |
| attn_idx | sparse_attn | 1.00 | 15.2 | 15.2 |
| copy | hc_post8 | 1.00 | 14.5 | 14.5 |
| gemv_f8_tiles | swiglu | 0.99 | 13.5 | 13.6 |
| copy | hc_mixes_part1 | 0.12 | 11.9 | 99.1 |
| gemv_f8 | act_quant | 0.99 | 11.8 | 11.8 |
| gemv_f8_tiles | gemv_f8_tiles | 0.99 | 11.6 | 11.6 |
| act_quant | gemv_bf16 | 1.00 | 10.3 | 10.3 |
| moe_scan | moe_place | 0.99 | 9.3 | 9.3 |
| memset | moe_count | 0.99 | 9.0 | 9.0 |
| hc_pre8 | rmsnorm32 | 2.11 | 8.9 | 4.2 |
| engram_gate | hc_mixes_part1 | 0.12 | 8.8 | 70.4 |
| gemv_f8_tiles | act_quant | 0.99 | 8.1 | 8.1 |
| hc_post8 | hc_mixes_part1 | 1.74 | 7.8 | 4.5 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 447.0 | 180.3 | 266.6 | 4% |
| 0/15 | 436.7 | 173.3 | 263.3 | 4% |
| 0/17 | 437.0 | 179.3 | 257.7 | 4% |
| 0/19 | 466.0 | 197.6 | 268.4 | 4% |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 8 | 1.0 | 10.556 | 1379.5 |
| cudaLaunchKernel_v7000 | 1616 | 202.1 | 5.799 | 3.6 |
| cuStreamSynchronize | 4 | 0.5 | 2.482 | 620.5 |
| cuMemcpyDtoDAsync_v2 | 386 | 48.3 | 1.654 | 4.3 |
| cuStreamWaitEvent | 584 | 73.0 | 1.226 | 2.1 |
| cuMemsetD8Async | 108 | 13.5 | 0.407 | 3.8 |
| cuEventRecord | 200 | 25.0 | 0.315 | 1.6 |
| cuMemAllocAsync | 15 | 1.9 | 0.230 | 15.0 |
| cuKernelGetName | 1616 | 202.1 | 0.205 | 0.1 |
| cuMemcpyHtoDAsync_v2 | 16 | 2.0 | 0.093 | 5.9 |
| cuMemcpy2DAsync_v2 | 15 | 1.9 | 0.085 | 5.5 |
| cuEventSynchronize | 8 | 1.0 | 0.025 | 3.2 |
| cuMemFreeAsync | 8 | 1.0 | 0.015 | 2.0 |

host threads, time inside CUDA calls per step (ms): 5.939, 5.852, 5.735, 5.565

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 231 | 7.23 | 516.8 |
| v41_gemv_f8_tiles_kernel | 95 | 2.98 | 220.5 |
| v41_gemv_bf16_kernel | 38 | 1.19 | 167.8 |
| v41_gemv_ffma_narrow_kernel | 40 | 1.24 | 158.9 |
| v41_gemv_ffma_kernel | 4 | 0.12 | 138.6 |
| v41_rmsnorm32_kernel | 136 | 4.24 | 94.0 |
| v41_act_quant_kernel | 359 | 11.22 | 92.0 |
| v41_hc_mixes_part1_kernel | 64 | 1.99 | 59.2 |
| v41_swiglu_kernel | 64 | 1.99 | 55.2 |
| v41_sparse_attn_kernel | 32 | 1.00 | 53.9 |
| v41_hc_finish_kernel | 64 | 1.99 | 43.0 |
| v41_hc_post8_kernel | 64 | 1.99 | 25.7 |
| v41_route_kernel | 32 | 0.99 | 25.4 |
| v41_moe_scan_kernel | 32 | 0.99 | 22.9 |
| v41_moe_combine_kernel | 32 | 0.99 | 20.5 |
| v41_moe_place_kernel | 32 | 0.99 | 18.6 |
| v41_rope_kernel | 104 | 3.26 | 17.8 |
| v41_hc_pre8_kernel | 68 | 2.11 | 11.5 |
| v41_engram_gate_kernel | 4 | 0.12 | 10.8 |
| v41_moe_owner_kernel | 32 | 0.99 | 10.5 |
| v41_moe_count_kernel | 32 | 0.99 | 6.8 |
| v41_attn_idx_kernel | 32 | 1.00 | 6.3 |
| v41_fp4_qdq_kernel | 8 | 0.26 | 5.1 |
| v41_topk_kernel | 4 | 0.12 | 1.5 |
| v41_index_score_kernel | 4 | 0.12 | 1.4 |
| v41_comp_pool2_kernel | 2 | 0.07 | 1.2 |
| v41_engram_dequant_kernel | 4 | 0.12 | 1.2 |
