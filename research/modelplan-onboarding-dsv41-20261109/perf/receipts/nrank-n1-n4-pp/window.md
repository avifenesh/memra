decode window: 1536 marker launches = 24.00 steps of 4 ranks x 8 layers; measured over 23.00 steps, 173.727 ms -> 7.553 ms/step, 944.2 us/layer

| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |
|---|---|---|---|---|---|
| 0/13 | 386 | 136 | 3.576 | 447.0 | 47% |
| 0/15 | 390 | 127 | 3.670 | 458.8 | 49% |
| 0/17 | 388 | 128 | 3.662 | 457.7 | 48% |
| 0/19 | 384 | 128 | 3.516 | 439.5 | 47% |

some stream busy 6.949 ms/step (92% of wall); every stream idle 0.604 ms/step

stream 0/13: idle 3.977 ms/step = 497.2 us/layer, by the events around the gap

| after | before | gaps/layer | idle us/layer | us/gap |
|---|---|---|---|---|
| copy | copy | 8.05 | 111.3 | 13.8 |
| act_quant | gemv_f8 | 7.23 | 44.4 | 6.1 |
| gemv_f8 | copy | 1.99 | 35.4 | 17.7 |
| gemv_bf16 | copy | 1.00 | 33.8 | 33.8 |
| gemv_f8_tiles | gemv_f8_tiles | 0.80 | 24.8 | 31.1 |
| act_quant | gemv_f8_tiles | 1.60 | 21.4 | 13.4 |
| gemv_f8_tiles | swiglu | 0.80 | 18.6 | 23.3 |
| gemv_f8 | engram_gate | 0.12 | 18.5 | 148.3 |
| copy | act_quant | 3.05 | 18.5 | 6.1 |
| gemv_f8 | act_quant | 0.99 | 17.4 | 17.5 |
| gemv_ffma | copy | 0.12 | 17.0 | 142.1 |
| gemv_f8_tiles | act_quant | 0.80 | 9.2 | 11.6 |
| act_quant | gemv_bf16 | 1.00 | 8.6 | 8.6 |
| swiglu | act_quant | 1.79 | 8.4 | 4.7 |
| memset | gemv_ffma | 0.12 | 8.1 | 68.0 |
| hc_post8 | copy | 0.12 | 7.6 | 60.5 |
| rmsnorm32 | memset | 0.12 | 7.3 | 61.2 |
| gemv_f8 | swiglu | 0.99 | 7.0 | 7.0 |
| copy | hc_post8 | 1.00 | 6.6 | 6.6 |
| copy | moe_combine | 0.99 | 6.5 | 6.6 |
| rmsnorm32 | gemv_ffma_narrow | 0.99 | 6.3 | 6.3 |
| sparse_attn | rope | 1.00 | 6.0 | 6.0 |
| copy | engram_dequant | 0.12 | 4.6 | 36.7 |
| gemv_ffma_narrow | route | 0.99 | 4.1 | 4.1 |
| rope | act_quant | 2.00 | 4.0 | 2.0 |
| hc_pre8 | rmsnorm32 | 2.11 | 3.9 | 1.8 |
| moe_place | copy | 0.99 | 3.7 | 3.7 |
| engram_gate | hc_mixes_part1 | 0.12 | 3.6 | 28.5 |

| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |
|---|---|---|---|---|
| 0/13 | 273.3 | 17.7 | 255.6 | 6% |
| 0/15 | 288.8 | 24.2 | 264.6 | 5% |
| 0/17 | 305.0 | 24.4 | 280.6 | 5% |
| 0/19 | 310.5 | 27.0 | 283.4 | 5% |

| host CUDA call | calls/step | calls/layer | ms/step | us/call |
|---|---|---|---|---|
| cuMemcpyDtoHAsync_v2 | 12 | 1.5 | 6.083 | 516.2 |
| cudaLaunchKernel_v7000 | 1549 | 193.7 | 5.556 | 3.6 |
| cuStreamSynchronize | 4 | 0.5 | 2.148 | 537.1 |
| cuMemcpyDtoDAsync_v2 | 370 | 46.2 | 1.604 | 4.3 |
| cuStreamWaitEvent | 567 | 70.9 | 0.635 | 1.1 |
| cuMemsetD8Async | 107 | 13.4 | 0.333 | 3.1 |
| cuKernelGetName | 1549 | 193.7 | 0.193 | 0.1 |
| cuEventRecord | 200 | 25.0 | 0.168 | 0.8 |
| cuMemAllocAsync | 15 | 1.9 | 0.091 | 5.9 |
| cuMemcpyHtoDAsync_v2 | 16 | 2.0 | 0.081 | 5.2 |
| cuMemcpy2DAsync_v2 | 15 | 1.9 | 0.071 | 4.6 |
| cuEventSynchronize | 8 | 1.0 | 0.024 | 3.0 |
| cuMemFreeAsync | 8 | 1.0 | 0.020 | 2.6 |

host threads, time inside CUDA calls per step (ms): 5.828, 3.771, 3.769, 3.639

| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |
|---|---|---|---|
| v41_gemv_f8_kernel | 231 | 7.23 | 513.7 |
| v41_gemv_f8_tiles_kernel | 79 | 2.48 | 218.1 |
| v41_gemv_bf16_kernel | 38 | 1.19 | 165.8 |
| v41_gemv_ffma_narrow_kernel | 40 | 1.24 | 158.1 |
| v41_gemv_ffma_kernel | 4 | 0.12 | 136.2 |
| v41_rmsnorm32_kernel | 136 | 4.24 | 92.9 |
| v41_act_quant_kernel | 348 | 10.89 | 88.6 |
| v41_hc_mixes_part1_kernel | 64 | 2.00 | 58.9 |
| v41_sparse_attn_kernel | 32 | 1.00 | 54.3 |
| v41_hc_finish_kernel | 64 | 1.99 | 42.7 |
| v41_swiglu_kernel | 58 | 1.82 | 34.3 |
| v41_hc_post8_kernel | 64 | 1.99 | 26.0 |
| v41_route_kernel | 32 | 0.99 | 25.3 |
| v41_moe_scan_kernel | 32 | 0.99 | 21.1 |
| v41_moe_combine_kernel | 32 | 0.99 | 20.3 |
| v41_rope_kernel | 104 | 3.26 | 17.6 |
| v41_moe_place_kernel | 32 | 0.99 | 16.0 |
| v41_hc_pre8_kernel | 68 | 2.11 | 11.7 |
| v41_engram_gate_kernel | 4 | 0.12 | 10.4 |
| v41_moe_count_kernel | 32 | 0.99 | 5.9 |
| v41_attn_idx_kernel | 32 | 1.00 | 5.9 |
| v41_fp4_qdq_kernel | 8 | 0.26 | 5.1 |
| v41_topk_kernel | 4 | 0.12 | 1.5 |
| v41_index_score_kernel | 4 | 0.12 | 1.5 |
| v41_comp_pool2_kernel | 2 | 0.07 | 1.3 |
| v41_engram_dequant_kernel | 4 | 0.12 | 1.1 |
