| device | stream | kernel | launches a layer | us a layer |
|---|---|---|---|---|
| 0 | 0 | v41_swiglu_f4s_tiles_kernel | 1.04 | 52.9 |
| 0 | 0 | v41_gemv_bf16_kernel grid 32x1x4 b128 | 1.04 | 36.6 |
| 0 | 0 | v41_gather_kernel grid 1x1x1 b1024 | 4.21 | 34.6 |
| 0 | 0 | v41_gemv_f8_kernel grid 320x1x1 b256 | 2.00 | 28.3 |
| 0 | 0 | v41_gemv_ffma_narrow_kernel grid 96x1x1 b128 | 1.00 | 26.3 |
| 0 | 0 | v41_gemv_ffma_kernel grid 2020x1x1 b128 | 0.05 | 23.7 |
| 0 | 0 | v41_gemv_f4s_tiles_kernel | 1.04 | 23.2 |
| 0 | 0 | v41_gemv_f8_kernel grid 2048x1x1 b256 | 1.00 | 15.9 |
| 0 | 0 | v41_rmsnorm32_kernel grid 1x1x1 b256 | 3.05 | 14.6 |
| 0 | 0 | v41_sparse_attn_kernel | 1.04 | 11.8 |
| 0 | 0 | v41_gemv_f8_kernel grid 3200x1x1 b256 | 0.10 | 11.1 |
| 0 | 0 | v41_route_kernel | 1.04 | 9.1 |
| 0 | 0 | v41_gemv_f8_kernel grid 160x1x1 b256 | 1.00 | 8.4 |
| 0 | 0 | v41_moe_plan_small_kernel | 1.04 | 6.8 |
| 0 | 0 | v41_hc_post8_kernel | 2.08 | 4.2 |
| 0 | 0 | v41_moe_combine_kernel | 1.04 | 4.1 |
| 0 | 0 | v41_rope_kernel | 2.14 | 3.7 |
| 0 | 0 | v41_hc_pre8_kernel | 2.14 | 2.4 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 320x1x1 b256 | 0.08 | 2.0 |
| 0 | 0 | v41_engram_gate_kernel | 0.10 | 1.5 |
| 0 | 0 | v41_gemv_ffma_narrow_parts_kernel grid 96x1x1 b128 | 0.04 | 1.5 |
| 0 | 0 | v41_act_quant_kernel grid 120x1x1 b256 | 1.00 | 1.4 |
| 0 | 0 | v41_argmax_rows_kernel | 0.05 | 1.3 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 2048x1x1 b256 | 0.04 | 1.3 |
| 0 | 0 | v41_act_quant_kernel grid 63x1x1 b256 | 1.00 | 1.1 |
| 0 | 0 | v41_act_quant_kernel grid 20x1x1 b256 | 1.00 | 1.1 |
| 0 | 0 | v41_act_quant_kernel grid 32x1x1 b256 | 1.04 | 1.1 |
| 0 | 0 | v41_act_quant_kernel grid 64x1x1 b256 | 1.00 | 1.1 |
| 0 | 0 | v41_act_quant_kernel grid 5x1x1 b256 | 1.00 | 1.1 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 3200x1x1 b256 | 0.00 | 1.0 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 160x1x1 b256 | 0.04 | 0.9 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 64x1x1 b256 | 0.04 | 0.6 |
| 0 | 0 | v41_rmsnorm32_kernel grid 16x1x1 b256 | 0.12 | 0.6 |
| 0 | 0 | v41_gemv_ffma_narrow_parts_kernel grid 128x1x1 b128 | 0.01 | 0.5 |
| 0 | 0 | v41_gemv_bf16_kernel grid 1x1x1 b128 | 0.01 | 0.3 |
| 0 | 0 | v41_engram_dequant_kernel | 0.10 | 0.1 |
| 0 | 0 | v41_rmsnorm32_kernel grid 16x1x1 b128 | 0.04 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 1920x1x1 b256 | 0.04 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 320x1x1 b256 | 0.08 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 24x1x1 b256 | 0.10 | 0.1 |
| 0 | 0 | v41_fp4_qdq_kernel | 0.02 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 504x1x1 b256 | 0.04 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 1024x1x1 b256 | 0.04 | 0.1 |
| 0 | 0 | v41_act_quant_kernel grid 512x1x1 b256 | 0.04 | 0.1 |
| 0 | 0 | v41_gemv_f8_rows_kernel grid 512x1x1 b256 | 0.01 | 0.1 |
| 0 | 0 | v41_index_score_kernel | 0.01 | 0.0 |
| 0 | 0 | v41_attn_idx_kernel | 0.04 | 0.0 |
| 0 | 0 | v41_act_quant_kernel grid 80x1x1 b256 | 0.04 | 0.0 |
| 0 | 0 | v41_rmsnorm32_kernel grid 8x1x1 b128 | 0.01 | 0.0 |
| 0 | 0 | v41_blk_pack_kernel | 0.01 | 0.0 |
| 0 | 0 | v41_gemv_bf16_kernel grid 16x1x1 b32 | 0.01 | 0.0 |
| 0 | 0 | v41_gather_kernel grid 1x1x1 b64 | 0.01 | 0.0 |
| 0 | 0 | v41_topk_kernel | 0.01 | 0.0 |
| 0 | 0 | v41_comp_pool2_kernel | 0.01 | 0.0 |
| 0 | 0 | v41_act_quant_kernel grid 40x1x1 b256 | 0.01 | 0.0 |
| 0 | 0 | v41_act_quant_kernel grid 384x1x1 b256 | 0.00 | 0.0 |
| 0 | 1 | v41_gemv_f8_kernel grid 144x1x1 b256 | 2.00 | 17.4 |
| 0 | 1 | v41_hc_mixes_part1_kernel | 2.08 | 14.2 |
| 0 | 1 | v41_hc_finish_kernel | 2.08 | 12.9 |
| 0 | 1 | v41_gemv_ffma_narrow_kernel grid 128x1x1 b128 | 0.30 | 6.8 |
| 0 | 1 | v41_gemv_f8_kernel grid 64x1x1 b256 | 1.00 | 6.3 |
| 0 | 1 | v41_rmsnorm32_kernel grid 1x1x1 b128 | 1.30 | 3.6 |
| 0 | 1 | v41_act_quant_kernel grid 20x1x1 b256 | 2.00 | 3.4 |
| 0 | 1 | v41_gemv_bf16_kernel grid 4x1x1 b32 | 0.15 | 3.0 |
| 0 | 1 | v41_put_rows_kernel | 1.60 | 2.1 |
| 0 | 1 | v41_swiglu_kernel | 1.04 | 1.9 |
| 0 | 1 | v41_fp4_qdq_kernel | 0.45 | 1.9 |
| 0 | 1 | v41_gemv_f8_rows_kernel grid 144x1x1 b256 | 0.08 | 1.5 |
| 0 | 1 | v41_attn_idx_batch_kernel | 1.00 | 1.4 |
| 0 | 1 | v41_rope_kernel | 1.45 | 1.3 |
| 0 | 1 | v41_act_quant_kernel grid 2x1x1 b256 | 1.00 | 1.0 |
| 0 | 1 | v41_gemv_f8_kernel grid 512x1x1 b256 | 0.15 | 0.8 |
| 0 | 1 | v41_comp_pool2_keys_kernel | 0.15 | 0.6 |
| 0 | 1 | v41_gemv_bf16_kernel grid 16x1x1 b32 | 0.15 | 0.5 |
| 0 | 1 | v41_topk_kernel | 0.15 | 0.3 |
| 0 | 1 | v41_index_score_kernel | 0.15 | 0.3 |
| 0 | 1 | v41_act_quant_kernel grid 5x1x1 b256 | 0.15 | 0.2 |
| 0 | 1 | v41_act_quant_kernel grid 320x1x1 b256 | 0.04 | 0.1 |
