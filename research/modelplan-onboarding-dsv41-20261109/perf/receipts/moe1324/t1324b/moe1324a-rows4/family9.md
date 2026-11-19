4 rank(s), 6 calls x 40 layers: kernel own time 324.0 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 23.0 | 6.20 |
| 2 dense FP8 | 8.8 | 7.05 |
| 3 bf16 | 16.8 | 1.32 |
| 4 attention incl. keys | 16.4 | 6.33 |
| 5 mHC | 34.1 | 8.03 |
| 6 norms | 20.3 | 4.22 |
| 7 collectives | 54.8 | 4.00 |
| 8 head | 24.5 | 1.10 |
| 9 other | 125.3 | 7.40 |
| sum | 324.0 | 45.65 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | route | 1x1x1 | 1.00 | 7.63 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 6.93 |
| 1 | main | moe_combine | 4x2x1 | 1.00 | 3.95 |
| 1 | side | swiglu | 4x1x1 | 1.00 | 1.53 |
| 1 | side | act_quant_sf | 80x1x1 | 1.00 | 1.39 |
| 1 | side | act_quant | 8x1x1 | 1.00 | 1.35 |
| 1 | side | act_quant_sf | 20x1x1 | 0.20 | 0.23 |
| 2 | main | act_quant_sf | 480x1x1 | 1.00 | 1.51 |
| 2 | main | swiglu | 24x1x1 | 1.00 | 1.43 |
| 2 | main | act_quant_sf | 252x1x1 | 1.00 | 1.21 |
| 2 | main | act_quant_sf | 128x1x1 | 1.00 | 1.17 |
| 2 | main | act_quant | 128x1x1 | 1.00 | 1.16 |
| 2 | main | act_quant_sf | 80x1x1 | 1.00 | 1.15 |
| 2 | main | act_quant_sf | 20x1x1 | 1.00 | 1.12 |
| 2 | main | act_quant_sf | 96x1x1 | 0.05 | 0.06 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 14.85 |
| 3 | side | gemv_bf16 | 4x1x1 | 0.20 | 1.45 |
| 3 | side | gemv_bf16 | 16x1x1 | 0.10 | 0.26 |
| 3 | side | gemv_bf16 | 64x1x1 | 0.03 | 0.19 |
| 4 | main | sparse_attn | 4x1x1 | 1.00 | 6.13 |
| 4 | side | rope | 4x1x1 | 1.25 | 2.92 |
| 4 | main | rope | 4x1x1 | 2.00 | 2.12 |
| 4 | side | fp4_qdq | 4x1x1 | 0.20 | 1.43 |
| 4 | side | index_score | 1x1x1 | 0.20 | 1.14 |
| 4 | main | attn_idx | 4x1x1 | 1.00 | 0.98 |
| 4 | side | fp4_qdq | 1x1x1 | 0.20 | 0.79 |
| 4 | side | topk | 4x1x1 | 0.20 | 0.47 |
| 4 | side | comp_pool2 | 2x1x1 | 0.07 | 0.17 |
| 4 | side | rope | 2x1x1 | 0.15 | 0.12 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 4x8x7 | 2.00 | 16.37 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 11.12 |
| 5 | main | hc_post8 | 4x3x1 | 2.00 | 4.33 |
| 5 | main | hc_pre8 | 4x3x1 | 2.00 | 2.25 |
| 6 | main | rmsnorm32 | 4x1x1 | 3.00 | 14.64 |
| 6 | side | rmsnorm32 | 4x1x1 | 1.05 | 5.11 |
| 6 | side | rmsnorm32 | 2x1x1 | 0.15 | 0.40 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.12 |
| 7 | main | gather | 1x1x1 | 4.00 | 54.84 |
| 8 | main | gemv_ffma_ring | 96x1x1 | 1.00 | 17.48 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.61 |
| 8 | side | gemv_ffma_ring | 256x1x1 | 0.07 | 1.36 |
| 9 | main | moe_small_blk | 9x24x2 | 1.00 | 41.46 |
| 9 | main | dense_small | 160x1x1 | 2.00 | 21.28 |
| 9 | main | moe_small_blk | 20x24x1 | 1.00 | 20.74 |
| 9 | side | dense_small | 144x1x1 | 1.00 | 12.81 |
| 9 | main | dense_small | 150x1x1 | 1.00 | 11.72 |
| 9 | main | device | 1x64x1 | 1.00 | 9.59 |
| 9 | main | device | 1x200x1 | 0.05 | 5.65 |
| 9 | side | device | 1x32x1 | 0.20 | 1.16 |
| 9 | main | engram_gate | 4x4x1 | 0.05 | 0.70 |
| 9 | main | embed_rows | 4x1x1 | 0.03 | 0.08 |
| 9 | side | candidates | 4x1x1 | 0.03 | 0.06 |
