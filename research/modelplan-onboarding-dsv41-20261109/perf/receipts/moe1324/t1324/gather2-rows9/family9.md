4 rank(s), 6 calls x 40 layers: kernel own time 510.5 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 58.8 | 6.15 |
| 2 dense FP8 | 11.5 | 8.20 |
| 3 bf16 | 18.7 | 1.27 |
| 4 attention incl. keys | 14.2 | 6.12 |
| 5 mHC | 48.4 | 8.03 |
| 6 norms | 19.9 | 4.22 |
| 7 collectives | 89.0 | 4.20 |
| 8 head | 5.6 | 0.03 |
| 9 other | 244.2 | 7.34 |
| sum | 510.5 | 45.57 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | gemv_ffma_narrow_parts | 96x1x1 | 1.00 | 31.57 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 7.68 |
| 1 | main | route | 3x1x1 | 1.00 | 7.60 |
| 1 | main | gemv_ffma_narrow_parts | 128x1x1 | 0.15 | 4.79 |
| 1 | main | moe_combine | 9x2x1 | 1.00 | 4.38 |
| 1 | side | swiglu | 9x1x1 | 1.00 | 1.51 |
| 1 | side | act_quant_sf | 180x1x1 | 1.00 | 1.30 |
| 2 | main | act_quant_sf | 1080x1x1 | 1.00 | 1.98 |
| 2 | main | act_quant_sf | 180x1x1 | 1.00 | 1.65 |
| 2 | main | swiglu | 54x1x1 | 1.00 | 1.50 |
| 2 | main | act_quant_sf | 567x1x1 | 1.00 | 1.37 |
| 2 | main | act_quant | 18x1x1 | 1.00 | 1.28 |
| 2 | main | act_quant_sf | 288x1x1 | 1.00 | 1.20 |
| 2 | main | act_quant | 288x1x1 | 1.00 | 1.16 |
| 2 | main | act_quant_sf | 45x1x1 | 1.00 | 1.14 |
| 2 | main | act_quant_sf | 15x1x1 | 0.15 | 0.17 |
| 2 | main | act_quant_sf | 216x1x1 | 0.05 | 0.06 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 17.18 |
| 3 | main | gemv_bf16 | 4x1x1 | 0.15 | 1.05 |
| 3 | main | gemv_bf16 | 16x1x1 | 0.10 | 0.27 |
| 3 | main | gemv_bf16 | 64x1x1 | 0.03 | 0.22 |
| 4 | main | sparse_attn | 9x1x1 | 1.00 | 6.20 |
| 4 | main | rope | 9x1x1 | 3.05 | 3.32 |
| 4 | main | fp4_qdq | 3x1x1 | 0.17 | 1.16 |
| 4 | main | attn_idx | 9x1x1 | 1.00 | 1.07 |
| 4 | main | index_score | 1x1x1 | 0.15 | 0.74 |
| 4 | main | fp4_qdq | 1x1x1 | 0.17 | 0.72 |
| 4 | main | topk | 3x1x1 | 0.15 | 0.35 |
| 4 | main | rope | 3x1x1 | 0.15 | 0.21 |
| 4 | main | comp_pool2 | 4x1x1 | 0.07 | 0.18 |
| 4 | main | rope | 4x1x1 | 0.15 | 0.11 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 9x8x7 | 2.00 | 30.55 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 10.42 |
| 5 | main | hc_post8 | 9x3x1 | 2.00 | 4.87 |
| 5 | main | hc_pre8 | 9x3x1 | 2.00 | 2.54 |
| 6 | main | rmsnorm32 | 9x1x1 | 4.05 | 19.38 |
| 6 | main | rmsnorm32 | 4x1x1 | 0.15 | 0.40 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.12 |
| 7 | main | gather | 1x1x1 | 4.20 | 89.05 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.63 |
| 9 | main | moe_small | 9x61x2 | 1.00 | 114.11 |
| 9 | main | moe_small | 20x61x1 | 1.00 | 53.21 |
| 9 | main | dense_small | 160x1x1 | 2.00 | 25.06 |
| 9 | side | dense_small | 144x1x1 | 1.00 | 20.92 |
| 9 | main | dense_small | 150x1x1 | 1.00 | 13.98 |
| 9 | main | device | 1x64x1 | 1.00 | 9.41 |
| 9 | main | device | 1x200x1 | 0.05 | 5.75 |
| 9 | main | device | 1x32x1 | 0.15 | 0.91 |
| 9 | main | engram_gate | 9x4x1 | 0.05 | 0.71 |
| 9 | main | embed_rows | 9x1x1 | 0.03 | 0.08 |
| 9 | main | candidates | 3x1x1 | 0.02 | 0.06 |
| 9 | main | engram_dequant | 216x1x1 | 0.05 | 0.05 |
