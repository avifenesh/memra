4 rank(s), 6 calls x 40 layers: kernel own time 456.5 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 60.0 | 6.15 |
| 2 dense FP8 | 11.8 | 8.25 |
| 3 bf16 | 19.6 | 1.32 |
| 4 attention incl. keys | 29.2 | 10.32 |
| 5 mHC | 50.5 | 8.03 |
| 6 norms | 18.7 | 4.22 |
| 7 collectives | 65.6 | 3.20 |
| 8 head | 5.6 | 0.03 |
| 9 other | 195.4 | 7.40 |
| sum | 456.5 | 48.93 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | gemv_ffma_narrow_parts | 96x1x1 | 1.00 | 32.55 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 7.68 |
| 1 | main | route | 3x1x1 | 1.00 | 7.67 |
| 1 | main | gemv_ffma_narrow_parts | 128x1x1 | 0.15 | 4.83 |
| 1 | main | moe_combine | 10x2x1 | 1.00 | 4.46 |
| 1 | side | swiglu | 10x1x1 | 1.00 | 1.53 |
| 1 | side | act_quant_sf | 200x1x1 | 1.00 | 1.27 |
| 2 | main | act_quant_sf | 1200x1x1 | 1.00 | 2.25 |
| 2 | main | act_quant_sf | 200x1x1 | 1.00 | 1.58 |
| 2 | main | swiglu | 60x1x1 | 1.00 | 1.54 |
| 2 | main | act_quant_sf | 630x1x1 | 1.00 | 1.37 |
| 2 | main | act_quant | 20x1x1 | 1.00 | 1.28 |
| 2 | main | act_quant_sf | 320x1x1 | 1.00 | 1.21 |
| 2 | main | act_quant | 320x1x1 | 1.00 | 1.18 |
| 2 | main | act_quant_sf | 50x1x1 | 1.00 | 1.12 |
| 2 | main | act_quant_sf | 15x1x1 | 0.15 | 0.17 |
| 2 | main | act_quant_sf | 240x1x1 | 0.05 | 0.07 |
| 2 | main | act_quant_sf | 5x1x1 | 0.05 | 0.06 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 17.65 |
| 3 | main | gemv_bf16 | 4x1x1 | 0.20 | 1.37 |
| 3 | main | gemv_bf16 | 64x1x1 | 0.03 | 0.36 |
| 3 | main | gemv_bf16 | 16x1x1 | 0.10 | 0.27 |
| 4 | main | blk_pack | 3x1x1 | 4.00 | 14.32 |
| 4 | main | sparse_attn | 10x1x1 | 1.00 | 6.19 |
| 4 | main | rope | 10x1x1 | 3.05 | 3.33 |
| 4 | main | fp4_qdq | 3x1x1 | 0.17 | 1.16 |
| 4 | main | attn_idx | 10x1x1 | 1.00 | 1.06 |
| 4 | main | index_score | 1x1x1 | 0.20 | 0.90 |
| 4 | main | fp4_qdq | 1x1x1 | 0.15 | 0.77 |
| 4 | main | topk | 3x1x1 | 0.15 | 0.35 |
| 4 | main | fp4_qdq | 2x1x1 | 0.07 | 0.23 |
| 4 | main | rope | 3x1x1 | 0.15 | 0.21 |
| 4 | main | comp_pool2 | 5x1x1 | 0.07 | 0.18 |
| 4 | main | rope | 5x1x1 | 0.15 | 0.11 |
| 4 | main | topk | 1x1x1 | 0.05 | 0.10 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | rope | 1x1x1 | 0.05 | 0.07 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 10x8x7 | 2.00 | 31.37 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 11.57 |
| 5 | main | hc_post8 | 10x3x1 | 2.00 | 5.01 |
| 5 | main | hc_pre8 | 10x3x1 | 2.00 | 2.53 |
| 6 | main | rmsnorm32 | 10x1x1 | 4.05 | 18.20 |
| 6 | main | rmsnorm32 | 5x1x1 | 0.15 | 0.40 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.12 |
| 7 | main | gather | 1x1x1 | 3.20 | 65.59 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.63 |
| 9 | main | moe_small_blk | 9x68x2 | 1.00 | 77.05 |
| 9 | main | moe_small_blk | 20x68x1 | 1.00 | 40.10 |
| 9 | main | dense_small | 160x1x1 | 2.00 | 25.50 |
| 9 | side | dense_small | 144x1x1 | 1.00 | 21.37 |
| 9 | main | dense_small | 150x1x1 | 1.00 | 14.13 |
| 9 | main | device | 1x64x1 | 1.00 | 9.41 |
| 9 | main | device | 1x200x1 | 0.05 | 5.76 |
| 9 | main | device | 1x32x1 | 0.20 | 1.19 |
| 9 | main | engram_gate | 10x4x1 | 0.05 | 0.71 |
| 9 | main | embed_rows | 10x1x1 | 0.03 | 0.08 |
| 9 | main | candidates | 3x1x1 | 0.02 | 0.06 |
| 9 | main | engram_dequant | 240x1x1 | 0.05 | 0.05 |
