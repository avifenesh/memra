4 rank(s), 6 calls x 40 layers: kernel own time 351.3 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 129.9 | 12.40 |
| 2 dense FP8 | 46.5 | 10.05 |
| 3 bf16 | 16.7 | 1.32 |
| 4 attention incl. keys | 14.8 | 6.33 |
| 5 mHC | 33.4 | 8.03 |
| 6 norms | 18.3 | 4.22 |
| 7 collectives | 58.4 | 4.00 |
| 8 head | 26.4 | 1.10 |
| 9 other | 6.9 | 0.20 |
| sum | 351.3 | 47.65 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | swiglu_f4s_tiles | 36x24x1 | 1.00 | 53.85 |
| 1 | main | gemv_f4s_tiles | 80x24x1 | 1.00 | 23.48 |
| 1 | side | gemv_f8 | 72x1x1 | 2.00 | 17.90 |
| 1 | side | gemv_f8 | 64x1x1 | 1.00 | 8.45 |
| 1 | main | route | 1x1x1 | 1.00 | 8.00 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 6.94 |
| 1 | main | moe_combine | 4x2x1 | 1.00 | 4.02 |
| 1 | side | act_quant | 80x1x1 | 2.00 | 3.27 |
| 1 | side | swiglu | 4x1x1 | 1.00 | 1.42 |
| 1 | side | gemv_f8 | 512x1x1 | 0.20 | 1.27 |
| 1 | side | act_quant | 8x1x1 | 1.00 | 1.06 |
| 1 | side | act_quant | 20x1x1 | 0.20 | 0.22 |
| 2 | main | gemv_f8 | 160x1x1 | 3.00 | 29.31 |
| 2 | main | gemv_f8 | 1024x1x1 | 1.00 | 10.11 |
| 2 | main | act_quant | 128x1x1 | 2.00 | 2.18 |
| 2 | main | act_quant | 480x1x1 | 1.00 | 1.47 |
| 2 | main | act_quant | 80x1x1 | 1.00 | 1.14 |
| 2 | main | act_quant | 252x1x1 | 1.00 | 1.14 |
| 2 | main | act_quant | 20x1x1 | 1.00 | 1.07 |
| 2 | main | act_quant | 96x1x1 | 0.05 | 0.06 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 14.69 |
| 3 | side | gemv_bf16 | 4x1x1 | 0.20 | 1.51 |
| 3 | side | gemv_bf16 | 16x1x1 | 0.10 | 0.27 |
| 3 | side | gemv_bf16 | 64x1x1 | 0.03 | 0.19 |
| 4 | main | sparse_attn | 4x1x1 | 1.00 | 6.11 |
| 4 | main | rope | 4x1x1 | 2.00 | 2.35 |
| 4 | side | fp4_qdq | 4x1x1 | 0.20 | 1.44 |
| 4 | side | index_score | 1x1x1 | 0.20 | 1.14 |
| 4 | side | rope | 4x1x1 | 1.25 | 1.11 |
| 4 | main | attn_idx | 4x1x1 | 1.00 | 0.90 |
| 4 | side | fp4_qdq | 1x1x1 | 0.20 | 0.79 |
| 4 | side | topk | 4x1x1 | 0.20 | 0.46 |
| 4 | side | comp_pool2 | 2x1x1 | 0.07 | 0.17 |
| 4 | side | rope | 2x1x1 | 0.15 | 0.11 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 4x8x7 | 2.00 | 17.05 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 9.70 |
| 5 | main | hc_post8 | 4x3x1 | 2.00 | 4.40 |
| 5 | main | hc_pre8 | 4x3x1 | 2.00 | 2.26 |
| 6 | main | rmsnorm32 | 4x1x1 | 3.00 | 14.80 |
| 6 | side | rmsnorm32 | 4x1x1 | 1.05 | 3.03 |
| 6 | side | rmsnorm32 | 2x1x1 | 0.15 | 0.40 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.12 |
| 7 | main | gather | 1x1x1 | 4.00 | 58.39 |
| 8 | main | gemv_ffma_ring | 96x1x1 | 1.00 | 19.41 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.61 |
| 8 | side | gemv_ffma_ring | 256x1x1 | 0.07 | 1.37 |
| 9 | main | gemv_f8 | 3200x1x1 | 0.05 | 5.98 |
| 9 | main | engram_gate | 4x4x1 | 0.05 | 0.75 |
| 9 | main | embed_rows | 4x1x1 | 0.03 | 0.08 |
| 9 | side | candidates | 4x1x1 | 0.03 | 0.07 |
| 9 | main | engram_dequant | 96x1x1 | 0.05 | 0.05 |
