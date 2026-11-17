4 rank(s), 6 calls x 40 layers: kernel own time 528.0 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 234.3 | 10.15 |
| 2 dense FP8 | 83.7 | 13.45 |
| 3 bf16 | 19.2 | 1.32 |
| 4 attention incl. keys | 14.8 | 6.33 |
| 5 mHC | 49.1 | 8.03 |
| 6 norms | 18.8 | 4.22 |
| 7 collectives | 93.9 | 5.20 |
| 8 head | 5.6 | 0.03 |
| 9 other | 8.7 | 0.20 |
| sum | 528.0 | 48.93 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | swiglu_f4s_tiles | 36x68x1 | 1.00 | 99.63 |
| 1 | main | gemv_f4s_tiles | 80x68x1 | 1.00 | 48.49 |
| 1 | main | gemv_ffma_narrow_parts | 96x1x1 | 1.00 | 32.51 |
| 1 | side | gemv_f8_rows | 72x1x1 | 2.00 | 26.24 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 7.68 |
| 1 | main | route | 3x1x1 | 1.00 | 7.60 |
| 1 | main | gemv_ffma_narrow_parts | 128x1x1 | 0.15 | 4.73 |
| 1 | main | moe_combine | 10x2x1 | 1.00 | 4.63 |
| 1 | side | swiglu | 10x1x1 | 1.00 | 1.49 |
| 1 | side | act_quant | 200x1x1 | 1.00 | 1.30 |
| 2 | main | gemv_f8_rows | 160x1x1 | 3.00 | 45.17 |
| 2 | main | gemv_f8_rows | 1024x1x1 | 1.00 | 13.97 |
| 2 | main | gemv_f8_rows | 64x1x1 | 1.00 | 12.53 |
| 2 | main | act_quant | 200x1x1 | 2.00 | 2.73 |
| 2 | main | act_quant | 320x1x1 | 2.00 | 2.28 |
| 2 | main | act_quant | 1200x1x1 | 1.00 | 2.15 |
| 2 | main | gemv_f8 | 512x1x1 | 0.20 | 1.27 |
| 2 | main | act_quant | 158x1x1 | 1.00 | 1.19 |
| 2 | main | act_quant | 50x1x1 | 1.00 | 1.06 |
| 2 | main | act_quant | 20x1x1 | 1.00 | 1.05 |
| 2 | main | act_quant | 15x1x1 | 0.15 | 0.16 |
| 2 | main | act_quant | 240x1x1 | 0.05 | 0.06 |
| 2 | main | act_quant | 5x1x1 | 0.05 | 0.05 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 17.23 |
| 3 | main | gemv_bf16 | 4x1x1 | 0.20 | 1.37 |
| 3 | main | gemv_bf16 | 64x1x1 | 0.03 | 0.36 |
| 3 | main | gemv_bf16 | 16x1x1 | 0.10 | 0.27 |
| 4 | main | sparse_attn | 10x1x1 | 1.00 | 6.21 |
| 4 | main | rope | 10x1x1 | 3.05 | 3.29 |
| 4 | main | fp4_qdq | 3x1x1 | 0.17 | 1.16 |
| 4 | main | attn_idx | 10x1x1 | 1.00 | 1.03 |
| 4 | main | index_score | 1x1x1 | 0.20 | 0.89 |
| 4 | main | fp4_qdq | 1x1x1 | 0.15 | 0.77 |
| 4 | main | topk | 3x1x1 | 0.15 | 0.35 |
| 4 | main | fp4_qdq | 2x1x1 | 0.07 | 0.23 |
| 4 | main | rope | 3x1x1 | 0.15 | 0.20 |
| 4 | main | comp_pool2 | 5x1x1 | 0.07 | 0.18 |
| 4 | main | rope | 5x1x1 | 0.15 | 0.12 |
| 4 | main | topk | 1x1x1 | 0.05 | 0.11 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | rope | 1x1x1 | 0.05 | 0.06 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 10x8x7 | 2.00 | 31.27 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 10.25 |
| 5 | main | hc_post8 | 10x3x1 | 2.00 | 4.98 |
| 5 | main | hc_pre8 | 10x3x1 | 2.00 | 2.53 |
| 6 | main | rmsnorm32 | 10x1x1 | 4.05 | 18.22 |
| 6 | main | rmsnorm32 | 5x1x1 | 0.15 | 0.41 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.13 |
| 7 | main | gather | 1x1x1 | 5.20 | 93.86 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.60 |
| 9 | main | gemv_f8_rows | 3200x1x1 | 0.05 | 7.72 |
| 9 | main | engram_gate | 10x4x1 | 0.05 | 0.75 |
| 9 | main | embed_rows | 10x1x1 | 0.03 | 0.08 |
| 9 | main | candidates | 3x1x1 | 0.02 | 0.06 |
| 9 | main | engram_dequant | 240x1x1 | 0.05 | 0.05 |
