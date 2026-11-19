4 rank(s), 6 calls x 40 layers: kernel own time 510.1 us a layer a rank
| row | us a layer a rank | launches a layer a rank |
|---|---|---|
| 1 MoE incl. router | 225.1 | 10.15 |
| 2 dense FP8 | 79.2 | 13.20 |
| 3 bf16 | 18.3 | 1.27 |
| 4 attention incl. keys | 14.2 | 6.12 |
| 5 mHC | 48.4 | 8.03 |
| 6 norms | 18.9 | 4.22 |
| 7 collectives | 91.2 | 5.20 |
| 8 head | 5.6 | 0.03 |
| 9 other | 9.3 | 0.34 |
| sum | 510.1 | 48.57 |

| row | stream | kernel | grid | launches a layer | us a layer |
|---|---|---|---|---|---|
| 1 | main | swiglu_f4s_tiles | 36x61x1 | 1.00 | 96.39 |
| 1 | main | gemv_f4s_tiles | 80x61x1 | 1.00 | 46.52 |
| 1 | main | gemv_ffma_narrow_parts | 96x1x1 | 1.00 | 30.56 |
| 1 | side | gemv_f8_rows | 72x1x1 | 2.00 | 24.27 |
| 1 | main | moe_plan_small | 1x1x1 | 1.00 | 7.71 |
| 1 | main | route | 3x1x1 | 1.00 | 7.62 |
| 1 | main | gemv_ffma_narrow_parts | 128x1x1 | 0.15 | 4.72 |
| 1 | main | moe_combine | 9x2x1 | 1.00 | 4.60 |
| 1 | side | swiglu | 9x1x1 | 1.00 | 1.47 |
| 1 | side | act_quant | 180x1x1 | 1.00 | 1.26 |
| 2 | main | gemv_f8_rows | 160x1x1 | 3.00 | 44.10 |
| 2 | main | gemv_f8_rows | 1024x1x1 | 1.00 | 13.10 |
| 2 | main | gemv_f8_rows | 64x1x1 | 1.00 | 11.52 |
| 2 | main | act_quant | 180x1x1 | 2.00 | 2.78 |
| 2 | main | act_quant | 288x1x1 | 2.00 | 2.25 |
| 2 | main | act_quant | 1080x1x1 | 1.00 | 1.94 |
| 2 | main | act_quant | 142x1x1 | 1.00 | 1.18 |
| 2 | main | act_quant | 45x1x1 | 1.00 | 1.06 |
| 2 | main | act_quant | 18x1x1 | 1.00 | 1.04 |
| 2 | main | act_quant | 15x1x1 | 0.15 | 0.16 |
| 2 | main | act_quant | 216x1x1 | 0.05 | 0.06 |
| 3 | main | gemv_bf16 | 128x1x2 | 1.00 | 16.76 |
| 3 | main | gemv_bf16 | 4x1x1 | 0.15 | 1.04 |
| 3 | main | gemv_bf16 | 16x1x1 | 0.10 | 0.27 |
| 3 | main | gemv_bf16 | 64x1x1 | 0.03 | 0.22 |
| 4 | main | sparse_attn | 9x1x1 | 1.00 | 6.20 |
| 4 | main | rope | 9x1x1 | 3.05 | 3.29 |
| 4 | main | fp4_qdq | 3x1x1 | 0.17 | 1.16 |
| 4 | main | attn_idx | 9x1x1 | 1.00 | 1.04 |
| 4 | main | index_score | 1x1x1 | 0.15 | 0.73 |
| 4 | main | fp4_qdq | 1x1x1 | 0.17 | 0.72 |
| 4 | main | topk | 3x1x1 | 0.15 | 0.35 |
| 4 | main | rope | 3x1x1 | 0.15 | 0.20 |
| 4 | main | comp_pool2 | 4x1x1 | 0.07 | 0.18 |
| 4 | main | rope | 4x1x1 | 0.15 | 0.11 |
| 4 | main | blk_pack | 64x1x1 | 0.02 | 0.07 |
| 4 | main | blk_pack | 41x1x1 | 0.02 | 0.06 |
| 4 | main | blk_pack | 6x1x1 | 0.02 | 0.06 |
| 5 | side | hc_mixes_part1 | 9x8x7 | 2.00 | 30.82 |
| 5 | side | hc_finish | 1x1x1 | 2.00 | 10.07 |
| 5 | main | hc_post8 | 9x3x1 | 2.00 | 4.93 |
| 5 | main | hc_pre8 | 9x3x1 | 2.00 | 2.54 |
| 6 | main | rmsnorm32 | 9x1x1 | 4.05 | 18.34 |
| 6 | main | rmsnorm32 | 4x1x1 | 0.15 | 0.40 |
| 6 | main | rmsnorm32 | 1x1x1 | 0.03 | 0.12 |
| 7 | main | gather | 1x1x1 | 5.20 | 91.15 |
| 8 | main | gemv_ffma | 1010x1x1 | 0.03 | 5.60 |
| 9 | main | gemv_f8_rows | 3200x1x1 | 0.05 | 7.45 |
| 9 | main | gemv_f8 | 512x1x1 | 0.15 | 0.96 |
| 9 | main | engram_gate | 9x4x1 | 0.05 | 0.75 |
| 9 | main | embed_rows | 9x1x1 | 0.03 | 0.08 |
| 9 | main | candidates | 3x1x1 | 0.02 | 0.06 |
| 9 | main | engram_dequant | 216x1x1 | 0.05 | 0.05 |
