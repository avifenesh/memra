## 1 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
2 device(s), 24 forwards x 20 layers: kernels 444.5 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f8 | 7.2 | 89.4 | 20.1% |
| gemv_bf16 | 1.4 | 67.5 | 15.2% |
| swiglu_f4s_tiles | 1.0 | 53.4 | 12.0% |
| gemv_ffma_narrow | 1.3 | 33.0 | 7.4% |
| gather | 4.2 | 29.4 | 6.6% |
| gemv_ffma | 0.1 | 23.7 | 5.3% |
| gemv_f4s_tiles | 1.0 | 23.2 | 5.2% |
| rmsnorm32 | 4.5 | 19.2 | 4.3% |
| hc_mixes_part1 | 2.1 | 14.4 | 3.2% |
| hc_finish | 2.1 | 13.2 | 3.0% |
| act_quant | 9.6 | 12.3 | 2.8% |
| sparse_attn | 1.0 | 12.0 | 2.7% |
| route | 1.0 | 9.2 | 2.1% |
| gemv_f8_rows | 0.3 | 7.4 | 1.7% |
| moe_plan_small | 1.0 | 6.9 | 1.6% |
| rope | 3.6 | 5.2 | 1.2% |
| hc_post8 | 2.1 | 4.3 | 1.0% |
| moe_combine | 1.0 | 4.3 | 1.0% |
| hc_pre8 | 2.1 | 2.5 | 0.6% |
| put_rows | 1.6 | 2.1 | 0.5% |
| fp4_qdq | 0.5 | 2.0 | 0.5% |
| gemv_ffma_narrow_parts | 0.1 | 2.0 | 0.4% |
| swiglu | 1.0 | 1.9 | 0.4% |
| engram_gate | 0.1 | 1.6 | 0.4% |
| attn_idx_batch | 1.0 | 1.5 | 0.3% |
| argmax_rows | 0.1 | 1.3 | 0.3% |
| comp_pool2_keys | 0.1 | 0.7 | 0.1% |
| index_score | 0.2 | 0.4 | 0.1% |
| topk | 0.2 | 0.4 | 0.1% |
| engram_dequant | 0.1 | 0.1 | 0.0% |
| attn_idx | 0.0 | 0.0 | 0.0% |
| blk_pack | 0.0 | 0.0 | 0.0% |
| comp_pool2 | 0.0 | 0.0 | 0.0% |
