## 1 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
2 device(s), 24 forwards x 20 layers: kernels 419.6 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f8 | 7.2 | 89.5 | 21.3% |
| swiglu_f4s_tiles | 1.0 | 53.2 | 12.7% |
| gemv_bf16 | 1.4 | 40.9 | 9.7% |
| gemv_ffma_narrow | 1.3 | 33.3 | 7.9% |
| gather | 4.2 | 30.6 | 7.3% |
| gemv_ffma | 0.1 | 23.7 | 5.6% |
| gemv_f4s_tiles | 1.0 | 23.3 | 5.6% |
| rmsnorm32 | 4.5 | 19.2 | 4.6% |
| hc_mixes_part1 | 2.1 | 14.4 | 3.4% |
| hc_finish | 2.1 | 13.2 | 3.1% |
| act_quant | 9.6 | 12.1 | 2.9% |
| sparse_attn | 1.0 | 12.0 | 2.9% |
| route | 1.0 | 9.8 | 2.3% |
| gemv_f8_rows | 0.3 | 7.3 | 1.7% |
| moe_plan_small | 1.0 | 7.0 | 1.7% |
| rope | 3.6 | 5.2 | 1.2% |
| hc_post8 | 2.1 | 4.3 | 1.0% |
| moe_combine | 1.0 | 4.2 | 1.0% |
| hc_pre8 | 2.1 | 2.5 | 0.6% |
| put_rows | 1.6 | 2.1 | 0.5% |
| fp4_qdq | 0.5 | 2.0 | 0.5% |
| gemv_ffma_narrow_parts | 0.1 | 2.0 | 0.5% |
| swiglu | 1.0 | 2.0 | 0.5% |
| engram_gate | 0.1 | 1.6 | 0.4% |
| attn_idx_batch | 1.0 | 1.5 | 0.4% |
| argmax_rows | 0.1 | 1.3 | 0.3% |
| comp_pool2_keys | 0.1 | 0.7 | 0.2% |
| index_score | 0.2 | 0.4 | 0.1% |
| topk | 0.2 | 0.4 | 0.1% |
| engram_dequant | 0.1 | 0.1 | 0.0% |
| attn_idx | 0.0 | 0.0 | 0.0% |
| blk_pack | 0.0 | 0.0 | 0.0% |
| comp_pool2 | 0.0 | 0.0 | 0.0% |
