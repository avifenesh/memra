## 1 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
1 device(s), 24 forwards x 9 layers: kernels 722.8 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f8 | 7.3 | 144.8 | 20.0% |
| gemv_bf16 | 1.5 | 139.1 | 19.2% |
| gemv_f4s_tiles | 3.1 | 136.3 | 18.9% |
| gemv_ffma | 0.1 | 100.8 | 13.9% |
| gemv_ffma_narrow | 1.4 | 47.4 | 6.6% |
| rmsnorm32 | 4.7 | 20.9 | 2.9% |
| hc_mixes_part1 | 2.1 | 16.0 | 2.2% |
| act_quant | 10.8 | 15.8 | 2.2% |
| hc_finish | 2.1 | 14.3 | 2.0% |
| sparse_attn | 1.0 | 12.5 | 1.7% |
| gemv_f8_rows | 0.3 | 11.1 | 1.5% |
| moe_plan_small | 1.0 | 9.6 | 1.3% |
| route | 1.0 | 7.4 | 1.0% |
| rope | 3.8 | 7.4 | 1.0% |
| swiglu | 2.1 | 6.7 | 0.9% |
| argmax_rows | 0.1 | 6.0 | 0.8% |
| hc_post8 | 2.1 | 5.6 | 0.8% |
| moe_combine | 1.0 | 4.8 | 0.7% |
| fp4_qdq | 0.7 | 3.0 | 0.4% |
| hc_pre8 | 2.2 | 2.9 | 0.4% |
| put_rows | 1.9 | 2.4 | 0.3% |
| gemv_ffma_narrow_parts | 0.1 | 2.3 | 0.3% |
| engram_gate | 0.1 | 1.8 | 0.3% |
| attn_idx_batch | 1.0 | 1.5 | 0.2% |
| comp_pool2_keys | 0.2 | 1.0 | 0.1% |
| index_score | 0.2 | 0.6 | 0.1% |
| topk | 0.2 | 0.5 | 0.1% |
| engram_dequant | 0.1 | 0.2 | 0.0% |
| attn_idx | 0.0 | 0.0 | 0.0% |
| comp_pool2 | 0.0 | 0.0 | 0.0% |
