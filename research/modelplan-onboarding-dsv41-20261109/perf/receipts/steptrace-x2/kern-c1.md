## 1 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
1 device(s), 64 forwards x 9 layers: kernels 627.6 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f8 | 7.3 | 147.0 | 23.4% |
| gemv_f4s_tiles | 3.0 | 121.6 | 19.4% |
| gemv_ffma | 0.1 | 103.9 | 16.5% |
| gemv_bf16 | 1.5 | 55.2 | 8.8% |
| gemv_ffma_narrow | 1.4 | 48.0 | 7.6% |
| rmsnorm32 | 4.6 | 21.9 | 3.5% |
| act_quant | 10.5 | 16.2 | 2.6% |
| hc_mixes_part1 | 2.0 | 15.5 | 2.5% |
| hc_finish | 2.0 | 14.9 | 2.4% |
| sparse_attn | 1.0 | 13.7 | 2.2% |
| moe_plan_small | 1.0 | 9.6 | 1.5% |
| route | 1.0 | 8.5 | 1.4% |
| rope | 3.7 | 7.8 | 1.2% |
| swiglu | 2.0 | 6.9 | 1.1% |
| argmax_rows | 0.1 | 6.3 | 1.0% |
| hc_post8 | 2.0 | 5.8 | 0.9% |
| moe_combine | 1.0 | 4.9 | 0.8% |
| gemv_f8_rows | 0.1 | 4.2 | 0.7% |
| hc_pre8 | 2.1 | 3.3 | 0.5% |
| fp4_qdq | 0.7 | 3.1 | 0.5% |
| put_rows | 1.9 | 2.7 | 0.4% |
| engram_gate | 0.1 | 1.9 | 0.3% |
| attn_idx_batch | 1.0 | 1.5 | 0.2% |
| comp_pool2_keys | 0.2 | 1.0 | 0.2% |
| gemv_ffma_narrow_parts | 0.0 | 0.9 | 0.1% |
| index_score | 0.2 | 0.5 | 0.1% |
| topk | 0.2 | 0.5 | 0.1% |
| engram_dequant | 0.1 | 0.2 | 0.0% |
| attn_idx | 0.0 | 0.0 | 0.0% |
| comp_pool2 | 0.0 | 0.0 | 0.0% |
