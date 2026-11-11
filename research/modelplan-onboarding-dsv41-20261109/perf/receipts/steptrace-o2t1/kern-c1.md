## 1 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
1 device(s), 24 forwards x 9 layers: kernels 637.3 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f8 | 7.3 | 142.7 | 22.4% |
| gemv_f4s_tiles | 3.1 | 137.4 | 21.6% |
| gemv_ffma | 0.1 | 100.6 | 15.8% |
| gemv_bf16 | 1.5 | 56.6 | 8.9% |
| gemv_ffma_narrow | 1.4 | 44.9 | 7.0% |
| rmsnorm32 | 4.7 | 21.2 | 3.3% |
| hc_mixes_part1 | 2.1 | 16.0 | 2.5% |
| act_quant | 10.8 | 15.9 | 2.5% |
| hc_finish | 2.1 | 14.5 | 2.3% |
| sparse_attn | 1.0 | 12.6 | 2.0% |
| gemv_f8_rows | 0.3 | 11.0 | 1.7% |
| moe_plan_small | 1.0 | 9.3 | 1.5% |
| route | 1.0 | 7.6 | 1.2% |
| rope | 3.8 | 7.4 | 1.2% |
| swiglu | 2.1 | 6.8 | 1.1% |
| argmax_rows | 0.1 | 6.0 | 0.9% |
| hc_post8 | 2.1 | 5.7 | 0.9% |
| moe_combine | 1.0 | 4.8 | 0.8% |
| hc_pre8 | 2.2 | 3.1 | 0.5% |
| fp4_qdq | 0.7 | 3.0 | 0.5% |
| put_rows | 1.9 | 2.5 | 0.4% |
| gemv_ffma_narrow_parts | 0.1 | 2.3 | 0.4% |
| engram_gate | 0.1 | 1.8 | 0.3% |
| attn_idx_batch | 1.0 | 1.5 | 0.2% |
| comp_pool2_keys | 0.2 | 1.0 | 0.2% |
| index_score | 0.2 | 0.6 | 0.1% |
| topk | 0.2 | 0.5 | 0.1% |
| engram_dequant | 0.1 | 0.2 | 0.0% |
| attn_idx | 0.0 | 0.0 | 0.0% |
| comp_pool2 | 0.0 | 0.0 | 0.0% |
