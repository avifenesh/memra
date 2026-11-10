## 6 row(s): kernels by name, us a step a layer a rank (the run's prefill included)
1 device(s), 64 forwards x 9 layers: kernels 1470.2 us a layer a device
| kernel | launches a layer | us a layer | share |
|---|---|---|---|
| gemv_f4s_tiles | 3.2 | 604.0 | 41.1% |
| gemv_f8 | 7.3 | 214.4 | 14.6% |
| gemv_ffma | 0.1 | 181.4 | 12.3% |
| gemv_ffma_narrow_parts | 1.8 | 77.8 | 5.3% |
| gemv_bf16 | 1.5 | 75.0 | 5.1% |
| gemm_f8_pipe_tiles | 0.0 | 47.8 | 3.3% |
| gemm_f8_pipe | 0.5 | 32.4 | 2.2% |
| rmsnorm32 | 5.0 | 29.2 | 2.0% |
| hc_mixes_part1 | 2.1 | 23.8 | 1.6% |
| act_quant | 11.3 | 22.3 | 1.5% |
| sparse_attn | 1.1 | 17.7 | 1.2% |
| hc_finish | 2.2 | 14.8 | 1.0% |
| swiglu | 2.2 | 12.7 | 0.9% |
| gemm_bf16 | 0.1 | 12.6 | 0.9% |
| moe_plan_small | 1.1 | 11.6 | 0.8% |
| rope | 4.0 | 10.4 | 0.7% |
| route | 1.1 | 10.3 | 0.7% |
| gemv_f8_rows | 0.2 | 9.8 | 0.7% |
| moe_combine | 1.1 | 8.5 | 0.6% |
| hc_post8 | 2.2 | 7.9 | 0.5% |
| hc_mixes_part | 0.1 | 7.3 | 0.5% |
| argmax_rows | 0.1 | 6.8 | 0.5% |
| gemm_ffma | 0.0 | 6.1 | 0.4% |
| fp4_qdq | 0.7 | 4.5 | 0.3% |
| hc_pre8 | 2.3 | 4.4 | 0.3% |
| put_rows | 1.9 | 3.5 | 0.2% |
| index_score | 1.4 | 3.1 | 0.2% |
| topk | 1.4 | 2.6 | 0.2% |
| engram_gate | 0.1 | 2.3 | 0.2% |
| attn_idx_batch | 1.0 | 2.0 | 0.1% |
| gemv_ffma_narrow | 0.1 | 1.4 | 0.1% |
| comp_pool2_keys | 0.2 | 1.2 | 0.1% |
| engram_dequant | 0.1 | 0.2 | 0.0% |
| attn_idx | 0.1 | 0.1 | 0.0% |
| moe_scan | 0.0 | 0.1 | 0.0% |
| moe_place | 0.0 | 0.1 | 0.0% |
| comp_pool2 | 0.0 | 0.1 | 0.0% |
| moe_count | 0.0 | 0.0 | 0.0% |
