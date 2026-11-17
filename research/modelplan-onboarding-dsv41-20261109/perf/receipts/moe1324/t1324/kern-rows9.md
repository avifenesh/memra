# per layer a rank, us (FORWARDS 6, LAYERS 40); reference route2

| arm | ranks | kernel own (sum) | device busy (union) | idle in span |
|---|---|---|---|---|
| route2 | 4 | 510.13 | 452.02 | 63723.98 |
| gather2 | 4 | 510.52 | 452.76 | 69600.81 |
| main25 | 4 | 510.01 | 452.43 | 67356.27 |

## gather2 against route2: kernels by |delta| (us a layer a rank; launches a layer a rank)

| kernel <grid> | route2 us | gather2 us | delta | route2 n | gather2 n |
|---|---|---|---|---|---|
| `v41_moe_small_kernel <9,61,2>` | 0.00 | 114.11 | +114.11 | 0.00 | 1.00 |
| `v41_swiglu_f4s_tiles_kernel <36,61,1>` | 96.39 | 0.00 | -96.39 | 1.00 | 0.00 |
| `v41_moe_small_kernel <20,61,1>` | 0.00 | 53.21 | +53.21 | 0.00 | 1.00 |
| `v41_gemv_f4s_tiles_kernel <80,61,1>` | 46.52 | 0.00 | -46.52 | 1.00 | 0.00 |
| `v41_gemv_f8_rows_kernel <160,1,1>` | 44.10 | 0.00 | -44.10 | 3.00 | 0.00 |
| `v41_dense_small_kernel <160,1,1>` | 0.00 | 25.06 | +25.06 | 0.00 | 2.00 |
| `v41_gemv_f8_rows_kernel <72,1,1>` | 24.27 | 0.00 | -24.27 | 2.00 | 0.00 |
| `v41_dense_small_kernel <144,1,1>` | 0.00 | 20.92 | +20.92 | 0.00 | 1.00 |
| `v41_dense_small_kernel <150,1,1>` | 0.00 | 13.98 | +13.98 | 0.00 | 1.00 |
| `v41_gemv_f8_rows_kernel <1024,1,1>` | 13.10 | 0.00 | -13.10 | 1.00 | 0.00 |
| `v41_gemv_f8_rows_kernel <64,1,1>` | 11.52 | 0.00 | -11.52 | 1.00 | 0.00 |
| `device_kernel <1,64,1>` | 0.00 | 9.41 | +9.41 | 0.00 | 1.00 |
| `v41_gemv_f8_rows_kernel <3200,1,1>` | 7.45 | 0.00 | -7.45 | 0.05 | 0.00 |
| `device_kernel <1,200,1>` | 0.00 | 5.75 | +5.75 | 0.00 | 0.05 |
| `v41_act_quant_kernel <180,1,1>` | 4.03 | 0.00 | -4.03 | 3.00 | 0.00 |
| `v41_act_quant_sf_kernel <180,1,1>` | 0.00 | 2.95 | +2.95 | 0.00 | 2.00 |
| `v41_gather_kernel <1,1,1>` | 91.15 | 89.05 | -2.10 | 5.20 | 4.20 |
| `v41_act_quant_sf_kernel <1080,1,1>` | 0.00 | 1.98 | +1.98 | 0.00 | 1.00 |
| `v41_act_quant_kernel <1080,1,1>` | 1.94 | 0.00 | -1.94 | 1.00 | 0.00 |
| `v41_swiglu_kernel <54,1,1>` | 0.00 | 1.50 | +1.50 | 0.00 | 1.00 |
| `v41_act_quant_sf_kernel <567,1,1>` | 0.00 | 1.37 | +1.37 | 0.00 | 1.00 |
| `v41_act_quant_sf_kernel <288,1,1>` | 0.00 | 1.20 | +1.20 | 0.00 | 1.00 |
| `v41_act_quant_kernel <142,1,1>` | 1.18 | 0.00 | -1.18 | 1.00 | 0.00 |
| `v41_act_quant_sf_kernel <45,1,1>` | 0.00 | 1.14 | +1.14 | 0.00 | 1.00 |
| `v41_act_quant_kernel <288,1,1>` | 2.25 | 1.16 | -1.09 | 2.00 | 1.00 |
| `v41_act_quant_kernel <45,1,1>` | 1.06 | 0.00 | -1.06 | 1.00 | 0.00 |
| `v41_rmsnorm32_kernel <9,1,1>` | 18.34 | 19.38 | +1.04 | 4.05 | 4.05 |
| `v41_gemv_ffma_narrow_parts_kernel <96,1,1>` | 30.56 | 31.57 | +1.01 | 1.00 | 1.00 |
| `v41_gemv_f8_kernel <512,1,1>` | 0.96 | 0.00 | -0.96 | 0.15 | 0.00 |
| `device_kernel <1,32,1>` | 0.00 | 0.91 | +0.91 | 0.00 | 0.15 |

## main25 against route2: kernels by |delta| (us a layer a rank; launches a layer a rank)

| kernel <grid> | route2 us | main25 us | delta | route2 n | main25 n |
|---|---|---|---|---|---|
| `v41_moe_small_kernel <9,61,2>` | 0.00 | 114.09 | +114.09 | 0.00 | 1.00 |
| `v41_swiglu_f4s_tiles_kernel <36,61,1>` | 96.39 | 0.00 | -96.39 | 1.00 | 0.00 |
| `v41_moe_small_kernel <20,61,1>` | 0.00 | 53.13 | +53.13 | 0.00 | 1.00 |
| `v41_gemv_f4s_tiles_kernel <80,61,1>` | 46.52 | 0.00 | -46.52 | 1.00 | 0.00 |
| `v41_gemv_f8_rows_kernel <160,1,1>` | 44.10 | 0.00 | -44.10 | 3.00 | 0.00 |
| `v41_dense_small_kernel <160,1,1>` | 0.00 | 25.02 | +25.02 | 0.00 | 2.00 |
| `v41_gemv_f8_rows_kernel <72,1,1>` | 24.27 | 0.00 | -24.27 | 2.00 | 0.00 |
| `v41_dense_small_kernel <144,1,1>` | 0.00 | 20.89 | +20.89 | 0.00 | 1.00 |
| `v41_dense_small_kernel <150,1,1>` | 0.00 | 13.97 | +13.97 | 0.00 | 1.00 |
| `v41_gemv_f8_rows_kernel <1024,1,1>` | 13.10 | 0.00 | -13.10 | 1.00 | 0.00 |
| `v41_gemv_f8_rows_kernel <64,1,1>` | 11.52 | 0.00 | -11.52 | 1.00 | 0.00 |
| `device_kernel <1,64,1>` | 0.00 | 9.41 | +9.41 | 0.00 | 1.00 |
| `v41_gemv_f8_rows_kernel <3200,1,1>` | 7.45 | 0.00 | -7.45 | 0.05 | 0.00 |
| `device_kernel <1,200,1>` | 0.00 | 5.75 | +5.75 | 0.00 | 0.05 |
| `v41_act_quant_kernel <180,1,1>` | 4.03 | 0.00 | -4.03 | 3.00 | 0.00 |
| `v41_act_quant_sf_kernel <180,1,1>` | 0.00 | 2.93 | +2.93 | 0.00 | 2.00 |
| `v41_gather_kernel <1,1,1>` | 91.15 | 88.84 | -2.31 | 5.20 | 4.20 |
| `v41_act_quant_sf_kernel <1080,1,1>` | 0.00 | 1.97 | +1.97 | 0.00 | 1.00 |
| `v41_act_quant_kernel <1080,1,1>` | 1.94 | 0.00 | -1.94 | 1.00 | 0.00 |
| `v41_swiglu_kernel <54,1,1>` | 0.00 | 1.50 | +1.50 | 0.00 | 1.00 |
| `v41_act_quant_sf_kernel <567,1,1>` | 0.00 | 1.37 | +1.37 | 0.00 | 1.00 |
| `v41_act_quant_sf_kernel <288,1,1>` | 0.00 | 1.20 | +1.20 | 0.00 | 1.00 |
| `v41_act_quant_kernel <142,1,1>` | 1.18 | 0.00 | -1.18 | 1.00 | 0.00 |
| `v41_act_quant_sf_kernel <45,1,1>` | 0.00 | 1.13 | +1.13 | 0.00 | 1.00 |
| `v41_act_quant_kernel <288,1,1>` | 2.25 | 1.16 | -1.09 | 2.00 | 1.00 |
| `v41_act_quant_kernel <45,1,1>` | 1.06 | 0.00 | -1.06 | 1.00 | 0.00 |
| `v41_gemv_ffma_narrow_parts_kernel <96,1,1>` | 30.56 | 31.53 | +0.98 | 1.00 | 1.00 |
| `v41_rmsnorm32_kernel <9,1,1>` | 18.34 | 19.31 | +0.97 | 4.05 | 4.05 |
| `v41_gemv_f8_kernel <512,1,1>` | 0.96 | 0.00 | -0.96 | 0.15 | 0.00 |
| `device_kernel <1,32,1>` | 0.00 | 0.91 | +0.91 | 0.00 | 0.15 |
