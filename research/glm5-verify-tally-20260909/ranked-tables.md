## t=2, N=11 verify rounds

| Kernel | Launches/round | us/round | Mean us | Weight MB/round | Effective GB/s | % of 8 TB/s |
|---|---:|---:|---:|---:|---:|---:|
| `memra_mla_attn_gathered_kernel` | 11.000 | 7636.399 | 694.218 | n/a | n/a | n/a |
| `moe_gate_up_preclamp8_q8_rows_ilp` | 42.000 | 2525.681 | 60.135 | 6341.788 | 2510.9 | 31.39 |
| `qmatvec_e4m3_mmvq_b2` | 204.000 | 2273.201 | 11.143 | 3467.117 | 1525.2 | 19.07 |
| `moe_down8_fma_q8_rows_ilp` | 42.000 | 2032.267 | 48.387 | 3170.894 | 1560.3 | 19.50 |
| `qmatvec_q8_0_mmvq_b2` | 157.000 | 1507.265 | 9.600 | 2537.947 | 1683.8 | 21.05 |
| `quantize_q8_1` | 590.000 | 1102.687 | 1.869 | n/a | n/a | n/a |
| `qmatvec_nvfp4_mmvq_b2_rp` | 126.000 | 861.651 | 6.838 | n/a | n/a | n/a |
| `dsv4_hc_pre_v4_e16_kernel` | 90.000 | 849.558 | 9.440 | n/a | n/a | n/a |
| `hc_mixes_gemv_f32` | 180.000 | 768.030 | 4.267 | 283.116 | 368.6 | 4.61 |
| `scale_f32` | 330.000 | 539.582 | 1.635 | n/a | n/a | n/a |
| `memra_mla_decompress_v_wp_kernel` | 11.000 | 379.678 | 34.516 | n/a | n/a | n/a |
| `rms_norm_f32_v2` | 113.000 | 359.785 | 3.184 | n/a | n/a | n/a |
| `memra_mla_kpool_score_dsa_kernel` | 11.000 | 352.731 | 32.066 | n/a | n/a | n/a |
| `memra_mla_kpool_select_kernel` | 11.000 | 328.984 | 29.908 | n/a | n/a | n/a |
| `moe_router_sigmoid_topk_f32` | 42.000 | 288.608 | 6.872 | n/a | n/a | n/a |

Verify launches/round: 2707.000; trunk launches/layer/round: 60.044.
Kernel sum 24340.174 us/round; GPU span 25442.829 us/round; GPU gaps 1274.463 us/round; NVTX host span 17134.549 us/round.

## t=4, N=11 verify rounds

| Kernel | Launches/round | us/round | Mean us | Weight MB/round | Effective GB/s | % of 8 TB/s |
|---|---:|---:|---:|---:|---:|---:|
| `memra_mla_attn_gathered_dsa_kernel` | 11.000 | 7218.395 | 656.218 | n/a | n/a | n/a |
| `moe_gate_up_preclamp8_q8_rows_ilp` | 42.000 | 4807.482 | 114.464 | 12683.575 | 2638.3 | 32.98 |
| `qmatvec_e4m3_mmvq_b4` | 204.000 | 3811.757 | 18.685 | 3467.117 | 909.6 | 11.37 |
| `moe_down8_fma_q8_rows_ilp` | 42.000 | 3693.864 | 87.949 | 6341.788 | 1716.8 | 21.46 |
| `qmatvec_q8_0_mmvq_b4` | 157.000 | 2012.293 | 12.817 | 2537.947 | 1261.2 | 15.77 |
| `hc_mixes_gemv_f32` | 360.000 | 1511.437 | 4.198 | 566.231 | 374.6 | 4.68 |
| `quantize_q8_1` | 608.000 | 1297.005 | 2.133 | n/a | n/a | n/a |
| `qmatvec_nvfp4_mmvq_b4_rp` | 126.000 | 1075.606 | 8.537 | n/a | n/a | n/a |
| `dsv4_hc_pre_v4_e16_kernel` | 90.000 | 850.084 | 9.445 | n/a | n/a | n/a |
| `scale_f32` | 330.000 | 565.813 | 1.715 | n/a | n/a | n/a |
| `memra_mla_decompress_v_wp_kernel` | 11.000 | 543.692 | 49.427 | n/a | n/a | n/a |
| `qmatvec_nvfp4_mmvq_rp_ilp` | 36.000 | 476.088 | 13.225 | n/a | n/a | n/a |
| `memra_mla_kpool_score_dsa_kernel` | 11.000 | 370.563 | 33.688 | n/a | n/a | n/a |
| `rms_norm_f32_v2` | 113.000 | 354.386 | 3.136 | n/a | n/a | n/a |
| `memra_mla_kpool_select_kernel` | 11.000 | 338.615 | 30.783 | n/a | n/a | n/a |

Verify launches/round: 2933.000; trunk launches/layer/round: 65.067.
Kernel sum 31815.195 us/round; GPU span 32661.631 us/round; GPU gaps 1034.122 us/round; NVTX host span 21112.953 us/round.

## t=7, N=2 verify rounds

| Kernel | Launches/round | us/round | Mean us | Weight MB/round | Effective GB/s | % of 8 TB/s |
|---|---:|---:|---:|---:|---:|---:|
| `memra_mla_attn_gathered_kernel` | 11.000 | 10472.558 | 952.051 | n/a | n/a | n/a |
| `moe_gate_up_preclamp8_q8_rows_ilp` | 42.000 | 8252.845 | 196.496 | 22196.257 | 2689.5 | 33.62 |
| `qmatvec_e4m3_mmvq_b8` | 204.000 | 7871.242 | 38.585 | 3467.117 | 440.5 | 5.51 |
| `moe_down8_fma_q8_rows_ilp` | 42.000 | 6217.352 | 148.032 | 11098.128 | 1785.0 | 22.31 |
| `qmatvec_q8_0_mmvq_b8` | 157.000 | 2813.554 | 17.921 | 2537.947 | 902.0 | 11.28 |
| `hc_mixes_gemv_f32` | 630.000 | 2633.750 | 4.181 | 990.904 | 376.2 | 4.70 |
| `qmatvec_nvfp4_mmvq_b7_rpsc` | 126.000 | 1972.387 | 15.654 | n/a | n/a | n/a |
| `quantize_q8_1` | 635.000 | 1394.817 | 2.197 | n/a | n/a | n/a |
| `memra_mla_decompress_v_wp_kernel` | 11.000 | 1034.017 | 94.002 | n/a | n/a | n/a |
| `dsv4_hc_pre_v4_e16_kernel` | 90.000 | 850.096 | 9.446 | n/a | n/a | n/a |
| `qmatvec_nvfp4_mmvq_rp_ilp` | 63.000 | 818.962 | 12.999 | n/a | n/a | n/a |
| `scale_f32` | 330.000 | 599.328 | 1.816 | n/a | n/a | n/a |
| `memra_mla_absorb_q_wp_kernel` | 11.000 | 546.721 | 49.702 | n/a | n/a | n/a |
| `memra_kda_scan_s128` | 34.000 | 446.930 | 13.145 | n/a | n/a | n/a |
| `memra_mla_kpool_score_dsa_kernel` | 11.000 | 414.481 | 37.680 | n/a | n/a | n/a |

Verify launches/round: 3266.000; trunk launches/layer/round: 72.467.
Kernel sum 49611.703 us/round; GPU span 50458.073 us/round; GPU gaps 1061.202 us/round; NVTX host span 33605.075 us/round.

