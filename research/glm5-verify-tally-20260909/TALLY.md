# GLM verify kernel tally, 2026-09-09

A p32k sampled PP1 DFlash2 request at t4 issues **2933 kernels per verify round**,
**65.067 per trunk layer**. E4M3 input projections cost **3811.757 us/round**
for **3467.117 MB** of logical weight reads. Their 70%-HBM ideal is
**619.128 us**, giving **3192.629 us** of bandwidth screening headroom.
The candidate is six-projection fusion with shared quantization and in-store
scales. This is a target selection, not a measured optimization win.

## Identity and protocol

- Source base `dcfeab7c738912a150ebbfea277112724bb99de4` plus the exact
  `instrumentation.patch` in this directory. The diagnostic adds NVTX ranges
  and profiler start/stop with stream drains only at burst boundaries.
- Measured binary SHA256
  `368e8bbbf41323745470bd1d28ba261bd446409b7b88df69661fe6f595079630`.
  One B200, sm_100a, CUDA 13.1.115, Nsight Systems 2025.4.1.0. All GPU phases
  hold the common lock. No cargo, gate, bench or smoke server ran on the rig.
- Native GLM-5.3-Flash mint, DFlash2 armed, PMIN=0.7, vendor defaults
  temperature=1.0/top_p=0.95 supplied by model metadata. HTTP bodies omit
  sampling parameters. All three requests use identical p32k bodies:
  SHA256 `03ff1ee585666af42cfe3509a2a592cafed55216a307e15437c8121e0fd30b43`.
- PP2 launcher's performance pins are retained; PP1 uses stages=1 and no
  PP_DEVICES/PP_SPLITS. Resident budget is 188 decimal GB, context 65536,
  one session and 2048 MB prefix budget. Stock-server deployment-only admin
  settings are omitted. Runtime logs prove RESIDENT, grouped prefill, batched
  verify and device-built MoE tables, with the unpacked ILP rows kernels.

| Run | K policy | Prompt/output | Rounds | Drafted/accepted | Width census |
|---|---|---|---:|---|---|
| auto6 | auto, selected K=3 | 29781/64 | 24 | 48/39 | t2:11, t3:2, t4:11 |
| k6 | existing K=6 pin | 29781/64 | 21 | 51/42 | t2:10, t3:3, t4:3, t5:1, t6:2, t7:2 |
| phase | K=6, SPEC_PROF=1, SPEC_TRACE=2 | 29781/64 | 20 | 63/45 | See phase-summary.json and raw depth log |

All three completed HTTP 200 with usage, 64 output tokens and terminal SSE.
Auto6 supplies the selected t2/t4 rows; k6 supplies t7. The phase run is a
separate synchronized diagnostic and is not pooled into these kernel means.
These are single-request profiles, not ABBA results or untraced serving rates.
The small t7 N=2 is explicit. Auto K is a cap; PMIN truncates actual width.

## Attribution and byte accounting

`tally.py` joins each CUDA kernel correlation ID to its host launch API and
then to enclosing NVTX verify/layer/weight ranges on that thread. It does
not use GPU timestamp overlap with host ranges. Draft, prefill, acceptance,
rollback and maintenance are excluded. Every included round must contain
all 45 trunk layer labels. NVTX weight labels record in/out features,
qtype and resident bytes at the actual rows-exact dispatch.

For labeled batched qmatvec kernels, logical weight bytes are the resident
plane once per launch, shared across t rows. For the routed pair, the estimate
is t*top8*4096*2048*36/64 bytes per weight plane, two planes for gate/up and
one for down, over 42 layers. It includes repeated expert visits; L2 reuse
is unknown. HC mixes use 24*16384*4 bytes per row launch. GB/s is bytes/us/1000,
percent is GB/s/8000*100. These are **effective logical weight bandwidths**,
not measured HBM traffic. n/a means no bound weight estimate for that row;
it is not zero bandwidth. Activation, sector amplification and cache hits
are not hidden inside an invented HBM counter.

Kernel durations are summed. GPU span runs from first kernel start to final
kernel end; interval union handles overlap. Kernel-free gaps include transfers;
the idle column below additionally removes traced memcpy/memset occupancy.
Host NVTX time is asynchronous submission time and is not end-to-end verify
latency. Neither a gap nor an API duration alone proves pure host-launch cost.

| t | Kernel-free gaps us/round | Memcpy/memset duration us/round | Device idle us/round |
|---|---:|---:|---:|
| 2 | 1274.463 | 231.014 | 1089.954 |
| 4 | 1034.122 | 232.224 | 861.639 |
| 7 | 1061.202 | 265.120 | 888.497 |

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


## Target selection

The largest-duration row is gathered MLA attention, 7218.395 us at t4.
`cu/mla_attn.cu`'s existing DSA roofline mechanism identifies serial tiled
softmax, barriers and L2 traffic. Its faster warp-online arm is already
excluded from verify after a prior t4 argmax failure. This cell does not
reinterpret it as an HBM-bound GEMV or reopen that numeric verdict.

Among derivable weight-streaming targets, E4M3 has the largest t4 saving to
70% HBM: 3192.629 us, versus 2561.402 us for routed down and 2542.558 us for
gate/up. `CANDIDATE.md` specifies the selected exact six-projection fusion,
its launch arithmetic, oracle and unchanged >=0.5 ms weighted KEEP bar.

## Failed attempts and custody

The first stock-server launch rejected deployment-only MEMRA_ADMIN_ADDR.
The 170 GB residency attempt selected SLRU and declined grouped prefill;
it was stopped before any measured decode. At 188 GB the actual resident
allocation fits. The 187.54 GB estimate uses the first expert layer times
all layer slots; 42 routed NVFP4 layers contain 171228266496 logical bytes.
The next attempt failed a runtime peer probe because PP_DEVICES=0 is not a
multi-card device list. The scored PP1 runs omit that variable.

The killed partial Nsight capture had no CUDA kernel table. Explicit
profiler stop now flushes events at each burst boundary while the process
is alive; capture-range-end=none retains one report across bursts. A tiny
CUDA allocation/launch probe then recorded exactly 10/10 kernels. The
corrected full requests have complete CUDA tables. Failed attempts remain
in the archive and contribute no selected timing rows.

Full raw archive (private custody beside the Darklanes lane entry):
`tally-raw.tar.gz`, SHA256
`67b33bc8eb75d3a88198a26a50046259afcf91f7bcb59918e73edbf76511687e`.
It includes raw sqlite/nsys reports, requests, SSE, logs, launcher metadata,
source patches, binary hashes, failed attempts and the profiler probe.
`raw-manifest.json` binds every member. Full profiler reports contain process
environments and stay private; public summaries and the instrumentation
patch retain the reproducible numeric method.

Verdict: **TALLY BANKED; fusion KEEP at 1.789758071 ms/round**, see RESULTS.md. rev: 2026-09-23
