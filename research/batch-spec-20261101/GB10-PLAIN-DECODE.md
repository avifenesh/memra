# GB10 plain batched decode: where the c>=8 plateau comes from

Date: 2026-11-03. Device: one DGX Spark (GB10, sm_121a). Model: owner GGUF Qwen3.8-27B (NVFP4 + Q5_K).

Geometry: 64 layers, 16 of them full attention (4 KV heads x 256) and 48 GDN (48 value heads x 128 x 128).

**Instrument.** `decode-batch-bench --seqs B --chunk min(B,16) --ctx D` is the serving tick program (`MEMRA_DBB_SAMP=greedy`: the lean device-sampled batched step, `MEMRA_DECODE_BATCH_CAP=16` as serving uses on GB10). Two measurements per (D, B):

- the median tick of 3 reps x 16 steps;
- an nsys kernel summary over 16 steps, captured with `cuProfilerStart/Stop` around the decode loop only, so the primes stay out.

**Ceiling.** The part's own streaming ceiling, `bw-roofline`: **234 GB/s contiguous, 226 GB/s scattered slabs.**

Raw data: `gb10-decode-prof/` (profile), `gb10-ncu/` (ncu), `gb10-b16-rpsc/`, `gb10-b16-final-ab/`, `rig5090-b16-ab/` (the fix), `gb10-g4-rpsc/` (the twin not kept).

**Result.** Up to B=8 the matmuls stream at the ceiling. At B=16 the b16 kernel stalled on L1 latency at 130 GB/s, and attention runs at half bandwidth at every batch. One exact fix went in: the b16 tier now uses the rpsc schedule, bit-identical. It cuts the B=16 tick by 18.4% on GB10 (100.9 -> 123.6 tok/s at depth 2048, 77.6 -> 90.5 at depth 12288) and by 14.4% on the 5090. The remaining levers are listed under Findings.

## Tick time and aggregate tok/s

| depth | B=1 | B=4 | B=8 | B=16 | B=32 (2 x 16) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 74.5 ms, 13.4 | 81.9 ms, 48.8 | 92.4 ms, 86.6 | 159.8 ms, 100.1 | 318.9 ms, 100.4 |
| 12288 | 78.2 ms, 12.8 | 94.0 ms, 42.6 | 116.2 ms, 68.9 | 207.7 ms, 77.0 | 415.5 ms, 77.0 |
| 32768 | 83.6 ms, 12.0 | 112.9 ms, 35.4 | 153.6 ms, 52.1 | 281.6 ms, 56.8 | not run |

At depth 12288 (about the workload's median turn: p50 ~11k tokens, p90 ~29k) B=16 and B=32 give **77 tok/s**, which is the serving plateau (67-77). B=32 is exactly 2 x B=16: the exact decode tier stops at 16 rows, so a 32-session tick runs two full forwards.

## Per-kernel time at depth 12288, B=16 (199.2 ms per step), against each part's own ceiling

| part | kernels | ms/step | bytes/step | effective GB/s | vs 234 GB/s |
|---|---|---:|---:|---:|---:|
| dense matmuls | `qmatvec_nvfp4_mmvq_b16_rp`, `_group4_b16_rp`, `q5_K_mmvq_b16` | 121.1 | 15.7 GB of weights | 130 | **55%: latency-bound (ncu below)** |
| full attention | `fa_decode_vec_q_seqs_v4` + `fa_decode_combine_seqs` | 51.6 | 16 x 365 MB of KV | 113 | **48%** |
| GDN state | `gdn_scan_s128_b` | 22.7 | 16 x 302 MB (fp32 state read + write) | 213 | 91%: at the ceiling |
| rest | norms, quantize, conv, sampling | ~4 | | | |
| host gaps | tick minus kernel sum | ~8.5 | | | |

KV bytes per token per layer: q8_0 K at 34 B per 32 elements plus q5_1 V at 24 B per 32 elements, which is 1856 B. Over 16 layers that is 29.7 KB per token, so 365 MB per session at depth 12288. GDN state per session: 48 layers x 48 x 128 x 128 x 4 B = 151 MB, read and written each step.

The same matmuls at B=1 take 69.3 ms for the same 15.7 GB, 226 GB/s, which is the ceiling. So weight streaming is fine at one row, and still fine at eight. At 16 rows the b16 kernel falls to 130 GB/s. The ncu section below shows why: it is not compute-bound, it is starved for eligible warps.

## Why the b16 matmul and attention do not scale (ncu, depth 2048 B=16 and depth 12288 B=8)

`battery/gb10_ncu.sh` sections SpeedOfLight, Occupancy, LaunchStats, WarpStateStats, SchedulerStats, ComputeWorkloadAnalysis and MemoryWorkloadAnalysis, 6 launches each. Raw CSVs: `gb10-ncu/`.

| kernel | us/launch | SM throughput | memory throughput | issue slots busy | regs | occupancy | eligible warps/scheduler | L1TEX scoreboard stall per warp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mmvq_b16_rp` (before) | 532 | 16% | 35% | 16% | 68 | 58% | 0.25 | 47.7 cycles |
| `mmvq_b16_rpsc` (after) | 395 | 19% | 26% | 19% | 90 | 39% | 0.26 | 23.8 cycles |
| `group4_b16_rp` | 400 | 23% | 31% | 23% | 96 | 41% | 0.36 | n/a |
| `mmvq_b8_rpsc` | 279 | 17% | 22% | 17% | 72 | 53% | 0.27 | n/a |
| `fa_decode_vec_q_seqs_v4` (d12288 B8) | 1580 | 42% | 60% (L1/TEX 63%) | 28% | 68 | 37% (smem-limited, 3 blocks/SM) | 0.47 | 5.8 cycles |

The dp4a matvec is neither compute- nor bandwidth-bound at any width. Each scheduler issues once every 6.5-7.6 cycles, and the dominant stall is the L1TEX scoreboard: activation loads inside the column loop, each column's 32 bytes re-read from L1 for every weight row a warp handles. The r1 `rp` schedule re-reads all 16 columns per row. Register pressure caps occupancy at 4-7 warps per scheduler, which is too few to hide that latency.

Attention is L1/shared-bound at 3 blocks per SM: 27.9 KB of dynamic smem per block (Q staging plus a 32 x 256 bf16 V tile).

## The exact fix that went in: b16 on the rpsc schedule

`qmatvec_nvfp4_mmvq_b16_rpsc` is the existing rpsc body (`nvfp4_mmvq_batched_rp_sc`, two weight rows per warp, scale rows prestaged to smem) instantiated at MCOLS=16. `batched_variant` picks it at mcols 16 for split-plane NVFP4 when the scale plane fits smem (in_f % 256 == 0, in_f/64 <= 272: every 27B shape), and `b16_rp` otherwise. Two rows per warp halve the activation re-reads per weight row. The L1TEX stall per warp drops from 47.7 to 23.8 cycles, and the kernel goes from 532 to 395 us.

The per-(token, row) chain is unchanged: same quant load, scale bytes, dp4a order, adg factor and warp reduce. It is therefore bit-identical, a batch-invariance-preserving schedule change and not a new numeric class.

Exactness:

- kernel-check `RP BATCHED` now also covers m=12 and m=16 at mcols 16, on the model under test's `attn_gate`, `attn_qkv` and `ffn_down`: bit-bad=0 on all six (`gb10-b16-rpsc/kc.log`).
- `FUSED4` m=9/12/16 (group4 against the b16 singles) passes bit-bad=0.
- The decode token fingerprints (`TOKFP`, 16 sessions x 48 steps) are identical between the old and new binaries in every A/B cell.

Single-run A/B on GB10, old binary then new (`gb10-b16-rpsc/ab.txt`):

| cell | before | after | change |
|---|---:|---:|---:|
| B=16, depth 2048 | 158.6 ms, 100.9 tok/s | 129.4 ms, 123.7 tok/s | -18.4% tick |
| B=16, depth 12288 | 206.2 ms, 77.6 tok/s | 176.7 ms, 90.5 tok/s | -14.3% tick |
| B=12, depth 12288 | 160.5 ms, 74.8 tok/s | 143.3 ms, 83.8 tok/s | -10.7% tick |

Balanced interleaved A/B, both orders, 3 runs per arm, B=16 at depth 2048 (each run is the median of 3 reps x 16 steps):

| device | order | before (ms/tick) | after (ms/tick) | median change | tokens |
|---|---|---|---|---:|---|
| GB10 (`gb10-b16-final-ab/`) | old new new old old new | 158.62, 158.91, 158.62 | 129.40, 129.60, 129.43 | **-18.4%** (100.9 -> 123.6 tok/s) | TOKFP identical in all 6 |
| RTX 5090 laptop, same binary, `MEMRA_KS=0` as the before arm (`rig5090-b16-ab/`) | off on on off off on | 77.96, 88.61, 85.38 | 70.80, 73.10, 75.06 | **-14.4%** (187.4 -> 218.9 tok/s) | TOKFP identical in all 6 |

The GB10 runs spread by 0.2%. The 5090 ran hot (79-82 C, SM clock 1590-1912 MHz, logged per run), so its spread is wide, but every on run beats every off run. No PRO 6000 measurement: the pick is not keyed on the device, so a PRO 6000 A/B is still owed before this is a qualified multi-rig default. The same kernel serves pooled spec verify at 9-16 rows. The spec cost table is calibrated at boot, so the gate prices the faster verify without a change.

**Tried and not kept: a group4 b16 rpsc twin.** It is the same smem-scale staging on the GDN in-projection quartet, which already runs two rows per warp. It was bit-identical (FUSED4 m=9/12/16 bit-bad=0) and flat: 129.3 vs 130.1 ms at depth 2048, and 176.6 vs 177.1 ms at depth 12288 (`gb10-g4-rpsc/`). That locates the b16 win in the second row per warp, not in the scale staging. The twin was deleted.

## Findings

1. **The b16 matmul tier was the only part that got worse per row as the batch grew.** That was the jump from 72 ms at B=8 to 121 ms at B=16. The rpsc schedule fixes it exactly: it is now about 92 ms at B=16. The kernel is still latency-bound (0.26 eligible warps per scheduler, 19% of issue slots). The next exact levers are on the same per-row chain:
   - stage each g-iteration's activation slab in smem once per block, since the 4 warps of a block read the same columns;
   - software-pipeline the next g's quant load.

   Both keep the dp4a order and so stay bit-identical. Beyond that, a tensor-core decode matmul with a fixed K-tiling would be a new numeric class (owner decision, unchanged).
2. **The 16-row cap makes c=32 two full forwards.** A b32 tier on the same template (`<32,1>`, the register budget of `<16,2>`) would stream the weights once for 32 rows. It would be bit-identical per row. It needs the batch cap, the spec pool's 16-row verify bound and the cost table extended past 16. Not built.
3. **Attention runs at about 50% of the part's bandwidth at every batch and depth** (105-124 GB/s), and is 41% of the tick at depth 32768, B=8. It is smem-limited to 3 blocks per SM. Shrinking the V tile changes the online-softmax tiling, and with it the numerics. A version with the same tile but more memory-level parallelism (more warps per split, or a cp.async ring over the same tile) is the exact candidate.
4. **GDN state traffic is at the ceiling** from B=8 up. The only lever is a narrower state dtype, which is a numeric change.
5. **Host gaps are 2-18 ms per tick** (under 5%). Not the plateau.

## Every cell, per part (`gb10-decode-prof/parts.txt`, `battery/kern_parts.py`)

ms per decode step, effective GB/s in parentheses. Bytes: matmuls 15.7 GB of weights per forward; attention B x depth x 29.7 KB of KV; GDN B x 302 MB of fp32 state read plus write. Gap = tick minus the kernel sum.

| cell | tick | matmuls | attention | GDN | rest | gap |
|---|---:|---:|---:|---:|---:|---:|
| d2048 B1 | 74.5 | 68.8 (228) | 0.7 (83) | 0.8 (366*) | 1.9 | 2.3 |
| d2048 B4 | 81.9 | 70.7 (222) | 2.3 (107) | 4.1 (294*) | 2.1 | 2.7 |
| d2048 B8 | 92.4 | 72.0 (218) | 4.5 (109) | 10.4 (233) | 2.6 | 3.0 |
| d2048 B16 | 159.8 | 120.7 (130) | 9.0 (109) | 22.7 (213) | 3.8 | 3.7 |
| d2048 B32 | 318.9 | 241.9 (130) | 18.5 (105) | 45.4 (213) | 7.5 | 5.5 |
| d12288 B1 | 78.2 | 69.4 (226) | 3.6 (101) | 0.8 (365*) | 1.9 | 2.5 |
| d12288 B4 | 94.0 | 70.7 (222) | 13.5 (108) | 4.1 (293*) | 2.2 | 3.5 |
| d12288 B8 | 116.2 | 71.9 (218) | 26.2 (111) | 10.4 (233) | 2.6 | 5.0 |
| d12288 B16 | 207.7 | 121.1 (130) | 51.6 (113) | 22.7 (213) | 3.8 | 8.5 |
| d12288 B32 | 415.5 | 241.3 (130) | 103.3 (113) | 45.4 (213) | 7.5 | 18.0 |
| d32768 B1 | 83.6 | 69.3 (227) | 9.1 (107) | 0.8 (363*) | 1.9 | 2.6 |
| d32768 B4 | 112.9 | 70.6 (222) | 32.3 (120) | 4.2 (291*) | 2.1 | 3.7 |
| d32768 B8 | 153.6 | 71.8 (219) | 62.7 (124) | 10.4 (233) | 2.6 | 6.1 |

\* Above the DRAM ceiling: at B<=4 one layer's state (3.1 MB per session) is written back into L2 and leaves after the kernel ends, so the kernel's own time sees only the read. From B=8 the write-back no longer hides.

The d32768 B16 kernel capture was cut to save box time. Its tick (281.6 ms) matches the per-part model: B8's matmuls replaced by B16's (121 ms), with attention doubled (125 ms) and GDN at 22.7 ms. That sums to 269 ms plus about 10 ms of gaps.

What this says:

- **Matmuls are at the weight-streaming ceiling up to B=8** (218-228 GB/s, 69-72 ms whatever the batch). The break is the b16 kernel: the same weights cost 121 ms at 16 rows. Two 8-row forwards would cost 144 ms, so b16 is still the better of today's options.
- **Attention runs at 105-124 GB/s at every batch and depth**, about half the part's ceiling. It is linear in B x depth, so it becomes the largest part at long context: 41% of the d32768 B8 tick.
- **GDN sits at the ceiling** from B=8 up.
- **Host gaps stay small** (2-18 ms, under 5% of every tick).

So the plateau at c>=8 has two causes. The b16 matmul tier jumps from 72 to 121 ms, and attention grows linearly at half bandwidth. The b16 jump is the only part that gets worse per row as the batch grows.
