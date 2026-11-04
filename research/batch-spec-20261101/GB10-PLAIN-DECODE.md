# GB10 plain batched decode: where the c>=8 plateau comes from

Date: 2026-11-03. Device: one DGX Spark (GB10, sm_121a). Model: owner GGUF Qwen3.8-27B (NVFP4 + Q5_K).

Geometry: 64 layers, 16 of them full attention (4 KV heads x 256) and 48 GDN (48 value heads x 128 x 128).

**Instrument.** `decode-batch-bench --seqs B --chunk min(B,16) --ctx D` is the serving tick program (`MEMRA_DBB_SAMP=greedy`: the lean device-sampled batched step, `MEMRA_DECODE_BATCH_CAP=16` as serving uses on GB10). Two measurements per (D, B):

- the median tick of 3 reps x 16 steps;
- an nsys kernel summary over 16 steps, captured with `cuProfilerStart/Stop` around the decode loop only, so the primes stay out.

**Ceiling.** The part's own streaming ceiling, `bw-roofline`: **234 GB/s contiguous, 226 GB/s scattered slabs.**

Raw data: `gb10-decode-prof/` (profile), `gb10-ncu/` (ncu), `gb10-b16-rpsc/`, `gb10-b16-final-ab/`, `rig5090-b16-ab/` (the fix), `gb10-g4-rpsc/` (the twin not kept).

**Result.** Up to B=8 the matmuls stream at the ceiling. At B=16 the b16 kernel stalled on L1 latency at 130 GB/s, and attention runs at half bandwidth at every batch. One exact fix went in: the b16 tier now uses the rpsc schedule, bit-identical. It cuts the B=16 tick by 18.4% on GB10 (100.9 -> 123.6 tok/s at depth 2048, 77.6 -> 90.5 at depth 12288) and by 14.4% on the 5090.

**Afternoon levers** (section "The three exact levers, measured"):

- Activation staging and next-weight prefetch in the 16-row kernel lose. A column split won the bench but was flat in the step. None of it is kept.
- The 32-row tier is kept, decided per box at boot. On GB10 at 32 sessions it is -3.8% per tick at depth 2048 and -3.0% at depth 12288.
- The batched-tick attention deep walk is kept. On GB10 it is -9.6% at 8 sessions x depth 32768 and -1.7% at 16 x 12288; on the 5090 9B, -9.5%.
- Served on GB10, the agentic cell went 60.4 / 67.1 / 67.8 -> 61.8 / 75.8 / 79.0 tok/s at c = 8 / 16 / 32 (vLLM 96.7 / 121.4 / 135.5).

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

## The three exact levers, measured (2026-11-03, afternoon)

Every arm below keeps the per-(token, row) chain. Each was gated by kernel-check (`RP BATCHED`, `FUSED4`, `fa_decode_seqs_v4 vs per-seq loop`: bit-bad=0 on every cell) and by the decode token fingerprints (`TOKFP`: all 16 or 32 sessions x 48 steps identical between arms).

### Lever 1: activation staging and next-weight prefetch in the 16-row kernel: not adopted

`mmvq-b16-bench` streams every NVFP4 2D tensor of blocks 0-3 (31 tensors, 856 MB) through the batched dispatch at m=16, one launch per tensor, and hashes every output bit (`gb10-lever1/sweep.txt`, 3 rounds; m=12 in `sweep-m12.txt`).

| schedule (m=16) | ms per pass | GB/s | output hash |
|---|---:|---:|---|
| rp (before today) | 8.86-8.99 | 96 | same |
| rpsc (adopted this morning) | 6.43-6.48 | 133 | same |
| rpsc + next-quant register prefetch | 8.53-8.63 | 100 | same |
| rpsc + cp.async smem activation staging, double-buffered | 8.41-8.57 | 101 | same |
| rpsc, four rows per warp | 7.11-7.13 | 120 | same |
| column split 2 x 8 (rpcs8) | 5.65-5.67 | 151 | same |
| column split 4 x 4 | 6.49-6.81 | 126 | same |

The two asked-for variants lose. The prefetch costs registers that cut residency, and the smem staging adds two block barriers per 32-group step at three blocks per SM. The column split won the bench, so it went to the decode step:

| B=16, depth 2048, 3 + 2 + 2 runs interleaved (`gb10-lever1/e2e-ab.txt`) | ms/tick |
|---|---:|
| rpsc (base) | 129.31, 129.59, 129.21 |
| rpcs8 singles, 16-wide group4 | 130.46, 130.76 |
| rpcs8 singles, split group4 | 133.59, 133.56 |

It is flat to negative in the real step. nsys (`gb10-lever1/nsys-*`) shows why: the decode step's singles are the FFN-sized tensors, where rpcs8 runs 268 us per launch against rpsc's 264. Its bench win came from the small tensors, which in the step ride the group4 door. Nothing from lever 1 is kept, and its kernels were deleted.

### Lever 2: the 32-row tier: adopted, decided per box at boot

- **Kernels.** `qmatvec_nvfp4_mmvq_b32_rpcs8` and `qmatvec_nvfp4_mmvq_group4_b32_rpcs8` give 32 columns as four 8-column blocks per 8-row weight block, adjacent in launch order so the repeat weight reads hit L2. Every other qtype at 17..32 rows runs as two bit-identical 16-column halves (here, the Q5_K output head).
- **Scope.** `decode_step_batch` admits 17..32 rows for exact-16 models and sets a decode-wide scope (`Engine::set_decode_wide`). Only inside that scope do `small_m_tier_max` and the group doors widen to 32. Verify and prime never see it, and the PP entries keep 16.
- **Server.** `chunk_cap_for` returns 32 for exact-16 models without a PP split. The spec cost table now measures plain steps up to the wave cap (plain points 1, 2, 4, 8, 16, 32), and `spec_plain_ms` prices up to 32. Pooled verify stays at 16 rows (below).

m=32 bench (`gb10-lever2/bench-m32.txt`): the 4 x 8 split takes 9.94-10.00 ms per pass. The 2 x 16 split takes 19.30.

Decode step at 32 sessions, one wave of 32 against two waves of 16 on the same binary (`gb10-lever2/ab.txt`, order c16 c32 c32 c16 c16 c32):

| depth | two 16-row waves (ms/tick) | one 32-row wave (ms/tick) | median change | tokens |
|---|---|---|---:|---|
| 2048 | 259.15, 259.02, 259.50 | 249.17, 249.21, 254.84 | **-3.8%** (123.5 -> 128.4 tok/s) | identical |
| 12288 (1 + 1) | 353.72 | 343.24 | **-3.0%** (90.5 -> 93.2 tok/s) | identical |

Schedule choice in the same step (`gb10-lever2/schedules-ab.txt`): 4 x 8 singles plus 4 x 8 group4 took 249.2 ms, and 2 x 16 singles took 367.7. 4 x 8 singles plus 2 x 16 group4 took 258.6 ms (in `ab.txt`'s first, killed run). nsys at 32 rows (`gb10-lever2/b32-kern.csv`): singles take 475 us per launch, against 2 x 264 for two b16 launches. group4 takes 612 us, against 2 x 276. The Q5_K head runs twice (2 x 6.5 ms).

The win is small: the dp4a kernel is latency-bound at every width, so a second column block per weight row costs nearly as much as the first, even when its weight reads hit L2.

**Device check on the RTX 5090: inconclusive under throttling, so the server measures it.** On the local 5090 laptop (`rig5090-levers/`, `battery/rig5090_levers.sh`), the test model was Qwen3.5-9B NVFP4 GGUF: the 27B GGUF runs out of memory on 24 GB at 24 or more sessions, and at 8 sessions x depth 12288. Results at 32 sessions, depth 2048, 3 + 3 interleaved twice, tokens identical:

| run | two 16-row waves (ms/tick) | one 32-row wave (ms/tick) |
|---|---|---|
| first (`ab.txt`) | 52.71, 59.30, 59.97 | 56.25, 61.96, 62.79 |
| rerun (`ab-wave-rerun.txt`) | 54.09, 63.90, 62.62 | 57.16, 60.74, 66.42 |

The laptop ran at 81-87 C, and its SM clock drifted from 2002 down to 1477 MHz across each run. The first run favored two waves in every adjacent pair. The rerun split 2 to 1. The drift is larger than the effect.

So the server decides on the box. At boot, `decode_wave_probe` times one plain step over 32 sessions as one wave and as two 16-row waves, interleaved per rep (spec-cost depth 2048, median of 3). `measured_wave_cap` keeps the 32-row cap only if one wave is faster. The cap stays 16 when the probe cannot run (allocation failure) or `MEMRA_MAX_SESSIONS` <= 16, and an explicit `MEMRA_DECODE_BATCH_CAP` skips the probe. Boot lines:

| box | two waves | one wave | cap | probe time |
|---|---:|---:|---:|---:|
| GB10, 27B (`gb10-probe/boot-lines.txt`) | 276.2 ms | 266.2 ms | 32 | 79.3 s |
| RTX 5090, 9B (`rig5090-probe/boot-lines.txt`) | 52.8 ms | 49.6 ms | 32 | 13.7 s |

The probe adds its time to boot: priming 32 sessions at depth 2048 dominates. No PRO 6000 measurement.

Two follow-ups on the 32-row tick were measured and are not kept (`gb10-lever2/followups-ab.txt`, 32 sessions at depth 2048, 3 runs per arm interleaved, tokens identical):

| arm | ms/tick |
|---|---|
| kept: Q5_K head as two b16 launches, group4 4 x 8 | 247.49, 247.64, 247.65 |
| Q5_K head as one 2 x 16 column-split launch, group4 4 x 8 | 253.21, 253.45, 252.97 |
| Q5_K split, group4 as two 16-column b16 launches | 251.80, 251.93, 251.95 |

The column-split head costs 5.6 ms per tick against two plain launches. Group4 halves recover 1.3 ms of that, but only on top of the losing head. Both were deleted.

### Lever 3: attention loads in flight at the current tile: adopted

The batched tick's attention (`fa_decode_vec_q_seqs_v4`) ran the plain v4 tile walk: K-tile rows at a 64-word stride, which is a 32-way bank conflict on the score-phase operand read, and a stage -> sync -> compute -> sync loop with no load of the next tile in flight. The B=1 eager arm had run the deep walk (`fa_v4_deep_walk`) since 2026-08-02. That walk keeps the same tile, values and accumulation order, pads the K rows (4 words per row, 1 per scale row), and prefetches tile+2's K/V lines into L2 during the current tile.

`fa_decode_vec_q_seqs_v4_deep` puts the seqs frame (z = sequence, per-seq pointer table, per-seq split partition) around that walk, and replaces `fa_decode_vec_q_seqs_v4`. Each sequence now runs exactly its eager program. kernel-check `fa_decode_seqs_v4 vs per-seq loop` is bitdiff=0 at B=4 (depths 96/128/257/511) and B=8.

A = the lever-2 build, B = the same plus the deep seqs twin (`gb10-lever3/ab.txt`):

| cell | before (ms/tick) | after (ms/tick) | median change | tokens |
|---|---|---|---:|---|
| B=16, depth 12288, 3 + 3 (old new new old old new) | 178.50, 176.59, 176.57 | 173.38, 173.85, 173.62 | **-1.7%** (90.6 -> 92.2 tok/s) | identical |
| B=8, depth 32768, 1 + 1 | 152.02 | 137.46 | **-9.6%** (52.6 -> 58.2 tok/s) | identical |
| B=32, depth 12288, 1 + 1 | 342.30 | 337.37 | **-1.4%** (93.5 -> 94.9 tok/s) | identical |

RTX 5090, Qwen3.5-9B, 16 sessions at depth 12288, 3 + 3 interleaved (`rig5090-levers/ab.txt`): before 43.58, 44.75, 45.00 ms; after 40.47, 40.51, 41.36 ms. That is **-9.5%**, tokens identical. The deep walk wins on both devices and is not keyed on either.

nsys at B=16, depth 12288 (`gb10-lever3/fa-kern.csv`): attention takes 42.4 ms per step, down from 45.6 (138 GB/s, 59% of the part's ceiling); the combine stays at 6.0. The gain grows with depth because the prefetch covers more tiles per split.

The kernel is still held to three blocks per SM by its 28.5 KB of smem: Q, the padded K tile and a 32 x 256 bf16 V tile. The next exact attention lever is to keep V raw (6 KB) in smem and dequantize at use, which would allow five blocks per SM. Every warp of the GQA group would then dequantize the whole tile, about 6x the B3-phase instructions, so it needs its own measurement. A smaller tile changes the online-softmax order, so it is not exact.

### Raw-V attention in smem: not adopted (2026-11-03, evening)

`fa_decode_vec_q_seqs_v4_rawv`, built on a research branch and A/B'd behind a temporary `MEMRA_FA_RAWV` seam, now deleted. It kept the V tile as raw q5_1 bytes in smem (6 KB instead of the 16 KB bf16 tile) and dequantized each element at use, with the staging expression verbatim. Smem fell from 28.5 to 18.3 KB per block. It was bit-identical: kernel-check `fa_decode_seqs_v4 vs per-seq loop` bitdiff=0 at B=4/8, and tokens were identical in every run (`gb10-rawv/`).

| cell | deep walk (ms/tick) | raw V (ms/tick) | change |
|---|---|---|---:|
| 16 sessions, depth 12288, 3 + 3 | 173.75, 173.80, 173.69 | 176.52, 176.42, 176.59 | +1.6% |
| 8 sessions, depth 32768, 1 + 1 | 137.46 | 141.22 | +2.7% |

Two things cancel the occupancy gain. Every warp of the GQA group now dequantizes the whole tile, and the kernel was register-limited at 4 blocks per SM anyway, so lifting the smem limit bought at most one block. The code was deleted; the deep walk stays.

### Tensor-core GEMM tier for 16..32-row ticks: research arm, negative

The cheapest tensor-core prototype skips the exact scope for 16..32-row ticks, so the matmuls take the existing tensor-core GEMM tier (MMQ / F8F4). This was branch `research/decode-tc`, behind `MEMRA_DECODE_TC=1`, default off. Measured on the RTX 5090 with Qwen3.5-9B at depth 2048, 3 + 3 interleaved (`rig5090-tc/`):

| sessions | dp4a exact tier (ms/tick) | tensor-core GEMM tier (ms/tick) | sessions with identical 48-token streams |
|---:|---|---|---:|
| 16 | 27.40, 28.23, 30.00 | 73.98, 73.61, 81.87 | 9 / 16 (greedy) |
| 32 | 66.37, 66.57, 66.57 | 91.07, 90.69, 92.90 | 18 / 32 (greedy), 8 / 32 (sampled) |

Each arm is deterministic run to run. The prefill GEMM tier is 1.4-2.7x slower at these widths, and it diverges by construction.

On GB10 (second box, 27B, depth 2048, `gb10-tc-served/tc-bench-ab.txt`) the picture splits:

| sessions | dp4a tier (ms/tick) | tensor-core GEMM tier (ms/tick) |
|---:|---|---|
| 16 | 128.31, 128.30, 128.58 | 196.52, 196.55, 196.74 |
| 32 | 247.81, 248.49 | 232.23, 232.46 |

So the arm was narrowed to 17..32-row waves only and served through the agentic cell, both arms on the same box (`gb10-tc-served/`):

| c | dp4a (exact) | tensor-core GEMM above 16 rows |
|---:|---:|---:|
| 16 | 76.09 | 75.76 |
| 32 | 79.16 | 79.36 |

Served, it is flat: the cell's waves are rarely full 32-row waves, and the arm also gives up bit-identity. The research branch was not kept. A real tensor-core decode matmul is a new kernel. One exact design exists: run only the integer per-16-element dot products on `mma.sync` s8 (integer sums are order-independent), and keep the float epilogue (`ws0*s0 + ws1*s1`, `acc += adg*partial`, the warp butterfly) in the dp4a kernel's order. That would make it bit-identical. It saves only the dp4a issue slots of a latency-bound kernel, so its value is unclear. Not built.

### Exact tensor-core decode matmul for the 32-row tier: adopted, decided per box at boot (2026-11-03, night)

The dp4a batched chain per (row, column) splits into an integer part and a float part:

- **integer:** S_sl, the integer sum of the 16 int8 products of each scale half;
- **float:** per lane l over g = l, l+32, ...: `partial = ws0*S_0; partial += ws1*S_1; acc_l += adg*partial`, then the 32-lane xor butterfly, with lane 0's value written.

Integer sums do not depend on order. So `nvfp4_mmvq_tc` computes every S_sl with int8 `mma.sync` m16n8k16 (k16 is one scale half) and replays the float part in the same order:

- each warp carries the chains of virtual lanes w, w+8, w+16, w+24;
- the butterfly's first two levels fold in registers in lane w's own operand order;
- the last three levels replay from shared memory.

Kernels: `qmatvec_nvfp4_mmvq_b32_tc` (singles) and `qmatvec_nvfp4_mmvq_group4_b32_tc` (the group door; every out_f % 16 == 0). It is bit-identical on the first build:

- kernel-check with the tier forced on: `RP BATCHED m=24/32 mcols=32` bit-bad=0 on 6/6 cells, and `FUSED4 m=24/32` bit-bad=0 (GB10 and 5090);
- the bench output hash matches the dp4a tier;
- decode token fingerprints are identical in every A/B run.

Singles bench (`mmvq-b16-bench`, 31 tensors of blk.0-3, GB10; `gb10-tc/bench.txt`):

| m | dp4a tier (ms per pass) | tensor-core tier (ms per pass) |
|---:|---:|---:|
| 12 | 5.41 | 6.93 |
| 16 | 6.50 | 7.09 |
| 24 | 7.86 | 7.62 |
| 32 | 9.93 | 8.17 |

At 16 rows it loses in the step too: 128.3 vs 149.1 ms/tick (`gb10-tc/ab-b16.txt`). The b16 tier stays dp4a, and the tensor-core kernel serves only the b32 tier.

Decode step at 32 sessions, depth 2048, the same binary with `MEMRA_MMVQ_TC=0/1`, order dp4a tc tc dp4a dp4a tc (`gb10-tc/ab.txt`, `rig5090-tc2/ab.txt`):

| box | dp4a (ms/tick) | tensor-core (ms/tick) | median change | tokens |
|---|---|---|---:|---|
| GB10, 27B | 247.53, 247.52, 247.49 | 213.46, 213.44, 213.64 | **-13.8%** (129.3 -> 149.9 tok/s) | identical |
| RTX 5090, 9B | 54.60, 58.39, 60.16 | 51.70, 53.03, 55.11 | **-9.2%** (5090 throttling; tc faster in every adjacent pair) | identical |

Adoption: the boot wave probe now times three arms (two 16-row waves; one 32-row wave on dp4a; one on the tensor-core tier). It selects the tensor-core tier (`Engine::set_tc_b32`) only when that single wave is the fastest. `MEMRA_MMVQ_TC=0/1` forces it either way.

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

## Combined opt-in build and where the 8- and 16-row ticks go now (2026-11-04, night)

Build: `research/combo-tc-gdnbf16` at 19b3495bab (batch-spec with the exact tensor-core b32 tier, merged with `research/gdn-state-bf16`). Default stays fp32 GDN state; `MEMRA_GDN_STATE_BF16=1` opts in. Receipts: `gb10-combo/`.

Identity: flag off, 16 sessions at depth 2048, TOKFP ffd722c3ca34, the fp32 fingerprint.

### Served agentic cell, one fresh boot per arm, GB10

The boot probe chose the tensor-core b32 tier with the flag on (two 16-row waves 253.1 ms, one 32-row wave 243.4 ms, tensor-core 209.0 ms).

| c | fp32, dp4a b32 | bf16 state, dp4a b32 | fp32, tensor-core b32 | **bf16 state + tensor-core b32** | vLLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.34 | 23.05 | | **23.17** | 24.2 |
| 2 | 29.02 | 23.80 | | **23.89** | |
| 4 | 37.64 | 39.91 | | **39.51** | 68.7 |
| 8 | 62.59 | 60.22 | 62.41 | **60.80** | 96.7 |
| 16 | 76.82 | 77.90 | 75.84 | **77.49** | 121.4 |
| 32 | 79.23 | 80.60 | 83.61 | **85.46** | 135.5 |

aggregate gen tok/s, 48 turns, thinking on, T=1. The fp32 and bf16 dp4a arms are `research/gdn-state-bf16` (c3356b88b3); the tensor-core arm is `gb10-agentic/tc-b32/`.

- c32: 85.46, the best c32 so far (+7.9% over fp32 dp4a, +2.2% over the tensor-core tier alone).
- c2 drops to 23.9 in both bf16 runs (fp32: 29.0). It is the spec cost gate, not the decode. With bf16 state, 9 of 18 rung-3 decisions at n=2 choose K=0, each at "predicted 24.9 vs plain 24.9". The fp32 run never chose K=0 at n=2. The gate's predicted spec rate falls to the plain rate and plain wins the tie, so it stops collecting acceptance there. This is a gate issue, worth its own fix.
- c8: -2.9% vs fp32 dp4a, inside one run's spread at this cell (c8 ranged 60.2-62.6 across four runs today).

### Per-part tick at depth 12288 (the served cell's median turn), nsys

`decode-batch-bench --ctx 12288 --steps 16 --reps 3` for the tick, nsys over 8 steps for the kernels (`gb10-combo/step/parts.txt`). ms per step; GB/s in parentheses where the bytes are known (weights 15.7 GB per forward; KV 365 MB per session; fp32 GDN state 302 MB per session read plus write).

| cell | tick | matmul | attention (core + combine) | GDN | rest | gap |
|---|---:|---:|---:|---:|---:|---:|
| B8 fp32 | 112.8 | 71.7 (219) | 21.5 + 2.9 | 10.3 | 2.6 | 3.7 |
| B8 bf16 | 107.1 | 72.0 | 21.6 + 3.0 | 4.7 | 2.6 | 3.2 |
| B16 fp32 | 173.3 | 91.5 (172) | 42.4 + 6.1 | 22.4 | 3.7 | 7.2 |
| B16 bf16 | 162.0 | 91.5 | 42.3 + 6.1 | 11.5 | 3.7 | 6.8 |

B16 matmul split: `mmvq_b16_rpsc` 67.3 ms (256 launches), `group4_b16_rp` 17.7 (64), the Q5_K head 6.5. At B8 the same weights take 52.8 + 13.9 + 5.0.

What is left, per lever, at B16 depth 12288:

| part | now (ms) | floor at 226 GB/s (ms) | headroom (ms) | exact levers already tried |
|---|---:|---:|---:|---|
| matmul | 91.5 | 69.5 | 22.0 | rpsc (adopted), rpscp, rpscs, rpsc4, rpcs8, rpcs4, group4 rpsc, tensor-core b16 (all rejected) |
| attention core | 42.4 | 25.8 (KV) + 5.3 (partial writes) | 11.3 | deep walk (adopted), raw V (rejected) |
| partials memset + combine | about 5 + 6.1 | 5.3 (combine read) | about 5.8 | none |
| GDN | 22.4 fp32 / 11.5 bf16 | 21.4 / 10.7 | about 1 | bf16 storage (research arm) |

The attention partials are a large hidden byte stream. At depth 12288 the split is 64 keys, so 192 splits. Each layer writes B x 24 heads x 192 x 256 fp32 partials: 75.5 MB at B16. `fa_decode_batch_seqs_v4` zero-fills that buffer with `memset_zeros` before every launch, then the kernel writes it and the combine reads it back. That is 3 x 1.2 GB per B16 tick.

The memset is dead work. The seqs kernel writes every slot `split < n_splits_z` for each (z, head), including empty splits, which write acc=0 and m=-inf. The seqs combine reads only `s < n_splits_z`. The gap column fits the memset. Gap grows with B x depth:
- d12288: 2.5, 3.5, 5.0, 8.5, 18.0 ms at B1..B32;
- d2048 B16, with 32 splits: 3.7 ms.
The model predicts 5.3 ms of memset at B16 d12288 and 10.7 at B32.

### Top remaining exact lever

**Drop the three partial-buffer memsets in `fa_decode_batch_seqs_v4`.** It is bit-identical by construction: no value read by the combine changes.

| cell | expected saving | share of tick |
|---|---:|---:|
| B8 d12288 | about 2.7 ms | about 2.4% |
| B16 d12288 | about 5.3 ms | about 3% |
| B32 d12288 | about 10.7 ms | about 2.6% |
| d32768 | proportionally more | |

Gate: kernel-check seqs-vs-loop bitdiff=0, TOKFP equal, an nsys capture showing the memset time gone, then the 3+3 tick A/B.

It is small but certain. It does not close the vLLM gap by itself.

What would close the gap at c8 is the second row. The served gate already picks plain at c>=8 because verify does not pay: spec wins only while the 16-row tick costs less than 1 + acceptance (about 1.49) times the 8-row tick.
- fp32: 173.3 vs 1.49 x 112.8 = 168.
- bf16: 162.0 vs 1.49 x 107.1 = 159.6.

Both sit just above break-even. Two changes move it:
- With the memset gone (about -5.3 / -2.7 ms), bf16 reaches 156.7 vs 155.6, about level.
- A 16-row matmul at the 8-row tier's bandwidth (-20 ms) would put the 16-row tick near 137 ms. Spec at c8 would then predict about 8 x 1.49 / 0.137 = 87 tok/s, against vLLM's 96.7.

Every exact schedule of the dp4a 16-row kernel has been tried, so the matmul part needs a new kernel design, not another schedule.

### Dead partial-buffer memsets in the batched-tick attention: adopted (2026-11-04)

`fa_decode_batch_seqs_v4` zero-filled its three partial buffers before every launch. The seqs kernel writes every slot below `n_splits_max` for each (sequence, head), including empty splits, and the seqs combine reads only `s < n_splits_z`. So the fill was dead work. It is removed in c86ae1ac6e. The single-sequence path keeps its memsets: the TP token graph's retargeting keys on their three-memset signature.

Exactness: kernel-check `fa_decode_seqs_v4 vs per-seq loop` bitdiff=0 at B=4 and B=8 (`gb10-memset/kc.log`, rc=0); TOKFP identical in every run.

Ticks, GB10, old = batch-spec 6ad5af8032, new = c86ae1ac6e, both built in the same target (`gb10-memset/ab.txt`):

| cell | old (ms/tick) | new (ms/tick) | change |
|---|---|---|---:|
| 16 sessions, depth 12288, order old new new old old new | 173.75, 173.45, 173.67 | 167.31, 167.29, 167.22 | **-3.7%** |
| 8 sessions, depth 12288, old new new old | 113.05, 113.05 | 110.05, 110.12 | **-2.6%** |
| 32 sessions, depth 12288, new old (dp4a b32) | 337.40 | 324.38 | **-3.9%** |

nsys at 16 sessions depth 12288 (`p-old-mem.csv`, `p-new-mem.csv`): the memset time falls from 5.58 ms per step to 0, against the 5.3 ms the byte model predicted. The tick drops 6.4 ms, a little more than the memsets alone, since their launch gaps go too.
