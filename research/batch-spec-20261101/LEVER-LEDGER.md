# Lever ledger: GB10 plain decode, 2026-11-03

One row per lever tried today, adopted, rejected or broken: config, hardware, n, baseline, candidate, commit and receipt. Model: owner GGUF Qwen3.8-27B (NVFP4 + Q5_K) unless marked 9B (Qwen3.5-9B NVFP4 GGUF, on the 5090).

Hardware:

- **GB10:** one DGX Spark, sm_121a.
- **5090:** RTX 5090 Laptop, sm_120a, thermally throttling.

Bench conventions:

- **Bench:** `decode-batch-bench --seqs N --chunk N --ctx D --steps 16 --reps 3`, with `MEMRA_DECODE_BATCH_CAP=32` and `MEMRA_DBB_SAMP=greedy`. The value is the median ms/tick of 3 reps.
- **Exactness:** kernel-check bit-bad=0 and decode token fingerprints (TOKFP) equal between arms.
- **Interleaving:** "3+3" means three runs per arm, interleaved in both orders.

SHAs on `batch-spec` are pushed. The served tensor-core build was 35897adf14, the same code tree as 0e914ccf4c, kept on `research/tc-b32-wip`. Rejected arms live on `research/*` branches on origin, so every SHA stays reachable. The lever-1 sweep kernels were never committed; only their receipt survives.

| # | lever | config / flags | hw | n | baseline | candidate | sha | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | b16 tier on the rpsc schedule | auto at mcols 16 (rpsc) | GB10 | 3+3 | 158.62 ms/tick (B16 d2048, rp) | 129.43 | 9a1ef496fa | GB10-PLAIN-DECODE.md, gb10-b16-final-ab/ | **adopted** (-18.4%, exact) |
| 1b | same | same, `MEMRA_KS=0` as base | 5090 27B | 3+3 | 85.38 | 73.10 | 9a1ef496fa | rig5090-b16-ab/ | adopted (-14.4%) |
| 2 | group4 b16 smem-scale twin | group4_b16_rpsc | GB10 | 1+1 x2 | 129.3 / 176.6 | 130.1 / 177.1 | (uncommitted, deleted) | gb10-g4-rpsc/ | rejected (flat) |
| 3 | b16 next-quant register prefetch (rpscp) | MEMRA_BVAR=rpscp | GB10 bench | 3 rounds | 6.43-6.48 ms/pass (rpsc) | 8.53-8.63 | (uncommitted) | gb10-lever1/sweep.txt | rejected (exact, slower) |
| 4 | b16 cp.async smem activation staging (rpscs) | MEMRA_BVAR=rpscs | GB10 bench | 3 rounds | 6.43-6.48 | 8.41-8.57 | (uncommitted) | gb10-lever1/sweep.txt | rejected (exact, slower) |
| 5 | b16 four rows per warp (rpsc4) | MEMRA_BVAR=rpsc4 | GB10 bench | 3 rounds | 6.43-6.48 | 7.11-7.13 | (uncommitted) | gb10-lever1/sweep.txt | rejected |
| 6 | b16 column split 2x8 (rpcs8) | auto at mcols 16 | GB10 | bench 3 rounds; step 2-3 per arm | bench 6.47; step 129.3 | bench 5.66; step 130.5 | (uncommitted) | gb10-lever1/ | rejected (bench win, flat in the step) |
| 7 | b16 column split 4x4 (rpcs4) | MEMRA_BVAR=rpcs4 | GB10 bench | 3 rounds | 6.47 | 6.49-6.81 | (uncommitted) | gb10-lever1/sweep.txt | rejected |
| 8 | 32-row tier, one wave (b32 rpcs8 4x8) | wave cap 32, decode-wide scope | GB10 | 3+3 | 259.15 (2x16 waves, B32 d2048) | 249.21 | fb90a89dc9 | gb10-lever2/ab.txt | **adopted** (-3.8%; d12288 -3.0%) |
| 9 | b32 singles 2x16 schedule (rpcs16) | MEMRA_B32V=cs16 | GB10 | 2 | 249.2 | 367.7 | 1a3decf498 lineage (research/batch-spec-wip-levers) | gb10-lever2/schedules-ab.txt | rejected |
| 10 | group4 b32 as 2x16 halves | in-tree variant | GB10 | 3 | 249.2 | 258.6 | 808119c506 (research/batch-spec-wip-levers) | gb10-lever2/ (first killed run) | rejected |
| 11 | Q5_K head b32 2x16 split + group4 halves | MEMRA_G4H=0/1 | GB10 | 3x3 | 247.5 | 253.2 (g4) / 251.9 (g4h) | 4959f37392 (research/b32-followups-rejected) | gb10-lever2/followups-ab.txt | rejected |
| 12 | boot wave probe (32-row vs 2x16 waves) | measured_wave_cap | GB10 / 5090 9B | probe median 3 | 276.2 / 52.8 (2x16) | 266.2 / 49.6 (1x32) | 8e271ade80 | gb10-probe/, rig5090-probe/ | **adopted** (per-box decision) |
| 13 | batched-tick attention on the deep walk | fa_decode_vec_q_seqs_v4_deep | GB10 | 3+3, 1+1 | 176.59 (B16 d12288) / 152.02 (B8 d32768) | 173.62 / 137.46 | fb90a89dc9 | gb10-lever3/ | **adopted** (-1.7% / -9.6%, exact) |
| 13b | same | same | 5090 9B | 3+3 | 44.75 (B16 d12288) | 40.51 | fb90a89dc9 | rig5090-levers/ab.txt | adopted (-9.5%) |
| 14 | raw-V attention in smem | MEMRA_FA_RAWV=1 | GB10 | 3+3, 1+1 | 173.75 / 137.46 | 176.52 / 141.22 | d14e4ae5c6 (research/fa-rawv-rejected) | gb10-rawv/ | rejected (exact, +1.6% / +2.7%) |
| 15 | tensor-core GEMM tier (non-exact) at 16..32 rows | MEMRA_DECODE_TC=1 | 5090 9B | 3+3 | 27.40 (B16), 66.37 (B32) | 73.98, 91.07 | b207de734c (research/decode-tc-gemm-arm) | rig5090-tc/ | rejected (slower; 9/16 and 18/32 streams identical) |
| 15b | same, >16 rows only | MEMRA_DECODE_TC=1 | GB10 | 3+3 bench; served 1 run | bench 247.8 (B32); served 79.16 (c32) | bench 232.2; served 79.36 | a28a2f4fc6 (research/decode-tc-gemm-rejected) | gb10-tc-served/ | rejected (served flat, not exact) |
| 16 | exact tensor-core b32 tier (int8 mma, dp4a float order) | b32 auto via boot probe; MEMRA_MMVQ_TC=0/1 forces | GB10 | 3+3, 1+1 | 247.53 (B32 d2048) / 337.27 (B32 d12288) | 213.46 / 302.00 | 0e914ccf4c | gb10-tc/ab.txt | **adopted** (-13.8% / -10.5%, exact) |
| 16b | same | same | 5090 9B | 3+3 | 58.39 | 53.03 | 0e914ccf4c | rig5090-tc2/ab.txt | adopted (-9.2%) |
| 16s | same, served | boot probe picked the tensor-core tier | GB10 | 1 run | 79.0 tok/s c32 (lever 3) | 83.6 | 0e914ccf4c | gb10-agentic/tc-b32/ | adopted (+5.8% served c32) |
| 16c | same at the b16 tier | MEMRA_MMVQ_TC=1 at mcols 16 | GB10 | 2+2 | 128.3 (B16 d2048) | 149.1 | 0e914ccf4c parent tree (v2, step-tc) | gb10-tc/ab-b16.txt | rejected (b16 stays dp4a) |
| 17 | bf16 GDN state, rounding emulation | MEMRA_GDN_STATE_BF16=1 (round on store, fp32 bytes) | GB10 | panel 264; agentic 1 run | 158/164, 97/100 (fp32) | 160/164, 97/100 | cdda787cde (research/gdn-state-bf16) | GDN-STATE-BF16.md (branch) | quality in band; served speed confounded |
| 18 | bf16 GDN state, real storage | MEMRA_GDN_STATE_BF16=1 (bf16 in the first half of each fp32 buffer) | GB10 | identity; decode 3+3, 1+1; served 1 run per arm | 247.44 (B32 d2048) / 173.48 (B16 d12288); served 22.34/29.02/37.64/62.59/76.82/79.23 (c1..c32) | 225.38 / 162.07; served 23.05/23.80/39.91/60.22/77.90/80.60 | c3356b88b3, receipts 2d8617de0f (research/gdn-state-bf16) | GDN-STATE-BF16.md (branch) | research arm, owner decides: exact to the emulation (TOKFP 73fe2628), decode -8.9%/-6.6%, served -18%..+6% by c |
| 19 | b16 tensor-core form (v1 kernel, 2 butterfly levels in smem) | MEMRA_MMVQ_TC=1 at mcols 16/32 | GB10 bench | 1 | 6.52 (m16) / 9.93 (m32) | 7.21 / 8.41 | ca62d9df98 lineage (research/tc-b32-wip) | gb10-tc/ (step-tc-v1, superseded) | superseded by v2 (registers fold two levels) |
| 20 | combined opt-in: exact tensor-core b32 tier + bf16 GDN state storage | MEMRA_GDN_STATE_BF16=1; boot probe chose the tensor-core b32 tier | GB10 | identity; decode 3 reps; served 1 run | fp32 dp4a served 22.34/29.02/37.64/62.59/76.82/79.23 (c1..c32); tensor-core alone c8/c16/c32 62.41/75.84/83.61 | served 23.17/23.89/39.51/60.80/77.49/85.46; decode B8/B16 d12288 112.8 -> 107.1, 173.3 -> 162.0 | 19b3495bab (research/combo-tc-gdnbf16) | gb10-combo/, GB10-PLAIN-DECODE.md (combined section) | research arm, owner decides: best c32 so far (+7.9%), c2 -18% from a spec-gate tie at n=2 under bf16 |
| 21 | per-part profile at the served depth (d12288, B8/B16) | nsys kernel sums, fp32 and bf16 | GB10 | 1 capture per cell | - | B16 fp32: matmul 91.5, attention 42.4 + combine 6.1, GDN 22.4, gap 7.2 | 19b3495bab | gb10-combo/step/parts.txt | measurement: next exact lever is the dead partial-buffer memset (about -5.3 ms at B16 d12288), not yet built |
| 22 | drop the dead partial-buffer memsets in fa_decode_batch_seqs_v4 | default (no flag) | GB10 | 3+3 (B16), 2+2 (B8), 1+1 (B32) | 173.75/173.45/173.67 (B16 d12288), 113.05 (B8), 337.40 (B32) | 167.31/167.29/167.22, 110.05/110.12, 324.38 | c86ae1ac6e | gb10-memset/, GB10-PLAIN-DECODE.md | **adopted** (-3.7% / -2.6% / -3.9%, exact, memset 5.58 -> 0 ms/step) |
| 23 | spec gate v1: stale spec slots stop pricing (64 s TTL) + tie probe within plain EMA noise | default (no flag) | GB10 served, fresh boot | 1 run c1..c32 | fp32 base 22.34/29.02/37.64/62.59/76.82/79.23 (c1..c32, dp4a b32) | 22.58/28.25/38.70/**49.21**/71.35/83.35 (with memset + TC b32) | 2ac067a1a9 (research/spec-gate-ttl-rejected) | gb10-gate/v1-ttl-default-fp32/ | rejected: without stale ratio anchors the table priced c8 spec at 98.7 vs realized 49.3, c8 -21% |
| 24 | spec gate v2: prices keep every slot; probe within 2 SE of the plain EMA | default (no flag) | GB10 served, fresh boot | 1 run c1..c8 (stopped after c8) | as row 23 | 22.45/**23.95**/39.80/62.25 | 781ecc64ef (research/spec-gate-v2-rejected) | gb10-gate/v2-default-fp32/ | rejected: c2 still plain (K=3 priced 25.8 vs plain 26.1 from K=4/5 slots measured at low acceptance; plain SE too small to call a tie); c8 unhurt |
| 25 | spec gate v3: spec slots priced at the live acceptance; tie probe within 2 SE of the difference | default (no flag) | GB10 served, fresh boot | 1 run c1..c32 (fp32); 1 run c2,c8 (bf16 combo) | c2 fp32 29.02 (old gate) / bf16 23.89 (old gate); c8 62.4 / 60.80 | fp32 22.60/28.94/38.10/62.10/75.77/83.11; bf16 c2 27.98, c8 59.42 | 062ecac658 (bf16: caed240c24 on research/combo-tc-gdnbf16) | gb10-gate/v3-default-fp32/, gb10-gate/v3-combo-bf16/, GB10-AGENTIC.md | **adopted**: c2 recovers on both arms, c8 holds on fp32 |
| 26 | default-build served rerun (memset removal + v3 gate + boot-chosen TC b32) | default | GB10 served, fresh boot | 1 run | TC-only c8/c16/c32 62.41/75.84/83.61 | 62.10/75.77/83.11 | 062ecac658 | gb10-gate/v3-default-fp32/ | measurement: served flat within single-run noise |
