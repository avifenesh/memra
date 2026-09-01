# samplat — filtered device-sampling latency (2026-08-21, box4)

Campaign: beat vLLM on the realistic cells; standing decode walls after v0.100.0.

## Box4 decode-only nsys anatomy (the instrument this lane exists for)

nsys 2026.1.3 works on the cloudbox driver (595.91) — box3's 2025.6.1 could not capture CUDA.
Decode-only window (delay past load+prime), B=8 ctx4700 filtered sampling, 910 ticks:

| kernel | share | per tick |
|---|---|---|
| trunk `qmatvec_nvfp4_mmvq_b8_rpsc` | 28.4% | 2.99 ms, 310 launches (~3x off BW floor) |
| MoE `gate_up_csr` + `down8` | 28.9% | 3.04 ms (~82% of expert-bytes BW floor — near-optimal) |
| `fa_decode_vec_q_seqs_v4` | 11.1% | 1.17 ms (~8x off KV BW floor) |
| **`filter_stats_f32`** | **5.9%** | **620 us, ONE call — 8 blocks on 142 SMs** |
| GDN scan+conv+prep | ~6% | |

GPU ~96% busy → kernel efficiency, not launch gaps. MoE decode near its memory floor
**vindicates dropping the expert wide-load increment** (2b): the dot bodies already carry
the wide-load form and the bytes are the wall.

## Increment: filter_stats cooperative multi-block form

Same 24-iteration bisection, each row split across 16 blocks, per-iteration totals via
grid sync (cooperative launch). Slice-partial f32 sums = the accepted device-sampling
class (distribution-equal, deterministic per grid); admission `16*nrow <= SM count`;
`MEMRA_FILTER_COOP=0` rollback.

- sample-check ALL GREEN, both arms.
- Engine B=8 filt tick: 744.5/757.5 -> 770.8/784.6 tok/s (**+3.5%**, interleaved both orders).
- Serve c8 (vendor sampling): old 678/701/693/702 vs coop 696/718/711/728 (**+2.5-3%**,
  pairwise-consistent).
- run-spec K=1..8 PASS; serve-smoke 0 failed (box4).
- 5090 sample-check pending (rig held by a co-tenant task's battery at the time) — run
  before merge.


## Increment 2: GDN-quartet fused mmvq at the batched tick (fused4-b8)

The decode anatomy's #1 item (trunk mmvq 28.4%, 310 launches/tick) — the m=1 fused4 kernel
existed but the host gated it to m==1, and a naive grid.y=m lift would re-read weights B
times. `qmatvec_nvfp4_mmvq_fused4_b8_rpsc`: fused4's block-range dispatch over the BATCHED
seg body (`nvfp4_mmvq_batched_rp_sc` verbatim, o0 rebased) — weight rows read once for all
B columns, bit-identical per (tensor,row,column) to the four bN_rpsc singles. Host admission
mirrors the singles' batched gates + scale==1.0 (the bN program has no in-kernel scale).

## DEFECT FOUND AND REVERTED: CSR-NVFP4 batch-composition dependence (shipped v0.99.0)

decode-batch-gate on the ornith artifact at B=8 (first time this gate ran on the MoE model
itself): gate2 FAIL — B=8 logits bit-differ from isolated (maxdiff up to 1.1e0 at step 0),
gate3b sampled streams diverge. Bisect: MEMRA_MOE_CSR=0 -> ALL GREEN; =2 byte-compare at
t=8 shows the csr_nvfp4 kernel drifts last-ULP vs the rows program on 11041/32768 ACT
elements — batch-composition-dependent logits, the eosclass law class, IN SHIPPED SERVING
(any batched tick at t<=10 on NVFP4-expert MoE, v0.99.0+).

Chain-pinning attempts (explicit fmul/fmaf close; helper-shaped cached dot mirroring
expert_dot_nvfp4_g's return-value contract) did NOT close the drift — diffs identical
(11041) across all three cached variants, np=1 pairs included. A source-verbatim per-pair
`expert_dot_g_v` form IS bit-identical (gates ALL GREEN) but loses the dedup win (-3% vs
rows). VERDICT: NVFP4 CSR admission reverted to the rows twins; the cached kernel stays in
history for a SASS-level diff. Qualification hole recorded: increment-1's =2 byte-compare
ran across run-spec (solo shapes) and decode-batch-gate only on a dense model — the batch
gate must run ON THE MoE MODEL for any future expert-kernel change.
CLOSED (box5, 2026-08-21): the pre-existing IQ4_XS/IQ3_S CSR kernel passes
decode-batch-gate B=4 and B=8 ALL GREEN on an IQ-MoE artifact (AgentWorld-35B UD-IQ4_XS)
— the composition defect was specific to the NVFP4 twin; production IQ-MoE serving is
unaffected. Boxes note: box4 (ohio spot) was reclaimed ~5h in, mid-battery; box5
(same class) relaunched and re-ran the full ornith battery at the lane tip (incl.
fused3-b8): batch-gate B=8, kernel-check, sample-check, run-gen argmax, run-spec — ALL
GREEN. 5090: sample-check GREEN (coop), q38 run-gen MATCH + run-spec PASS; the 27B/35B
batch-gates OOM on the 24GB card by size (not a defect) — box-class gates cover them.

## Lane A/B (box4, all gates green: B=2/4/8 batch-gate, kernel-check, sample-check,
## run-spec, serve-smoke)

Engine B=8 filt tick: base (all seams off) 730.5/747.8 -> lane naked 797.7/812.8 =
**+8.8%** (coop filter_stats + fused4-b8, net of the CSR revert).
Serve c8 (vendor sampling): ~678-702 pre-lane -> 727.9/743.4.

## Ranked remaining decode walls (from the anatomy)

1. trunk mmvq b8 BW efficiency (28.4%, 3x off floor) — multirow/vectorization tuning.
2. fa_decode at B=8 (11.1%, 8x off KV floor).
3. GDN state ops (day-class each; scan is 17us x 30 launches).
vLLM c8 control on box4 note: flashinfer JIT tax pollutes its first visit (393 -> 994
warm); warm-cache measurement is the honest control. memra unaffected (no JIT).
