# orndecode — beat every line by a margin (owner goal 2026-08-22)

Owner: "plan the work, learn 27b work, freeze vllm numbers, beat them by a margin on
each line, beat yourself on every current line." Model: Ornith-1.5-35B-A3B (3B-active
qwen35moe hybrid). Rig: box9-class 1x RTX PRO 6000 Blackwell WS 600W. Owner framing:
q38-27B DENSE serves 259 tok/s c1 on this engine — a 3B-active MoE has no excuse to be
slower; vLLM's 277 on the same silicon proves the headroom.

## Frozen scoreboard (vLLM 0.27.1 column FROZEN 2026-08-22 — never re-measured)

Raw: `research/ornith15-vision-20260822/raw/`. Vendor sampling, N in receipts.

| line | vLLM (frozen) | ours today | target |
|---|---|---|---|
| c1 decode, short | 277 tok/s | 188 plain / 118 spec-recipe | **>277 + margin** |
| c1 decode, longdoc | 289 tok/s | 210 plain | **>289 + margin** |
| c16 shared-prefix agg | 1190 tok/s | ~700 (spec-on AND off) | **>1190 + margin** |
| TTFT true-cold 14.7k | 0.126 s | 0.49 spec-armed / 1.37 plain | **<0.126 s** |
| TTFT under c16 load | 0.058 s | 0.062 s | **<0.058 s** |
| session 8-turn wall | 21.9 s | 12.89 s best | **<12.89 s** (self-beat) |
| shared c8 agg | 733 | 913.7 best | **>913.7** (self-beat) |
| TTFT warm repeat | 0.063 s | 0.025 s best | **<0.025 s** (self-beat) |

## What the 27b/q35moe campaign already banked (learn-from ledger)

- `moebatch-q35moe-20260821`: CSR-NVFP4 gate_up owner-scan (+6.8% B=8); f16g NVFP4
  prefill admission (pp14715 2,331→10,785 = 4.63x); message-boundary prefix seed
  (sharedc8 209→481); batched filtered device-sample (sampled c8 478→640-678).
- `moeprime-nvfp4-direct-20260821`: NVFP4 direct sk tile loader (+7-10% pp, kq-direct
  precedent); GDN K4/K5 mma naked on sm_120a (+5-8% pp). Prime anatomy after both:
  moe ~520 ms, gdn_linear ~360 ms, attn_full ~273 ms at 14.7k.
- `samplat-20260821`: decode anatomy (B=8 serve): trunk mmvq 28.4% (3x off BW floor),
  MoE experts 28.9% (~82% of floor — near-optimal), fa_decode 11.1% (8x off), GDN ~29%
  of B=8 tick; filter_stats fixed via coop grid.
- `draftcost-moe-20260820`: t-parallel verify admission + graph draft for resident-MoE
  heads; ornith spec-on 1.165x code / 1.048x agentic vs plain at acc ~0.43.
- q38's 259 route (v0.101): DFlash2-class drafter + rank-1..3 round-cost engine +
  masked-ranks trim — the acceptance economics ornith's baked MTP head lacks.
- Standing next-increments from moebatch (unfinished): NVFP4 expert-dot wide-load for
  decode (iq4 down8 precedent, rows+CSR+down bodies); GDN at batch (state ops + out
  block); serve logits D2H + host split at batch (10.4% bench); message-boundary
  CHECKPOINT INSERT (engine snapshots GDN state mid-prefill).

## Lanes (dependency order; every A/B interleaved both orders N>=5 per doctrine)

**A. Anatomy re-rank (running):** nsys c1-plain + c16 at v0.102 tip → kernel-time
shares per shape. Grounds B-F priorities. Output: table in RESULTS.md.

**B. m=1/small-m decode trunk fusion (1-2 agent-days):** fused q/k/v trio + GDN quartet
mmvq twins exist only at b8 (v0.100.1); c1 rides separate warp-per-row launches
(DRAM 40-50% vs experts' 82%). Build m=1 twins from the b8 source. Expected +10-25% c1.

**C. NVFP4 expert-dot wide-load, decode bodies (1-2 agent-days):** the iq4 down8
precedent ("47% of byte-math wall") applied to rows + CSR + down NVFP4 bodies —
decode ALU, benefits every c-level.

**D. GDN decode share (2-3 agent-days):** ~29% of B=8 tick; batched projections fine,
state ops + out block are the candidates. Also biggest prime residue (360 ms).

**E. c16 batch scaling (1-3 agent-days):** agg flat c8→c16 while vLLM triples; NOT the
spec/cache forfeit (controls). Suspects from nsys-c16: decode-batch width ceiling
(engine B=8 ~800), admission wave cap, per-tick host work (logits D2H 10.4% — lane F),
MoE grouped kernel batch ceiling. Config-class fix possible.

**F. Serve per-tick host/bus audit (1 agent-day):** logits D2H + host split at batch;
device-sample covers greedy — audit what still crosses per tick under vendor sampling.

**G. Prefill + TTFT (2-3 agent-days):** pp14715 12.9k tok/s → cold 14.7k ≈ 1.14 s; vLLM
TTFT 0.126 s. Sub-lanes: (i) the 0.49 spec-armed vs 1.37 plain true-cold anomaly —
MEMRA_PRIME_ANATOMY both boots, explain or exploit (a 3x on plain prime may be free);
(ii) GDN-linear + attn prime shares (360+273 ms); (iii) message-boundary checkpoint
insert (mid-prefill GDN snapshot) for warm-turn + shared-peer TTFT; (iv) prime-while-
decode overlap for TTFT-under-load.

**H. Spec economics (week-class, parallel):** v3 head — MEMRA_DUMP_HSEED tap on the
NVFP4 trunk, chain-train D=3, masked-ranks trim (q38 recipe). acc 0.43→0.65+ turns
K=3 multiplier from 2.3x to ~3x over plain. Bridge experiment first (hours): serve A/B
the PUBLISHED masked frspec-owngen draft (945 MB) vs baked head via MEMRA_DRAFT on the
draftcost graph-draft path.

**I. Recipe assembly + reps (last):** per-shape posture (spec off at c1-short today is
+59%; K-policy per shape), 3-rep medians on every scoreboard line, receipts.

## Kill rules

Per flags doctrine: losing/flat arms deleted, winners naked. Every claim: raw jsonl in
`raw/`, N, thermal regime, both-orders interleave. One scored campaign on the box at a
time; teardown by compute-apps PID between arms.

## Lane A results (2026-08-22, box9, decode-batch-bench full-run nsys 2026.1.3)

Server-attach captures with --delay produced empty reports twice (2025.6.3 AND
2026.1.3 — injection misses the already-created context); CLI full-run capture works.
Bench rates: B=1 141.4 tok/s, B=8 agg 799.9 (N=3 medians; serve graph path is faster —
serve plain c1 = 188).

**B=1 kernel shares (200 steps, prefill-class kernels excluded ≈8%):**
trunk mmvq family 29.5% (mr2_rp 17.1% — the NON-fused projections, 8 launches/layer;
fused4_rp 9.8%; fused3_rp 2.6%) · MoE 27% (gate_up 11.0, down 6.5, router chain 9.3 =
topk 4.5 + gemv 3.3 + sigmoid 1.5) · quantize_q8_1 7.4% (100,773 launches = 12.6/
layer-step — a launch storm) · lm_head Q5_K 6.9% (221 us/launch, 2/step, vocab 248,320)
· norms 6.5% · gdn_scan 2.6% · fa_decode 1.8%.

c1 levers ranked: mr2_rp coverage (fuse or wide-load the 8 unfused projections/layer),
quantize storm (batch per-tick), router chain fuse (one kernel), lm_head → NVFP4 or
w4a8 head route, then GDN/fa floors. Sum recoverable ≈25-30% → plain c1 ~235-245.
**Crossing 277 needs Lane H's acceptance economics on top — plan accordingly.**

**c16 wall ROOT-CAUSED:** `decode_batch_exact16_ok` refuses `Ffn::Moe(_)`
unconditionally (decode_batch.rs:288) — MoE checkpoints never enter the exact-16 tier,
so serve chunks c16 into two B<=8 waves (agg flat ~700). The naive door
(MEMRA_DECODE_BATCH_CAP=16) measures 290 agg — B=16 without the exact tier rides the
m>=16 GEMM tier for the trunk and collapses; opening the cap is NOT the fix. Lane E =
qualify the MoE stage at B=9..16 (pair-list kernels are m-agnostic by construction;
gate2-style byte-compare B=12/16 vs isolated is the bar, the CSR-NVFP4 batch-composition
defect is the cautionary precedent) + flip the policy per-tensor. Trunk b16 kernels for
NVFP4/Q5_K already exist and are admitted.

Lane B re-scope: m=1 fused3_rp/fused4_rp already exist (v0.100.x) — the m=1 gap is the
mr2_rp remainder + the small-op storm, not missing fusion twins.

## Standings correction (2026-08-22, frspec-ab ABBA, fresh boots, N=6/shape/arm)

The earlier c1 record was POLLUTED: those probes ran on servers whose
MEMRA_SPEC_ADAPT state had been shaped by a prior 8-turn session + c8 cell in the same
boot. Fresh-boot ABBA (baked head, spec recipe):

| line | polluted record | fresh boot | vLLM frozen | real gap |
|---|---|---|---|---|
| c1 short decode | 118-122 | 228-244 (med ~234) | 277 | **1.18x** |
| c1 longdoc decode | 189-229 | 263-270 (med ~266) | 289 | **1.09x** |
| c1 TTFT true-cold 14.7k | 0.485-0.514 | 0.405-0.450 | 0.126 | 3.2x |

Consequences: (1) "spec net-negative at c1 short" is REFUTED — fresh-boot spec-on is
+24% over plain (188) at that shape; the polluted number was adapt-state carryover.
Board drafter note corrected this commit. (2) Adapt-state boot-history dependence is
itself a finding: serve throughput for a shape depends on what shapes the boot served
before it — needs either per-shape adapt pools or state decay (sub-lane of H).
(3) Kernel lanes alone (25-30% recoverable) now clear BOTH c1 lines; H remains the
margin-maker and the c16/TTFT lanes are the hard walls.
(4) First frspec-ab pass attached NO draft (MEMRA_DRAFT is the single-model seam;
MEMRA_MODELS needs the `name=trunk+draft` spelling — FLAGS.md table) — frspec arms
rerun with `+draft`; proof line is `[worker] m: regime draft attached`.

## Lane H bridge: external masked frspec draft REFUTED at serve (2026-08-22)

`m=trunk+draft` attach verified (`[worker] m: regime draft attached`, head_vocab 32768
trimmed d2t). ABBA vs baked head, fresh boots, N=6/shape: c1 short 107-126 vs 228-244,
longdoc 109-159 vs 263-270, TTFT true-cold 0.78-0.84 s vs 0.41-0.45 s — the external
head halves throughput and doubles TTFT (it forgoes the resident-MoE graph-draft path
the baked head rides, and adds its own prime cost). Do not re-try the external attach
as a perf lever on this class; H's route is a better BAKED-style head (v3 retrain).

## Lane H refinement (2026-08-22, late): the draft lm_head is the measurable third of H

The baked MTP head shares the trunk's Q5_K lm_head (248,320 rows, ~221 us/launch at
m=1) — every K=3 round pays ~3 draft lm_head passes (~660 us) for ~2.3 accepted
tokens. The published ranks file (`ornith15-ranks-owngen-32768.gguf`) caps the draft
vocab at 32k rows (~29 us) — an ~10% c1 round saving IF the rank mask can ride the
BAKED head's graph-draft path (MEMRA_FRSPEC_TRIM is the safetensors spelling; the GGUF
spelling today is the pre-trimmed EXTERNAL head, which was refuted at serve because
external heads forgo graph draft). Concrete increment: teach the graph-draft path a
rank-masked lm_head view (row-slice of the resident head by the ranks list — same
kernel, smaller out_f, d2t remap at accept). Pairs with the v3 retrain; does not wait
for it. Hunter cycle shortened 26→13 min (sleep 1200→420) to raise clean-window
sampling for the two hair-lines.
