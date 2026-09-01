# TP-2 box battery — results (lane/glm5-tp2-battery, window 2026-08-31)

Box: the glm53 second 4-card (4x RTX PRO 6000 Blackwell Workstation 96 GB, 600 W
confirmed, host RAM 755 GB). Build **4a680d0ca** = pin **2c9e7fff6** (merge-forward head)
+ probe-bin-only commits; `git diff 2c9e7fff6 -- crates/` = the probe bin + its Cargo row
(receipts/build.log carries shas + the 4 TP announce strings probes + the worker-refusal
literal). Topology: ALL pairs NODE, one NUMA node — no weak edge on this box class; pair
0,1 used. **P2P capability: `nvidia-smi topo -p2p r|w` ALL OK** (receipts/topology/).

## Cell 1 — real-artifact class gate: verdict BAND, decomposed, no silent-wrong

Instrument: `glm5-tp2-box-probe` tape mode, teacher-forced on the plain reference tapes
(the rig-gate shape), full-vocab f32 logit dumps (prime + first 8 decode steps),
5 prompts (4 decode-pool + WARM), plus a tiny-prime (12-token) pool that keeps prime
inside the exact small-t regime.

| arm (vs plain single-card reference) | worst norm_rel (max_abs/scale) | tape |
|---|---|---|
| plain RP=0 vs RP=1 (rp mirror contract) | **0.0 — byte-identical 56/56 files** | identical |
| plain grouped-prefill=0 vs =1 | **0.0 — byte-identical 56/56** | identical |
| TP layers 0-2 (pure KDA shard, dense), tiny prime | **0.0 — byte-identical** | identical |
| TP layers 0-2, deep prime (250-560 tok) | 5.9e-2 | forks at soft margins (7/15) |
| TP one MoE layer (KDA+EP, layer 4), tiny | 4.8e-2 | holds 32/32 |
| TP one MoE layer (MLA+EP, layer 3), tiny | 2.9e-2 | holds 32/32 |
| TP all@0,1 (full trunk), tiny | 5.2e-2 | holds 32/32 |
| TP all@0,1, deep prime | 1.2e-1 | forks at tokens 7-114, ALL at margin <= 0.23 |
| RED swap-wo, tiny | **0.93-1.05**, argmax rank blown to 1.5e4-1.4e5 | forks at token 0 |

Decomposition (each row measured, not argued):

1. **KDA-shard decode t=1 is BYTE-EXACT** on the real classes (BF16-resident MMV column
   shard, f32 b_proj cuBLAS [64,4096]->[32,4096], conv/scan head split, wo
   column-over-gather): zero differing bits over 9 full-vocab dumps. The lane's b_proj
   near-tie suspicion does NOT materialize at t=1.
2. **The EP-MoE decode class owns the band**: both single-MoE-layer arms carry
   ~3-5e-2 regardless of mixer class, and the full 42-EP-layer trunk lands at 5.2e-2 —
   SATURATING, not accumulating. This is the SHARD-MAP §3 pre-registered named gate arm
   firing its pre-registered fallback: the sequential per-slot `qmatvec + ffn_act_lim +
   slot-ordered axpy(macro_scale)` walk does not bit-reproduce the fused NVFP4 epilogue
   kernels (`moe_gate_up_preclamp8_q8` + `moe_down8_fma_q8`); byte-identity scopes to the
   non-MoE classes, MoE carries the measured band.
3. **Deep primes ride the documented batched near-tie class** amplified by 34 recurrent
   KDA layers: greedy forks appear only at soft positions (measured margins at every fork
   <= 0.23 vs typical margins 1-16; 44/45 argmax steps agree).
4. **Reds bite 20-100x above green** (norm_rel ~1.0 vs green worst 1.2e-1; fork at
   token 0 vs green tapes holding through 32 tiny-prime steps).
5. Output-sample gates: TP free-run fluent, loop-law 0 flagged (0/5 tape + 0/102 timed).
6. **Named deviation**: TP arms REQUIRE `MEMRA_RP=0` — the shard builders refuse the rp
   split-plane mirror layout by name. RP=0 proven bit-identical on plain (row 1).
   Wiring the mirror layout into the shard path is a named engine follow-up.

NOT a serving qualification: the band verdict is against the plain walk, not a truth
oracle; serving admission (increment 6) still needs the truth-anchored gate. No
silent-wrong signature (bounded, class-attributed, soft-margin-only forks, loud reds).

## Cell 2 — transport receipt

v1 transport = host-canonical (every boot announces `transport=host-canonical` on all
four seams). Native P2P is NOT wired to the glm5 seam (LANE stage-3 decision); the box
IS P2P-capable both directions (topology receipt), so the native-P2P engagement A/B is a
real arm on this host class — NAMED follow-up, ladder inherited (`configure_native_p2p`).

## Cell 4 — bare TP-2 pricing (timed, marker held, interleaved x3 — spreads pp3 0.04% / tp2 0.17%, no escalation)

| arm | instrument | pool decode tok/s | deep decode | TTFT 0.4-0.5k | TTFT 3.7k | vendor row |
|---|---|---|---|---|---|---|
| PP-3 recipe (15,30) | **SERVED** (calibration boot, same build) | **35.36** | 29.99 | 0.42 s | 2.21 s | 34.59 |
| PP-3 recipe | engine twin x3 | 8.99 | 5.44 (A4630 3.35) | 0.47 s | 2.83 s | 9.20 |
| plain 1-card (SLRU 12000) | engine twin x1 | 19.87 | — | — | — | — |
| **TP-2 pair v1** (`all@0,1`, RP=0) | engine twin x3 | **22.65** | 21.4 (A4630 20.45) | 4.66 s | **94.2 s** | 22.6 |

- Served calibration reproduces the banked baseline to the hundredth (35.408-class),
  proving the pinned build serves baseline-identical.
- **Instrument finding (receipted, load-bearing):** the eager engine driver
  (`prime_cache`+`decode_step`, the card3-probe program class) is **0.254x of served on
  the PP-3 placement** and collapses further with depth (0.11x at 3.7k) — an arm-specific
  PP pathology of the naive per-step driver; on single-engine walks the same driver reads
  ~0.8x (plain1 control 19.87 vs its 24-26 served A-class, cross-box caveat). Engine
  twins CANNOT price PP arms. TP-2 is a single-root walk: served-class projection
  **~27-30 tok/s** (stated as a projection; the real number needs the serving wiring).
- EP engagement: 1,012,568 peer-slot dispatches per pool boot, 946,203 per deep boot —
  identical across all 3 repetitions of each shape (greedy determinism receipt).
- Vendor-default sampled rows track greedy on TP-2 (22.6 vs 22.65) and on served PP-3
  (34.59 vs 35.36). 128-token floor applied; loop-law 0/102.

## Cell 3 — the measured v1 join tax

TP-2 engine wall = 44.15 ms/token. Decode-gap table terms (ATTRIBUTION.md §4):
bandwidth 15.2/2 = 7.6 ms (9.7 ms with the EP-2 slowest-rank 1.57x haircut) + latency
class ~10 + drain ~4.4 = 22.0-24.1 ms, x the measured single-engine driver tax
(~1.2-1.3) = 26-31 ms. **Measured residual: ~13-18 ms/token of v1 join+dispatch tax vs
the table's assumed 1-2 ms.** Named contributors (code-anchored): per-token host-canonical
fan-out (z htod to peer per MoE layer x42), ~4-5 peer-slot sync dtoh->htod returns per
layer, per-slot sequential qmatvec dispatch (~32+ launches/layer vs 2-3 fused), wo gather
hops on 45 mixers. Additionally the **EP prime gap**: TP-2 v1 prefills at 39-58 tok/s
(no grouped prefill under EP; per-token per-slot walk) => TTFT 4.7 s @0.5k / 94 s @3.7k —
serving-blocking on its own.

## Cell 5 — composition gate NOT fired

TP-2 v1 does not beat PP-3 on the honest comparison (22.65 engine / ~27-30 projected vs
35.36 served). The TP-2+PP-2 4-card composition stays refused-at-preflight, gated on the
join diet landing first.

## Verdict vs the prediction, and the honest path to the 100 bar

**Bare TP-2 v1 does NOT pay the 42-43 roofline row.** The gap is fully attributed:
~13-18 ms/token of v1 correctness-transport join tax (the step37 ladder's exact
territory: 43.1 -> 69.7 was bought by direct join +0.6/+1.85, prestage +2.5, prejoin
overlap +5.1) plus the unported EP prime. The class gate holds (KDA byte-exact; EP band
pre-registered and bounded; reds loud), so the seam is CORRECT and the remaining work is
data-movement, not arithmetic.

Follow-up lanes, ranked by what these numbers say (with the diet doors dead at 1.025x and
SHIP at 62.4 needing 1.60x more):

1. **EP dispatch diet + batched EP decode walk** (biggest single lever this lane names):
   batch the per-slot chains per rank (one grouped launch per projection per rank),
   device-router or routed-slot prestage, direct join for the peer rows (P2P store into a
   root buffer — playbook-blessed), prejoin overlap with the root-owned shexp (the
   step37 +5.1 lever, placement law applies). Target: reclaim most of the 13-18 ms —
   lands TP-2 decode in the ~29-31 ms class (~33-35 tok/s engine, ~40+ served-class).
2. **EP grouped prime** (unblocks TTFT): route the existing grouped-prefill machinery
   per rank over the EP halves; without it TP-2 cannot serve any real prompt shape.
3. **Native P2P transport A/B** (box is P2P-OK both directions): replaces the
   host-canonical hops the diet does not remove.
4. **Serving wiring (increment 6)** after 1-2: per-session TP admission + the worker
   refusal lift, then the REAL served TP-2 number replaces the ~27-30 projection.
5. **TP-2 x spec composition gate**: the vrest head's DFlash2+PMIN loop is 1.77x on
   PP-3; if post-diet TP-2 reaches ~40+ served-class, the composed shape is the first
   credible 70-90; the 100 bar additionally needs the census's matvec-efficiency lever
   (bf16-mmv+moe = 65% of GPU time at 57-70% efficiency vs q38's 87%).
6. TP-2+PP-2 composition and the rp-mirror TP wiring ride behind 1-4.

Wall: window open ~05:05Z, cells closed ~09:0xZ box time (build 5 min, cell 1 + controls
~75 min incl 6 extra isolation boots, cell 4 ~105 min incl calibration, banking the rest).

Note: the l3 prompt pool texts are private surface (real transcripts) and stay in
darklanes/on-box (/root/l3-ab-prompts.json); only tapes and stats are banked here.
