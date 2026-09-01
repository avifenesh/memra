# Gemma-4-31B NVFP4mix single-stream gap: diagnosis + verdicts (lane/gemma-fused2, 2026-08-16)

Owner bar: "the bar is our best results, not other best results, go for it."
Instruments: gguf-census (exact byte budget), decode-window-profile + nsys
(cuProfilerApi-bracketed eager decode, depth 512, n=64, Japan GPU1 @450W), q38_bench
serving cells c1. Raw kernel sums + cells: gap-receipts/.

## The headline: the gap is BYTES, and we now sit ABOVE byte-parity

Per decode token, every weight is read once (dense trunk + head):

| artifact | bytes/token | c1 measured | achieved BW |
|---|---|---|---|
| Q4_0 (+Q6_K head) | 16.42 GiB | 72.4 tok/s | 1189 GiB/s |
| NVFP4mix (Q8_0 v/down/embd) | 20.48 GiB | 58.4 tok/s (post-lane) | **1196 GiB/s** |

NVFP4mix carries **+24.7% bytes** (ffn_down 6.86 GiB Q8_0 vs 3.63 Q4_0; attn_v 1.09 vs
0.58; embd/head 1.40 vs 1.08 — the owner-validated quality recipe). At Q4_0-equal
efficiency the NVFP4mix ceiling is 72.4/1.247 = **58.05 tok/s**. Post-lane serving
measures **58.4** — the artifact now runs at slightly BETTER achieved bandwidth than the
Q4_0 path. **There is no remaining kernel-efficiency gap.** The residual ~14 tok/s IS
the recipe's byte mass; only a recipe change (owner's call) or lifts that speed BOTH
artifacts can move the number further.

## Per-op budget, NVFP4mix eager step (µs/token, nsys /64, Q8RP on)

| op | µs/tok | share | achieved |
|---|---|---|---|
| NVFP4 fused2 pairs (attn q,k + ffn gate,up) | 6626 | 38% | 1412–1517 GiB/s |
| Q8_0 `_rp` singles (v, down, lm_head) | 6397 | 37% | 1431–1519 GiB/s |
| NVFP4 wo single (mr2_rp) | 1159 | 7% | ~1394 GiB/s |
| norm/elementwise zoo (rms×121, add_scale×59, add_rms×60, rope-norm×60) | 1192 | 7% | latency-bound (~4.3µs ea) |
| attention (fa_decode + combine + rows, d512) | 822 | 5% | KV-bound |
| gelu + quantize + append + misc | 490 | 3% | latency-bound |
| launch/idle gap (wall − GPU busy) | ~620 | 3.6% | host launch |

Every matvec class sits at 85–91% of the card's ~1669 GiB/s peak. Q4_0's budget has the
same shape (matvecs 79%, non-matvec 2327 µs, gap 611 µs) — the two artifacts differ in
byte mass, not in structure.

## Per-fix verdicts

**Shipped (all bit/byte-identical, default-on, composing kill switches):**
1. `matmul_nvfp4_fused2` dispatch (FUSED2-RESULTS.md): +1.8% serving. Pin 32/32
   bit-identical across two artifacts; `MEMRA_NVFP4_FUSED2=0` kill switch.
2. **Q8RP capacity-keyed default** (owner default-on ruling): the Q8_0 split-plane
   mirror machinery existed, default-off outside Hopper for VRAM. Unset env now means
   ON iff free VRAM ≥ mirror mass + 8 GiB (96GB boxes on; 24GB rigs keep today's OFF;
   Hopper keeps compile-time ON). Bit-identical by the rp law. Window 58.3→58.8.
3. **NVFP4mix joins the gemma4 slotted/graph path** (`matmul_nvfp4_fused2_into` +
   three chained refusal sites; SWA trio = fused2_into(q,k) + generic m1 slot matvec
   for the Q8_0 v). Graph-gate stream 512/512 IDENTICAL. Serving-neutral today (see
   verdict 4) — shipped for parity with Q4_0's slotted coverage; `MEMRA_NVFP4_FUSED2=0`
   composes as the full kill switch (fused2_into declines → capture refuses → exact
   pre-lane eager).

Serving cells (interleaved ×3, dead-flat): kill-switches 57.2 → default **58.4**
(+2.1%). Spec battery on the new binary: acceptance 78/142 = **0.549 exactly** matches
the banked cell, 128/128 stream agreement, 131.7 tok/s — dflash gates green.

**Measured dry / dropped:**
4. **The graph loop itself loses on this card**: eager-vs-graph gate at N=512 reads
   NVFP4mix 58.8 vs 51.6 and Q4_0 **72.3 vs 63.4** — replay+drain overhead exceeds the
   620 µs launch-gap it collapses, for BOTH artifacts. The server's c1 path is
   (correctly) not riding it; the 72.4 Q4_0 serving figure is its eager rate. Verdict:
   enablement stays (identity-proven, parity, future-proof), engagement stays off.
5. **Q8_0 mmvq tuning beyond rp**: dry. All classes at the 1400–1520 GiB/s plateau;
   the only lever left is split-K reduction reorder, which breaks FP-order identity —
   refused per identity law.
6. **Mixed-type trio kernel (nvfp4 q,k + Q8_0 v one launch)**: dropped as not worth a
   new kernel class — v already runs at plateau BW; the fusion could only save one
   launch slot per SWA layer (~60 µs/tok upper bound, +0.2%).

**Sized, not funded here (next arcs, in value order):**
7. **pn-fold backport to the decode-step trio** (+1–2 tok/s, lifts Q4_0 too): the E4B
   glue-fusion machinery (`tail_core_pn`, `rms_pre_add_rms_norm_q8z`,
   `rms_pre_add_q8_1`) kills the 2 extra launches/layer the trio keeps — already
   documented as left on the table ("the rows arms took this fold at 550fcfa5"). BUT
   the E4B receipts prove the fused reduction is NOT FP-order-identical (verify/decode
   split, greedy tie-flips, gate 135/256, resolved only by moving ALL arms together).
   Backporting means re-basing the 31B byte-identity receipts and re-validating the
   owner-banked drafter acceptance numbers. Days-class re-gate → own lane, not this one.
8. **PDL for the nvfp4/q8 mmvq families** (identity-safe): the q4 mr1 kernel already
   carries MEMRA_PDL_ENTRY; extending programmatic dependent launch to the other decode
   matvec classes attacks the same 620 µs gap the graph loop failed to win, without
   replay overhead. Unknown gain — probe-sized.
9. **embd/head Q8_0 → Q6_K on the NVFP4mix artifact** (+~1.6%): 0.32 GiB/token. This is
   a QUANTIZATION RECIPE change — the owner quality law names v/down; embd/head were
   not adjudicated. Flagged for owner decision, untouched here.

## Bottom line

55.2 → 58.4 tok/s c1 shipped this lane (+5.8% over the original finding's baseline,
+2.1% over the corrected same-protocol baseline), bit/byte-identical throughout, spec
untouched at 0.549/128-128. The artifact now runs above Q4_0-parity bandwidth; the
remaining distance to 72 is the recipe's +24.7% byte mass, not kernel work. The
next real speed unlocks are verdicts 7–9.

## Follow-up 1 — PDL wave-B (verdict 8): SHIPPED, +0.5% serving

`MEMRA_PDL_ENTRY` into the gemma NVFP4mix chain's hot kernels (nvfp4 fused2_rp +
mr2_rp, q8_0 rp singles) + `launch_pdl` arms in the fused2 wrapper and the generic
single dispatch (the `_into` twin stays plain — graph capture body untouched).
`MEMRA_PDL_NVFP4=0` reverts wave-B alone.

Gates: suites green; pin 12/12 (gemma) + 8/8 (Q38) both seam arms; greedy gemma-gate
BYTE-IDENTICAL seam off/on. Interleaved ×5 window: 58.8 → 58.9. Interleaved ×3 serving
cells (GPU0, dead-flat): **58.34 → 58.64 tok/s (+0.5%)**. Ships default-on per the
owner every-drop law. (A first window run was contaminated by a concurrent lane taking
GPU1 mid-interleave — reps discarded, re-run clean on GPU0; the interleave protocol
caught it exactly as designed.)

Batched-lane interaction: the gemma4 batched arm lives on lane/gemma-batched (not
merged here — decode-batch-gate on this binary reports "no gemma4 arm"). Wave-B keeps
every shared kernel bit-identical, so their identity gates stay green by construction;
their serve-stream gate re-runs at merge.

Serving c1 ledger this lane: 55.2 (original finding) → 57.2 (corrected baseline) →
58.4 (fused2 + Q8RP-auto) → **58.64 (PDL wave-B)**.

## Follow-up 2 — CRITICAL regression + certification (2026-08-17)

The capacity-keyed Q8RP default (shipped above, eb915bfc) exposed a pre-existing loader
defect: `build_q4_rp_swap` took whatever `rp4` mirror it found — the gemma4-dense q4rp
walk hijacked the 110 Q8RP mirrors on NVFP4mix, replacing GGUF Q8_0 bytes with the
split-plane layout in place. DECODE read the swapped bytes correctly via the `_rp`
dispatch (masking it from every decode pin), PREFILL read the fp16 d-plane as weights →
layer-0 v NaN → <pad>-spam serving on every fresh 96GB boot. Isolated by
lane/gemma-recipe-probe (RECIPE-PROBE.md); root-caused and fixed here (abf155e8): the
qtype guard now lives in the swap itself — impossible by construction, any walk order.

**Pin gap closed:** the fused2 pin was self-referential for the mirror dimension (both
arms read the same hijacked bytes). New receipts on the fixed build: prefill trace
finite (v0 nan=0), run-gen's prefill-vs-decode single-position assert completed without
MISMATCH on the exact invocation that panicked pre-fix (`CUDA_VISIBLE_DEVICES=1
MEMRA_G4_PRIME_TRACE=1 MEMRA_CHAT=1 MEMRA_NGEN=32 run-gen <NVFP4mix> --prompt "Explain
binary search briefly."` — single-position assert at this depth, NOT the calibrated
board-2048 argmax-margin gate; that gate's gemma calibration + banked runs came later,
see ZOO-FUSION.md gates + gap-receipts/argmax-gate/), mirror-on greedy
output BYTE-IDENTICAL to mirror-off (end-to-end, prefill included), pin 12/12, and
every serving cell now carries a fresh-boot output-sample gate (real prompt → assert
non-degenerate text) before its tok/s counts.

**CONTAMINATED RECEIPTS (garbage-token throughput, marked, superseded):** every
NVFP4mix cell on a 96GB boot without explicit MEMRA_Q8RP=0 since eb915bfc — this doc's
def-r* 58.4 and the Q8RP window pair 58.3→58.8 (both arms of the PDL cells 58.34/58.64),
and PN-FOLD's NVFP4mix cells (see that doc). CLEAN: kill-r* 57.2 (explicit Q8RP=0), the
original fused2 A/B 57.2→58.2 (pre-default binary), all Q4_0 cells, all dflash
acceptance cells (Q4_0 trunk), all 5090 runs (capacity key never engages).

**CERTIFIED serving c1 (fixed build, interleaved ×5, Japan GPU1 @450W, healthy-output
gate on every boot, dead-flat):**

| arm | decode p50 |
|---|---|
| NVFP4mix, all lane seams off (pre-lane baseline) | 57.17 |
| NVFP4mix, default stack (fused2 + Q8RP + PDL wave-B + pn-fold) | **60.46 (+5.8%)** |
| Q4_0, default stack (×2) | **74.22** |

Corrected cumulative ledger: 55.2 (original finding) → 57.17 (certified baseline) →
**60.46 certified** — the stack's claim survives certification and slightly improves
(healthy prefill vs the contaminated 60.05). Exactness on the fixed state: dflash
0.570 (81/142) spec 136.5, agreement 128/128; graph parity 256/256 on healthy streams;
batched lane's gemma arm still merges from its own branch (gate re-runs there).
