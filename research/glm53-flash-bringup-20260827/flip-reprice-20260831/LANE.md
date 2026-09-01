# glm5 FLIP RE-PRICE on the batched verify walk (box B, 2026-08-31)

Third run of the flip battery (cells 3+4 shape of `../flip-battery-20260830/`), now
against `lane/glm5-verify-batch` @ `c62677352` — the batched verify walk
(`MEMRA_GLM5_VERIFY_BATCH`, default ON). Question: does glm5 spec (DFlash2, the drafter
of record) beat plain once the K+1 verify rows share their heavy math?

Prior verdicts, both NO-FLIP:
- 3way (f8f35bd91): plain 35.41 unbeaten; round wall fit 31.6 + 20.1*K ms.
- flip re-battery (bb8d9e3cc, loop-ported): plain 35.408 | K1 35.043 (0.9897x) /
  K2 34.474 (0.9736x) / K3 31.919 (0.9015x); round walls 52.49/71.31/91.08 ms; the wall
  is VERIFY-ROW-bound (~24-26 ms/row ~= one plain step per row); phase shares K3
  draft 8.65 / verify 96.47, K1 8.57 / 51.65; tau 0.5/0.7 = 33.47/33.79 at old K=3;
  K=1 tie bar 51.93 ms (0.56 ms short).

What changed: the verify-batch lane (LANE.md section 2) batches KDA projections/conv/
gates (one t=K+1 call per layer, recurrence sequential inside one scan launch), MLA
(one rows-exact t-call per layer), and the lm head (tcols twin, weight read once).
Predicted (section 4, verbatim): "verify K=3 96.5 -> ~40-55 ms ... round ~50-64 ms at
acc/cyc 2.907 -> ~45-58 tok/s. K=1: verify 51.6 -> ~28-33 ms."

Pins: build `c62677352` (= handoff head 41fe867cc tree + lane-doc row; contains bringup
34e0c0bf2), own clone `/root/memra-vb`; model /root/models/glm53-nvfp4; drafter
/root/models/glm53-dflash2 sha256 re-verified `b33c0347...e410b`; serving env
byte-identical to the flip re-battery (3-card recipe, STAGES=3 SPLITS=15,30, BF16_MMV=1
— load-bearing for the tcols class, port 18400); pools: in-repo decode pool (pinned by
the build sha) + /root/l3-ab-prompts.json sha `de57a7a4...b53e46` verified.

## Cells

1. Identity first: served spec-vs-plain greedy byte identity, DFlash2 K in {1,3} x 6
   prompts (batched walk ON, incl. rejection-heavy d02/d04). ANY divergence STOPS the
   window (rig proved bit identity; the real artifact is the final word). Walk-engagement
   receipt per spec boot: `[glm5-spec] verify walk BATCHED per layer` present, PER-ROW
   line absent.
2. Phase receipt (`MEMRA_GLM5_SPEC_TRACE=2`): DFlash2 K=3 + K=1 batched, plus K=3 with
   `MEMRA_GLM5_VERIFY_BATCH=0` (the instrument-level seam receipt: the old walk must
   reproduce the flip-battery verify shares ~96.5 ms @ K=3). Banks `[glm5-phase]` +
   `[glm5-phase-v]` (vkda/scan/vmla/vrest). SHARES, never walls.
3. THE FLIP TABLE (timed, marker up): plain vs DFlash2 K=1/2/3 batched, interleaved x3
   per the amended owner protocol (x5 on anomaly: spread >0.5% or verdict within 2x
   pooled spread). Decode tok/s both pools, TTFT @0.4k/@3.7k cold, vendor-default
   sampled row per boot (128-token floor), engagement receipts, loop-law. Plus ONE
   `MEMRA_GLM5_VERIFY_BATCH=0` control boot (zctl, untimed-rows-ok): the wall-level seam
   receipt — reproduces the old ~91.08 ms K=3 round wall, never enters the table.
4. If any spec arm beats plain: K sweep refinement + c=4 row + PMIN 0.5/0.7 overlay on
   the winner.

## Status log

- Window opened 2026-08-30T21:58:30Z after the hbatch-battery done-line (queue protocol:
  waited under PENDING; build launched only after their timed marker dropped, nice-19,
  ~80s tail overlap with their late c=15 rung disclosed in BOX-QUEUE.md). BUILD GREEN
  @ c62677352: real 5m32s, BUILD_EXIT=0, bin mtime==BUILD_END, strings probes hit
  ([glm5-phase-v], "verify walk BATCHED per layer", dflash2 source, PMIN gate).
  Drafter + l3-pool sha256 full-matched; decode pool pinned inside the build clone.
- CELL 1 GREEN (identity first): 12/12 spec-vs-plain tapes byte-identical (DFlash2
  K in {1,3} x 6 prompts incl. d02/d04, greedy 256, served path, batched walk ON).
  Walk-engagement receipts GREEN both spec boots (BATCHED line present, PER-ROW absent).
  All boots GATES GREEN nonce-verified; VRAM-at-ready == flip-battery byte-for-byte
  (plain 51444/62772/66166, dflash 51444/62772/66774). Loop-law 0/18. The rig's
  bit-identity claim HOLDS on the real artifact. `receipts/c1/`.
- CELL 2 DONE (phase receipt, trace=2, 18 bursts/arm, 4 prompts greedy 128; SHARES,
  never walls — traced totals carry the phase+per-layer sync tax):

  | arm | draft | verify | accept | roll | maint | traced total | vkda (scan) | vmla | vrest |
  |---|---|---|---|---|---|---|---|---|---|
  | DFlash2 K=3 BATCHED | 8.64 | 69.72 | 0.042 | 0.25 | 0.11 | 78.76 | 17.57 (0.43) | 6.26 | 45.61 |
  | DFlash2 K=1 BATCHED | 8.57 | 43.72 | 0.025 | 0.09 | 0.08 | 52.40 | 12.40 (0.36) | 4.28 | 26.70 |
  | DFlash2 K=3 VERIFY_BATCH=0 | 8.65 | 96.477 | 0.044 | 0.19 | 0.11 | 105.443 | 0 | 0 | 96.477 |

  (medians; rounds-weighted means banked alongside.) THE SEAM RECEIPT: the =0 arm
  reproduces the flip-battery cell-2 old-walk shares TO THE THOUSANDTH (96.468/105.441
  then, 96.477/105.443 now) — same instrument, same walk, one build. The batched walk
  moves verify -26.8 ms/round at K=3 and -7.9 at K=1. Read against the lane prediction
  ("verify K=3 96.5 -> ~40-55 ms; K=1 51.6 -> ~28-33 ms"): the collapse LANDED but sits
  ABOVE the predicted band — the batched-class terms (vkda 17.6 + vmla 6.3 at K=3) are
  no longer the wall; vrest (MoE per-(token,expert) + glue + head classes, the named
  out-of-scope term) is 45.6 of the 69.7. Sub-split accumulators are all-zero on the
  per-row arm as built (they only tick inside the batched walk). Tapes vb-vs-vb0 4/4
  byte-identical; the c2 compare rc=1 is a banking-layout artifact (the phase-line
  dumps live in the tape dir and get swept by the *.txt compare) — named here, tapes
  themselves 4/4. Loop-law 0/18. `receipts/c2/`.
- CELL 3 DONE — THE FLIP TABLE (timed, marker held, x3 interleaved SUFFICIENT — no
  escalation rule fired, spreads 0.014-0.067%; `receipts/c3/c3/flip_check-x3.txt`).
  **FIRST FLIP IN THREE BATTERIES:**

  | arm | dec tok/s | ratio | deep tok/s | pool TTFT | TTFT@0.4k | TTFT@3.7k | vendor tok/s | round wall | verdict |
  |---|---|---|---|---|---|---|---|---|---|
  | plain | 35.423 | 1.0 | 30.04 | 0.362 | 0.422 | 2.207 | 33.56 | (28.23/step) | (==35.408 prior, 0.04%) |
  | DFlash2 K=1 | 41.221 | 1.1637x | 35.66 | 1.282 | 1.366 | 3.724 | 38.79 | 44.62 ms | **FLIP** |
  | DFlash2 K=2 | 44.245 | 1.2491x | 39.26 | 1.223 | 1.269 | 3.769 | 43.22 | 55.56 ms | **FLIP** |
  | DFlash2 K=3 | 43.420 | 1.2258x | 39.46 | 1.287 | 1.425 | 3.961 | 46.21 | 66.95 ms | **FLIP** |
  | zctl (=0) | 31.898 | — | 26.19 | 1.603 | 1.776 | 4.601 | 30.33 | 91.14 ms | seam receipt |

  Readings: (a) plain reproduces both prior batteries (35.408/35.41) — box comparable.
  (b) The zctl =0 control reproduces the flip-battery K=3 arm to 0.07% (31.898/91.14 vs
  31.919/91.08) — one build, both walks, the A/B seam is the only mover. (c) Acceptance
  is byte-identical to both prior batteries (tok/cyc 1.839/2.458/2.907) — the batched
  walk moved TIME only. (d) Round-wall fit ~33.5 + 11.2K ms vs the old 33.0 + 19.6K:
  the verify marginal HALVED; the fixed term did not move (verify-row fixed cost +
  draft ~8.6). (e) The vendor-default sampled rows (the real traffic shape) flip too:
  38.79/43.22/46.21 vs plain 33.56 — K=3 is the best SAMPLED arm (sampled acceptance
  does not decay on this drafter, the 3way finding, now priced on the batched walk).
  (f) TTFT stays the known near-constant per-session drafter setup + ctx ingest
  (+0.9s pool, +1.5-1.8s @3.7k), not O(prompt). One plain vendor row excluded by the
  128-token floor (ct=35, named in flip_check). Loop-law 0/182. 13 boots 0 failures.
- CELL 4 DONE (gate fired; timed, marker held; `receipts/c4/`). Three parts:

  (a) K SWEEP refinement (x3 interleaved with plain, spreads <=0.06%): K4 41.799
  (1.1805x, wall 76.56), K5 37.442 (1.0575x, wall 91.38). Full measured curve
  K1..K5: 41.22 / 44.25 / 43.42 / 41.80 / 37.44 — **the peak is K=2 with K=3 close
  behind**, every K still above plain. Wall fit holds ~33.5 + 11.2K across all five.

  (b) c=4 CONCURRENCY row: plain 30.26 aggregate | nopin 30.34 (the c>2 K-shed fires,
  [spec-gate] demote receipt at 4 active >= HIGH=4 — nopin IS plain at c=4, protective
  as designed) | K2 PINNED through c=4: 31.67 (+4.7% vs plain — the pin no longer
  costs 24% as on the old walk; a follow-up lane may re-price the shed thresholds,
  named, not changed here).

  (c) PMIN overlay (single boots, timed, cross-tau greedy tapes 14/14 + 14/14
  identical, armed lines receipted; K3 overlay added as a NAMED ADDENDUM since K2/K3
  straddle the greedy/sampled peak): K2+tau0.5 44.661 / K2+tau0.7 43.847 /
  K3+tau0.5 45.196 / **K3+tau0.7 45.650 = 1.2887x plain, the best arm measured**
  (drafted/rnd 2.31, accrate 0.742, wall 59.36 ms — the tau arithmetic predicted
  ~59.4). tau still rescues the larger-K arm; at the halved marginal it now CREATES
  the peak instead of merely softening a loss.

  8-TURN LARGER-PROMPT TWIN (owner multi-turn law; vendor-default sampled, 4.6k->7.9k
  ptok): plain TTFT 2.22->3.39s; K2 spec 3.54->5.22s (delta +1.3->+1.8s, the
  per-session ctx-ingest class, NOT O(prompt)); per-turn sampled tok/cyc holds
  2.19-2.69 at depth. Cache: no cached_tokens on any turn of any arm (usage carries no
  cache field on this tree) — glm5 prefix cache remains structurally dead (the 3way
  loud-refusal + hbatch cached_tokens=0 receipts, same base tree), so cache-on twins
  and the cached->K=2 policy row stay unreachable, receipted not skipped.
- CELL 5 DONE (confirm; timed, marker held; `receipts/c5/`): x3 interleaved plain vs
  THE DEPLOYABLE CONFIG (DFlash2 + auto K policy nopin + `MEMRA_SPEC_PMIN=0.7`, batched
  walk default) — the exact env a flip would ship. Policy receipts every ship boot:
  `route=spec K=3 ... cold=1` (nopin routes K=3) + `draft confidence gate armed:
  PMIN=0.700`. Result: plain 35.408 (boots 35.431/35.408/35.407, spread 0.067%) vs
  SHIP **45.654** (boots 45.654/45.661/45.648, spread 0.030%) = **1.2894x**; deep
  40.29 (1.342x); vendor-default sampled 46.66; pool TTFT 1.259 vs 0.362; TTFT@3.7k
  3.770 vs 2.206; tok/cyc 2.71, drafted/rnd 2.31, wall 59.36 ms. Gap 10.25 tok/s >>
  2x pooled spread — no escalation rule fired. 8-turn twin on the ship config: TTFT
  3.53->5.19s across 4.6k->7.9k ptok (plain twin 2.22->3.39; delta +1.3->+1.8s,
  per-session ctx-ingest class, not O(prompt)); per-turn sampled tok/cyc 2.33-3.39.
  Loop-law 0/84.

## VERDICT (the flip decision)

**FLIP.** On the batched-verify head c62677352, on the deployed 3-card serving shape,
EVERY DFlash2 arm beats plain — the first flip in three batteries — and the best
deployable config is:

    MEMRA_GLM5_SPEC=1  MEMRA_GLM5_DFLASH=<pinned b33c0347 drafter>  MEMRA_SPEC_PMIN=0.7
    (auto K policy — routes K=3 cold; no MEMRA_SPEC_K pin; MEMRA_GLM5_VERIFY_BATCH
     stays at its default ON; 3-card recipe unchanged)

= **45.65 tok/s decode vs 35.41 plain (1.289x), deep 40.29 vs 30.02 (1.342x), vendor
sampled 46.66 vs 33.56** (x3 interleaved, spreads <=0.067%, boot-nonce arm identity).

What moved and why, receipted: the batched walk halves the verify marginal (round-wall
fit 33.5 + 11.2K ms vs 33.0 + 19.6K; trace=2 sub-split: vkda 17.6 + vmla 6.3 at K=3
with the scan floor at 0.43 ms — vrest = MoE/glue/head, the named out-of-scope classes,
is the remaining 45.6). The =0 rollback arm reproduces the old walk at both the
instrument level (verify 96.477 vs flip-battery 96.468) and the wall level (31.898
tok/s / 91.14 ms vs 31.919 / 91.08). Acceptance is byte-identical across all three
batteries — the seam moved time only. Correctness: 12/12 identity re-gate + 4/4 seam
tapes + 14/14 + 14/14 cross-tau, loop-law 0 flagged across every screen
(18/18/182/196/28/84 per cell), 44 boots 0 failures.

Costs that ride the flip, stated: pool TTFT +0.9s and deep TTFT +1.6s per session (the
DFlash2 per-session setup + ctx-ingest class, near-constant in prompt); VRAM +608 MiB
on dev2 (q4 drafter); the c>2 auto-shed serves plain at c>=4 (unchanged, protective —
though pinning K=2 through c=4 now GAINS +4.7%, re-pricing the shed thresholds is a
named follow-up, not this lane's change).

Caveats, named: tau arms were single boots (the x3 confirm covers the shipped tau=0.7
config; the tau LADDER shape rode the flip-battery precedent); vendor rows are one row
per boot (median-of-3 for the table, single rows noisy 34.5-52.4); prefix cache remains
structurally dead for glm5, so the cached->K=2 policy rows stay unreachable.

### Proposed FLAGS.md row amendments (draft — lands with the serving-flip lane, not this
receipts branch; the current rows carry the superseded NO-FLIP verdicts)

- `MEMRA_GLM5_SPEC` row, replace the verdict sentence block with: "REAL-ARTIFACT FLIP
  RECEIPTS (2026-08-31, `flip-reprice-20260831/`, batched-verify head): with
  `MEMRA_GLM5_VERIFY_BATCH` (default ON) the round-wall fit drops to ~33.5 + 11.2K ms
  and EVERY DFlash2 arm beats plain — K1 41.22 / K2 44.25 / K3 43.42 vs 35.41 (x3
  interleaved, spreads <=0.07%); the deployable shape (DFlash2 + auto K policy +
  `MEMRA_SPEC_PMIN=0.7`) measures 45.65 tok/s = 1.289x plain, vendor-default sampled
  46.66 vs 33.56. Flip to default-ON for glm5 serving rides the deploy lane's pinned
  rollout (rollback seam: unset the flag; `MEMRA_GLM5_VERIFY_BATCH=0` additionally
  restores the per-row walk, receipted at 0.9x plain)."
- `MEMRA_GLM5_DFLASH` row, replace "(2) spec still loses to plain..." with: "(2) ON the
  batched-verify walk (2026-08-31) spec BEATS plain at every K (peak K=2 44.25 greedy /
  K=3 best sampled 46.21; +PMIN 0.7 = 45.65); the flag ships ON wherever
  `MEMRA_GLM5_SPEC` ships ON — the native head stays the fallback source only."
- `MEMRA_SPEC_PMIN` glm5 note: "glm5 tau priced on the batched walk 2026-08-31:
  0.5/0.7 = 45.20/45.65 at K=3 (44.66/43.85 at K=2); ship 0.7 with the auto K policy;
  cross-tau greedy tapes identical (the gate moves drafts, never output)."
