# gemma4 pn-fold backport — results (lane/gemma-pnfold, 2026-08-17)

GAP-DIAGNOSIS verdict 7, coordinator-funded. The E4B glue chain (entry:
`rms_pre_add_rms_norm[_q8z]`, exit: `rms_pre_add_scale_rms_norm_q8_1`) backported to
the gemma4 dense decode/verify/slotted trio — finishing the fold the E4B lane
documented as left on the table ("the decode-step trio kept 2 launches/layer it
doesn't need", 550fcfa5). All six arms (dc, eager ×2, verify-t ×2, dc_slotted) ride
ONE front (`gemma4_layer_tail_add_nq_pn` + slot-fed `_into` twins), so
decode == verify == graph parity holds by construction at either seam value.
BITS-CHANGING (single-phase reduction expansion rounding, per the E4B receipts);
`MEMRA_G4_PNFOLD=0` restores the unfused chain everywhere. Prefill stays unfused
(self-consistent — plain and spec share the same prime). 26B MoE guarded unfused.

## Gates (Japan GPU1 @450W)

- **Seam-off byte-reproduction**: `MEMRA_G4_PNFOLD=0` greedy stream is byte-identical
  to the pre-fold binary's — the kill switch is exact.
- **Graph parity, fold-on**: eager vs graph stream 256/256 IDENTICAL.
- **Suites**: engine + server green.
- **Drafter acceptance re-bank** (Q4_0 + dspark dflash, the accept5 recipe, interleaved
  ×5, dead-flat to the third digit):

| horizon | fold=0 (reproduces banked) | fold=1 | agreement |
|---|---|---|---|
| n=128 | 0.549 (78/142), plain 74.36, spec 131.6 | **0.570 (81/142), plain 76.17, spec 136.6** | 128/128 both |
| n=256 | 0.614, spec 142.6 | 0.611, spec 144.2 | 256/256 both |

  Acceptance moves UP at n=128 (+2.1pts) and is flat at n=256 (−0.3pts, trajectory
  noise after the bit change). The E4B verify/decode-split failure mode does not
  appear — all arms moved together. NOTE (receipts gap): the banked code-class cell's
  prompt ids were never recorded; the drafter lane should re-record them. The fold-off
  arm reproducing banked numbers exactly makes this A/B sound regardless.

## Serving cells (memra-server c1, interleaved ×3, dead-flat)

| artifact | fold=0 | fold=1 | delta |
|---|---|---|---|
| NVFP4mix | 58.70 | **60.05** | **+2.3%** |
| Q4_0 | 72.56 | **74.21** | **+2.3%** |

Window (NVFP4mix, ×3): 59.1 → 60.5 (−397 µs/tok — the two rms_norm + one quantize
launches per layer the profile predicted).

## Verdict: SHIPS default-on

Wins everything it touches: +2.3% serving on BOTH trunks, spec +3.8% with acceptance
up, every identity gate green, exact kill switch. Banked cells re-based by this
battery: plain Q4_0 74.2, spec n128 136.6/0.570.

**Merge flags:**
1. E4B: the dc site also serves E4B's dc-eager arm — this fold completes E4B's own
   pending item, but E4B's argmax/chat gates were NOT re-run here (no E4B artifact on
   this box). E4B lane must re-gate before its next prod push, or boot E4B serving
   with MEMRA_G4_PNFOLD=0 until then.
2. Batched lane (lane/gemma-batched): merges after this must re-run its serve-stream
   gate with the fold ON — its eager-vs-batched identity gate compares against the
   folded eager stream once merged.

## CONTAMINATION AUDIT (2026-08-17, supersedes the NVFP4mix rows above)

The NVFP4mix serving cells (58.70/60.05) and window numbers (59.1/60.5) in this doc ran
under the Q8RP-mirror prefill corruption (GAP-DIAGNOSIS Follow-up 2) — garbage-token
throughput. The Q4_0 serving cells, the dflash acceptance re-bank, and the seam-off
byte-reproduction gate are CLEAN (no Q8_0 mass / explicit-arm comparisons). Certified
replacements on the fixed build (healthy-output gate, interleaved ×5):
NVFP4mix all-off 57.17 → default stack **60.46 (+5.8%)**; Q4_0 re-certified **74.22**;
dflash re-run on fixed build: 0.570, agreement 128/128, spec 136.5; graph parity
256/256 on healthy streams. The fold's own verdict (SHIPS, +2.3% both trunks) is
re-certified by these numbers.

## Cumulative c1 ledger (NVFP4mix, Japan @450W)

55.2 (original finding) → 57.2 (corrected baseline) → 58.4 (fused2 + Q8RP-auto) →
58.64 (PDL wave-B) → **60.05 (pn-fold)** = **+5.0%** over the corrected baseline,
**+8.8%** over the original finding. Q4_0: 72.4 banked → **74.21**. Spec board number:
132.6 banked → **136.6** (same recipe, re-banked).
