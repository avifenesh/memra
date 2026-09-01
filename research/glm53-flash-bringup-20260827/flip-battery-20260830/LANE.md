# glm5 FLIP RE-BATTERY on the loop-ported head (box B, 2026-08-30)

The 3way decision window's cells 4-6 re-run against `lane/glm5-loop-port` @ `bb8d9e3cc`
(bringup merge base `a76b5b398` + ports 0-3 + fold-ins A/B). The owner's flip decision
rides on this window. Question: does glm5 spec (DFlash2, the drafter of record) now beat
plain after the loop port?

Baseline = the 3way decision window (`../3way-decision-20260830/`, build f8f35bd91),
serving env byte-identical (3-card recipe, STAGES=3 SPLITS=15,30, port 18400):

- plain 35.41 tok/s decode / 30.01 deep / TTFT 0.422s@0.4k / 2.208s@3.7k
- DFlash2 K curve (dec tok/s, ratio, tok/cyc, round wall ms):
  K1 34.98 0.988x 1.839 52.6 | K2 34.32 0.969x 2.458 71.6 | K3 31.72 0.896x 2.907 91.6
- round wall fit `31.6 + 20.1*K ms` vs plain step 28.24 ms; the K=1 flip needs a 0.67 ms
  fixed-cost saving (needs 51.93 vs measured 52.6)
- native TTFT@3.7k 12.303s (the O(prompt) sequential MTP-plane warm, ~400 tok/s)

Pins: model /root/models/glm53-nvfp4; drafter /root/models/glm53-dflash2
`model.safetensors` sha256 `b33c0347...e410b` RE-VERIFIED == the dflash-draft-src pin
(full hash matched byte-for-byte at window open). Pools: in-repo decode pool
`decode-attribution-receipts/prompts.json` (pinned by the build sha) + `/root/l3-ab-prompts.json`
sha256 `de57a7a4...b53e46` VERIFIED. Build attribution: real build time + `git log -1` +
mtime + strings probe (`[glm5-phase]` literal — the port-0 instrument only exists on the
loop-ported head). First build launch failed exit 127 (cargo not on the nohup PATH) and the
receipt caught it per the rebuild-attribution law; relaunched with `/root/.cargo/bin`.

## Owner protocol change (2026-08-30, mid-window, before cell 3 ran)

Cell 3 (and 6) run interleaved **x3** fresh boots per arm by default; escalate to x5 ONLY
on anomaly: (a) within-arm relative spread of the decode-tok/s boot-medians > 0.5%, or
(b) verdict too close to call at any K — the spec arm's median within 2x the pooled spread
of the plain median (either side). Escalation extends BOTH the affected spec arm and plain,
still interleaved; the receipt states which rule fired. Every arm reports its spread either
way (`box/flip_check.py` — definitions in its docstring are the receipt).

## Cells

1. Boot + byte-identity re-gate: plain + DFlash2 K in {1,3} x 6 prompts (both pools incl.
   the rejection-heavy d02/d04), greedy 256, served path. ANY divergence STOPS the window.
2. PHASE TIMER receipt (`MEMRA_GLM5_SPEC_TRACE=1`, port 0): DFlash2 K=3, K=1 + native K=3,
   4 prompts each. Attribution SHARES, never round walls (phase boundaries sync the stream)
   — read against the 3way `31.6 + 20.1*K` split, never as a perf row.
3. THE FLIP TABLE (timed, marker up): plain vs DFlash2 K=1/K=2/K=3, interleaved fresh
   boots, x3 + escalation rules above. Decode tok/s both pools, TTFT @0.4k + @3.7k cold,
   one vendor-default sampled row per boot (128-token floor guard from the 3way trap),
   engagement receipts, loop-law screen.
4. Native TTFT spot receipt (one boot): fold-in A (batched MTP-plane warm) should collapse
   the 12.30s TTFT@3.7k to near-plain — spec-battery flip condition 1.
5. PMIN tau ladder (count-based): DFlash2 K=3, MEMRA_SPEC_PMIN in {0.3, 0.5, 0.7} —
   acceptance, drafted/round, tok/s deltas; prices glm5's tau (per-model measurement law).
   Greedy tapes must be identical across taus (truncation moves drafts, never output).
6. If a spec arm beats plain: K sweep refinement + c=4 concurrency row on the winner.

## Status log

- Window opened, BOX-QUEUE line written (wall estimate ~2.5h). Checkout bb8d9e3cc, build
  running, drafter + pool shas verified. Scripts staged (`box/`).
- BUILD GREEN @ bb8d9e3cc: real 1m19.075s, BUILD_EXIT=0, bin mtime==BUILD_END, strings
  probes hit ([glm5-phase], "draft source = dflash2 @", "confidence gate armed: PMIN=",
  glm5_mtp_plane_fill). Receipt `receipts/build-bb8d9e3cc.log`.
- CELL 1 GREEN: 3 boots (plain / dfl-k1 / dfl-k3) all GATES GREEN (nonce-verified, 3x
  RESIDENT, fresh-boot sample fluent). Byte identity 12/12 spec-vs-plain (K in {1,3} x 6
  prompts incl. d02/d04), loop-law 0/18. VRAM-at-ready == the 3way receipts byte-for-byte
  (plain 51444/62772/66166, dflash +608 on dev2). The loop-port byte-identity claim HOLDS
  on the served path. Receipts `receipts/c1/`.
- CELL 2 DONE (phase-timer receipt, the port-0 instrument's first real-artifact run;
  SHARES-not-walls law applies — traced totals carry the phase-boundary sync tax and are
  never perf rows): 54 `[glm5-phase]` bursts banked (`receipts/c2/`), per-round ms medians:

  | arm | draft | verify | accept | roll | maint | traced total |
  |---|---|---|---|---|---|---|
  | DFlash2 K=3 | 8.648 | 96.468 | 0.044 | 0.191 | 0.114 | 105.441 |
  | DFlash2 K=1 | 8.566 | 51.649 | 0.026 | 0.065 | 0.078 | 60.309 |
  | native K=3 | 6.900 | 96.379 | 0.044 | 0.278 | 0.018 | 103.548 |

  Reading vs the 3way `31.6 + 20.1*K` fit: the marginal is VERIFY, almost alone — verify
  22.41 ms/K between the two DFlash2 points (total marginal 22.57), draft FLAT across K
  (~8.6 ms/round, the constant per-round drafter cost by design), accept 0.026-0.044 ms
  (the port-1 accept DtoH + host argmax is GONE from the accept phase), roll+maint <0.3.
  Verify prices at ~24-26 ms per row (K=1: 51.6/2 rows; K=3: 96.5/4 rows) ~= one plain
  step per row — the sequential per-row verify walk is the remaining structure; the lane's
  named next lever (MLA fa-rows batched verify) is exactly this term. Loop-law 0/15.
- CELL 3 DONE — THE FLIP TABLE (timed, marker held, interleaved x3 per the owner
  protocol; X3 SUFFICIENT — no escalation rule fired; within-arm decode-median spreads
  0.011-0.051%, all |gaps| >> 2x pooled spread; `receipts/c3/flip_check-x3.txt`):

  | arm | dec tok/s | ratio | deep tok/s | pool TTFT | TTFT@0.4k | TTFT@3.7k | vendor tok/s | verdict |
  |---|---|---|---|---|---|---|---|---|
  | plain | 35.408 | 1.0 | 30.00 | 0.362 | 0.422 | 2.208 | 33.55 | (==3way 35.41) |
  | DFlash2 K=1 | 35.043 | 0.9897x | 28.62 | 1.433 | 1.523 | 3.978 | 34.29 | NO-FLIP |
  | DFlash2 K=2 | 34.474 | 0.9736x | 28.21 | 1.456 | 1.492 | 4.177 | 36.60 | NO-FLIP |
  | DFlash2 K=3 | 31.919 | 0.9015x | 26.19 | 1.602 | 1.772 | 4.601 | 32.06 | NO-FLIP |

  Round walls (tok_cyc/dec): K1 52.49 ms, K2 71.31, K3 91.08 (3way: 52.6/71.6/91.6);
  fit ~33.0 + 19.3*K vs the 3way 31.6 + 20.1*K. The ports bought 0.11/0.29/0.52 ms at
  K=1/2/3 — the marginal (KDA diet share) moved ~0.3-0.8 ms/K-point at the higher Ks, but
  the K=1 FIXED term moved only ~0.11 ms of the 0.67 needed: K=1 still 0.56 ms short of
  the 51.93 tie bar. Acceptance is bit-identical to the 3way (acc/cyc 0.839/1.458/1.907),
  so ONLY loop time moved — a clean A/B of the ports. One vendor row excluded by the
  128-token floor (k2-1, ct=97, named in flip_check). Loop-law 0/168. Where the wall
  lives (cell-2 shares agree): verify ~24-26 ms PER ROW ~= one plain step per row — the
  sequential per-row verify walk is the whole remaining structure; the named next lever
  is the MLA fa-rows batched verify (deliberately out of this port).
- CELL 4 DONE — native TTFT spot receipt (one boot, marker held): TTFT@3.7k cold
  **4.153s vs the 3way's 12.303s** — fold-in A (batched MTP-plane warm fill) KILLED the
  O(prompt) sequential plane warm (spec-battery flip condition 1, closed with receipts).
  ttft@0.4k 1.691 (was 2.299), pool TTFT 1.786, dec 28.06 (was 27.49), deep 22.40. The
  residual +1.9s at 3.7k is the per-session setup + ctx-ingest class DFlash2 carries, no
  longer O(prompt). Gates GREEN, loop-law 0/14. Receipts `receipts/c4/`.
- CELL 6 NOT FIRED: its gate is "a spec arm beats plain in the flip table" — none did.
- CELL 5 DONE — PMIN tau ladder, glm5's tau PRICED (DFlash2 K=3 base, control = the
  cell-3 K=3 arm at 31.92; timed rows under the held marker; `receipts/c5/`):

  | tau | dec tok/s | deep | drafted/round | acc/cyc | accrate | eff wall ms |
  |---|---|---|---|---|---|---|
  | (off) | 31.92 | 26.19 | 3.000 | 1.907 | 0.636 | 91.08 |
  | 0.3 | 32.19 | 26.11 | 2.921 | 1.900 | 0.650 | 90.08 |
  | 0.5 | 33.47 | 26.96 | 2.644 | 1.833 | 0.693 | 84.62 |
  | 0.7 | 33.79 | 28.01 | 2.306 | 1.710 | 0.742 | 80.21 |

  Monotone UP with tau in the measured range: truncation rides DOWN the 33.0+19.3K line
  faster than it sheds acceptance (the port-2 design claim, confirmed on the real
  artifact). But the best tau arm (+1.87 over no-PMIN K=3) still sits UNDER K=2 (34.47)
  and K=1 (35.04) — consistent with the smallest-K optimum; tau rescues large-K arms, it
  does not create a flip. Zero-draft arithmetic says PMIN0 cannot flip K=1 either: a
  gated round still pays draft (~8.6 shares) + one verify row (~26) > one plain step
  (28.2), so no gating configuration beats plain while a verify row costs a plain step.
  Cross-tau greedy tapes 28/28 identical (the gate moves DRAFTS, never output); armed
  boot lines banked (`receipts/c5/c5/armed-lines.txt`); loop-law 0/84. If glm5 spec ever
  ships, carry MEMRA_SPEC_PMIN~0.5-0.7 with the smallest K, on the automatic policy.

## VERDICT (the flip decision)

**NO-FLIP.** On the loop-ported head bb8d9e3cc, on the deployed 3-card serving shape,
no spec sub-arm beats plain: K=1 0.9897x, K=2 0.9736x, K=3 0.9015x (x3 interleaved,
spreads 0.011-0.051%, escalation rules armed and not fired). Keep `MEMRA_GLM5_SPEC` +
`MEMRA_GLM5_DFLASH` **default OFF** (the loop's rollback seam, still the prod default).

What the ports DID buy, receipted:
1. Correctness held: 12/12 served byte-identity re-gate, cross-tau 28/28, loop-law 0/281
   across the window — the four ports moved time only.
2. The native O(prompt) TTFT defect is DEAD (fold-in A): 12.303s -> 4.153s at 3.7k cold.
   Spec-battery flip condition 1 closed with receipts.
3. Round walls moved 52.6/71.6/91.6 -> 52.49/71.31/91.08 ms (K=1/2/3): ~0.3-0.8 ms/K off
   the marginal (the KDA-diet share) but only ~0.11 ms off the K=1 fixed term against the
   0.67 needed — the predicted accept-DtoH/tap-sync savings did not price at 0.3-1.1 ms
   on this box's round; the phase shares say why: the wall was never sync-bound, it is
   VERIFY-ROW-bound (~24-26 ms per row ~= one plain step per row, K+1 rows per round).
4. Confidence gating works as designed and is priced (tau ladder above).

Flip condition, restated sharper than the 3way's: the verify walk must stop paying one
plain-step per row — the MLA fa-rows BATCHED verify (T-parallel verify rows in one
kernel class, the lane's deliberately-deferred item) is now the ONLY named lever with
the size to close a 0.56 ms (K=1) / 3.4 ms (fixed-term) gap. Sync diet and confidence
gating are done and banked; re-run cells 3+4 of THIS window unchanged when that lands.

Window totals: 22 boots (3 c1 + 3 c2 + 12 c3 + 1 c4 + 3 c5), 0 boot failures, all gates
green, 40/40 byte-identity tapes, loop-law 0 flagged of 281 tapes screened, 1 vendor row
excluded by the 128-token floor (named). Cell 6 not fired (gate: a spec arm beats plain).
