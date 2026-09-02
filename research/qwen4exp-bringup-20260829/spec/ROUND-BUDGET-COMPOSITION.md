# Is 200+ tok/s structurally reachable on one card? (banked-receipt analysis, 2026-09-01)

> **CORRECTED 2026-09-01 by `../ep2/EP2-DESIGN.md`.** This file called the 15.1 ms flat term a
> "weight-read floor" and sized two-card EP2 as halving it. The EP2 lane's dispatch audit shows
> the sel (routed-expert) section scales with SLOT COUNT, not bytes, and is 22.9-26.4% of the
> round: EP2 halves ~5.6% of the flat term and ~32% of the 2.10 ms/row slope — a SLOPE lever,
> not a floor lever — so its two-card ceiling is 154-180 tok/s (centre ~157), not 200+. The
> flat term is dense-trunk + attention work. The decomposition table and the per-K arithmetic
> below still hold; the mechanism paragraph and the "EP2 is the precondition" verdict do not.


Data source: the mtp11 K-ladders already in this repo
(`spec/mtp11/ab-defer-k{1,2,3,4,5,6,8}-m11-ladder-raw.tsv`, host arm, raw shape,
single-card route, ckpt q48fn-nvfp4, binary acd01bd5). No new measurement — this file
is arithmetic over rows that carried the full receipt headers when they were cut.

## The per-K decomposition

| K | chain ms | verify ms | accept_len | round ms | ms/token | tok/s |
|---|---|---|---|---|---|---|
| 1 | 0.97 | 19.27 | 1.97 | 20.24 | 10.28 | 97 |
| 2 | 2.35 | 22.55 | 2.81 | 24.90 | 8.86 | 113 |
| 3 | 3.60 | 25.41 | 3.51 | 29.00 | 8.26 | 121 |
| 4 | 4.62 | 28.76 | 4.00 | 33.37 | 8.34 | 120 |
| 5 | 5.30 | 30.31 | 4.41 | 35.61 | 8.07 | 124 |
| 6 | 5.69 | 31.59 | 4.57 | 37.29 | 8.16 | 123 |
| 8 | 6.53 | 34.00 | 5.02 | 40.53 | 8.07 | 124 |

## The model the table pins

- **verify(t) ≈ 15.1 ms flat + 2.10 ms/row** (least-squares over t=2..9). The flat
  15.1 ms is the single-card weight-read floor — it matches the measured plain t=1
  trunk step (14.86 ms) because the verify chunk reads the same weights once
  regardless of t (verify_mt weight-sharing, the fact that killed vfuse).
- **chain ≈ 1.06 ms/draft step** (already cheap; FR-Spec's draft-cost half has almost
  nothing to buy here, unlike q38 where the draft was the expensive leg).
- **acceptance decays**: per-position ~0.86 at the front, worse deeper —
  accept_len grows 1.97 → 5.02 while K grows 1 → 8. This decay is why tok/s
  plateaus at ~124 on this shape: each extra K costs 3.16 ms (2.10 verify + 1.06
  chain) and buys ~0.45 accepted tokens at the margin ≈ 7 ms/token, the current
  average — the ladder IS its own fixed point.

## What each lever can and cannot do (ceilings, not estimates)

- **Perfect acceptance** (accept_len = K, unattainable): ms/token = 3.16 + 15.1/K
  → 198 tok/s at K=8, 213 at K=10. So even a perfect draft only *touches* 200 —
  the flat floor is that dominant.
- **selgroup** (receipted; default ON since 2026-09-02, box cells: +3.0-4.0% K=5, flat at depth): cuts the 2.10 slope's MoE
  share, ceiling 6.42% end-to-end → ~145 tok/s. Real, worth landing, not the gap.
- **Accept-side gains** (FR-Spec-style masking, guard tuning): at a plausible
  per-position 0.93, accept_len(K=10) ≈ 7-8 → ~160 tok/s. Also not the gap.
- **The floor itself is the gap.** 200+ needs the 15.1 ms weight-read floor roughly
  halved, and the only mechanism this architecture offers is real two-card weight
  parallelism — the EP2/TP2/PP2 registration lane (memra issue #70), where each card
  reads half the expert bytes concurrently. The existing TP2 route does NOT deliver
  it at these shapes (TP2 short-spec rows sit at the same ~120-136 as single-card;
  PCIe allreduce eats the split — see TP2 join-diet receipts).

## Verdict

**Single-card 200+ is structurally out of reach on this artifact** (flat floor +
acceptance decay bound every composition below ~160-180 even stacking all receipted
levers). The receipted path to the owner's 200+ target is:
1. EP2 across two cards (memra #70; copy the glm/hy3 placement work; co-activation
   traces already wired for the placement map) → floor toward ~8-9 ms;
2. on top of that, the same K=8-10 + accept-side stack (then 200+ needs only
   accept_len ≈ 5-6, which K=8 already delivers today).
With the floor halved and today's measured K=8 numbers untouched:
(7.5 + 2.10×9 + 6.53) / 5.02 ≈ 6.6 ms/token ≈ 152 — and with selgroup + modest
accept gains ≈ 190-210. Two cards is not an optimization here; it is the target's
precondition, matching the owner's original "need to see if we can afford it on two
cards" framing — 262k CONTEXT fits one card, but 200+ THROUGHPUT does not.
