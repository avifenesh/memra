# Composition-lane VERDICT: the composed TP route does NOT reach the 100 tok/s bar

> FRAMING SUPERSEDED 2026-09-01 (owner pivot): the 100@262k bar is RETIRED as a launch
> gate — the qualified 3-card PP3 shape serves. Every MEASUREMENT below stands; the bar
> framing is historical. The current reading (upgrade path) and the follow-on-arc handoff
> live in `B200-TRANSFER.md` + the LANE.md close entry.

Charter: prove or refute the composed route to the owner's serving bar — 100 tok/s decode
for GLM-5.3-Flash — by composing TP × speculative decode (DFlash2) × the matvec doors.

**VERDICT: REFUTED as composed.** Measured composed decode on the qualified 2-card shape is
**29.684 tok/s** (interleaved x3, spread 0.195%). The bar is 100. Worse for the route: the
composed TP shape lands **BELOW the already-shipped 3-card PP3 + spec configuration** (banked
45.654 tok/s greedy / 46.66 vendor-sampled, flip-reprice-20260831), so composing TP does not
merely miss the bar, it does not beat what the fleet can serve today.

The composition itself is CORRECT and it does pay on the TP shape (spec is worth 1.151x
there vs 1.010x on one card). The route fails on the TP BASE, not on the composition.

## The measured chain (one box, one binary, interleaved x3, real artifact)

Box: a shared 2-card dev host — 2x RTX PRO 6000 Blackwell **Server Edition**, driver
595.91.07, 600 W class (provider/instance identity lives in the private ops repo), host ST
0.66-0.74 s (ABOVE the 0.6 s bar — every ABSOLUTE row below carries the host-class caveat;
the RATIOS are within-box and unaffected). memra @ ca5e7cd48 / 83b9f0f28 lineage, artifact
`Avifenesh/GLM-5.3-Flash-NVFP4` with all 20 shards sha256-verified against
`hf-publish-receipts/SHA256SUMS.txt`, drafter `incoai/GLM-5.3-Flash-DFlash2` @ b33c0347.
Prompts: the house decode-attribution pool (6 real prompts), max_new 200, doors pinned
(`MEMRA_RP=0 MEMRA_MLA_TC_PREFILL=0`), `NVIDIA_TF32_OVERRIDE=0`.

### W2-T plain decode, x3 interleaved (`box-receipts/w2t-*`)

| arm | boot medians | aggregate | spread | vs 1-card |
|---|---|---|---|---|
| plain1 (1 card, streaming experts) | 19.53 / 19.53 / 19.53 | **19.529** | 0.036% | 1.000x |
| tp2 host-canonical | 19.79 / 19.75 / 19.73 | **19.754** | 0.278% | 1.012x |
| tp2 peer-pull | 22.90 / 22.91 / 22.86 | **22.902** | 0.210% | 1.173x |
| tp2 peer-pull + EP diet | 25.81 / 25.74 / 25.78 | **25.784** | 0.246% | 1.320x |

Bare host-canonical TP-2 buys ~nothing (1.012x — the banked "TP-2 v1 bare does not pay"
verdict reproduced on a second card class). peer-pull is worth **1.159x** over it and the EP
dispatch diet a further **1.126x**; together **1.305x**. Vendor-default sampled rows track
greedy within 0.4% on every arm.

The transport lane's PREDICTION MISSED, in both dimensions, and is recorded as a miss: it
projected TP-2 peer-pull at 29.0-35.7 tok/s engine-twin (measured 22.902) and transport worth
~1.4-1.6x on the TP-2 arm (measured 1.305x for peer-pull AND diet together). Future transport
budgets use the measured ratios.

### W2-S composed spec rows, x3 interleaved (`box-receipts/w2s-*`)

| arm | boot medians | aggregate | spread | tok/cyc | acc | vendor |
|---|---|---|---|---|---|---|
| spec1 (1 card + DFlash2) | 19.72 / 19.72 / 19.72 | **19.720** | 0.028% | 2.746 | 0.582 | 19.34 |
| **stp2 (spec × TP-2 peer-pull + diet)** | 29.71 / 29.68 / 29.66 | **29.684** | 0.195% | 2.921 | 0.640 | 27.82 |

K ladder (single boot): K=1 spec1 20.905 / stp2 29.481; K=3 as above; K=5 spec1 16.271 /
stp2 25.081 — K=3 is the peak on both, matching the banked K-curve shape.

- **spec × TP composition multiplier: 1.151x** (29.684 / 25.784) — spec pays far better on
  the TP shape than on one card (1.010x). MECHANISM NOT ESTABLISHED: the obvious "an
  expensive base amortizes better" story runs BACKWARDS here (the TP walk is *cheaper* per
  token, 38.8 ms vs the one-card 51.2 ms), so this stays an observation with a named
  follow-up, not an explanation.
- Acceptance is NOT identical across the two shapes on this box: one-card tok/cyc 2.746 /
  acc 0.582 vs sharded 2.921 / 0.640 — so part of the 1.151x is acceptance, and an
  acceptance-parity gate is still owed on any sharded serving arm. What IS byte-exact is the
  OUTPUT (composed tapes 6/6 identical to plain sharded, below). The sharded 2.921 happening
  to sit beside the deployed 3-card head's 2.907 is a cross-box coincidence, not evidence of
  sharding invariance.
- Engagement proven from the log, never boot-trust: `[glm5-spec] spec x TP composition
  ARMED`, `[glm5-tp-transport] armed transport=peer-pull` + ladder PASS
  (`byte_ladder=[16384, 65536, 1048576, 67108864] mismatches=0 same_device_gate=false` — a
  REAL fabric, unlike the rig's emulation), `[glm5-tp-ep] verify rows ride the SEQUENTIAL EP
  walk (t=4)`.

### Correctness on the real artifact (`box-receipts/G1-IDENTITY-SUMMARY.txt`)

- W2-G0: `glm5-tp-gate` **ALL ARMS PASS on box silicon** (both fixtures, both rank counts).
- W2-G1 plain-vs-TP-2, teacher-forced, class metrics: worst norm_rel **2.715e-01** on prime
  rows, **0 argmax forks over every prime and 32 decode-step dumps** — the pre-registered
  EP-band class (the banked band is 3-5e-2/layer saturating; deep primes amplify through the
  recurrent trunk exactly as `QUIRK:glm53:tp2-class-gate-band-ep-owned` says).
- Transport twin: TP-2 host-canonical vs peer-pull **44/44 files byte-identical** (every
  logit dump, every tape) on real fabric — the transport moves bytes, it does not compute.
- Composed spec tapes vs plain TP tapes: **6/6 byte-identical**. The composition emits
  exactly what the plain sharded walk emits.

## Why the route cannot reach 100, with the arithmetic

Best composed measurement: **29.684**. The remaining named multipliers:

- **TP-4 over TP-2: 1.03-1.05x** (the transport lane's own prediction, driver-primitive
  bound: ~3750 primitives/token eat the 1.15x memory-system gain). Even at a generous 1.10x
  this is **~32.7**.
- **matvec doors T/X/K/W + D/H: 1.146x measured** (1.1288 x 1.0154; the charter's 1.25 does
  not exist — moe-loc refuted the premise for the MoE half). And on the composed shape the
  doors' MoE arm is preempted: the EP walk owns verify-width MoE, so the door term is
  optimistic here. Applying it anyway: **~37.5**.
- The bar is **100**. At the lane's own quoted TP-4 arm (1.05x) the ceiling is **35.7**; the
  ~37.5 above needs the generous 1.10x. Either way the gap is **2.7-2.8x with every named
  lever spent**, and the doors term is optimistic because the EP walk owns verify-width MoE.

And the comparison that ends the route: the fleet already serves **45.654 tok/s** (3-card
PP3 + DFlash2 + PMIN 0.7, flip-reprice-20260831) and holds **71.49** as best-single-stream
(struct-battery, doors + D/H). The composed TP shape at 29.684 is **0.65x of the shipped
spec configuration**. TP is the wrong axis for this model on this fabric: its base sits
~1.0-1.3x of one card because the trunk is join-bound, while PP-3 keeps whole layers local.

At ctx 262144 the verdict only strengthens — decode slows with depth, and the banked 2-card
262k cell already refused the shape (workspace OOM above ~8k prompt tokens, "NOT a 262k
SKU"). No 262k row can rescue a shape that is 2.7x short at 2k.

## What this lane BANKS as reusable value

1. **TP-4 exists and is gated** (PR #78, merged): the rank envelope is [2, 4], every arm
   byte-identical at four ranks, own calibrated quad band 4e-3 (measured 4.013e-4).
2. **spec × TP is wired, gated and default-OFF** (PR #80, merged): `MEMRA_GLM5_SPEC_TP`, the
   per-rank verify/rollback contract, and the first receipts that the composition is
   byte-exact AND profitable on the TP shape (1.151x).
3. **The transport re-price the tp-transport lane asked for** (this doc): peer-pull is worth
   1.159x and the EP diet 1.126x on real fabric, with byte identity across the swap. Those
   two doors are now MEASURED, not predicted — they belong to any future TP work.
4. **A named engine lever with a price tag**: the EP walk preempts the batched vrows MoE pair
   at verify widths, so the composed round re-inherits the sequential vrest wall (~45.6 ms of
   the unsharded K=3 round the vrows lane removed). An EP-aware vrows arm is the single
   biggest remaining term on the composed shape.

## Recommendation

- **Do not spend the 4-card on-demand box on this route.** The 2-card measurement plus the
  TP-4 prediction bound the composed ceiling at ~37 tok/s; the owner's spend would buy a
  confirmation of a refutation. If a 4-card box is rented for other reasons, W4-B/W4-S are
  pre-registered in `box/CELLS.md` and cost ~1 h.
- **The 100 bar needs a different axis.** On the evidence: the shipped PP3 + spec + doors
  shape at 71.49 is the live frontier; the named unlocks with receipts behind them are the
  EP-aware vrows arm (composed and unsharded), CUDA-graph capture of the decode round, and
  the batched-EP prime. TP's remaining upside (peer-pull + diet, 1.305x) applies to a base
  that starts 40% below PP-3.
- **Keep `MEMRA_GLM5_SPEC_TP` default OFF.** It is correct, gated and now priced; nothing
  about these numbers argues for exposing it to serving.
