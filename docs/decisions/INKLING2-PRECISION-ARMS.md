# INKLING2-PRECISION-ARMS: f32 where it does not move the model, f64 where it does (2026-11-21)

Scope: the Inkling 2 Mini GPU route (crates/memra-engine/src/inkling2_gpu.rs, cu/inkling2_gpu.cu), all
three classes (shipped, nvfp4-rig, float). The #1397 route ran every norm, clip, mix, controller,
convolution, router, epilogue and attention reduction in f64. That precision was the route's own choice,
not part of the checkpoint's program, so it is a numeric decision measured by KL (owner: "the kl matter").

## Chosen

The route's program (Prec::default()):

| Group | Program | Why |
|---|---|---|
| controller | f32 | round 1 |
| norms and clips (embed and RMS norms, q / k / r clip, tau, value norm) | f32 | round 1 |
| convolutions | compensated f32 (f32c) | round 4, with the router and attention below |
| router | compensated f32 (f32c) | round 4 |
| attention | f32 tiles of 256 keys (scores, max, exp, value sums), the tiles merged in f64 (f32t) | round 4: 32k prefill 1.7x, 32k decode 1.5x over f64g |
| depth mix, epilogues (output gate, softcap) | f64 | mix f32 fails with the set (round 2); the epilogue was backed off (round 1) |

History: round 1 adopted {controller, norm, f64g} (the epilogue backed off); round 4 replaced f64g with
f32t and added conv and router under the revised gate below.

Prec::all_f64() is the #1397 program; ink2-gpu --prec all=f64 runs it (the seam is bit-identical to the
merged route before adoption: digests equal on all 48 slots and the logits). Every measurement below is
against it.

## Rule (coordinator, revised 2026-11-21)

First rule: p3 8k mean KL <= 1e-5 against f64 and in-distribution dNLL not worse. Revised after the
measurements: mean KL <= 1e-4 on p3 at both chunk sizes (512 and 1024), AND in-distribution dNLL not worse
(rawdoc and chatwrap sequence-bootstrap 95% CIs; a CI entirely below 0 means the arm predicts better and
passes), AND p2 argmax equal. Reason: the 1e-5 bound is stricter than the noise it measures. p3 KL at
chunk 1024 is set by one cluster of router near-tie flips: every perturbing arm, in unrelated groups too,
lands on the same 2.54e-5 with 69 to 300 of 396,480 route rows different. It measures the tie set, not a
precision loss. The revised bound matches the FP8 KV adoption (<= 1.6e-4 per slot).

## Measurements (float class unless named; receipts research/modelplan-onboarding-inkling2mini-20261121/perf/arms/ and cum/, per-token JSON as .json.gz)

Single arms, p3 (8260 tokens) mean KL vs the all-f64 route at chunk 512 (corpus mode) and vs the f64
oracle tape at chunk 1024 (--score; the all-f64 route itself reads 8.2e-13 there), route rows equal at
chunk 1024, rawdoc and chatwrap dNLL (real positions):

| Arm | p3 KL c512 | p3 KL c1024 | route rows equal c1024 | rawdoc dNLL [CI] | chatwrap dNLL [CI] | p2 KL / argmax |
|---|---|---|---|---|---|---|
| controller f32 | 6.0e-7 | 2.54e-5 | 396411 / 396480 | -9.9e-6 [-3.0e-5, -4.5e-9] | -1.3e-6 [-3.8e-6, 1.1e-8] | 1.9e-12, 288/288 |
| controller f32c | 6.0e-7 | 2.54e-5 | 396411 | -1.4e-5 [-3.8e-5, -3.3e-9] | -7.2e-9 [-2.6e-8, 1.0e-8] | 1.7e-12 |
| norm f32 | 3.8e-6 | 2.54e-5 | 396384 | -2.9e-4 [-8.6e-4, -4.6e-9] | -2.3e-4 [-6.9e-4, 1.3e-5] | 7.2e-12, 288/288 |
| norm f32c | 2.4e-2 | 2.13e-3 | 350866 | +5.4e-6 [-2.9e-5, 4.6e-5] | +6.9e-6 [-4.3e-8, 1.7e-5] | 1.7e-12 |
| mix f32 | 4.2e-5 | 2.54e-5 | 396177 | -4.0e-6 [-1.2e-5, 2.7e-7] | +1.8e-5 [-1.1e-8, 5.4e-5] | 6.7e-12 |
| mix f32c | 3.0e-2 | 1.40e-3 | 349986 | +9.8e-8 | +9.8e-6 | 2.0e-12 |
| conv f32 | 2.3e-2 | 5.24e-4 | 357350 | -1.4e-5 | +1.4e-6 | 2.4e-12 |
| conv f32c | 4.2e-5 | 2.54e-5 | 396177 | +1.3e-6 | +1.4e-6 | 2.2e-12 |
| router f32 | 2.9e-2 | 1.02e-3 | 350926 | -2.7e-4 | +7.9e-6 | 2.6e-12 |
| router f32c | 4.1e-5 | 2.54e-5 | 396178 | -2.9e-4 | +8.2e-6 | 4.8e-12 |
| epilogue f32 | 6.3e-7 | 2.54e-5 | 396409 | -1.4e-5 [-3.6e-5, 2.7e-8] | +2.2e-5 [-2.0e-8, 6.7e-5] | 7.8e-12 |
| attention f32 | 2.6e-5 | 2.44e-5 | 396265 | +5.3e-6 | +6.2e-6 | 6.4e-12 |
| attention f32c | 3.3e-3 | 1.51e-3 | 383119 | -2.9e-4 | +5.5e-6 | 2.1e-12 |
| attention f64g | 4.7e-14 | 7.7e-13 | 396480 | +2.7e-9 [-5.9e-10, 7.7e-9] | -4.6e-10 [-1.1e-9, 0] | 2.2e-12 |

The route is deterministic: three all-f64 runs of p2 and the all-f64 arm on chatwrap give KL exactly 0.
KL in both corpora is a few discrete outcomes: arms in unrelated groups land on the same values to four
digits (the same fragile positions flip), so a single-arm KL says which tie set it crossed, not how much
precision it lost.

Cumulative sets (p3 c512 / c1024 KL, route rows equal c1024):

| Set | p3 c512 | p3 c1024 | rows equal | rawdoc dNLL [CI] | chatwrap dNLL [CI] |
|---|---|---|---|---|---|
| controller + norm + epilogue + f64g | 1.14e-2 | 9.5e-4 | 371186 | -4.6e-5 [-1.3e-4, 1.9e-7] | -4.6e-5 [-1.7e-4, 2.0e-5] |
| norm + epilogue + f64g | 1.13e-2 | 9.5e-4 | 371390 | | |
| controller + epilogue + f64g | 6.0e-7 | 2.54e-5 | 396411 | | |
| **controller + norm + f64g (adopted)** | 6.5e-7 | 2.44e-5 | 396389 | -2.9e-4 [-8.5e-4, 7.6e-9] | -2.3e-4 [-6.9e-4, 1.3e-5] |

The adopted set on p2: KL 2.2e-7, argmax 288/288, dNLL -1.7e-5 [-5.1e-5, -1.2e-7]. Its in-distribution KL
is 1.7e-4 (rawdoc) and 7.2e-5 (chatwrap), top-1 agreement 99.88% and 99.97%; both dNLL means are negative
and both CIs reach 0 only at the 1e-5 level, so it is not worse.

The full set of four crossed the 1e-2 cluster at both chunk sizes; leave-one-out puts it on norm plus
epilogue together. The epilogue arm (the fewest kernels, the least time) was backed off.

## Rejected

- mix f32c, norm f32c, conv f32, router f32, attention f32c: the 1e-3 cluster at chunk 1024.
- epilogue f32: passes alone, but with norm f32 the set lands in the 1e-2 cluster.
- mix f32, conv f32c, router f32c, attention f32: pass at chunk 1024 (2.5e-5) but not adopted in this round;
  they stay f64 until measured inside the adopted set.
- An f64-to-f32 conversion by field moves (bits equal, 3.9% slower) and a controller with its norms formed
  once (0.8% slower): deleted (perf/README.md).

## Round 2: the remaining arms inside the adopted set (2026-11-22)

Same rule. Each set below is {controller f32, norm f32} plus the arms named; attention f32t is arm 4
(f32 scores, max, exp and value sums in tiles of 256 keys, the tiles merged in f64; a flash-decode
split). p2 argmax is 288/288 for every set. Receipts: perf/set2 and perf/set3.

| Added arms | p3 KL c512 | p3 KL c1024 | route rows equal | rawdoc dNLL [CI] | chatwrap dNLL [CI] | verdict |
|---|---|---|---|---|---|---|
| f64g, mix f32 | 4.2e-5 | 2.54e-5 | 396176 | -1.4e-5 [-3.8e-5, -2.7e-9] | +1.7e-5 [-4.5e-9, 5.0e-5] | pass |
| f64g, conv f32c | 6.4e-7 | 2.55e-5 | 396407 | -1.3e-5 [-3.4e-5, 1.3e-8] | -9.4e-5 [-3.0e-4, 1.6e-5] | pass |
| f64g, router f32c | 4.2e-5 | 2.39e-5 | 396163 | -1.4e-5 [-3.8e-5, 1.8e-7] | -2.2e-4 [-6.9e-4, 2.3e-5] | pass |
| attention f32 | 6.8e-5 | 2.43e-5 | 396031 | +5.3e-6 [-3.0e-5, 4.5e-5] | +2.2e-5 [-6.5e-6, 6.9e-5] | pass |
| attention f32, mix, conv, router | 1.7e-3 | 2.67e-5 | 393610 | | | fail (p3) |
| attention f32, conv f32c | 2.7e-5 | 2.44e-5 | 396253 | -3.0e-5 [-8.3e-5, -1.5e-8] | +2.8e-5 [9.2e-9, 7.6e-5] | fail (chatwrap worse) |
| attention f32t | 1.1e-2 | 9.5e-4 | 371374 | | | fail (p3) |
| attention f32t, conv f32c | 1.3e-2 | 1.1e-3 | 371031 | | | fail (p3) |
| attention f32t, conv f32c, mix f32 | 1.6e-3 | 2.70e-5 | 393746 | | | fail (p3) |
| attention f32t, conv f32c, router f32c | 6.1e-7 | 2.44e-5 | 396392 | -1.0e-5 [-3.0e-5, 2.5e-8] | -1.4e-4 [-3.8e-4, 3.1e-6] | pass |
| f64g, mix f32, conv f32c, router f32c | 4.2e-5 | 2.54e-5 | 396176 | -9.8e-6 [-3.0e-5, 2.3e-8] | +4.0e-6 [-3.8e-6, 1.6e-5] | pass |

The outcome is not monotone in precision: attention f32t fails alone and passes with conv and router
in f32 as well. Every set lands on one of three discrete levels (about 6e-7, 2-7e-5, or 1e-3 to
1e-2 at chunk 512), set by which near-tie router positions of p3 the perturbation crosses. The rule
selects among the arms, but a passing set has no margin beyond the tie set it happened to miss.

## Round 3: replication on two more long prompts, and the route-flip rate (2026-11-22)

Two 8448-token prompts built from the corpus documents (not p3): L1, six code and documentation
documents; L2, fifteen multilingual chat and tool documents. Each set runs against the all-f64 route in
corpus mode. Flips: route rows (slot, position) whose expert set differs from the all-f64 route's, per
100k route rows: a continuous measure beside the KL. Receipts: perf/set4.

| Set ({controller, norm f32} plus) | L1 KL / flips / dNLL | L2 KL / flips / dNLL | p3 KL / flips / dNLL |
|---|---|---|---|
| attention f64g (the program now) | 1.2e-2 / 2540 / -1.8e-3 | 6.1e-10 / 0.5 / +3.4e-9 | 6.5e-7 / 23 / -4.2e-7 |
| attention f32 | 3.0e-2 / 8823 / -9.3e-3 | 9.1e-5 / 394 / +2.9e-5 | 6.8e-5 / 113 / -1.9e-4 |
| attention f32t, conv f32c, router f32c | 3.2e-2 / 8947 / -1.5e-2 | 5.6e-7 / 0.7 / -8.1e-6 | 6.1e-7 / 22 / -4.4e-6 |

- Corpus mode at chunk 512 and 1024 is bit-equal for every set: the route is chunk-invariant by
  construction (the calls-of-8 vs calls-of-1 cells). The second chunk size matters only against the CPU
  f64 tape, which exists for p3 alone.
- L1 fails the 1e-4 bound for every set, the program now adopted included. Once a long code sequence
  diverges, a perturbation reaches 2.5 to 9% of route rows. Every set's dNLL on L1 is negative (the
  perturbed program predicts the text better than the all-f64 route). Over 8k tokens, KL against f64
  measures how far a perturbation spreads through the router, not a loss of quality.
- Discrete outcomes: on a single long prompt a set lands on one of a few levels, set by which router
  near-tie positions it crosses, and the levels do not order with precision (attention f32t fails
  alone at 1.1e-2 and passes with conv and router in f32 at 6.1e-7). One pass on one prompt is weak
  evidence; the flip rate across prompts is the steadier measure.
- Per the coordinator's fallback (neither candidate passes everywhere) the route keeps
  {controller f32, norm f32, attention f64g}.

## Round 4: the gate revised, f32t + conv + router adopted (coordinator ruling 2026-11-22)

The gate from now on:
1. teacher-forced dNLL not worse on every corpus, with bootstrap CIs (over sequences, and over 64-token
   blocks of positions for a single long prompt); a CI entirely above 0 is worse;
2. short-context KL <= 1e-4 against the all-f64 route, where KL still measures precision: p1, p2 and the
   corpus documents under 1k tokens each;
3. the route-flip rate per 100k route rows, reported per corpus every time as the stability metric,
   with no hard bound.
Reason: over 8k-token prompts KL against f64 measures how far a perturbation spreads through the
router (round 3: the program adopted in round 1 reads 1.2e-2 on L1), so it cannot be the quality gate
there. p2 argmax must stay equal.

{controller f32, norm f32, conv f32c, router f32c, attention f32t} on the gate:

| Corpus | dNLL [95% CI] | KL | flips / 100k |
|---|---|---|---|
| rawdoc (15 x 1024) | -1.0e-5 [-3.0e-5, 2.5e-8] | 1.3e-5 | |
| chatwrap | -1.4e-4 [-3.8e-4, 3.1e-6] | 5.6e-5 | |
| L1 (8448, block bootstrap) | -1.5e-2 [-2.6e-2, -4.1e-3] | 3.2e-2 | 8947 |
| L2 (8448, block bootstrap) | -8.1e-6 [-2.4e-5, 5.5e-8] | 5.6e-7 | 0.7 |
| p3 (8260) | -4.4e-6 | 6.1e-7 | 22 |
| p2 | argmax 288/288 | 5.3e-11 | 0 |
| p1 | argmax equal | 2.2e-12 | 0 |

Every dNLL CI reaches or lies below 0 (L1: the arm predicts better), and the short-context KL is at
most 5.6e-5. The program adopted in round 1, for comparison: L1 flips 2540, L2 0.5, p3 23. The pins
of the adopted program (slot and logit digests, shipped / float / nvfp4-rig, p1 at chunk 1, 4 and 512, p2,
nvfp4-rig p3) are in perf/pin-a16; the program is chunk-invariant (digests equal at chunk 1, 4 and 8,
perf/s10-attention-rows/gr2-inv). Speed on the adopted program: perf/README.md.

## Speed

Shipped class, c=1, 32 tokens from p1, three interleaved takes, full clocks: all-f64 49.7 tok/s (p50 19.57
ms), the first adopted set (with the epilogue) 54.6 tok/s (p50 17.79 ms). The backed-off set is measured in
perf/README.md.
