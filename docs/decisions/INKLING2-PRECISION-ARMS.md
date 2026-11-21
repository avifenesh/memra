# INKLING2-PRECISION-ARMS: f32 where it does not move the model, f64 where it does (2026-11-21)

Scope: the Inkling 2 Mini GPU route (crates/memra-engine/src/inkling2_gpu.rs, cu/inkling2_gpu.cu), all
three classes (shipped, nvfp4-rig, float). The #1397 route ran every norm, clip, mix, controller,
convolution, router, epilogue and attention reduction in f64. That precision was the route's own choice,
not part of the checkpoint's program, so it is a numeric decision measured by KL (owner: "the kl matter").

## Chosen

The route's program (Prec::default()):

| Group | Program | Why |
|---|---|---|
| controller | f32 | passes alone and in the set |
| norms and clips (embed and RMS norms, q / k / r clip, tau, value norm) | f32 | passes alone and in the set |
| attention | f64, a chunk's scores split over blocks (f64g) | the same f64 program in another evaluation order; KL 1e-13, prefill attention about 4x faster |
| depth mix, convolutions, router, epilogues (output gate, softcap), attention arithmetic | f64 | see Rejected |

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

## Speed

Shipped class, c=1, 32 tokens from p1, three interleaved takes, full clocks: all-f64 49.7 tok/s (p50 19.57
ms), the first adopted set (with the epilogue) 54.6 tok/s (p50 17.79 ms). The backed-off set is measured in
perf/README.md.
