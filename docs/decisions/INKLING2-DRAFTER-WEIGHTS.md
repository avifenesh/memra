# Inkling 2 Mini drafter: weight format and output head

Date: 2026-11-22. Lane S (DSpark drafter of Inkling 2 Mini). Receipts: `research/modelplan-onboarding-inkling2mini-20261121/drafter/quant-cpu/`, `quant-gpu/`, `timed-graph-taps-rig-heads/`.

## Chosen

- The drafter's projections, markov_w1, rnn_proj and the fold stay BF16 as shipped (the fold in f32), on both classes.
- On nvfp4-rig, the drafter reads the class's shared NVFP4 unembed for its base logits and its fold. It does not load a separate BF16 head.

## Rejected

- NVFP4 drafter weights (lane M's class quantizer), per component and all together.
- The FP8 control arm (e4m3 with one scale per row), per component and all together.
- A separate BF16 output head beside the NVFP4 unembed (+0.92 GB).

## Measurement

GPU per-component sweep:

- Setup: RTX 5090, clampcheck OK, no void telemetry sample, declared class.
- Each component's drafter was loaded alone, and its draft time at K = 1..7 was measured with `--bench` reps.
- Acceptance: K = 7 on 12 prompts of 64 tokens, greedy and T 1.0. Every greedy run equals plain greedy.

Draft time at K = 7 is 5.88 ms in BF16. Every NVFP4 component is as fast or slower:

| arm | draft time at K = 7 | change |
|---|---|---|
| BF16 | 5.88 ms | |
| every NVFP4 component together | 8.13 ms | +2.25 ms |
| the NVFP4 fold alone | | +0.86 ms |
| every FP8 component together | | +1.13 ms |

The best single NVFP4 component saves 0.17 ms, inside run noise. The drafter's GEMVs are latency-bound at these shapes: a BF16 projection reads about 14 MB, under 10 µs at the card's bandwidth. Fewer bytes buy nothing.

Acceptance is unchanged within noise:

- NVFP4 all: 2.790 vs 2.776 accepted per K = 7 verify greedy, and 2.311 vs 2.262 sampled. Single components move it by -0.056 to +0.019.
- CPU study (simulated K = 7 over 12 prompts, KL against the BF16 drafter on the same taps): every layer component is within ±0.035 of 2.032. KL is 8e-5 to 5e-3.

The greedy cumulative set, chosen by acceptance lost per millisecond saved, is empty: no component saves time. End-to-end the quantized drafter can only be slower. The draft is 3 to 8 ms of a 20 to 72 ms speculative step. Even a kernel that halved it would buy at most a few percent.

Output head on nvfp4-rig (run-spec 28/28 with both heads):

| head | gate accepted / steps | adaptive tok/s vs plain (two runs) |
|---|---|---|
| shared NVFP4 unembed | 1238 / 542 | 1.101 / 1.093 |
| separate BF16 head | 1239 / 542 | 1.097 / 1.084 |
| declared class, for reference | 1254 / 532 | |

The heads are equal, so the shared head wins on memory.

## Revisit when

- A drafter GEMV fuses several projections or reaches bandwidth at these shapes.
- Memory, not speed, binds the drafter on a smaller card. NVFP4 would save about 0.5 GB of drafter weights at no measured acceptance cost.
