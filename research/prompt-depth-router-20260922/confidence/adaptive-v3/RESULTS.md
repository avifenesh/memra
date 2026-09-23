# Qwen K=3 live learned confidence: held-out native result

The earlier C=0/0.15/0.30 grid used preset cutoffs. This run tested a live cutoff derived from verified accepted prefixes and measured K-dependent cost.

## Complete-request comparison

| Arm | tok/s | Gain vs K=3/C=0 (95% conversation bootstrap) | Output ratio | Time ratio | Fenced code | One-case function probes |
|---|---:|---:|---:|---:|---:|---:|
| Live learned C, K=3 | 120.96 | -13.44% [-14.58, -11.83] | 1.045 | 1.207 | 48/48 | 48/48 |
| Monitor control, K=3/C=0 | 121.75 | -12.87% [-13.13, -12.41] | 1.000 | 1.148 | 48/48 | 48/48 |
| Calibrated fixed C, K=3 | 137.72 | -1.44% [-4.05, +0.48] | 0.960 | 0.974 | 48/48 | 48/48 |
| Fixed K=3/C=0 | 139.74 | +0.00% [+0.00, +0.00] | 1.000 | 1.000 | 48/48 | 48/48 |
| Fixed K=2/C=0 | 132.87 | -4.91% [-6.07, -3.52] | 1.421 | 1.495 | 48/48 | 48/48 |

Live learned C versus the calibrated fixed C: -12.17%, conversation bootstrap [-13.63%, -9.33%]. Versus the equal-budget monitor: -0.65%, [-1.88%, +1.01%].

The learner applied a nonzero cutoff on 27/48 turns, changed its cutoff after 47 turns and made 20 full-offer probes. Controller and trace work took 37.49s within 297.12s of complete request time.

Live learned C offered K=1/2/3 on 24/216/12540 native rounds. Mean actually offered draft tokens per round were 2.979 live, 2.373 fixed C and 3.000 K=3/C=0. The fixed-C path did not retain per-round chosen-confidence traces. 0.609 drafts were accepted. Draft acceptance is a diagnostic, not the throughput or code-quality score.

## Requested reference-length cells

| Padding characters | Live C tok/s | K=3/C=0 tok/s | Live gain | Fixed C tok/s | K=2/C=0 tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 116.46 | 137.39 | -15.23% | 134.72 | 124.62 |
| 1,024 | 117.03 | 135.01 | -13.32% | 133.93 | 131.42 |
| 4,096 | 124.41 | 142.64 | -12.78% | 141.95 | 135.49 |
| 16,384 | 122.46 | 139.19 | -12.02% | 131.54 | 138.73 |

These cells are grouped by frozen requested reference padding, with realized prompt-token counts retained in `RESULTS.json`. They contain fewer whole conversations than the pooled result.

## Calibration bound

The three disjoint calibration conversations provided 3,184 eligible full K=3 offers. Measured eligible round costs for K=1/2/3 were 14.067/15.717/17.486 ms. The optimistic hindsight cost oracle was +9.12% above K=3/C=0 [+7.87, +9.84] and is not a live result. Calibration selected fixed C=(0.434978, 0.850344); its modeled gain is in-sample and may be optimistic.

## Scope

Included conversations: [0, 1, 2, 3, 4, 5]; matched whole-conversation loop exclusions: []. All timed arms used sampled temperature 0.7, top-k 20, top-p 0.95, the full embedded Qwen MTP head, one non-production RTX 5090 and eight-turn native KV continuation. The function probes cover one valid-domain case per request across all 48 held-out requests per arm, including any performance-loop exclusions; they are not general code-correctness proof. Conversation bootstrap intervals with at most six synthetic conversations are exploratory. Greedy identity and controlled small-vocabulary sampling checks do not qualify a served positive-C policy; Memra #673 remains the sampled serving gate.
