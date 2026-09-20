# Explaining the full-head MTP results

The full vocabulary is identical in each comparison. The measured variable is how many MTP draft steps are requested.

Round yields below use the engine-emitted counts in the traces, including terminal
overshoot. The primary request E2E metric uses only returned output tokens. Terminal
rounds do not train the learner. These round summaries describe observed work; they
are not counterfactual same-state measurements at alternative depths.

## Qwen: cheaper rounds did not compensate for fewer emitted tokens

Fixed K=3 emitted 2.274 tokens per timed round at 18.351ms. The learner averaged K=2.33, emitting 2.069 tokens at 17.254ms.
Round time fell 6.0%, but tokens per round fell 9.0%. That tradeoff loses throughput. The final request E2E loss is 2.45% versus fixed K=3.

Native adaptation averaged K=2.10 and reacted to each accepted prefix. The learner averaged K=2.33 overall, including its probes. Its exploitation depth was close to native, but it also sampled larger depths periodically.

The replayed schedule assigns 12.85% of learned round time to periodic probes, plus 3.28% to initial exploration. Probe rounds averaged 104.27 tokens/s versus 120.88 during exploitation. These are observed rates in different states; they do not establish a counterfactual speedup from removing probes. Slower response to changing acceptance and stale cost estimates remain possible contributors, not separately measured causes.

## Gemma: fixed K=5 drafts more than this workload rewards

Moving from fixed K=5 to learned mean K=2.88 reduced timed-round cost from 15.002ms to 11.481ms (23.5%). Tokens per round fell from 3.074 to 2.568 (16.5%). Here the cost reduction is larger, so request E2E improves 7.74%, with eight wins in eight pairs.

Native adaptation already chose mean K=2.41. The learner's somewhat deeper rounds gained tokens and cost time in nearly equal proportions. Its E2E advantage over native is only 0.43%, with a 0.36% paired median and two losing pairs. This supports adapting depth relative to the specified fixed K=5, while showing only a small incremental gain for this cost learner.

## Startup does not explain the whole Qwen loss

Grouping the selected requests by position in each eight-turn conversation gives:

| Model / request position | Learned vs fixed | Learned vs native adaptive |
| --- | ---: | ---: |
| Qwen, first request | −7.05% | −5.53% |
| Qwen, requests 2..8 | −1.94% | −3.18% |
| Gemma, first request | +6.25% | −0.81% |
| Gemma, requests 2..8 | +7.92% | +0.58% |

The first-request penalty is larger on Qwen, but the loss remains in later requests.
Gemma's small advantage over native adaptation appears in the later requests.
Request position also changes prompt length and content, so this is a diagnostic
of where the effect appears, not a pure initialization-cost ablation.

`diagnose_requests.py` reproduces this table after running the full selected-data
audit. Its totals reconcile exactly with the overall metrics; the JSON outputs are
`qwen-request-diagnostics.json` and `gemma-request-diagnostics.json`.

## What remains open

This tests one new online cost controller on one frozen workload and quantization per model. It does not reproduce the unavailable original H/C/D implementation. It also does not compare against a calibrated best fixed depth: K=3 and K=5 were preselected native MTP starting points.

A stronger next test holds the full head fixed, selects fixed-depth controls on separate calibration data, and evaluates a frozen controller on held-out workload regimes. The existing round traces can diagnose the current policy, but cannot supply unobserved per-state counterfactual costs for every alternative depth.

The next controlled tests are specified in [NEXT-EXPERIMENTS.md](NEXT-EXPERIMENTS.md).
