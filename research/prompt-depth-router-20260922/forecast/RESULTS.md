# Cheap forecasting pilot and output-coverage audit

A shared shallow tree is inexpensive enough to investigate: 17 nodes, maximum
depth four, and 1.873 microseconds p99 for incremental observation, feature
extraction, prediction and K-policy selection on the measured CPU. No LLM or
draft-head weights were trained or changed.

The first tree did not outperform the simple prefix baseline overall. More
importantly, this corpus does not support a cross-model code-transfer verdict:
Qwen has only 17 code-labelled training prefixes and zero code-labelled
evaluation prefixes. All 12 of its held-out code requests have empty recorded
final answers at the 512-token cap. Inspected returned token tapes contain
reasoning about the requested code rather than a final code answer.

The earlier throughput measurements remain valid for their bounded native
workload. They do not settle whether cheap prediction of actual prose/code
output can improve K selection.

## What was measured

The tree uses 16 bounded byte features and the recorded prompt-class prediction.
It predicts the machine-defined format of the next eight returned tokens.
Training uses six calibration conversations across Qwen3.8-27B NVFP4+Q5_K and
Gemma 4 12B QAT Q4_0 / Q8_0 assistant, with fixed K=2,3,4. Evaluation uses six
different conversations per model and their recorded fixed K=3 trajectories.

| Predictor | Qwen supported-class macro F1 | Gemma supported-class macro F1 |
|---|---:|---:|
| Prompt label | 56.94% | 79.22% |
| Recorded current-mode rule | 80.80% | 87.59% |
| Prefix rule with code-introduction cue | 80.36% | 87.22% |
| Shared tree, 80% confidence / otherwise abstain | 79.19% | 69.32% |

Macro F1 averages only classes present in each model's evaluation set, so the
Qwen and Gemma columns have different class coverage. Abstentions count against
recall. Fences and numeric-density rules define the labels; these are not
independent human semantic judgments.

On Gemma's 2,364 code-labelled evaluation prefixes, code F1 is 99.49% for the
recorded current-mode rule and 99.79% for the shared tree. This establishes that
cheap format recognition is feasible in that covered stratum. Most such
prefixes are inside a code block; these scores alone do not establish useful
anticipation of transitions or generation speedup.

## Coverage changes the interpretation

| Model / split | Prose prefixes | Code prefixes | Numeric prefixes |
|---|---:|---:|---:|
| Qwen calibration | 4,017 | 17 | 517 |
| Qwen evaluation | 8,178 | 0 | 965 |
| Gemma calibration | 2,102 | 1,458 | 540 |
| Gemma evaluation | 4,429 | 2,364 | 891 |

Training samples one prefix every eight tokens; evaluation uses actual eligible
round boundaries. These counts are not independent conversations.

With minimum leaf size 64 and an 80% confidence requirement, a confident code
leaf needs at least 52 code examples. The entire Qwen-only training fold has 17.
Consequently, its zero code recall on Gemma cannot be read as a general failure
of portable trees. The opposite transfer direction has no Qwen code targets
to evaluate. Code transfer is **not qualified by this pilot**.

The 12/12 empty Qwen final answers and 0/12 empty Gemma final answers were
checked directly against the sealed `turn-2.answer.txt` and `turn-5.answer.txt`
records in the six held-out fixed-K conversations. The precise paths and
counts are retained in the coverage audit.

## Cost and validation

- One shared fit plus two diagnostic transfer fits: 0.308 CPU wall seconds.
- 8,651 known-label training prefixes; maximum tree depth four.
- Python/Rust feature and prediction agreement: 321 sampled prefixes.
- Five Python tests and two Rust tests passed.
- 100,000 raw timed calls after 1,000 untimed warmup calls.
- Median 1.693 us; p95 1.823 us; p99 1.873 us; maximum 54.322 us.
- AMD EPYC 7763; Rust 1.97.1; hosted CPU CI, one in-process classifier.
- State-copy and timer overhead are included. No real-time deadline is claimed.

CI run `35738157799` completed all four jobs successfully at source
`2f11a16f2ffb996649b919dd93b5ed7e286fe2a9`. The shared model SHA-256 is
`807a19573ce5239d074f6da48e55b01dd841342227b2b83a35b9489c58769fbd`.
The input manifest remains
`2c6558fbc158ff936ea1d47c54c66659fd884982ea6afd7222a41cb8a842352d`.

The complete CPU artifact includes models, features/labels, predictions, text
inspection samples, agreement cases, every timing sample, exact executables,
source hashes and the workflow record. The model, raw timings and scientific
report are exported beside this note; complete operational custody is banked
privately.

The next research comparison is defined in [NEXT-NATIVE.md](NEXT-NATIVE.md):
actual code/prose coverage on every model, shared cheap prediction, fixed
K=2,3,4 controls, and complete-request timing. This pilot promotes no K policy.
