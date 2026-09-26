# Cheap prediction of upcoming output format

This continuation tests the owner's research hypothesis: a small, shared
predictor of upcoming prose/code/numeric output can select K from {2, 3, 4}
without training or changing an LLM or draft head.

The earlier prompt lookup and retrospective span controller are measured
baselines. Their negative aggregate results do not close this hypothesis.
Their reports and sealed records remain unchanged.

## Fixed first pilot

- Source: the closed native receipt manifest
  `2c6558fbc158ff936ea1d47c54c66659fd884982ea6afd7222a41cb8a842352d`.
  Model artifacts, sampling settings, source and hardware are those recorded
  in the parent `NATIVE-PROTOCOL.md`.
- Predict the format of the **next eight returned tokens**, before observing
  those tokens. Features use the latest prompt classifier result and at most
  128 recent committed bytes plus bounded running counters. No model identity,
  vocabulary IDs, future bytes, timings or target logits are features.
- Train from the three calibration conversations per model, using their fixed
  K=2,3,4 arms and a prefix every eight output tokens. Keep every arm from a
  conversation in the same split.
- Evaluate on the six held-out conversations per model, using their fixed K=3
  arm and actual eligible round boundaries. Also train on one model's calibration
  conversations and evaluate on the other model's held-out conversations.
  These are reused, synthetic archived conversations; this is an offline
  forecasting pilot, not a fresh generation-speed or broad generalization claim.
- One pooled decision tree, maximum depth four, minimum 64 training examples
  per child, class-balanced Gini splitting, deterministic thresholds and ties.
  A leaf needs 80% unweighted class agreement to make a confident prediction;
  otherwise it abstains. Hyperparameters are not selected on held-out results.
- Comparisons: the recorded prompt prediction; the recorded causal span label
  (current-mode persistence); a small prefix-only rule; the pooled tree; and
  cross-model transfer trees.
- Report class coverage, per-class precision/recall, macro F1, abstention,
  and results at upcoming format transitions separately from stable interiors.
  Accuracy on long uninterrupted prose alone is insufficient.

## Labels and interpretation

The first pilot uses explicit, reproducible format labels. Markdown fenced
program text is code. Fences explicitly marked as plain text/Markdown are
ambiguous. Outside fences, a window with at least four digits and more
digits/operators than ASCII letters is numeric. Other predominantly unfenced
text is prose. Windows crossing formats without a 75% majority, or containing
fewer than four non-whitespace bytes, are ambiguous.

These are machine-defined format labels, not independent human judgments of
semantic intent, difficulty, output quality or the optimal K. The labeler sees
future bytes only to construct the training/evaluation target. The predictor
never sees those bytes.

The proposed policy starts each request at K=3. Confident prose proposes K=2;
confident code/numeric proposes K=4; abstention proposes K=3. Two consecutive
proposals confirm a change, which moves at most one step per round. This fixed
mapping is an experimental hypothesis, not an assumed law across models.

Historical token streams can measure prediction and shadow decisions. They
cannot establish counterfactual throughput after changing K. A later native
comparison must execute the policy against fixed K=2,3,4 and the strongest
calibrated control, retain exactness checks, and measure complete requests
including switching and predictor costs.

## Runtime and evidence

The exported tree has at most 31 nodes and uses integer comparisons. A
dependency-free Rust implementation shares the bounded byte features and policy.
Hosted CPU CI checks Python/Rust agreement, prefix causality, split isolation,
and the raw latency distribution for incremental observation plus prediction.
The initial component target is p99 below 5 microseconds on the recorded CPU.
There is no additional model forward pass or model-specific neural training.

All fitting and benchmarks run on hosted CPU CI. No local rig gate, GPU rental,
production host, or change to a model checkpoint is part of this pilot.
