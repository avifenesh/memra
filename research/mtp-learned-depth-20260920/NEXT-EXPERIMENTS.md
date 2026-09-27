# Tests needed to explain the controller difference

The existing receipts establish the cost/yield tradeoff. Qwen saves 6.0% per timed
round but emits 9.0% fewer engine-emitted tokens; Gemma saves 23.5% while losing only 16.5%
of tokens per round. They do not yet establish which policy component caused the
depth choices, or whether a better fixed depth would match the learner.

## 1. Does a calibrated fixed depth match Gemma's improvement?

Hold the full target and MTP heads, quantization, sampler and priming fixed. Sweep
fixed K=1..5 for Gemma and K=1..7 for Qwen on a separate calibration prompt pool.
Choose one K per predeclared workload regime before opening evaluation results.
On held-out prompts, compare that frozen K, native adaptation and the unchanged
cost learner with matched seeds and balanced orders. Keep the current K=3/K=5
controls as reference points rather than silently replacing their published results.

This separates improving an overly deep starting point from an advantage that
requires online learning. It also establishes the available depth-selection headroom.

## 2. What is the net effect of periodic probing?

Keep the current learner's initial calibration, candidate set, cost objective,
EWMA and hysteresis unchanged. Compare periodic probes on versus off, with fresh
fixed/native controls in the same paired sets. All arms retain full heads and MTP.
Report request E2E, timed-round cost/yield, depth choices and recovery after a
predeclared workload transition. Keep stationary and transition workloads separate.

This measures the total causal effect of periodic probing, including its effect
on later decisions. Subtracting probe time from the current receipts would not:
removing probes changes observations, depth choices and sampled trajectories.

## Interpretation and discipline

- If calibrated fixed K matches Gemma's learner, the observed K=5 improvement is
  principally an opportunity to choose a better depth, not evidence that learning is needed.
- If probes-off improves Qwen on stationary workloads but harms transition recovery,
  the next question is when to probe, rather than whether exploration is universally bad.
- The current per-phase rates compare different states. They are diagnostics, not
  counterfactual estimates of what the same round would cost at another K.
- Preserve original controls, full heads, cold priming, minimum run duration, paired
  order and whole-set rejection on GPU interference. Freeze the protocol before scoring.

These tests are proposed follow-ups. They have not been run and supply no numbers
to the published result.
