# After the sparse row oracle

Registered while collection is running, before inspecting held-out results.
These are separate gates, not additional claims from the current experiment.

1. **Physical replacement qualification.** Replay the same preserved draft hidden
   vector through an actual 4,608-row head with exactly one mutable slot replaced.
   Copy the original Q8_0 row bytes; use the normal active-head projection and
   argmax. Compare the replay winner with the score-based prediction. Include
   beneficial, harmful, neutral, and near-tie cases. Do not continue a changed
   block using the old hidden-state suffix. Keep unqualified score comparisons
   labelled estimates, even when overlapping original rows agree numerically.
2. **Candidate discovery alone.** Replace the full-vocabulary diagnostic with a
   bounded candidate pool built from prior prompt/verification observations and
   logged exploration. Match that pool across selectors. Measure discovery recall
   against the oracle and total staging/projection/readback/statistics cost. The
   current top-16 full-vocabulary candidates are privileged expensive features;
   a table selected from them is not yet a deployable policy.
3. **Admission alone.** Freeze the eviction rule and capacity. Compare utility
   selection with target frequency on fresh family-separated streams, including
   no-op and instrumentation-only controls. Freeze the training profile and
   calibration choices before evaluation. Preserve early cold-start losses and
   domain-switch transients. A tie with frequency does not establish added value.
4. **Eviction alone.** Freeze the admitted candidate stream. Compare FIFO, LRU,
   frequency and complete-swap utility under the same update count, residency
   constraints and capacity. Include the value of the row being removed. Do not
   infer a multi-row gain by summing independent one-row labels.
5. **Actual online head-only benefit.** Run changed blocks from their real initial
   state, with target verification unchanged, K fixed and confidence disabled.
   Count all cost and bank identical initial state per arm. Require balanced
   AB/BA runs, whole-output correctness and per-stream uncertainty. Promote only
   with a measured advantage over the strongest matched simple policy.

The first sparse oracle can stop the current *implementation* if opportunity is
too small or diagnostics dominate its plausible benefit. It cannot rule out all
adaptive draft learning. Test a revised candidate source, capacity or draft
prediction hypothesis in a new registered experiment with fresh evaluation data.

Confidence learning and depth learning still require their own isolated gates
before pairwise or joint trials. Qwen needs its own row-update implementation;
the current Gemma flag does not provide it. Serving claims additionally require
sampling, session persistence, concurrency and tail-latency qualification.
