# Candidate-source ablation — registered before results

The committed-only bounded32 probe is one candidate source. A mandatory simple
alternative is already present in Memra's append-only learner: prior verifier
predictions, including the abandoned suffix, can discover potentially useful
future rows. This is existing behavior, not the research novelty.

Hold the head, capacity, cadence, candidate budget, recency rule, deterministic
exploration, confidence and depth fixed. Change only the observed candidate stream:

- Control: prompt plus accepted-prefix/correction IDs from completed verification.
- Alternative: prompt plus all verifier argmax IDs from completed verification.

Both are causal at the following block. Suffix predictions are candidate hints,
not labels from the real continuation. The bounded state recorder must still
stop correctness labels at the first rejection. Never insert the current block's
verifier results into its already-selected candidate pool.

On the same48 diagnostic prompts, record one request per source with native full
draft traces to audit the exact prior observations. Join to full-head oracle
states to compare repairable-target discovery; this reuses qualification data,
not a fresh policy-generalization test. Measure source ON/OFF cost on eight
calibration-code prompts with six balanced repetitions, full draft traces disabled
in both timed arms. Preserve complete output equality, raw hashes and250ms telemetry.

No learned head updates or changed confidence/depth are admitted here. If a
candidate source is useful, retain it as a matched baseline for the subsequent
fresh-data admission test; a better candidate pool must not be credited to the
learned selector.
