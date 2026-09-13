# Gates after the component baseline

The current `isolated.py` campaign is a causal baseline for existing mechanisms.
Completing it does not mean the proposed learned policies have been tested.
The owner's isolation-first requirement also applies to each new implementation.

1. **Learned row policy alone.** Compare fixed-capacity learned replacement against
   both frozen rows and the existing append-only learner. Keep confidence disabled
   and draft depth fixed. The current cold 128-token requests do not fill 512 spare
   rows, so they cannot exercise replacement. Register a continuing stream and a
   domain switch, bank identical starting state for both arms, and include update
   and persistence cost. Record actual-continuation corrections separately from
   counterfactual verifier suffixes. The target verifier stays unchanged.
2. **Learned confidence policy alone.** Freeze the exact draft head within each
   evaluation block. Compare a train/calibration-selected static threshold with
   calibrated stopping at fixed maximum depth, accepted-length adaptation disabled.
   Use a pre-banked set of independently trained head states to test calibration
   after a head change without simultaneously enabling online row learning. Retain
   any already drawn proposal; decide only whether to draw another. Inference and
   confidence-normalization overhead count toward the objective. The previous
   head-aware predictor's narrow held-out loss is retained, not tuned away on that
   same set.
3. **Learned depth policy alone.** Freeze rows and disable confidence stopping.
   Compare with both calibration-selected fixed K and existing accepted-prefix+1
   control. Extend K only when the expected additional accepted prefix pays the
   marginal draft and verification cost. The July 30 flat marginal-rate controller
   is prior work, not a new contribution. Collect exploratory actions with their
   probabilities before choosing a policy, and freeze it for held-out evaluation.
4. **Interactions only afterward.** Bank separate implementation correctness,
   mechanism engagement, and cost-inclusive verdicts for 1–3, including losses.
   Then measure all pairwise combinations and the full joint policy against strong
   independent controllers. No combined aggregate substitutes for component rows.

For Qwen, `spec.rs` has a distinct p-min gate and an opt-in accepted-length policy;
source comments explicitly record earlier Qwen adaptive-K losses. The Gemma trim
freeze switch does not supply Qwen with an adaptive head. Its own row-update path,
actual confidence semantics and request-time harness require separate qualification
before cross-family replication. Current Qwen evidence is artifact/correctness and
own-generation ranks only.

Production conclusions additionally require sampled correctness, continuing chat
sessions, representative non-code workloads and serving concurrency. These are
different evidence gates from the current greedy one-shot component study.
