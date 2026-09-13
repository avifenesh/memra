# Bounded candidate discovery — registered before measurement

Keep the head frozen at 4096+512, K4, confidence cuts zero, eager greedy and
128-token outputs. Sample rounds1,17,33,... as in the full oracle. Neither probe
changes the live head. Full and bounded probes must refuse simultaneous use.

Maintain up to256 unique out-of-head IDs in recency order, initialized from the
request's prompt. After each completed verification, observe only the accepted
prefix and its correction/bonus token; never observe the rejected suffix. Before
the next sampled block, select up to24 most recent IDs and fill to32 with distinct
deterministic xorshift exploration IDs outside the head. Seed each block with
0x9e3779b97f4a7c15 XOR its zero-based round index. This is deterministic exploration,
not a claim of unbiased importance-weighted sampling. Keep candidates fixed for
the whole block. The current verifier label cannot choose a candidate.

Gather only these32 original Q8_0 rows and score with the native projection.
Log candidate IDs/scores, history-row count, round/context/position and reached
verifier label. Discard labels beyond the first rejection or output budget.
No full-vocabulary projection or full-head GPU allocation occurs in bounded mode.

First qualify complete output equality with probe-OFF controls. Reuse the earlier
48 prompts for discovery diagnostics, not fresh learned-policy evaluation. Join
the same reached states to the full-head oracle; verify numerical scores where
candidate IDs overlap and measure how many oracle-repairable targets are found.
Audit the history pool from the original prompt and committed output prefix.
Measure bounded ON/OFF request cost using eight calibration-code groups and six
balanced repetitions per group; capture all allocation, gathering, projection,
readback and recording cost. Preserve250ms telemetry and all failures.

No fixed-token table is promoted by this experiment. Poor discovery recall is a
negative result for this candidate source, not permission to use current target
labels as pre-verification features. A revised discovery mechanism, online
selector or workload split needs a separately registered test. Continue to an
online head-only comparison only after physical and discovery qualification.
