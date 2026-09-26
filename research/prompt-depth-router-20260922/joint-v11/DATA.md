# C/K/D training labels and their limits

**Prepared, not yet measured.** The v10 battery stopped prematurely
after the mainline #673 discard path was mistakenly attributed to
the pinned v8 research binary. Its archived source instead retains
sampled picks before C stops. A clean transfer rerun is using the
already frozen v11 training prompts. The extractor expects that
complete replayed archive, not the stopped partial custody; the
v11 validation and final prompts remain untouched. Sampled
distribution qualification on the exact binary remains a separate
gate for equal-distribution claims.

The replayed training archive carries only the qualification and
training projection of the original frozen split. It contains no
validation or final prompt files or answer keys. The training
projection commits to the later validation projection and original
full manifest by hash. Validation files are released after fitting;
final files only after selection.
The full v10 archive also holds benchmark source datasets and stays
private off the v11 training host. A verified subset archive carries
only the v10 native observations needed for training and behavior
analysis, plus grader code. Its manifest and custody receipt name
the complete replayed v10 parent hash.

The frozen v11 follow-up uses four observation groups. The archived
v9 code training rows supply K turn outcomes, eligible randomized D
round outcomes, and C offer labels. The v9 older code rows add K turn
utility with a source-specific reference rate and D/C acceptance
observations in the augmented candidate. The replayed v10 non-code fixed arms
supply observed K turn and D round outcomes, but no C offer labels.
Fresh v11 non-code exploration supplies balanced fixed-K arms,
randomized per-turn draft-K assignments, randomized D rounds,
and conditional C offer labels.
Validation compares the K-only router trained on these non-code rows
with the mixed code/non-code K router and their model-running no-ops.

Each K row records draft top-k, the first 32 user token IDs, prior-turn
acceptance when available, returned tokens, and complete native
request seconds. K utility uses the source's own fixed-K=20
reference rate. Each eligible D row records selected depth, recent
committed tokens, accepted prefix, and native round time. Each fresh
C row records the bounded first 32 user-token IDs, an observed
proposal probability, and whether that offer was accepted; only
positions that were actually offered and observed are labels.

`measurement_rows.py` excludes a whole conversation when an arm
needed for its matched fixed K/D observations loops. It excludes a
looped randomized K arm separately. It marks fixed-arm outcomes as
observed actions, randomized K turns as assigned actions, and
randomized D exploration as assigned rounds. `training_seal.py` and
`training_replay.py` preserve and verify
the randomized native conversations before row extraction. The
extractor reads those sealed v11 bytes and requires the v10 training
projection's private custody receipt, which binds its parent hash to
the complete independent replay. Validation and final task IDs in the v11
manifest never enter these row files.

`fit_mixed.py` uses only the new v11 GPU's randomized D rows for
marginal time-cost fits and that GPU's fixed K=20 rows for its runtime
reference rate. Earlier K turns use source-specific rates in their
training labels; earlier D/C rows add acceptance evidence. The fit
allows positive or negative measured marginal
differences and rejects nonfinite costs. The model files are
validation candidates, not throughput or quality results.

`behavior.py` is a separate posthoc read of the v10 training projection.
It reports full-request tok/s for first versus later turns and
observed D choices after committed token classes. Its accepted-prefix
and round-time groups are diagnostics: selected D actions are not
randomized counterfactuals and cannot prove that a larger D caused a
throughput change.

`feature_audit.py` checks C's history and prompt-token hypotheses on fresh
v11 randomized training labels. It holds out whole training
conversations, compares proposal-probability-only calibration with
recent-token, prior-round and bounded prompt-prefix features for each
K and offer position,
and reports log loss separately for instruction and math tasks.
Those held-out groups are within the training split and may enter
the later full candidate fit. Validation and final prompts remain
separate. Prompt features are a training-only calibration probe; the
pinned native C/D learner still does not read the prompt prefix.
Calibration can motivate a candidate; it is not a tok/s result.
