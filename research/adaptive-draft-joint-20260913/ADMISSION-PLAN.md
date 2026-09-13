# Fresh admission test — registered before collection

After candidate-source qualification, test admission alone on a frozen head and
fixed victim slot4096. Keep K4 and both confidence cuts zero. This remains an
offline, reached-state single-swap experiment; runtime replacement and whole-block
benefit require a subsequent experiment.

Choose committed-only or suffix-assisted discovery using calibration repairable
target recall, with ties preferring committed-only. Never choose from fresh test
results. Fit models only on training traces, with all hypotheses and parameters
selected on the old calibration split. Earlier held-out prompts are diagnostic
history, not a new held-out evaluation.

Compare no-op, highest candidate score, training target frequency, the original
per-ID utility table, a per-ID table conditioned on changing the proposal, and
a small context regression. All selectors share candidates, numerical eligibility
and the rule that an operation which cannot change the current proposal is a
no-op for this one-step objective. This rule does not claim such an insertion
could never help a future state.

The frequency baseline uses all committed training-output tokens available in
the logs, not just sampled probe labels. Regression gives each nontrivial state
total weight one, divided across its eligible candidate interventions, so a state
with many candidates does not impersonate many independent observations.

The context model predicts delta correctness with ridge regression. Features are:
bias; candidate-minus-surviving-winner margin clipped to[-20,20]/8; current top-two
logit gap clipped to[0,20]/8; draft position/3; candidate in the prompt; candidate
in previously verified actual output; candidate from the history rather than
exploration slots; and whether the current winner occupies the mutable victim.
These features are available without the current verifier answer. There are no
target-frequency encodings in the context feature vector. Use ridge penalties
{1,10,100}; both per-ID tables use four zero-utility pseudo-exposures. Thresholds
are{0,0.01,0.025,0.05,0.1,1.0}. Select prompt-mean calibration delta, then fewer
actions, then stronger regularization/higher threshold on ties. Freeze coefficients,
tables, thresholds, source choice, input hashes and source version before test
generation. Refuse to fit the regression with fewer than eight nontrivial training
interventions; report the insufficient-data result rather than invent coefficients.

Fresh test:24 synthetic prompts in six new task families, four cases each:
inventory reconciliation, constrained routing, structured schema conversion,
terminology-controlled rewriting, state-machine execution and conflicting-evidence
summaries. Publish exact prompt bytes before model execution. Use128-token outputs
and paired full/bounded diagnostic runs, with complete target-output identity.
Record a full native draft trace for causal provenance, not just correct output
text. Fit/selection never read test labels. Report benefits, harms, action counts,
per-prompt and family means. With only six families, uncertainty is descriptive;
do not claim broad generalization or serving speedup from this pilot.

Advance a learned selector to an online head-only experiment only if it improves
held-out utility over both frequency and highest-score selection. Preserve a tie
or loss as a result; do not retune on this test or mix confidence/depth to hide it.
