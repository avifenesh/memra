# Qwen draft-only C/K/D follow-up

K is MTP draft sampler top-k; target top-k remains 20. D is offered draft
depth. C decides whether to extend the offer after seeing the sampled
proposal probability. The Qwen target and MTP head stay frozen. The
evaluation objective is pooled returned tokens divided by complete native
request seconds, with code quality, output length, and native KV reuse
reported separately.

## Data boundary

The v6 randomized development sessions and all six v8 fresh conversations
are training material for this follow-up. They cannot select a policy or
count as a final evaluation. Recheck their archive member hashes before
reading them. Keep every turn, round, and alternative arm of one
conversation in one split; never split by row or by turn.

v8 fixed K=3/10/20 turns provide observed full-turn token and time rewards
for K training. v8 spans provide realized D, accepted-prefix and elapsed
round observations. Their D=2/4 actions under the learned controller were
selected by that controller, so they are model calibration observations,
not randomized evidence that D=2/4 is better. v8 did not retain chosen
proposal probabilities per offer, so it adds no C acceptance labels.
The v6 randomized D receipts do retain probabilities and supply censored
conditional C labels.

Hardware-specific timings remain separate: v6/v8 ran on one RTX 5090.
If new data is collected on another GPU, reuse acceptance observations
with their model/artifact identity but fit the time cost and throughput
baseline on that GPU's new measurements. Do not pool raw milliseconds
across GPUs or claim that the two GPUs have the same optimal policy.

## New collection and selection

Freeze disjoint prompts before a GPU run, using distinct tasks from the
pinned Google Research sanitized MBPP revision recorded by `workloads.py`. This is a
custom continuing-conversation workload and must not be called an official
MBPP benchmark score:
24 eight-turn training conversations, 8 validation conversations, and
16 final conversations. Each turn starts with the task description so
the bounded first-32-token K input contains task information, followed
by one source test as an example;
two or more remaining tests are retained for code evaluation outside the
native request clock. Requests use `max_new=4096`, `ctx=65536`,
temperature 1.0 and target top-k=20 throughout this v9 comparison.
For each training conversation, run balanced
fixed K=3/10/20, D=3 controls and a randomized-D exposure at each K.
The fixed K arms expose every task at every K; the randomized-D arm
offers D=1/2/3/4 at eligible rounds. Record the assignment seed and
actual action for each exposure. Retain the sampled chosen-token
probability, proposal position, acceptance/rejection, committed-token
history, round time, and full-turn outcome. C labels past
the first rejection are censored, not zeroes.
If any training arm on a task group exact-loops, exclude all six arms
of that group from fitting and report it. Require at least 20 of the
24 groups to remain. Keep the raw receipts for every arm.

Fit a small K policy from bounded first-user-token prefix and previous
turn acceptance, with K=10 included as an action. Fit D and C using
only information available at their native decision points. Calibrate
expected accepted prefix and marginal time, and select actions by
predicted time-adjusted output, not acceptance alone. Compare against
fixed K=3/10/20, D=3, C=0, a fixed C sweep chosen on training, learned
K only, learned C/D at fixed K=20, joint C/K/D, and each learned arm's
model-running no-op in the qualifier and final battery. At fixed K=20, compare C/D token-only,
recent-token history, and history with previous-round features as a
native throughput ablation. The validation split chooses the one frozen policy
per family. No final-set feedback changes weights, features, cutoffs,
candidate lists, or quality rules.

All arms use identical pinned model, source, binary, target sampler,
prompts, seeds, and request settings. Rotate arm order by conversation.
Require a byte-identical sampled no-op qualifier, nonzero native C
decisions with stops and continuations, and positive cached and new
input tokens on every later turn. Exclude exact loops from throughput
aggregates and report them. Publish paired whole-conversation intervals,
tokens, complete request seconds, actual C/K/D actions, and code-format
and MBPP test checks. Acceptance is a model input and diagnostic,
never the evaluation score.

For validation selection, an arm is quality-eligible when it has no
more exact-loop turns than fixed K=20/D=3/C=0 and its fenced syntax and
all-tests-pass rates are each no more than five percentage points below
that control. Its output-cap rate may be at most five percentage points
higher than that control. This is a frozen noninferiority guard, not a claim that
the benchmark tasks measure broad code correctness. Choose one fixed,
one K-only, one C/D, and one joint arm from quality-eligible validation
points. The final split reports every fixed K=3/10/20 control, each
selected learned arm, and the learned K and joint no-op twins. Claim a
new default only if a learned arm passes quality and its paired
whole-conversation lower confidence bound is above the best eligible
fixed control.
