# Cost-aware C learning after the Qwen K=3 grid

K=3 is the ceiling for the code-request research arm. A confidence gate may
offer zero to three MTP draft tokens in each round. A learned cutoff should
change that gate only when the extra accepted output is worth its draft,
confidence-read and verification cost on the running artifact and hardware.
The completed fixed-C grid and code-only held-out set did not establish a
stable win for the current raw-probability gate; see `VERDICT.md`.

## What the learner may observe

- The drafter's confidence for each **offered** token, its draft position, and
  the target's first rejection position. Only the accepted prefix has valid
  survival labels; target logits after a rejected suffix were conditioned on
  draft tokens that never committed. Require finite confidence in `[0, 1]`
  before updating a bin; a non-finite score is an observation failure.
- Memra's current C signal is the **raw** MTP softmax probability of the
  sampled pick at temperature 1, even when the proposal uses temperature
  0.7 and top-k/top-p filters (`cu/kernels.cu` and `cu/spec_sample.cu`).
  Keep that signal distinct from the target's filtered acceptance
  probability and calibrate it against verified prefixes on this artifact.
- Committed output tokens, complete round/request wall time, and current
  prompt/output context. A token never offered because C stopped the chain
  has no acceptance label. The controller must sometimes lower C and offer
  that slot to learn its value.
- The current serving sampler, quantized target/draft artifacts, concurrency
  and GPU shape. A threshold learned on BF16 or another batch regime is not
  automatically calibrated for this NVFP4+Q5_K research path.

## Policy to qualify

Maintain per-position confidence bins with accepted-prefix counts and
attempt counts. Estimate the next slot's conditional survival chance from
eligible offered prefixes only. Pair that estimate with measured draft,
confidence and verify cost to choose whether the next slot should be
offered. The reward is committed output tokens divided by full cycle time;
accepted/drafted alone is a diagnostic. Carry the statistics and C decision
across requests in a continuing session. Probe lower and higher cutoff
values with a small, recorded exploration budget, including the probe time
in the primary metric.

DSpark (arXiv:2607.05147, §3.2.1) explicitly models survival at position
`j` conditional on earlier draft tokens being accepted, then calibrates the
prefix-survival product. Its trained confidence head and concurrent verify
scheduler are different from this native MTP chain; the transferable part is
the conditional-survival target and the need for calibration. Learning to
Draft (arXiv:2603.01639, §3.1 and §6.3) shows why cycle throughput is a
better reward than acceptance length alone.

The post-result 512-output-token phase control separated the confidence
read from actual cuts on the same binary. Off and probability-only arms
returned identical sampled token tapes and K=3 draft lengths in both
orders. Probability-only added 0.85% to draft-phase time per 100 tokens,
while complete-request rate was nearly equal (101.84 versus 101.80
tok/s). This capped diagnostic does not explain the free-generation
held-out variability by itself. A cheaper confidence path would still
need its own exactness and paired wall-time cell before a learner uses
it. The negative cutoff recipe does not rule out other calibrated
confidence signals.

One concrete cost candidate is the sampled draft path's existing filtered
`filter_stats` output (`cu/spec_sample.cu`): it already emits the row max,
filtered renormalization mass and cutoff. For a kept sampled token, its
filtered proposal probability is `exp((logit[token] - max) / T) / Z`, which
a chosen-logit gather could compute without another full-vocabulary scan.
This would change the C signal from raw to filtered proposal probability,
so it needs a matched exactness gate, acceptance calibration and a direct
cost/quality comparison; it is not an assumed optimization.

## Controls and promotion boundary

Establish a cost-inclusive oracle upper bound before fitting a controller:
record full K=3 offers and verified accepted prefixes on disjoint code
requests, then price the best hindsight stopping choices against the
best fixed C on the same target-token tape. Never-offered slots have no
labels, so the oracle corpus must include an uncensored full-offer arm;
the current separately sampled C arms cannot supply an attainable
counterfactual oracle by choosing their fastest request after the fact.
If the bounded oracle cannot clear the fixed-control gain floor after
confidence-read and exploration costs, stop the learner work there.

Compare live adaptive C with C=0, the best calibrated fixed C, fixed K=2
at C=0, and native depth adjustment on disjoint code requests. Keep H and
model weights fixed. Match sampler, prompt, seed, output budget and host;
report full request wall, output tokens, draft-length histograms, zero-draft
rounds, probe cost and format coverage. Greedy target identity and sampled
reproducibility are gates. These native single-GPU results would still need
vendor-default HTTP, warm-session and concurrent-serving qualification
before changing a served policy.
