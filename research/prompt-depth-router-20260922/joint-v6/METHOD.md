# What is learned in the draft-only K/C/D study

The Qwen target and its full-vocabulary MTP head stay frozen.
This study trains small controller weights from randomized
development conversations. K means **MTP draft sampler top-k**;
the target sampler remains top-k=20. D is offered draft length
and C is the decision to stop offering another draft position
after observing the current sampled proposal probability.

The K controller is a small ridge model trained separately for
draft K=3 and 20. Its inputs are the first 16 or 32 tokenizer
tokens of the current user turn, with one variant also using
the previous turn's observed acceptance. Its training utility
is returned output tokens minus a fixed development throughput
rate times complete turn seconds. A model can still decide K=20
on every turn; fitting weights is not evidence of useful routing.

At each eligible draft K, the D controller fits accepted-token
and measured round-time regressions from randomized D=1..4
offers. It selects the D with the largest predicted
`1 + accepted - rate × seconds` utility from committed output
tokens and available prior-round history. The C controller fits
conditional acceptance from offered draft probabilities and
committed history, censoring positions beyond the first
rejection. It weighs the expected next accepted token against
the measured marginal draft/verify cost. The learned arm does
not select among preset C threshold values.

All selection and controller work runs inside the native request
clock. K selection has a separate `k_model_ns` counter; the
reported C/D policy counter includes both D inference and
after-offer C inference. No-op twins run the same model work
while holding the fixed K/D/C actions. Acceptance and local
round utility train or explain the controller; they are not
the evaluation score.

The primary evaluation score is pooled returned output tokens
divided by complete native request seconds. Selection uses
only old development conversations. Six fresh eight-turn code
conversations are reserved for the final frozen-policy
comparison, with code-format and bounded function checks,
exact-loop exclusion, output-token ratios, paired
whole-conversation uncertainty and native KV continuation
reported alongside tok/s.

The research source uses an after-offer C stop: it proposes a
sampled token before deciding whether to offer another slot.
That rule avoids the earlier chosen-token discard bias under
the stated speculative correction assumptions. This study's
format and timing checks are not full-model sampled
distribution parity or a vendor-default endpoint
qualification. Public Memra issue #673 remains that gate.
