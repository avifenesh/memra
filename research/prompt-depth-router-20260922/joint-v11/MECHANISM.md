# What the learned C/K/D policy must predict

SpecDec++ already trains an acceptance prediction head to decide when
to stop a draft round (Huang, Guo and Wang,
https://arxiv.org/abs/2405.19715). Its `K` is candidate length;
this study's `K` is draft sampler top-k. Its reported gains concern
different draft/target model pairs and hardware. A trained C head or
adaptive depth alone is therefore not a novelty claim here. The
question is whether cheap first-token and recent-history features
improve complete native request throughput for this embedded Qwen MTP
head against the best measured fixed K/C/D controls, with inference
cost, response quality and cap behavior included.

Training-free acceptance-history length policies are also prior art
(GammaTune, https://arxiv.org/abs/2504.00030). The archived Memra
binary has its own opt-in accepted-run depth law, but this bounded
study does not run that native heuristic. A win over the measured
fixed menu would still need an equal-budget heuristic comparison
before a claim that learning is the best way to adapt depth.
EAGLE-2 also adapts a draft tree using context-dependent confidence
(https://arxiv.org/abs/2406.16858). Its tree mechanism is different
from this single embedded MTP head, so its published rate does not
establish a Qwen C/K/D gain on this GPU.

For this Qwen research request shape, K is the **MTP draft sampler**
top-k. Target top-k stays fixed at 20. D is the proposed draft depth.
C decides whether to extend an offer after observing its probability.
The pinned native K router reads up to the first 32 user tokenizer
tokens and prior-turn acceptance. The small D/C learners instead see
the last committed output tokens and prior-round acceptance/time;
C also sees the offered draft probability. The first round of the
first turn has no generated-token history or prior round; later
turns may carry prior-round state. This experiment does not test
direct prompt-prefix conditioning of C or D. A negative
history-conditioned result would not rule out that separate design.
The fresh C training rows do retain the first 32 user tokens for a
conversation-heldout calibration audit. That audit can measure
whether prompt context predicts acceptance beyond q and generated
history; it does not make the current native C policy prompt-aware.
The offered q already reflects the draft head's context, so a
prompt-prefix feature is useful only if it adds predictive signal
beyond that proposal probability in the measured cells.

The useful quantity is marginal returned-token utility at the current
hardware rate. For a D choice at context `x`, the training proxy is
`E[emitted tokens | D,x] - λ × E[native round seconds | D,x]`, where
`λ` is the measured fixed-control tokens per complete native request
second on the GPU being optimized. C should continue only when the
expected next offer adds enough emitted-token value to repay its
marginal time. K uses the analogous per-turn proxy
`output tokens - λ_source × complete request seconds`, with a separate
reference rate for each older measurement source.

These proxies guide the small controller. They are **not** the
evaluation result. An action can increase accepted draft tokens and
still reduce complete-request tok/s because it spends more draft,
verification, controller, or KV time. It can also alter response
length or task quality. The decisive evidence is paired native
returned tokens / complete request seconds, with IFEval and GSM8K
quality, cap, and exact-loop guards reported separately.

Randomized per-turn K exposure adds action labels without relying
only on fixed-K conversations whose later histories can diverge.
Randomized D exposure identifies its measured acceptance and time
costs over several contexts. Offer-level C labels are conditional on
offers that were actually observed. Fixed or selected policy logs
describe outcomes for their chosen actions; they do not establish
counterfactual benefit from a different D. The v10 final archive
contains observed K/D outcomes but no per-offer C traces, so it can
enter v11 training only with that boundary.

The earlier code result set a demanding control: learned C/D beat its
model-running no-op but was only about 0.3% ahead of the strongest
fixed C arm, with an interval crossing zero. A transferable policy
cannot treat those positive-C rates as a serving result without
sampled distribution qualification on the exact binary. The pinned
research source retains the sampled pick before C stops; Memra
#673's discard counterexample applies to a different mainline path.
A transferable policy
must show live nonconstant decisions where the input supports them,
beat its exact no-op **and every eligible fixed arm** on fresh
conversation-level comparisons, and preserve task quality. If K
again picks one value on every turn, the experiment has shown a
trained fixed choice, not useful K routing.

The follow-up has three falsifiable questions. First, v9 chose K=10
on every final code turn; a learned K router is useful
only if the final controller varies K and beats the best eligible
fixed K on native request tok/s. Second, test whether recent-token
and prior-round C features improve held-out training calibration
over proposal probability alone. The native comparison with fixed
C decides throughput benefit regardless of that diagnostic;
calibration by itself is not a speed result.
Third, a larger D can increase accepted draft count while reducing
full-request throughput. Randomized D rounds identify its emitted
tokens and time cost, while the final paired rate decides whether
the learned D behavior helped.
