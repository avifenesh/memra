# Input-conditioned speculative control: C, K and D

This continuation tests the owner's proposal to measure fixed settings
on a bounded input corpus, then train a small controller that chooses
settings from the current and previous tokens. The previous
[Qwen live-C result](../confidence/adaptive-v3/VERDICT.md) measured
120.96 versus 139.74 complete native tok/s for learned C and
K=3/C=0. Its C=0 tracing/controller twin was 121.75 tok/s. A new
controller must run in-process and beat the strongest fixed setting
after its own inference, switching and observation costs.

## Decision timing and notation

Pending the owner's terminology check, this protocol uses:

- **K**: maximum draft-token ceiling allocated for a request.
- **D**: actual draft depth chosen at a safe round boundary, `1 ≤ D ≤ K`.
- **C**: stopping rule applied *after* offering a sampled draft token,
  before another draft slot. Positive sampled PMIN0 is refused.

K and D are not two names for the same action: K fixes the eligible
range and resource/graph shape for a request; D is the selected depth
within it. C may shorten an attempted D. Measure the incremental
effect of each knob separately before treating a joint vector as a
win. If K meant sampler top-k or vocabulary-head size, version the
action contract **before** scoring: changing sampled top-k changes
the target distribution, while changing head size changes the
proposal model.

## Pinned model and evidence

Use the full 248,320-row embedded MTP head of
`tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@0f82b27dbb264b731e7d20f576582c871ef1969c`,
file `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
Start from the separately sealed sampled-after-offer v3 research
source archive SHA-256
`49015ce03f02b3fee00223b5914377b04b13ab47a88a4d7c36fe96444006223a`.
Any v4 patch gets a distinct source, binary and receipt hash. The
v3 native-data archive is **development material only** for feature
feasibility and fixed-grid choice; all v4 model selection and final
evaluation require disjoint new conversations.

Sample with temperature 0.7, top-k 20, top-p 0.95 and default
thinking in every arm. Keep `max_new=8192`, `ctx=65536`, a complete
eight-turn native continuing session, and a full final-format gate.
These are study settings, not vendor-default HTTP qualification.
No model/draft-head weights or serving defaults change in the first
controller experiment.

## What the controller may know

At **request start**, a controller may use only a bounded first user
token prefix, request length and prior turns' committed tokens. At a
**round boundary**, it may additionally use the last committed token,
a bounded history of 4 and 16 committed tokens, previous *completed*
round acceptance and elapsed time, current offered draft confidence,
and current K/D. The current sampled draft token can be read only at
the after-offer C decision. Never feed a future target logit, a
rejected suffix, a later output token, a later round's acceptance, or
an unmeasured cache state into a decision.

Train and ablate three nested feature sets:

1. Current token and current draft confidence only.
2. Those features plus 4/16 prior committed tokens.
3. Those features plus previous-round accepted-prefix and cost
   summaries.

Check content transitions such as reasoning-to-code fence and
prose-to-JSON separately from stable interiors. Keep token identity
features small (hashed or typed, not a 248K-way memorization table).
Every arm records source feature bytes, decision time, action and
availability timestamp. A no-op feature/controller twin pays the
same feature and model inference cost while applying the fixed
K=3/C=0 action.

## Labels, static controls and training

The primary Qwen question is **code**. Re-run six v3 development
topics (three old calibration topics and heldout topics 0–2) for
training, and use the remaining three v3 topics for model selection;
none is a v4 heldout result. `workloads-v4/manifest.json` freezes
one new qualifier and six new heldout eight-turn code conversations
before their outputs exist. Require actual final code and cover
reasoning-to-code transitions. Do not claim prose, JSON or another
model family from this code-only experiment; those need separate
fresh splits after this result.

Group every turn and every static arm of a conversation in one
split. Freeze seeds, format/loop rules, prompt hashes, context
budget, model/binary hashes and balanced arm order before any
held-out request.

On calibration only, execute fixed K/D/C settings on matched input
conversations. Include K=3/C=0, fixed K=2 and K=4 at C=0, native
depth adaptation, and fixed C candidates derived from the v3
development q-distribution, with a control at the v3
calibration-selected C=(0.434978, 0.850344). Keep the strongest
executed fixed control from calibration for held-out comparison.
Report any per-input best-of-arm oracle as **hindsight over different
sampled outputs**, not a deployable policy or a same-tape E2E result.
An input's highest individual tok/s can lower the pooled ratio after
switching arms: use the globally optimal ratio `λ` and label an
input by `argmax_a(tokens(input,a) − λ·seconds(input,a))`, then
independently validate that this choice improves pooled
`sum(tokens)/sum(seconds)`. Never train an action classifier on
per-input tok/s argmax as if it were the E2E objective.
The v3 same-tape +9.12% oracle and −1.44% executed fixed-C result
show why acceptance-derived oracle gains cannot stand in for
throughput.

For round-level training, record actual offered D, accepted-prefix
labels only while all earlier proposals were accepted, committed
tokens, draft/verify/commit wall, and complete request time. Cover
each eligible D action on calibration with a preregistered randomized
schedule and retain action propensities, so a learner is not trained
only on states selected by its own favorite depth. Acceptance is a
**calibration target and feature for later rounds**; it is never the
evaluation metric. Train a small frozen predictor of conditional
survival/cycle utility and select actions with measured marginal
cost. Export an in-process Rust inference program: no Python
subprocess, per-turn JSON rewrite, extra target forward, or new
GPU-to-host hidden-state transfer in the scored arm.

Use model-selection conversations only to choose one controller and
its hyperparameters. Compare token-only, +history and +prior-round
ablations at matched training budget. Freeze the chosen binary,
weights and action mapping before touching held-out results. A
side-head trained on frozen MTP features is a separate candidate
only after this cheap controller; retraining the token-prediction
head changes proposal quality and needs its own artifact/exactness
and equal-cost controls.

## Final decision

Execute the frozen learned controller on fresh held-out continuing
conversations against K=3/C=0, the strongest calibrated fixed C/K/D
setting, native adaptation and the same-budget no-op controller.
Interleave both orders on the same model, GPU, source, prompts, seeds
and sampled decode. The **primary metric is pooled returned output
tokens divided by complete native request seconds**, including
controller/feature work, tokenization, new-input prefill, draft,
verification and detokenization. Also report paired whole-conversation
uncertainty, output-token ratio, total wait time, format and
task-specific functional coverage, prompt-length and output-phase
cells, actual action distributions, and exact-loop exclusions.
Acceptance, predicted survival, classifier F1 and per-round cost
are explanatory diagnostics.

Later turns must prove native checkpoint/KV reuse through stable
prefix digests and positive cached/new-token receipts. Run all gates
and benches on an explicitly non-production rented GPU, never the
local rig or a production Memra host. Before renting compare Nebius,
Verda and Jarvis spot/on-demand and applicable Israel AWS/GCP
offers; no Vast or RunPod use. Verify a real CUDA allocation before
staging, keep the pod on goal-linked work, and destroy it with a
direct provider-list absence check when receipts are banked. No
served policy moves without Memra #673's exact sampled and
vendor-default endpoint gates.
