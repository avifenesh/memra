# Next native comparison: output-aware K without head training

Owner and scope: continuation of memra issue #635. The research hypothesis is
that inexpensive prediction of upcoming prose/code/numeric output can choose
K from {2,3,4} across models without model-specific neural training. LLM and
draft-head checkpoints remain frozen.

## First correct the coverage

Use calibration-only runs to choose tasks and output budgets that actually
reach the requested prose and code phases. The 512-token Qwen code-request
records do not meet this requirement. Retain normal reasoning behavior in the
main experiment and allow enough generation to reach code; report reasoning
and final-output coverage separately.

Every model must have sufficient actual prose and code examples in calibration,
not only prompts requesting those formats. Require at least 64 known-label
examples per training class, and at least 100 code-labelled evaluation windows
spread across five conversations, before interpreting code-transfer metrics.
Record the count of requested-code turns that reach a final code answer.

Include fenced code, unfenced code/JSON, prose quoting code markers, and
reasoning-to-final transitions. Retain inspectable content annotations for a
stratified sample. The detector being evaluated must not supply its own labels.

Freeze tasks, budgets, seeds and prediction rules before the new evaluation.
If evaluation fails coverage, report the under-covered comparison and retain
all runs. Do not replace seeds or silently discard inconvenient requests.
Code/prose continuations with supplied assistant prefixes may be separate
diagnostics; they must not be pooled with free-generation session results.

## Cheap predictors and explicit K controls

Compare fixed K=2, K=3 and K=4, plus:

1. The existing prompt-only rule with the explicit requested mapping.
2. A bounded decoder-prefix rule tracking actual code fences and numeric spans.
   Fence tracking must distinguish a real line-opening fence from a quoted
   literal marker in prose.
3. One shared shallow tree, fitted once on the covered calibration mixture.
   Test transfer using a model excluded from fitting, with all relevant formats
   represented in that training fold.

All adaptive arms start at K=3. Confident prose proposes 2, code/numeric proposes
4 and uncertainty proposes 3. The initial comparison uses the recorded
two-proposal, one-step switching rule; a different switching rule is a separate
arm rather than an unrecorded adjustment. Report both request-only selection
and changes during generation, since reasoning and final code can share a turn.

A first-two-draft-token preview is a possible later predictor. It requires real
proposal logging and an executed implementation. Verified future output from
historical traces must never be substituted for an available draft preview.

## What decides the result

Pin the runtime commit, checkpoints, tokenizer/template, hardware and sampled
decode settings. Build and qualify the code before allocating development GPU
time. Re-run legal-depth correctness, dynamic-policy engagement and replay
against the recorded K schedule. Verify native checkpoint reuse on later turns.

Use paired, interleaved complete requests with the same inputs and both arm
orders. Include predictor, switching, prefill and verification costs in request
time. Keep the strongest fixed control. Forecasting F1 and shadow K agreement
remain diagnostics.

Report code/prose/numeric support, transition behavior, abstention, actual K
distribution, request throughput and paired results per model. Preserve every
run and any predeclared matched-set loop exclusions. A shared predictor that
helps one model and not another is a scoped result; it is not a reason to switch
this research toward draft-head retraining.
