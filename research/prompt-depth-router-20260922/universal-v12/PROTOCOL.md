# One Qwen C/K/D policy across code, prose and math

**Preparation only. No V12 GPU or learned-policy result exists.**
The active V11 training run keeps its frozen source and topic-specific
selection. This separate lane uses V11 measurements as possible
training evidence, then makes one shared decision on fresh inputs.

## Serving question

Can one small controller choose MTP draft top-k K, confidence stop C
and draft depth D from bounded prompt tokens and generated history
without a user-declared task label, while improving complete native
request tok/s and preserving code, open-prose and math quality?

The Qwen backbone and MTP head remain frozen. The controller may
adapt its numeric actions to the input, but the **same weights,
binary, action rules and target decode** run on every domain. The
domain labels in the benchmark are used to balance tasks and check
regressions. They are never passed to the controller or used to
choose a different runtime arm.

## Fresh source and split

`workloads.py` pins full Google MBPP, OpenAI GSM8K test, and AllenAI
WildBench v2 sources. The latter is attributed to WildBench under
CC BY 4.0; selected prompt and checklist text stays in private
Darklanes receipts. NVIDIA SPEED-Bench supplied category context
only. Its evaluation-only dataset is not used to fit this policy.

The generator excludes every V9 code task, the first V10
instruction/math task set, and the full V11 instruction/math split.
WildBench tasks must be standalone English user prompts with a
quality checklist, within 30 to 5,000 characters, and without
redacted/toxic flags or the recorded contact/key patterns.

The frozen private manifest is SHA-256
`b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477`.
It has one eight-turn qualifier, 16 eight-turn training
conversations, eight eight-turn validation conversations and 24
eight-turn final conversations **per domain**. The 392 task IDs
within each domain are distinct across phases. Final prose has
192 tasks, including 128 from creative writing, editing, roleplay,
brainstorming and advice categories. These are custom continuing
conversations, not official MBPP, GSM8K or WildBench scores.

The private phase seal keeps validation and final prompt files off
the GPU host until their releases. Its training projection SHA-256
is `2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00`,
validation projection
`bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313`,
and full final manifest is the hash above. The final group has a
separate sealed commitment.

## Fit and choose once

1. Use only qualification/training prompt files and replayed V9,
   V10 and V11 measurements to build candidate controller weights.
   Collect new randomized K/D/C observations on prose if training
   source coverage is insufficient. Older selected-policy outcomes
   remain observational; C labels come only from actual offers.
   Source-specific complete-request rates price K utility, while
   same-GPU randomized rounds price D/C actions.
   Seal and replay the fresh prose native bytes and the training-only
   phase package before converting them into K/D/C rows. The replay
   verifies all 112 prose training sessions and rejects any validation
   or final prompt file in the training archive.
   The fresh prose collector runs K=3/10/20 fixed-D3 and randomized-D
   sessions plus one randomized-K session on each of 16 training
   conversations. The fit pools those labels with the pinned code and
   V11 non-code rows. Candidate weights use the same bounded prompt
   prefix, generated-token history, and prior-turn features at runtime.
   A prose-balanced candidate explicitly triples the weight of fresh
   prose rows during fitting; validation remains disjoint and unweighted.
   Before freezing arms, a training-conversation-heldout preflight
   must show that the same first-16/32 token buckets used by the K
   controller distinguish code, open prose and math above a 0.6
   balanced-accuracy floor, with every stratum recall at least 0.5.
   This classifier is a diagnostic only and never routes requests.
2. Before validation, freeze one candidate inventory and a bounded
   fixed C/K/D menu that includes D=1/2 as controls, plus exact
   model-running no-op twins. Hold the target sampler at top-k 20,
   temperature 1 and top-p 0.95 on one nonproduction GPU. Keep
   `max_new=4096`, `ctx=65536`, full embedded MTP engagement and
   eight-turn native KV receipts. Price and tag the rental under
   the development-provider policy before allocating it.
   The menu contains nine fixed controls including K=3, D=1/2/3/4
   and three C cutoffs measured from training offers, plus three
   selectable joint C/K/D candidates, two component diagnostics,
   and their exact model-running no-op twins. The K-only and C/D-only
   diagnostics cannot become the selected universal controller.
3. Score every arm on the **same** mixed validation conversations.
   Choose one global fixed control among quality-eligible fixed arms
   by pooled returned tokens / complete native request seconds.
   Retain each domain's validation-best fixed arm as a diagnostic
   regret ceiling; it is not a per-request route.
4. Select one immutable learned policy label and model manifest hash
   on validation. It must clear the quality/cap/loop guards and have
   positive paired native tok/s point estimates against its no-op
   and each quality-eligible fixed control in code, prose and math.
   Rank survivors by their **worst domain** margin, then pooled
   margin. If none survives, record a global no-go and leave final
   prompts unopened. There is no `chosen_by_domain` primary.

Only the single selected policy, its no-op, the K20/D3/C0 reference,
the global fixed control and the validation-best fixed controls enter
the final arm file.
Every listed arm runs on every domain. No field switch occurs in
the native command.
Seal the mixed native outputs, hidden-task grades, both-order prose
judge receipts, model weights, selection and opened phase packages in
a private archive. Recompute the rate and quality summaries from the
archived bytes before declaring pipeline completion. Raw prompts,
generated answers and judge analysis remain private.

## Final result and prose protection

The decisive rate is pooled returned output tokens divided by
complete native request seconds, with paired bootstrap intervals
resampling whole conversations. Startup, prompt preparation and
post-timing graders stay outside request clocks. Report per-domain
rates and output/time ratios beside the pool. Acceptance is a
training signal and diagnostic, not the performance score.

The one selected policy must beat its model-running no-op and the
single global fixed setting on pooled tok/s with a positive lower
paired interval bound. It must also have no negative lower paired
interval bound against each domain's validation-best fixed setting
for code, prose and math. Failure of any domain is a no-go even if
the pool rises. No exact-loop conversation enters a rate; caps,
loops and actual K/D/C action counts remain visible.

Code quality uses two hidden MBPP tests per task, format and syntax
checks in a credential-free bubblewrap namespace with no network or
host data mounted. The sandbox must pass a runtime preflight before
any generated code is executed. Math
quality requires a complete numeric `#### <number>` final line.
Prose quality uses the frozen WildBench checklist for the current
standalone prompt in a blinded pairwise comparison, with both
response orders judged. The independent judge model, prompt
template, parsing rule and budget are pinned in the private
`judge-config.json` with SHA-256
`dd01fb5c3fa3cc22919dc3ef6f09931935a2b9fe5db67d4151b79cd6a7806fe7`.
The pinned Bedrock global Sonnet 5 profile was reported active
by the provider control plane on 2026-09-26. The conservative
accounting ceiling is not a provider price quote; actual pricing
must be checked before judge requests. A disagreement between
reversed judgments counts as a tie. The prose comparison with its
validation-best fixed arm
must have point win fraction at least 0.5 and a nonnegative lower
paired 95% bound relative to 0.5. A weaker prose quality result
blocks a universal claim, regardless of tok/s.

Sampled target-distribution parity and vendor-default endpoint
qualification are additional requirements before a serving
setting changes. A bounded final result does not prove optimal
C/K/D choices for every possible prompt.
Report the actual K/D/C action counts. If K never varies, the
result cannot be called adaptive K, even if a learned C/D controller
improves the global rate.
