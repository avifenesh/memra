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
   Collect new randomized K/D/C observations on all three domains
   on the same GPU. Older selected-policy outcomes
   remain observational; C labels come only from actual offers.
   Source-specific complete-request rates price K utility, while
   same-GPU randomized rounds price D/C actions.
   Seal and replay the fresh mixed native bytes and the training-only
   phase package before converting them into K/D/C rows. The replay
   verifies all 336 code, prose and math training sessions and rejects any validation
   or final prompt file in the training archive.
   The fresh collector runs K=3/10/20 fixed-D3 and randomized-D
   sessions plus one randomized-K session on each of 16 training
   conversations per domain, interleaving domains in time. The fit
   compares fresh-only, mixed-history and augmented-history candidates
   using pinned V9/V10/V11 rows as the historical options. Candidate
   weights use the same bounded prompt prefix, generated-token
   history, and prior-turn features at runtime. Validation remains
   disjoint and unweighted.
   Historic K utility uses a reference rate from each source. D/C
   acceptance labels may use historic rows, while D timing and
   marginal C cost come only from randomized rounds on the current
   V12 GPU.
   Before freezing arms, a training-conversation-heldout preflight
   measures whether the same first-16/32 token buckets used by the K
   controller distinguish code, open prose and math above a 0.6
   balanced-accuracy floor, with every stratum recall at least 0.5.
   A weaker result is recorded and limits claims about first-token
   routing. It does not suppress validation of generated-history
   behavior. This classifier never routes requests.
2. Before validation, freeze one candidate inventory and a bounded
   fixed C/K/D menu that includes D=1/2 as controls, plus exact
   model-running no-op twins. Hold the target sampler at top-k 20,
   temperature 1 and top-p 0.95 on one nonproduction GPU. Keep
   `max_new=4096`, `ctx=65536`, full embedded MTP engagement and
   eight-turn native KV receipts. Verify the same physical GPU UUID
   at training and every evaluation phase, and carry it through
   validation selection and final scoring. Price and tag the rental under
   the development-provider policy before allocating it.
   During validation and final, finish prose native cells first. Run
   the judge on reserved CPU cores while the GPU completes code and
   math native cells on disjoint CPU cores. Seal the CPU split with
   the evaluation receipts.
   Before the long training battery, run D=1 and D=2 fixed controls
   on the pinned binary and replay their native depth, sampler,
   and KV receipts from the training-only archive. The same pilot
   loads pinned historical C/D weights and checks model-running
   no-op byte identity against fixed D3/C0 at K=3/10/20 before
   collecting the 336 fresh training sessions.
   Also require two synthetic, reversed-order independent checklist
   judgments to pass the frozen JSON parser on the trusted research
   host before downloading the large Qwen artifact. This access pilot
   uses no final prompt or customer content and is sealed with training.
   The menu contains 21 fixed controls: every K=3/10/20 and
   D=1/2/3/4 combination at C=0, plus three D3 C cutoffs
   measured separately from training offers at each K.
   It also contains five joint C/K/D candidates, three fixed-K
   learned C/D candidates, one K-only diagnostic, and exact
   model-running no-op twins. The joint and fixed-K C/D candidates
   are selectable as one universal policy. The K-only arm remains
   a diagnostic.
   Each no-op is byte-compared with the fixed D3/C0 control at
   its own draft K. A selected fixed-K C/D arm carries that
   same-K control into final.
   Two fresh-only joint ablations compare D/C's last generated token
   with its 4/16-token history windows, then test K's previous-turn
   acceptance feature. The shortest D/C variant still reads the
   last generated token.
3. Score every arm on the **same** mixed validation conversations.
   Choose one global fixed control among quality-eligible fixed arms
   by pooled returned tokens / complete native request seconds on
   the conversations shared by all eligible fixed controls.
   Retain each domain's validation-best fixed arm as a diagnostic
   regret ceiling using that same common cohort; it is not a
   per-request route.
4. Select one immutable learned policy label and model manifest hash
   on validation. It must clear the quality/cap/loop guards and have
   positive paired native tok/s point estimates against its no-op
   and each quality-eligible fixed control in code, prose and math.
   Its pooled margin against the global fixed control also uses
   identical unlooped conversations for both arms.
   Rank survivors by their **worst domain** margin, then pooled
   margin. If none survives, record a global no-go and leave final
   prompts unopened. There is no `chosen_by_domain` primary.

Only the single selected policy, its no-op, the full fresh controller,
both fresh feature ablations with their no-ops, the K20/D3/C0
reference, the global fixed control and the validation-best fixed
controls enter the final arm file.
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
Report paired final native-rate intervals for the full fresh trio.
The window-history arm versus last-token arm isolates the longer
D/C token window. Full fresh versus window-history isolates K's
previous-turn acceptance feature. Comparisons involving a
historically trained winner are diagnostics, since data source also
changes. Apply the same quality guards against fresh trio outputs
distinct from the selected policy. These diagnostics do not become
per-request routes.

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
`624cbb8478326ec7662d6e5aaa959e713cb3bf0330128dd42a7e0dc9b8a05bdd`.
The private judge custody pins its model, price source and cumulative
spend cap. At most three format attempts are allowed per packet;
every provider response is sealed and its usage counts toward that
cap, while only a parseable response casts a quality vote. The access
pilot verifies the model and template before
the native battery starts. A disagreement between
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
improves the global rate. A static K does not turn a qualified
one-policy throughput and quality win into a no-go when that K
is the best tested choice across the mixed workload. Report the
selected arm kind and configured draft K alongside adaptive
component flags.

At `origin/main` `e3a8402cb9f2d37ef91e7107b6f251cf3ca7d9ef`
on 2026-09-26, `docs/MODELS.md` names DFlash2 as Qwen3.8's
qualified served route. This study
compares settings within one pinned MTP artifact. A learned MTP
gain over fixed MTP does not establish a serving gain against
DFlash2. Any serving proposal must compare the qualified served
route on the relevant sampled workload and hardware with its
own exact artifact and endpoint receipts; artifact-specific rates
cannot be compared as if the artifacts were identical. Recheck the
serving registry at the time of any deployment decision.
