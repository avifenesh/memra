# Qwen C/K/D non-code transfer evaluation

**Execution stopped before final scoring after a mistaken source
attribution.** Memra #673 proves the mainline sampled-C discard path
can change the target distribution. The archived v8 source tied to
this study's exact binary SHA instead appends each sampled pick
before a fixed or learned C stop on its sampled graph and eager paths,
and refuses positive sampled PMIN0. That specific counterexample
does not apply to this binary. The research VM completed both
qualifiers and 78 of 320 intended held-out arms before the scheduler
was stopped. Those raw native sessions were sealed and replayed
privately. No topic-level result or serving change is claimed. A
clean complete rerun is required; sampled distribution qualification
remains separate. The protocol below records the intended frozen
battery, not a completed one.

This is a transfer test of the **frozen v9 code-trained controller
weights**, not new non-code training. The Qwen3.8-27B target, its
full-vocabulary embedded MTP head, and the native v8 research binary
remain pinned. K is MTP draft sampler top-k; target top-k stays at 20.
D is offered draft length and C is the after-offer continuation choice.

## Frozen task strata

Use two separate non-code strata. Each has one eight-turn qualification
conversation and 16 disjoint eight-turn final conversations:

- **Instruction-following prose:** pinned Google Research IFEval prompts,
  excluding prompts with code-related terms by the frozen regex in
  `workloads.py`. Use each official prompt unchanged. Grade each final
  response with the pinned IFEval strict evaluator, reporting both
  all-instructions pass and instruction-level pass rates.
- **Grade-school math:** pinned OpenAI GSM8K test questions, with the
  same short instruction after each question to end the solution
  `#### <number>`. Grade the final numeric answer against the pinned
  source answer. This is a continuing-conversation request shape,
  not an official standalone GSM8K score.

Each conversation consists of eight distinct tasks. Tasks and
qualification prompts are frozen by source SHA-256 and generator
SHA-256 before any GPU output. The model receives only the current
task prompt and prior committed conversation turns, never the answer
key or the IFEval grader metadata. Later turns must prove positive
native cached and new input tokens.

## Native alternatives

All alternatives run on one explicitly nonproduction research GPU with
the same model, binary, prompts, seeds, target sampler and request
shape: temperature 1.0, top-p 0.95, `max_new=4096`,
`ctx=65536`. Rotate arm order across conversations.

1. Fixed draft K=3/10/20, D=3, C=0.
2. Fixed draft K=20, D=4, C=0.
3. Fixed draft K=20, D=3, C=(0, 0.50844276) and
   C=(0, 0.937437713), both carried from v9 code training.
4. v9-selected new-only learned C/D at fixed draft K=20 and its
   model-running K=20/D=3/C=0 no-op.
5. v9-selected augmented joint K/C/D policy and its model-running
   K=20/D=3/C=0 no-op.

The qualifier must show full 248,320-row embedded MTP engagement,
target top-k=20, native KV reuse on later turns, live learned C
decisions with nonzero controller time, and byte-identical
sampled no-op outputs versus fixed K=20/D=3/C=0. Final no-op bytes
must match that control on every turn of every conversation.
Record whether C stopped and continued; a uniform policy action
is a transfer observation if the native C model ran.

## Score and claim boundary

For each stratum separately, the primary score is **pooled returned
output tokens / complete native request seconds**. The clocks include
controller inference but exclude model startup and post-timing
graders. Report paired whole-conversation 95% bootstrap intervals,
output-token and elapsed-time ratios, actual K/D/C choices,
controller time, exact loops and capped turns. Acceptance is a
diagnostic, not the score.

Exclude an exact-loop conversation from a pair's rate; report it
separately. An arm is quality-eligible when it has no more looped turns
than fixed K=20/D=3/C=0 and its per-stratum task pass rate falls by no
more than five percentage points, while its cap rate rises by no more
than five percentage points. Report syntax/format details where the
source evaluator exposes them.

Do not choose model weights, thresholds or candidate arms using either
final stratum. Call a policy a transfer winner on a stratum only if it
passes that stratum's quality guard and its paired lower confidence
bound exceeds both its exact no-op and **every** quality-eligible fixed
alternative. Report prose and math independently. Even two positive
strata would not turn the earlier code-only result into a general
serving default; full sampled distribution parity and vendor-default
endpoint qualification remain separate under Memra #673.
