# Qwen code-trained C/K/D: clean non-code transfer rerun

The first v10 battery stopped prematurely after the mainline
sampled-C discard bug in Memra #673 was applied to a different
research binary. The archived v8 source tied to binary SHA-256
`84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f`
retains sampled picks before fixed and learned C stops in sampled
graph and eager paths, and refuses positive sampled PMIN0. Full
sampled-distribution and serving qualification still require
separate gates.
Before any qualifier output, the supervisor runs
`source_exactness.py` against that exact source archive and binary.
It checks all three sampled branch orders, positive sampled PMIN0
refusal, and a small two-token discard counterexample. The receipt
is sealed and replayed with the native data; it is not a full-model
distribution test.

This rerun holds the v9 code-trained controller weights, Qwen3.8-27B
GGUF SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
full-vocabulary embedded MTP head, native binary, and ten v10 arms
fixed. K is MTP **draft** top-k; target top-k stays at 20. Target
sampling is temperature 1.0, top-p 0.95, `max_new=4096`, and
`ctx=65536`. C=0 and positive fixed C controls, learned C/D and
joint K/C/D, plus their model-running no-op twins all run on one
newly accepted nonproduction GPU. Rotate arm order across
conversations. Later turns must prove native KV reuse.

The fresh workload manifest is SHA-256
`dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8`.
It uses the already frozen v11 **training** split as this
code-trained controller's held-out transfer set: 16 eight-turn
IFEval instruction conversations and 16 eight-turn GSM8K math
conversations, plus disjoint eight-turn qualifiers. Their IDs do
not overlap the stopped v10 prompts. The source v11 split is
SHA-256
`71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df`
and was frozen without using v10 partial rates to choose prompts.
No arm, threshold, controller weight, or target sampler was
selected from the interrupted data. After this transfer test,
these prompts may become training-only measurements for a later
controller; the separate v11 validation and final prompts remain
untouched.

Qualification must show the full 248,320-row embedded MTP path,
target top-k 20, pinned model and binary, positive cached and new
tokens on later turns, live C/K controller time, and byte-identical
sampled no-op output IDs versus fixed K=20/D=3/C=0. Final no-op
bytes must match on every turn. Task quality is the pinned IFEval
strict evaluator and a complete numeric `#### <number>` on the final
answer line for GSM8K.
These are internal continuing-conversation quality checks, not
official standalone benchmark scores.

Score each topic separately by pooled returned output tokens /
complete native request seconds. Controller time is included;
startup and post-timing graders are excluded. Report paired
whole-conversation 95% bootstrap intervals, output/time ratios,
actual K/D/C actions, loops, caps, and task pass counts. Acceptance
is a training signal and diagnostic, not the performance score.
Exclude exact-loop conversations from each paired rate and report
them separately. An arm must pass the frozen task-pass and cap
guards versus K=20/D=3/C=0. A learned transfer winner must have a
positive lower interval bound versus its exact no-op and every
quality-eligible fixed arm on that topic.

Even a positive native transfer rate cannot be promoted to a
serving setting without sampled-distribution qualification on the
exact model/source/binary/GPU tuple and vendor-default sampled
endpoint gates. The stopped partial archive is a separate
diagnostic receipt and is never pooled with this rerun.
