# Gemma 4 12B QAT numerical contract

Scope: issue #398. This is a numerical diagnostic on one local GPU. It does not
estimate base-model quality, select a serving default, or qualify model support.

Pinned model SHA256:
`93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
Pinned llama.cpp source: `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29`.
The instrumentation patch records raw logits immediately before the server sampler.

The first cell uses the named `clinical_knowledge/test/52` counterexample and the
first three agreeing controls below 800 tokens in the original 600-item order.
Inputs retain the original prompt SHA256 and prior result. All requests use raw
prompts, BOS enabled, one output token, temperature zero, seed zero, no prompt
cache. HTTP and direct API run sequentially against the same freshly built
libraries. CPU threads are capped at two. Context is 8192; microbatch is 2048.

Arms: pinned HTTP invocation; HTTP with `--ctx-checkpoints 0`; previously
aligned direct API; direct N-4/4 split; direct output limit set to one; and the
server's common initialization with that output limit. HTTP adds only test
authentication, a local port, and verbose logging. All API arms print every input
token ID and dump all vocabulary logits. The original API-default and common
context-conversion modes remain available to reproduce the earlier controls.

Compare token identity, argmax and raw top-five logits. Agreement on argmax alone
does not establish a matched numerical contract. Full-vocabulary and layer/operator
captures follow only if the first cell reproduces a discrepancy. No FP16 model
mirror is allocated. The model remains read-only.

Static lead before execution: the pinned server creates checkpoints at prompt
length minus four tokens (`tools/server/server-context.cpp`, checkpoint offsets).
All 600 banked raw HTTP captures report `tok_idx=3`; the banked direct probe
processes each short prompt in one `llama_decode` call. The decisive follow-up
pairs default HTTP with a direct two-call split at N-4, and checkpoint-disabled
HTTP with the original unsplit direct API. This is a hypothesis until measured.

A read-only evaluation callback is prepared for the split/unsplit direct probe.
It captures the last token row at layer-zero projection boundaries and each layer
output. Its own logits must match the uninstrumented arm before a first differing
stage is interpreted. If the callback changes logits, it is intrusive evidence
and cannot identify a production first-bad-layer by itself.

Current main already has a different Gemma prime program from the issue's pin:
PRs #561 and #564 route dense text through quantized-KV chunked prime. Their
measurements do not retroactively describe the original v0.136.0 run. The native
reproduction therefore needs its source pin named explicitly; current-head
behavior is a separate row.

The pinned CUDA source also makes the suspected numerical distinction concrete:
`ggml_cuda_mul_mat` selects MMVQ for eligible narrow batches and MMQ for wider
ones. Its Q8_1 matvec quantizer stores `(scale, sum)` in half precision; the Q4_0
MMQ activation path uses D4 float scales. This source observation is a candidate
mechanism, not a measured attribution. The trace control and last-row comparisons
are required before naming the first observed divergent stage.
