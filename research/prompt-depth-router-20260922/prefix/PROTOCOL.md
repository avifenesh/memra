# Prompt-prefix routing across prompt lengths

This is the primary continuation requested by the owner. The forecaster reads
only the beginning of the user prompt, up to X tokenizer tokens, once per
request. The earlier decoder-prefix tree is retained as a separate diagnostic.
No LLM or draft head is trained or changed.

## Primary comparison

- User-prompt length targets: 256, 1,024, 4,096 and 16,384 tokenizer tokens.
  Retain exact user-token and complete model-input counts; report any small
  preparation tolerance rather than presenting approximate lengths as exact.
- Requested final output: prose and code, reported separately at every length.
- Prefix budgets: X=64,128,256 user-content tokens.
- Fixed control: **K=3 for every prose and code request**.
- Adaptive arms: predict from the bounded prefix, then use K=2 for prose,
  K=4 for code/numeric and K=3 for mixed/unknown input. Keep the selected K
  for the request. The baseline is not replaced by a separately tuned fixed K.
- The first predictor is the existing small lexical rule. A shared shallow
  tree is optional if it improves the prefix prediction; per-model neural
  training is outside this research.

Use the same prompt, model/artifact, seed, sampler, output budget and hardware
for the fixed/adaptive comparison. Freeze scenario and order lists before
scoring. Rotate and reverse arm order across paired scenarios. The proposed
starting sample is six scenarios per model/type/length.

Each length/type entry is an independent request with a fresh native cache.
Do not accumulate earlier requests into its prompt length. Model weights are
warm; K=2,3,4 all receive the same pre-measurement warmup. No continuous-session
or generated-KV-reuse claim is made for this matrix.

## Enforce the input bound

Reuse normal serving tokenization. Do not tokenize the whole user prompt a
second time for classification. Locate the user-content token span using
tokenizer/template metadata, then give the forecaster only its first X tokens.
Template-boundary handling is bounded and its added time is recorded.

The prediction function receives only the bounded token slice and its decoded
prefix. It cannot scan the rest of the prompt, its filename or the expected
output label. Do not scan all text before applying the cap, inspect a tail
excerpt, or re-run the forecaster per generated token.

Changing content after the inspected prefix must not change its prediction
when those first X token IDs are unchanged. Record the inspected token IDs,
consumed count, prefix bytes/hash, predicted kind, chosen K and routing time.
Test actual tokenizer-token limits, not whitespace-word substitutes.

The main instruction-first prompts keep the request near the beginning.
Separate robustness prompts put the task after a long reference. Insufficient
prefix evidence falls back to K=3. Conflicting tails attached to an identical
prefix are explicitly ambiguous: no prefix-only predictor can distinguish
them, and it must not secretly inspect their tails.

## Cover the output phases

Use default thinking/sampled behavior for the main free-generation comparison.
Choose output budgets on calibration examples that actually reach final code
and prose. Do not repeat the 512-token Qwen coverage failure. Retain reasoning
and final-output counts separately, plus real code/prose annotations.

Try 8,192 returned tokens on the separate versioned-record qualification task,
then 12,288 only if coverage fails. Use the first budget at which all eight
qualification requests cover their requested format without an exact repetition
loop; freeze it per model before any scored request. The context limit is
32,768 tokens. Closed Python fences must contain at least 80 bytes of parseable
Python; prose must contain at least 200 bytes and no source-code fence. These
are format checks, not a claim that the generated program solves its task.

Each scored scenario contributes one matched request for every length/type.
If any arm triggers the pinned exact-repetition screen, exclude that same
request from all four arms. Never replace its seed. The primary requested-format
table retains capped responses and format failures and reports them; also
show the separate subset where every arm covers the requested final format.

If controlled assistant-prefix or non-thinking cells are useful, label them as
separate diagnostics and do not pool them with default free generation. Failed
format coverage is reported; no evaluation seed is replaced or silently dropped.

## Metrics and qualification

Primary: complete native request seconds and returned-token throughput,
including normal tokenization, prefix extraction/decoding, forecasting,
configuration, prefill and generation. Keep model-load/warmup and receipt I/O
outside that clock and identify the boundary.

Also report forecaster time against both X and full prompt length, classification
and fallback rates, actual K engagement, output phase coverage, per-cell paired
changes and relevant decode-cycle diagnostics. Fixed K=3 is shown for prose
and code independently at every prompt length.

Qualify greedy target identity and sampled identity against the selected fixed-K
schedule. Keep all raw runs, exact inputs/outputs, model/source/binary bindings
and predeclared matched-set loop exclusions. Run checks/benchmarks on hosted
CPU CI and an explicitly non-production GPU host; never on the local rig.
