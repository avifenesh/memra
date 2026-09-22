# Native request-depth check

Freeze the classifier before observing native timings. The runtime parent is
`5450580fe2e5eb5c34cd17452f472ed368034f41`, source archive
`943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8`,
the exact runtime used for the preceding latest-engine comparison. All arms use
the same resulting executable per model. `prepare_native.py` changes only the
two research drivers and their shared routing/I/O files; model math, sampler and
native prompt-checkpoint reuse stay at that pin.

## Workload and clocks

- Qwen3.8-27B NVFP4+Q5_K with its full embedded MTP head; Gemma 4 12B QAT Q4_0
  with the full Q8_0 assistant. Artifact locks come from the pinned source.
- One non-production RTX 5090, one scored process at a time.
- Eight turns with approximately 16K initial input tokens, 512 new tokens per
  turn, 49,152 context capacity, temperature 0.7, top-k 20, top-p 0.95 and the
  model's default thinking template.
- Requested output includes three prose, two code, two numerical and one mixed
  task per conversation. Reference material is explicitly marked as data.
  The classifier receives the actual latest user message, not a workload label
  or an excerpt selected using the expected class.
- The output cap includes reasoning tokens. This measures bounded native
  generation, not task-completion quality, HTTP or vendor-default serving.
- The primary metric is returned output tokens / complete native request
  seconds, including routing, controller installation, prompt rendering,
  prefix restore/suffix prefill and generation. Model loading, warmup and
  receipt I/O are outside this clock. No artificial CPU/prefill overlap is
  assumed.

## Equal-budget calibration

Use three calibration conversations per model. Sweep fixed K=1..7 on Qwen and
K=1..5 on Gemma in rotated/reversed order. No held-out output selects a profile.

From these same records:

1. Select the global fixed K by pooled complete-request throughput.
2. Select K per requested-output class by pooled complete-request throughput.
   A class change from global K needs improvement in at least two independent
   calibration conversations. Mixed/unknown requests retain global K.
3. Fit the span-based policy from the same fixed-K span records, using its
   established class definitions. Its starting global K is the same global
   choice as the fixed and prompt policies.

The old 2/4/4 and 3/3/4 span profiles are prior observations, not imposed answers
for this new request-level mapping.

## Correctness and held-out comparison

Before scoring, require greedy target-token correctness on the prompt policy,
sampled token-tape equality against a separately specified per-turn fixed-K
schedule, positive speculative engagement, and native checkpoint reuse on every
later turn. Check recorded routing decisions and actual full-round depths.
Terminal budget shortening is distinct from a policy change.

Run six fresh held-out conversations per model under four arms: native,
calibrated global fixed K, calibrated frozen span policy, and prompt routing.
Pair consecutive seeds with reversed arm order; rotate the first order across
pairs. Retain every requested run, including errors and interruptions. An error
stops scoring until repaired; no missing or incomplete arm is silently dropped.

Exclude an entire matched scenario/policy set if any member has independently
confirmed exact output looping. Keep every raw output and the exclusion reason;
do not replace the selected seed. Calibration applies the same matched-set rule.
Fewer than two clean calibration scenarios or five clean held-out pairs do not
support a performance promotion. Loop thresholds and the scenario/seed list are
frozen with the driver before the first native generation.

Report per-model/per-class rates, paired deltas and wins, requested versus
observed output behavior, fallbacks, routing time, exclusions, full input/token
tapes and hardware/source/binary bindings. A native gain is not inferred from the
CPU classifier's latency.
