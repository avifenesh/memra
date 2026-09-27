# Full-head MTP learned-depth research

Status: complete. Both eight-set matrices are audited; see `RESULTS.md`.

Owner scope: MTP models only; Qwen first, then Gemma. Full vocabulary on both
sides. DFlash2 results belong to the earlier, separate lane and do not answer this
research question. H learning, vocabulary trimming, and confidence cuts are off.

This reuses the disclosed new Memra-native cost learner. The original remote H/C/D
implementation remains unavailable; this is not a claim to reproduce that source.

## Frozen design

- Qwen3.8-27B: pinned NVFP4+Q5_K target with its own embedded MTP block. Both
  target and draft projections cover all 248,320 tokens. K is actual sequential
  MTP draft steps, allowed 1..7, learner initialized at K=3. Fixed K=3 is chosen
  before measurement from the existing native MTP CLI baseline; it is not a
  best-fixed-depth claim for this workload.
- Gemma 4 12B: pinned QAT Q4_0 target and matching QAT Q8_0 MTP assistant, full
  262,144-token head. K=1..5; fixed K=5; learner initialized at K=5.
- Four arms: fixed depth (instrumented), native adaptive depth (floor 1), the same
  native adaptive depth with timing fences, and learned depth. Fixed and learned
  share timing instrumentation. The native/measured pair detects instrumentation
  cost and must produce identical sampled prompt and output tapes.
- Learner unchanged: 16 complete rounds per observation block, pooled useful
  tokens / complete-round time, EWMA 0.5, 1% hysteresis, one round-robin probe after
  eight exploitation blocks, every candidate initially explored. Probe costs count.
- Qwen maps controller action to K+1 to include the bonus token in the reward
  bound. Only K controls drafting. Terminal rounds do not train; their time and
  any native commit overshoot stay in request E2E. Public output stops at EOS or
  the length cap. Every request is primed cold in all arms; learner state alone
  carries between conversation turns. This holds prompt numeric lineage constant.
- Real repository code and concrete analysis tasks in `workload-code.txt`, with
  source hashes. No repeated padding or flattened chat transcripts. Each family
  uses its native template; Qwen reasoning is replayed separately from content.
- Eight continuing turns per Qwen run; two independent eight-turn conversations
  per Gemma run, resetting learner/history between them. Temperature 0.7, top-k 20,
  top-p 0.95, no penalties, output cap 1,024, context reservation 49,152. Actual
  prompt token counts are recorded per family; no claim of equal cross-family counts.
- Eight repetitions using two cycles of a four-treatment Williams design. Every
  policy has the same seeds within a repetition. Reports compare learned against
  both fixed and adaptive controls, retain every pair, and pool tokens / E2E seconds.

## Qualification and execution

1. Controller tests, native reasoning-template tests, runner validation tests, and
   real CUDA build on the pinned RTX 5090 test machine.
2. Short and code-context greedy gates across all four arms, with nonempty outputs,
   full-head/MTP engagement, cache invariants, candidate-depth engagement and exact
   token comparisons. A mismatch is a failed gate, not a reason to loosen it.
3. Serialized sampled measurements using the existing `/tmp/memra-5090.lock`, at
   least 10 seconds warmup and 60 measured seconds per run. A too-short run requires
   a uniformly revised protocol and fresh controls; it cannot be accepted as scored.
4. Save commands, immutable source/binary/artifact/workload hashes, GPU telemetry,
   stdout/stderr, prompt/output tapes, decoded text, per-turn and per-round metrics.
   Native-session results are not HTTP serving measurements or deployment changes.

No model, learner, workload or runner changes during scored measurements. Complete
Qwen before starting Gemma GPU measurements. Preserve old evidence separately.

## Pre-measurement gate correction

The first Qwen short gate used conditional exact-prefix reuse. Native commit
overshoot varied with K: on turn 3, native/measured reused the cache while
fixed/learned primed cold. Outputs first differed at token 39 despite identical
prompt tapes. The plain-target K=1..8 oracle had passed. No scored runs were
started. This is retained as `qwen-mtp-short-v1`, a failed gate, not performance
evidence. The revised harness primes every request cold to isolate depth from
cache reuse; all four arms must pass the same exactness gate again. Gemma already
uses fresh requests. This correction is made before any scored measurement.

## Timing interruptions and complete-set restart

Unrelated GPU unit tests interrupted the first timing attempt and later the fourth
matched set. The monitor stopped only this study's process. In v3, cycles 0..2
completed without detected interference and passed native/measured identity checks.
Cycle 3 is excluded in full, including its two completed arms; it is retried using
the original seed and order. Subsequent execution checkpoints one complete four-arm
set at a time. Hardware interference triggers a whole-set retry, never selection
based on measured speed. The eight-cycle schedule and every model/policy/sampling
setting remain unchanged. Partitioned receipts record cycle indices and runner hashes.
