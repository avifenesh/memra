# Cold-prefill head-of-line design

Date: 2026-08-12

## Problem and invariant

The Qwen3.6-27B mixed90 single-card workload has a formal clean-throughput
knee at c=12. At c=16, every repetition admits two cold misses in the first
wave. The batched worker then executes these phases in a fixed order:

1. admit requests;
2. synchronously prefill every eligible interactive session;
3. advance all decode-ready sessions in batched decode.

The configured `MEMRA_PREFILL_TICK` budget is per session, not per scheduler
iteration. With two long misses, phase 2 therefore makes two sequential
1,024-token `prime_cache` calls before phase 3 can begin. The diagnostic trace
measured the one-call phase at 290.45 ms median and the two-call phase at
580.2 ms median. Cache-hit sessions are ready throughout that interval but
cannot advance on the sole CUDA worker.

The implementation must preserve each request's token order, cache position,
sampler history, prefix-cache boundaries, and exact decode behavior. A session
may consume at most one scheduler-level prefill chunk per iteration. Gemma-4's
eager-only prime and Step35's request-level `seq_end` contract must not be
silently admitted to a generic chunk batch.

## Candidate ranking by blast radius

1. **Decode-ready phase first.** Move phase 3 before phase 2 when both classes
   exist. This is the smallest edit and gives ready rows one token before the
   cold hold, but the same cold work still serializes later in the iteration.
   It is a latency-ordering change, not a credible saturated-throughput knee
   mover, so it is not the first scored candidate.
2. **One global interactive prefill slice per iteration.** Rotate cold sessions
   and spend at most one 1,024-token chunk before returning to decode. This
   bounds each hold and directly improves fairness, but total single-card GPU
   work is unchanged and cold completion is delayed. It is a useful QoS lever,
   not the best first mechanism for the lane's throughput verdict.
3. **Continuation-capable cross-request chunk batching.** Take one bounded
   chunk from each eligible cold session and execute one existing
   `prime_cache_batch` call rather than N sequential `prime_cache` calls. This
   changes the prefill numeric configuration and scheduler bookkeeping, so its
   blast radius is larger than reordering, but it reduces the measured critical
   path and can therefore move the knee. The engine already supports and gates
   fresh and carried batches; the server currently uses that support only for
   complete short interactive prompts and bounded dark-lane groups.
4. **Overlap prefill and decode on independent CUDA streams.** This could hide
   work rather than only batch it, but it needs new same-device stream, cache,
   scratch, and publication ordering. CUDA streams merely express possible
   concurrency; actual overlap depends on resources and synchronization, and
   this engine explicitly quarantines same-device PP stream placement. This is
   the highest-blast option and is not justified before the existing batch path
   is tried. See NVIDIA's current
   [asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Selected increment

Extend the interactive prime-batch scheduler to admit bounded chunks for
non-eager, non-Step35 sessions whose cache position equals their fed-token
count and which have no prefix/checkpoint boundary pending.

- Each selected session contributes at most
  `min(prefill_budget, MEMRA_PRIME_BATCH_MAX_T, queued_tokens)` tokens.
- A selected session is marked advanced for the iteration, so the existing
  multi-round batch-former cannot drain a long prompt before decode.
- Success appends only that chunk to `fed` and sampler history, keeps the
  remainder queued, and marks prefill complete/seeds the prefix cache only when
  the queue becomes empty.
- Failure restores each drained chunk to the front of its original queue and
  falls through to the existing per-session path.
- Existing complete-prompt batching, prefix fanout, boundary-stopped primes,
  eager-only models, and Step35 remain unchanged.

No new flag is needed. `MEMRA_PRIME_BATCH=1` already disables cross-request
prime batches and is the rollback/control arm. If this increment wins, its
expanded scope will be documented in `docs/FLAGS.md` and the naked default will
remain enabled. If it does not move the knee, the scheduler change will be
removed; the negative raw evidence, not a dead experimental path, will remain.

## Gates and decision rule

Before and after the change, on the required GPU surface:

- `kernel-check`: ALL GREEN;
- `run-gen`: prefill/decode argmax MATCH;
- `run-spec`: K=1..8 self-consistency PASS.

The after arm also needs a targeted Q27 carried-prime batch gate and a real
server trace proving that long cold requests used batch calls rather than the
old serial single-call signature.

The scored verdict uses the frozen c=8,12,16,20,24 mixed90 harness, N=5 per arm,
interleaved by round under one Box1 GPU-lock hold, with the base and candidate
binaries fixed before the run. Promotion requires all integrity and output
checks clean, the formal first-decline knee above c=12, no material c=8/c=12
throughput regression, and cache-hit TTFT p95 held. Otherwise the result is a
refutation and the published default does not change.
