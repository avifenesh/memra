# Lane cx-sec3 results

## Verdict

**PASS — the constrained-decoding compile path no longer runs on the memra worker tick.**

Oversized or structurally pathological JSON schemas fail before worker submission. Accepted
schemas compile on a bounded per-model background thread; the request waits asynchronously for a
pre-header verdict while normal sessions and the worker heartbeat continue. There is no inline
factory or matcher fallback in `admit()`.

## What changed

- Pre-admit JSON-schema envelope:
  - maximum 512 KiB compact serialized schema;
  - maximum 64 raw JSON levels; and
  - maximum 32,768 JSON values as a coarse complexity bound.
- One lazy compiler thread per loaded model, with an eight-job bounded queue. The thread owns:
  - the first-time full-vocabulary TokTrie build;
  - `ConstraintFactory::new`; and
  - every request's `matcher(spec)` construction and initial error check.
- Five-second request deadline at both the HTTP wait and worker pending state. Timeout or compiler
  saturation is a retryable 503. Pre-admit bound violations are clear 400s; compiler-detected
  invalid schemas remain 400 with `response_format` named by the worker error taxonomy.
- Fresh constrained HTTP calls wait on a one-shot compile verdict before committing headers, so
  streaming calls cannot turn a compile rejection into an HTTP-200 in-band error.
- Successful matchers return to a pending-request map and re-enter normal admission. Expired,
  disconnected, and stale results are discarded. A worker-side state mismatch is an internal
  error; it never falls back to synchronous compilation.
- The tokenizer is shared immutably with the compiler through `Arc`; unconstrained requests keep
  `grammar == None`, allocate no compiler job, and take the existing admission path.

## Evidence

Final required gate:

```text
cargo test -p memra-server
test result: ok. 176 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Focused regressions:

```text
constrained::tests::json_schema_bounds_fail_before_compile                         PASS
worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress PASS
tests::deep_schema_fails_while_normal_decode_keeps_stepping                       PASS
tests::valid_response_format_preflight_preserves_generation                       PASS
```

The concurrent CPU harness holds a deep accepted-schema compile on its background thread past the
request deadline. During that interval the simulated normal decode publishes at least ten token
events, the heartbeat remains younger than its stall threshold, the constrained request receives
one retryable timeout verdict, and the late compiler result cannot reach the client. The handler
regression separately keeps a 64-step normal decode live while an over-depth schema receives a
pre-admission 400.

Source audit:

```text
ConstraintFactory::new  -> constrained.rs background compiler only
factory.matcher(spec)   -> constrained.rs background compiler only
TokTrie::from           -> ConstraintFactory::new on that compiler thread only
```

`git diff --check` passed. No GPU path, kernel, performance board, or generated performance surface
changed; this was the requested CPU-only serving lane. `cargo fmt` was not run.

## Upstream validation

- OpenAI's current Structured Outputs guide allows up to 5,000 object properties, 10 semantic
  nesting levels, and 120,000 characters across schema names/enum/const strings. The memra bounds
  count raw JSON shape and retain headroom around that compatibility envelope:
  <https://developers.openai.com/api/docs/guides/structured-outputs>
- llguidance 1.7.6 also applies parser fuel, lexer-state, and grammar-size limits. Those remain a
  second line of defense after memra's cheap pre-admit envelope:
  <https://docs.rs/llguidance/1.7.6/llguidance/api/struct.ParserLimits.html>

## Residual boundary

Rust cannot forcibly cancel CPU code already executing on a blocking OS thread. A compile that
crosses the five-second request deadline may therefore finish in the background, but it cannot
hold the CUDA worker tick: concurrency is one compiler thread per configured model, queued work is
bounded to eight jobs, already-expired queued jobs are skipped, the request has already failed,
and any late result is discarded. If one compiler were to remain stuck indefinitely, constrained
requests for that model would time out or receive 503 while unconstrained and already-active
serving continues normally.
