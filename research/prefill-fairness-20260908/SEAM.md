# Shared prime walker contract

2026-09-08. Early seam commit, before route adapters and GPU qualification.

Engine API in `crates/memra-engine/src/prime_walker.rs`:

```rust
pub type PrimeError = Box<dyn std::error::Error>;

pub trait PrimeWalker {
    type Output;
    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError>;
    fn remaining_chunks(&self) -> usize;
    fn finish(self) -> Result<Self::Output, PrimeError>;
}
```

`PrimeChunk { phase: &'static str, rows: usize }` describes one actual operation.
`advance_prime(&mut walker, yield_after_chunk, observer)` runs exactly one chunk
when ON and drains when OFF. It checks that every advance consumes exactly one
frozen range. `finish_prime(walker)` refuses incomplete work before calling finish.
The observer receives each chunk's wall time in both arms. The adapter must fence
shared device scratch before returning, own all live buffers/caches/carries, and
freeze its numerical range program before the first advance. `finish` cannot hide
another prompt-length loop. Once-only captures and setup belong in measured work.

Worker API in `crates/memra-server/src/prime_fairness.rs`:

- Every `Session` has `prime_service: PrimeService`. The route calls
  `s.prime_service.advance(&mut walker)?`; false means save the adapter's owned
  state and return `Ok(true)` without draining the remaining request or decoding.
  True means call `s.prime_service.finish(walker)`, install its outcome, then do
  first-token work. This additive helper keeps failed final ingestion out of reuse
  pools; the engine trait and the original driver signatures remain unchanged.
- The service records `pending` and cumulative yield count; `MEMRA_TICK_TRACE=1`
  prints `[prime-chunk] phase=... rows=... wall_ms=...` on either arm. ON yields
  print `[prime-yield] count=... remaining=...`.
- `PrimePolicy::order` rotates eligible spec peers and puts saved primes after fresh
  peers. Existing tick-top command drain/admission runs before the next advance.
  Each selected row appears once. Pending primes cannot publish captures, demote,
  or park. Cancellation/error must drop their owned state through normal retirement.
- `PrimeService::decode_target` limits peers to one public progress quantum
  while any prime is pending: an initial seed or at most one committed spec round.
  A round's committed surplus is preserved; this is
  scheduler cadence, not a request budget. Existing plain prefill/decode batching
  is retained. All route burst targets use this common policy, including the
  future GLM5 adapter's existing decode path; this commit implements no GLM5 prime.

Adapters must retain their logical speculative routing identity while pending so
the session cannot accidentally enter plain prefill/decode. Route-specific state
storage, cold/resume setup and capture details belong to those follow-up commits.
Keep this trait and service API stable for stacked route work.

The bound and per-route receipt slots are in the `MEMRA_PRIME_YIELD` FLAGS row.
An unadapted route still primes synchronously; the seam alone establishes no GPU
latency or exactness qualification. No context/output envelope changes.

CPU validation uses only the authorized remote CPU, nice 19, two jobs,
`MEMRA_CUDA_ARCH=120a`, `CARGO_TARGET_DIR=/root/target-prefill`:

- `cargo test --release -p memra-engine --lib prime_walker`: 3 passed. Same OFF/ON
  operation tape including non-divisible ranges; failure and finalization guards.
- `cargo test --release -p memra-server --lib prime_fairness`: 3 passed. New
  arrivals precede a saved prime, the long request progresses under continuous
  arrivals, peer service rotates without duplicates, and no-prime behavior stays
  unchanged. `cpu-seam.json` binds the checked source files.
- Remote fmt and release library clippy passed before route integration. Hosted CI
  passed all checks at the early seam commit `534040262`. Adapter validation and
  full all-target clippy are recorded separately in `CPU.md`.

No GPU process launched. No local build/test/gate. Pushes use
`MEMRA_SKIP_PERF_CI=1`; local hooks are disabled per invocation to honor the
owner's prohibition on local gates. Hosted PR CI remains required.
