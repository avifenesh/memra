# Self-review: integ26 (A day 16: memra#536 census, stall cell, the prime cancellation point)

Author's review of the full diff `main..lane/spill-integ26-20260921`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/progress.rs`: a thread-local `Option<Box<dyn Fn() -> bool>>` predicate, `PrimeCancelScope`
  (install replaces, drop restores the previous one, nested scopes tested), typed `PrimeCancelled`, and
  `prime_cancel_point(chunk, rows_done, rows_total)` which returns `Ok` when no scope is installed, when the predicate
  says the client is present, or when the take is complete. Two unit tests: the semantics and a source-level wiring
  gate (at least three live call sites in `hybrid_forward.rs`, comment-stripped).
- `crates/memra-engine/src/hybrid_forward.rs`: the check at the chunk boundary of the three sequential walks, after
  a chunk completed and before the next starts. I read each site: the call is placed after the odometer stamp and
  passes `end` (rows primed so far) and `t` (rows of the call), so the last chunk can never fire it. The pipelined PP
  walks and `prime_cache_batch` are untouched, as the module note says and why.
- `crates/memra-server/src/worker.rs`: the scope is installed around the one `prefill_tick` prime call on the worker
  thread with `tx.is_closed()`; both error call sites match `PrimeCancelled` first (`prime_cancelled_abort`) and fall
  through to the existing error path otherwise. The abort path is the tick-top sweep's (`abort_log`, retire refuses
  the park, cache freed at drop); no `Event::Error` on a closed channel.
- `tools/prime-cancel-gate.sh`: the serving-shape fault cell (client disconnects mid-prime; asserts the receipt line,
  no publication, and the next request's cold and warm digests against a run without the disconnect).
- Research: census, design note, stall receipts (five cells, N=5 per arm per order, both orders), gate receipts on
  both cards, DAY16, STATE, INDEX row; the lead record section; this file; battery receipts.

## What I checked
- One numeric program: the check does not alter any chunk schedule or any bytes of a prime that completes; a
  cancelled prime returns no logits, so the capture sites after `Ok` are unreachable. A's gate shows the next request's
  cold and warm digests identical to a run without the disconnect on both cards, and the hit and continuation gates
  are green on the changed binary.
- The predicate is read on the thread that installed it (the worker thread runs the walk); no other thread reads the
  thread-local. The downcast in the worker works because the walks return the typed error through `?` unwrapped (the
  gate's receipt line proves the match fired).
- No flag, no new `MEMRA_*` read (flags census clean); lock names in the new script canonical.
- Stall numbers are stated as stall per class on one card class, N and regime named, no cross-box comparison and no
  default changed by them.
- INDEX.md: A's row kept, the inherited stray marker dropped, no row of either parent lost; the marker census passes.
- Battery on this tree in the receipts (fmt, portable suites, server suite, clippy, censuses, collector pytest, engine
  CPU lib tests, engine and server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check) and the
  local 5090 serve-smoke.

## Limits, stated
- Pipelined PP primes and the batch prime keep the tick-top sweep as their only cancellation point (a drain plus the
  tainted-cache contract would be needed; A's note says so).
- Nothing here changes what runs on the owner thread; the offload design is a note with its decision cells.
