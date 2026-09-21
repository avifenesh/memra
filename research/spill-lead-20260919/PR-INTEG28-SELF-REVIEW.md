# Self-review: integ28 (C day 20: memra#365, the DFlash prefill tap bounded on the standalone entry)

Author's review of the full diff `main..lane/spill-integ28-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/dflash.rs`: `generate_spec_dspark` no longer allocates a whole-prompt tap sink
  (`tp x n_taps x n_embd` f32) and no longer runs the monolithic `prime_cache` that also holds the whole-prompt
  hiddens; it primes through the existing serving walker `prime_dflash_taps` (one 4,096-row chunk sink plus the
  256-row carry, the same walker the serving cold and resume paths use since #370). The carry copy's `expect` is now a
  typed refusal (no live sink, or a copy beyond the rows the trunk wrote). One CPU test drives `TapBatchCarry` over
  thirteen prompt lengths and asserts every copy stays inside the chunk the trunk wrote and every ingest stays behind
  the copies. The oracle hook (`DflashPrimeOracle` under the allocation trace) is kept.
- `crates/memra-engine/src/bin/dspark_q38_gate.rs`: a `[dflash-taps]` receipt line (bytes of the chunk sink, the
  carry, and the former whole-prompt sink) for the replay.
- Research: DAY20 (census, before and after cells, gates), receipts, replay script, STATE, INDEX row; the lead record
  section; this file; battery receipts.

## What I checked
- One numeric program: the standalone entry moved from the monolithic prime to the chunked walker. The two rungs that
  ran on both sides (8,194 and 16,382 tokens) read the same `spec_sha256`, `spec_len` and acceptance, which is the
  bit-identity the law asks for at those lengths; the hit gate and the continuation gate are green on the changed
  binary. Serving paths are untouched (they already used the walker).
- The refusal replaces a panic; it fires only on a bookkeeping violation that the new CPU test forbids by
  construction.
- No flag, no new `MEMRA_*` read (census clean); the receipt line is diagnostic output only.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, engine clippy `-D warnings`, marker census, workflow keys, C's replay, perf board, diff-check)
  and the local 5090 serve-smoke.

## What I did not do
- No target-card cell (device-independent bookkeeping; the target-card battery is the release lane's).
- The 131k rung and the serving 128k and 262k cells stay with the issue (#370, #377).
