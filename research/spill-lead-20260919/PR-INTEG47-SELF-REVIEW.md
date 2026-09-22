# Self-review: integ47 (A day 30: the recurrent f32 state rides the demote's copy stream, the D2H half; C day 38: the per-tick split and the 9B byte split)

Author's review of the full diff `main..lane/spill-integ47-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/pinned_host.rs`: `PinnedHostBuf::new_unwritten` (cached pinned, no fill, unreadable until
  landed), `enqueue_from_device_f32` (`cuMemcpyDtoHAsync`, no host wait), `mark_landed`.
- `crates/memra-engine/src/tier_transfer.rs`: `D2hSpan`, `CudaTransfers::submit_d2h_spans` (admission first, the
  owner-stream fence and the copy-stream wait, one enqueue and one event per span, an enqueue error from the first on
  sets the ticket `unknown`), `take_d2h_spans`; `progress` folds the span events into `producer_done`; `retire` is
  `Busy` while spans are untaken; an unretired entry's drop forgets its spans. CPU census and one native cell.
- `crates/memra-server/src/worker.rs`: the per-context staging pool (`staging_take`, `staging_put`, cleared at
  `disable`), `host_spans_submit` in the OffTick route, the settle's span take before destinations, retire and require,
  the driver's attach of each staging buffer to its heap placeholder, the helper's copy-then-hash, the pool return on
  the landed path, the copy-complete line's `items=N (K KV, S f32 spans)` term. CPU census and two GPU door cells.
- `crates/memra-tier`: conformance rules 1 to 5 for the span batch (three schedules including a red arm) and a CPU
  binding. Additive; `WIRE_VERSION` stays 1.
- No new `MEMRA_*` name, no flag, no docs registry change.
- Research: A DAY30 with target-card and 5090 receipts; C DAY38 with its readers and CPU receipts; INDEX rows; the lead
  record section (ruling 42); this file; the integ47 CPU and 5090 battery receipts.

## What I checked
- `unsafe`: the pinned allocation, the async D2H enqueue (the source's byte length must equal the buffer's and be
  non-zero before the call; the source's read guard lives across the enqueue; the destination is pinned memory owned
  by the span until the take) and `mark_landed` (one call site, inside `take_d2h_spans`, reached only when `progress`
  observed every span event complete, census-pinned). The one byte view in `tier_transfer.rs` is a test helper over an
  `&[f32]`.
- Staging lifecycle: taken by exact length or allocated unwritten; `as_f32_slice` asserts `written`; returned on the
  landed path and on a refusal before submission; dropped (freed, not pooled) on a post-take abort, which A lists as
  owed (finding 4); cleared at `disable` after `hasher.close`. The set grows only by distinct plane sizes, and one
  demote is in flight at a time.
- Fail-closed: a span refusal hands every span back, returns every source plane to `dead` and the staging, and unwinds
  the ticket through `host_contract_abort` with a typed line; a staging buffer with no heap placeholder latches.
- Reachability: spans are submitted only in the OffTick contract route, which stays guarded `host.arena.is_none() &&
  !is_glm`. Door OFF never allocates staging.
- One numeric program per request: the spans are a byte copy of the same planes; the helper hashes the same bytes with
  the same `host_hash_payload_digest` program; the bitwise GPU cell and the hit gate's unchanged census on the 5090 hold.
- Noted, not a finding: `PinnedHostBuf`'s drop does not bind a CUDA context (unchanged by this diff). The staging is
  outside the tier governor's pinned ledger (157.9 MB per context on the 27B), owed by A.
- C day 38: the tick rule's commit precedes the reader's first run on day 37's receipts; the reader requires day 37's
  admissibility line; the verdict lines in the record are quoted from `day38-cpu/tick-split.log`.
- Battery on `160929a92`: fifteen of fifteen CPU steps rc=0 (server suite 836 passed, engine CPU lib 532 passed, tier
  contracts 90 passed, portable suites 357 passed, both clippy passes `-D warnings` clean, `git diff --check` clean).
  Local RTX 5090 under one collector hold: `serve-smoke: 0 failed`; engine GPU cells 6 passed; worker span cells 2
  passed; `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`; `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` 123 ok in the
  default and plain arms; `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 ok OFF and 68 ok ON with the day-24 census.
- BOX3 receipts are A's, on `a8d6b1df5`, before the lane merged #652's bounded latch close; not rerun on this tree.

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
