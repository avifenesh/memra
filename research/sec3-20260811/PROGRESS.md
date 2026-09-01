# Lane cx-sec3 progress

## Scope

Protect the memra-server worker tick from synchronous constrained-decoding setup:

- reject oversized or structurally complex JSON schemas before admission;
- build the per-model token trie and request matcher away from the worker thread, with a bounded wait;
- preserve valid constrained-decoding behavior; and
- add a CPU-only regression proving a pathological schema cannot stall scheduler progress or heartbeats.

## Status

- 2026-08-11: lane brief and repository law read; branch `lane/cx-sec3` is clean and isolated.
- 2026-08-11: added pre-admit JSON-schema bounds (512 KiB serialized bytes, 64 raw JSON
  levels, 32,768 JSON values) with focused deep/wide/oversized rejection coverage.
- 2026-08-11: moved each model's lazy TokTrie/factory and per-request matcher construction to
  one bounded background compiler (8 queued requests, 5 s request timeout). The worker polls
  completion without waiting, keeps pending compiles outside admission, and has no inline
  compilation fallback.
- 2026-08-11: constrained HTTP calls now await a one-shot compiler verdict before response
  headers, preserving clean 400/503 status for streaming calls. Focused CPU regressions prove a
  deep schema fails while a 64-step normal decode continues, and a stuck compile times out while
  normal token events and heartbeat stamps continue.
- 2026-08-11: final source audit found the only factory/matcher/TokTrie construction path inside
  the background compiler. Full `memra-server` suite passed 176/176; verdict recorded in
  `RESULTS.md`.

## Required gates

- `cargo test -p memra-server`
- focused pathological-schema/concurrent-progress regression

Focused receipt: `cargo test -p memra-server
constrained::tests::json_schema_bounds_fail_before_compile -- --exact` — PASS (1/1).

Focused receipt: `cargo test -p memra-server
worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress
-- --exact` — PASS (1/1).

Focused receipt: `cargo test -p memra-server
tests::deep_schema_fails_while_normal_decode_keeps_stepping -- --exact` — PASS (1/1).

Focused receipt: `cargo test -p memra-server
tests::valid_response_format_preflight_preserves_generation -- --exact` — PASS (1/1).

Final receipt: `cargo test -p memra-server` — PASS (176/176).
