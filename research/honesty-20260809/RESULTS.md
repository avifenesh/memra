# API honesty cluster — results

Verdict: **PASS**. The three confirmed serving-contract bugs are fixed in `5c585a16` without
changing speculative engine commit/cache-row semantics. No model or GPU run was used.

The lane began at `b752cf2d`. Steering arrived after the initial implementation pass and required
the local train's `cx-pinfix` merge; the branch was rebased onto train tip `04babb07` before the
fix commit and all gates below were rerun on that combined tree.

## Before / after receipts

| contract | before | after |
|---|---|---|
| `max_tokens` | The live cachespec arm returned more than requested on 5/17 requests at `max_tokens=768`: four 769-token responses and one 770-token response. | `spec_emission_clamps_engine_overshoot_to_the_request_budget` models a five-token engine burst with only three request tokens left: all five engine tokens remain present, exactly three token events/public tokens are selected, two surplus tokens stay outside the public prefix, and the public total ends exactly at the requested five. |
| per-token event ids | The audited native blocking receipt reported `n_tokens: 2048` beside an 803-id `tokens` array because one `Event::Token` represented a whole speculative round. | `spec_emission_publishes_one_event_per_visible_token_id` emits ids `[0, 1, EOS]` as three distinct events with deltas `["a", "b", ""]`; EOS remains text-suppressed but still has its own id event. `finish()` now asserts `tokens_emitted == generated.len()` before publishing `TokenSnapshot` or `Done`. |
| output metrics | The live cachespec run completed 17 speculative requests while `/metrics.tokens_out` remained 0. The legacy `MEMRA_SERVE_BATCH=0` loop also had no output accounting. | The spec counter test advances generated length `10 -> 14`: `tokens_out=4`, interactive `lane_tokens=4`, four 5 ms per-token samples from a 20 ms burst, and a refreshed interactive-decode timestamp. The legacy test records no output/timing for prefill (`0 -> 0`) and exactly one token plus one 7 ms sample for decode (`0 -> 1`). |

Pre-fix live sources: `research/cachespec-20260809/RESULTS.md` (overshoot and zero output
counter) and `research/code-audit-20260809/PAPER.md` Area 4 (2048/803 event receipt and traced
call sites). After receipts are device-free unit tests over the production helpers added here.

## Implementation contract

- The engine still returns and stores every accepted/bonus token needed to preserve
  `cache.pos == SpecSession.committed.len()`. `crates/memra-engine/src/spec.rs` is unchanged.
- The worker applies a separate request-owned emission prefix before touching `generated`,
  `fed`, sampler history, events, usage, or output metrics. A crossing round exposes at most the
  remaining request budget; engine surplus remains in `SpecSession` committed/pending state.
- Spec round cadence is retained, but each visible id gets its own `Event::Token`. Token bytes are
  decoded incrementally per id, so the fix does not repeatedly detokenize the full history.
  `MEMRA_SSE_PER_BURST=1` now changes only when those per-id events flush, not their 1:1 shape.
- EOS produces an empty-text token event on every path. This preserves marker-text suppression
  while making the terminal id/event/generated invariant exact.
- Batched spec and legacy round-robin accounting use the successful `generated.len()` delta.
  Spec wall time is divided by emitted tokens and recorded once per emitted token; budget surplus,
  rejected drafts, and prefill-only calls do not become output.

## Gates

All commands ran in `~/projects/wt-cx-honesty` on committed fix `5c585a16`.

| gate | result |
|---|---|
| focused overshoot + per-id event tests | PASS, 2/2 |
| focused spec + round-robin accounting tests | PASS, 2/2 |
| `cargo test -p memra-server` | PASS, 142/142, 0 failed |
| `cargo test -p memra-engine` | PASS, 55 passed across library/bin/integration targets, 0 failed, 2 GPU-required tests ignored by annotation |
| `cargo build --release` | PASS, optimized profile, 2m33s |

The original mission base had 132 server tests. Steering predicted a 137-test train baseline;
the fetched tip also contained the later `/v1/models` reasoning-capability test, making the actual
train baseline 138. The four tests in this lane produce the observed final 142.

No origin push, merge to train, tag, release, `rustup`, `nsys`, model load, or GPU execution was
performed.
