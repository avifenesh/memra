# CX spec/plain identity fix — results

## Verdict

PASS. `8392e9a5` restores serving text and token-id identity while preserving the shipped
honesty contracts. No engine code, default, perf board, release tag, or remote branch changed.

## Root cause

The honesty path used one value, `room = request_remaining.min(burst_t)`, for two different
limits:

1. the number of tokens the engine should target in this scheduler burst; and
2. the number of tokens the request may still expose.

Session-mode speculative generation may return a cache-authoritative surplus id past the
scheduler target. Hiding that id on an intermediate burst advanced `SpecSession` while the
worker's generated/sampler/fed history stayed behind. The next burst resumed after the hidden
state, so output ids diverged.

The live receipt pins the boundary: zero-based token 32, exactly the default 32-token burst
edge. Before the fix, the common prefix ended `[262, 348, 256, 42794, 25]`; the two continuations
were `[303, 348, 588, ...]` and `[424, 303, 348, ...]`. The text first differed at zero-based
character 110 (`Constraint: in` versus `Constraint: it in`). This is case (b), not a
detokenization-only fork.

## Fix

`worker.rs` now keeps the limits separate:

- `burst_target = request_room.min(burst_t)` goes to the engine.
- `request_room` governs per-id events and the public generated/token/usage vectors.

Intermediate engine surplus therefore remains public and keeps worker state aligned. On the
final burst, `request_room` still clamps exact `max_tokens`; surplus remains only in the engine
session as required. The per-id event path, terminal `tokens_emitted == generated.len()` assert,
token snapshot, spec accounting, and `spec.rs` are unchanged.

The regression test models the observed shape directly: a 32-token engine target returns 33 ids
while 64 request slots remain, and all 33 must be public. The existing final-budget overshoot test
still proves that only the remaining request slots are exposed.

## Raw A/B receipts

Before (`fee291f7`, server binary SHA-256
`0cb88b54b2d6063301a0833c291745140232fbc06fd73519c9c1b33a815dee95`):

- `raw/token-before/plain-text.txt` SHA-256
  `69a68b932a3097a4078bedee3249348731027cc39d6de0921bf5c7b04e1d33f3`
- `raw/token-before/spec-text.txt` SHA-256
  `36e34579f1f0fa04a9ff61538a2f08876a3f0fac5e0ecf0662049ee2bdeb5c20`
- token receipts also differ; full responses, server logs, and GPU state are beside them.

After (`8392e9a5`, server binary SHA-256
`cd4b41ed2775ffb10e30ed306b0fe013687948fe699499389abf99367f98ed63`):

- plain/spec text SHA-256 both
  `29a15b2c67c18467d359a19d6f490d8ebe27b67e17ca867b2bd3dfa71146cf6b`
- plain/spec token-array SHA-256 both
  `7b4f15411917effaabd57a65e1576eea5bc16a746a8954abcb9823d99662bcfe`
- both native receipts: `n_tokens == tokens.len() == 64`, stop `MaxNew`
- both OpenAI responses: `completion_tokens == 64`, with `usage.spec` present

## Gates

| Gate | Result | Receipt |
|---|---:|---|
| Focused spec-emission tests | 3/3 PASS | `raw/gates/cargo-test-memra-server.log` |
| `cargo test -p memra-server` | 154/154 PASS | `raw/gates/cargo-test-memra-server.log` |
| serve-smoke spec/plain | MATCH | `raw/gates/serve-smoke.log` |
| sampled truncation matrix | 4/4 PASS | `raw/gates/serve-smoke.log` |
| cache-meter | 23/23 PASS | `raw/gates/serve-smoke.log` |
| full serve-smoke | 0 failed | `raw/gates/serve-smoke.log` |
| q9 run-spec K=1..8 | 8/8 PASS | `raw/gates/run-spec-q9.log` |

All GPU work ran on the local RTX 5090 Laptop GPU under `/tmp/memra-gpu.lock`; pre/post state is
stored beside each receipt. A repository-wide `cargo fmt --check` was not a lane gate and remains
red on extensive existing formatting drift in untouched files; no formatting rewrite was made.
