# Q35 mixed c=2 completion-token mismatch — progress

Date: 2026-08-12

Branch: `lane/cx-q35bug`

## Objective

Reproduce and localize the Qwen3.6-35B-A3B mixed-c=2 sellgate rejection, distinguish
wire/content generation from accounting, fix the smallest proven defect, and rerun the required
exactness gates on the designated PRO 6000 host.

## Starting evidence

- Scored source: `79c3c0b2779101c7de89d6f822b9392d03e71702`.
- All five Q35 mixed-c=2 cells were invalid.
- Client-visible completion totals were `1165 / 1122 / 1165 / 1165 / 1122`; engine
  `tokens_out` totals were `1164 / 1120 / 1164 / 1164 / 1120`.
- Seven cached requests stopped at 17 or 25 completion tokens with `finish_reason=stop`; one more
  cached request did the same in the fifth c=12 mixed cell.
- Cached-token accounting remained exact.

The early stops and the aggregate 1–2 token response/engine deltas are tracked as separate facts
until per-request SSE events, usage, engine counters, and token-id hashes prove whether they share
one cause.

## Success gates

- Minimal Q35-only mixed-c=2 reproduction using the frozen workload prompts, with per-request
  client token count, SSE event count, engine `tokens_out`, `finish_reason`, usage, and token-id
  hash captured.
- Root cause established from quoted raw evidence, not inferred from aggregate totals.
- Regression coverage asserts client/engine agreement on every affected finish path.
- Relevant cargo tests pass.
- Q35 mixed-c=2 cell passes five repetitions.
- `run-gen` argmax is MATCH; if decode changes, the Q35-class decode-batch gate passes.
- `RESULTS.md` records the root cause, fix, and sealed rerun receipts.

## Work log

- Investigation opened. No root-cause claim yet.
- Checked `/home/avifenesh/.lanectl/inbox/cx-q35bug.md`; it contains the lane brief and no additional steering.
- Audited the sealed sellgate replay rows and harness. `sellgate_replay.py` labels the response `usage.completion_tokens` total as the client total; it did not independently count SSE token events or capture token ids.
- Localized the aggregate accounting discrepancy in the plain batched scheduler: `advance_sample_emit` appends and emits the terminal EOS token, then returns the row as finished; `n_tokens_out` is incremented only later for rows that survive into the next batched decode. The observed deficit is exactly one engine metric token per early-EOS request in every rejected cell. This is a server accounting defect; remote wire receipts are still required before calling the early stops a generation defect.
- The initial Step35 analogy was rejected by a live control: Q35 is `qwen35moe`, not the separate StepFun `step35` architecture, and `MEMRA_STEP35_BATCH=0` left its logged decode wave cap at 8. The applicable boundary is the generic `decode_step_b1_fast` eager fusion trunk at B=1 versus `decode_batch_layers` at B>=2.
- Fresh OpenAI-wire reproduction (`raw/pre-openai-stable/`) failed 5/5 cells. Every request had exact SSE-token-event == response-usage accounting; the five cell totals were `1165/1122/1165/1165/1165` on both surfaces versus engine `tokens_out` `1164/1120/1164/1164/1164`. The deficit was exactly the `1/2/1/1/1` early-EOS request count. No cache or transport mismatch occurred.
- Fresh native-wire reproduction (`raw/pre-native-stable/`) captured every token id. Serial hot seeds were one stable 60-id hash (`5bc2ab6255e54c6183320a792f4cce1b643d019a86508f32a6514ac9df69d034`); mixed c=2 produced several distinct sequences and nine early EOS requests. The early sequences end in EOS id `248046` at lengths 15, 17, or 25, proving content generation changed rather than only its reported count.
- Decisive control: setting only `MEMRA_SERVE_B1FAST=0` made Q35 run the generic batched trunk at B=1 and B=2. `raw/pre-native-b1-batched/` passed 5/5: 100/100 requests full-length, one token-id hash across seeds and mixed traffic, no cache/transport mismatch, and client event / response usage / engine totals all exactly `1200` per cell (`6000` total).
- Implemented the architecture-scoped decode fix: `Arch::Qwen35Moe` is ineligible for eager B1 in unsplit and PP-N paths; dense Qwen35 and unrelated families retain the fast path. The config-mode decode-batch gate treats eager comparison as inapplicable for this family and keeps its B=1-vs-B=N gate at bit strength.
- Implemented scheduler accounting from each request's successful `generated.len()` delta before survivor branching. This closes terminal EOS/callback/context-full loss in batched, graph, graph-demotion, and eager-only scheduler paths; MaxNew emits no token in its terminal check because its final token was counted on the preceding call.
- Focused unit tests pass for the architecture policy, terminal scheduler accounting, and SSE-event/usage equality across `Eos`, `Callback`, `MaxNew`, and `ContextFull`.
- Full local package tests pass: `memra-engine` 78 passed / 1 GPU-only ignored and `memra-server` 196 passed. Release builds for `memra-server`, `run-gen`, and `decode-batch-gate` also pass on sm_120a.
- Candidate `e953420156d9c53a693386efa7a54a56c665b094` passed the committed-default eu-west rerun on both native and OpenAI wire surfaces. Each was 5/5 clean, 100/100 full-length requests, and exactly `6000 == 6000 == 6000` for independently counted SSE token events, response usage, and engine `tokens_out`; native seeds and mixed requests were one token-id hash.
- Final eu-west gates passed under one exclusive flock block: `run-gen` prefill/decode argmax MATCH and batched-prime/tokenwise MATCH; Q35 B=2 config gate ALL GREEN at bit-strength isolation; equalized strict gate ALL GREEN. Raw logs and provenance are under `raw/gates-final/`.
- `RESULTS.md` seals the quoted root causes, scoped fix, hashes, and rerun receipts. Branch remains for orchestrator promotion; no merge, tag, push, board update, or formatting run was performed.
