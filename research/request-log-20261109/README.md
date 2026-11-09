# Per-request timing log (`MEMRA_REQUEST_LOG`): exactness and test receipts (2026-11-09)

Verdict: PASS. With the log set, every response of a fixed greedy request set was byte-identical
(per-response ids and wall-clock fields aside) to the same request with the log unset, on all
four generation routes, streaming and blocking, over four boots in the order A B B A on the
RTX 5090 Laptop GPU. The log's lines matched their responses: times ordered, status and token
counts present, one session key across both turns of the named session and of the unnamed one.

## Question

The log observes requests from four places: a middleware, the handler, the event receiver and
the tool parser, plus one copy of the prompt ids on the GPU worker. Does any of that change what
the engine generates or what a client receives? And does a line carry the fields its contract
names, with a session key that holds across turns?

## Device A/B

- Card: RTX 5090 Laptop GPU (sm_120a), one card, under a `gpulock` lease. One pass per run.
- Binary: `memra-server` built from the PR branch with the CI release profile
  (`CARGO_PROFILE_RELEASE_LTO=false`, `CODEGEN_UNITS=16`; device code is profile-independent).
  Its sha256 and build line are in each run's `summary.json` and `boot*/server.log`.
- Model: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (the serve-smoke default), `MEMRA_MODELS=m=<path>`,
  every other setting at its default (`MEMRA_*` cleared from the boot environment).
- Order: four boots, A B B A, A = log unset, B = `MEMRA_REQUEST_LOG=<run>/requests.jsonl`.
- Requests per boot, in order, all `temperature: 0`: a named chat session (`session_id`) turn 1
  blocking and turn 2 streaming, with a `get_weather` tool; the same two turns with no identity;
  a raw completion blocking (with `user`) and streaming; one `/v1/responses`; one
  `/v1/messages`. `max_tokens` 96 (64 on Responses and Messages).
- Pass rule (`ab.py`): every request's normalized response is equal across the four boots, and
  each B boot's lines pass `check_log` (one line per request, ordered times, status equal to the
  response's, `stop` or `length`, `prompt_sha256` and `generated_tokens` present, the named
  session's key equal to `sha256("ab-session-1")` on both turns, the unnamed session's turns
  sharing one `fingerprint` key, the completion keyed by `sha256("ab-user")` with
  `tool_calls: null`). Normalization drops `id`, `created`, `created_at`, `completed_at`,
  `system_fingerprint` and `elapsed_s` at any depth, and nothing else.

| run | commit | boots | requests | byte diffs | log failures | verdict |
|---|---|---|---|---|---|---|
| `device-ab/run1-32598d3947` | 32598d3947 (first push) | 4 | 8 | 0 | 0 | PASS |
| `device-ab/run2-124665c83a` | 124665c83a (after the review fixes) | 4 | 8 | 0 | 0 | PASS |

Run 2's eight normalized digests are also equal to run 1's: the review fixes changed what the log
records, not what the server answers.

Per request (boot 1 line of either run; normalized response digest equal in all eight boots):

| request | normalized sha256 | end_reason | generated | session_key_source | tool calls |
|---|---|---|---|---|---|
| chat-named-t1 | 8644955943be | stop | 56 | session_id | 1 (`get_weather`) |
| chat-named-t2-stream | 957bce9a6b43 | length | 96 | session_id | 0 |
| chat-unnamed-t1 | 873a1561ffb0 | stop | 56 | fingerprint | 1 (`get_weather`) |
| chat-unnamed-t2-stream | dd6cba6da48c | length | 96 | fingerprint | 0 |
| completion | 4147cd533891 | length | 96 | user | null (raw route) |
| completion-stream | b0062f2f3466 | length | 96 | null | null (raw route) |
| responses | aff0336d805a | length | 64 | null | 0 |
| messages | 60e10d1db7e2 | length | 64 | null | 0 |

The two unnamed turns share the fingerprint key on the real Qwen ChatML tokenizer, so the
implicit tier keys a session from its first turn when it has a system prompt.

## CPU tests (memra-server, release, CI profile)

- `request_log_two_turns_of_one_session_share_a_key_and_fill_every_field` (lib.rs): two turns of
  one session through the real handlers, middleware stack and writer thread, named and unnamed,
  plus a conversation with a different first turn, a raw completion, Responses, Messages, a
  refusal, a blocking request cut by its deadline and a stream whose client hangs up. Checks
  every field of every line and the cross-turn key.
- `request_log_on_and_off_serve_identical_bytes` (lib.rs): the HTTP half of exactness; the same
  five requests with the layer installed and not installed give the same status, content type
  and bytes.
- Unit tests: the end-reason order, the line's hashing and the absence of text, the namespace
  scope of the session key, the drop-and-count queue (`request_log.rs`), and
  `implicit_affinity_key_is_equal_exactly_at_the_nomination_bar` (`worker.rs`).

Red arms (`red_arm.py`), each on the final tree:

| arm | what it removes | two-turn test | on/off identity test | log |
|---|---|---|---|---|
| `bind` | `bind()` binds nothing: the server's view before this change | FAILED (`end_reason` `cancelled`, no session key, no first token) | ok | `red-bind-cut.log` |
| `both` | `bind` cut, and logged streams end after their first frame | FAILED | FAILED ("the request log changed a response") | `red-bind-cut-and-body-truncation.log` |
| none | | ok | ok | `green.log` |

Against the base commit 958ae32fed the two-turn test does not compile: there is no request log.
The whole memra-server suite on the final tree: `server-suite.log` (1105 passed, 0 failed,
10 ignored).

## Files

- `ab.py`: the device A/B driver. `bank_ab.py`: copies a run here with the home directory as `~`.
- `device-ab/run*/`: `summary.json`, and per boot `results.json` (normalized responses),
  `*.body` (raw responses), `requests.jsonl` (B boots: the log itself) and `server.log`.
- `red_arm.py`: the two red arms. `*.log`: the CPU test output.
