# API surfaces: `/v1/messages` and `/v1/responses`

memra-server speaks three request dialects over ONE serving core. `/v1/chat/completions`
(plus raw `/v1/completions`) is the native surface; `/v1/messages` (Anthropic Messages
API) and `/v1/responses` (OpenAI Responses API) are **translation surfaces**: the request
is rewritten into the internal chat request, flows through the same tenant auth, budget
admission, ledger receipts, metering, rate limits and capture posture as a chat
completion, and only the response rendering differs. A request costs and bills the same
number of tokens no matter which dialect carried it.

Both surfaces exist so agentic clients that are hard-wired to one dialect can point at
this server directly — an Anthropic-format client via `ANTHROPIC_BASE_URL`-style
configuration, a Responses-format client via a custom provider with
`wire_api = "responses"`. No proxy in between.

The house law applies on both: a semantic feature the engine cannot honor is a **clear
400 naming the field**, never a silent downgrade. Cosmetic or client-telemetry fields are
accepted and ignored.

## `/v1/messages` — Anthropic Messages API

`POST /v1/messages` (query strings such as `?beta=true` are accepted). Streaming and
blocking. `anthropic-version` is accepted and not enforced.

**Auth**: `x-api-key: <key>` **and** `Authorization: Bearer <key>` are both accepted and
resolve against the same tenant keyring / single-key / open-server rules as every other
surface. If both headers are present, each is tried.

**Supported request fields**

| Field | Handling |
|---|---|
| `model` | same registry as `/v1/models`; unknown model → 400 (`not_found`-coded body) |
| `max_tokens` | required, ≥ 1 (Anthropic contract) |
| `messages` | roles `user` / `assistant` / `system`; content string or block array |
| content blocks | `text`; `image` (base64 source only, becomes the vision path); `tool_use` (assistant) → internal tool_calls; `tool_result` (user; string or text-block content, `is_error` accepted) → internal tool turn; `mid_conv_system` → system turn; `thinking`/`redacted_thinking` in history are **dropped** (chat templates re-render history without think segments — same law as the template itself) |
| `system` | string or text-block array (concatenated; `cache_control` markers ignored — prefix caching here is automatic) |
| `tools` | client-defined tools (`name`, `description`, `input_schema`); rendered through the model template's tools branch. Server-executed tool types (`web_search_*`, `bash_*`, `code_execution_*`, …) → clear 400 |
| `tool_choice` | `{"type":"auto"}` / `{"type":"none"}`; `any` / `tool` need constrained decoding → clear 400. `disable_parallel_tool_use` ignored |
| `stop_sequences` | native stop strings; the matched sequence is reported as `stop_sequence` |
| `temperature`, `top_p`, `top_k` | native |
| `stream` | Anthropic SSE vocabulary (below) |
| `thinking` | `enabled` → model-native thinking on, `disabled` → off, `adaptive` → the model's own default (Anthropic's "the model decides", which is what that arm does). Any other `type` → 400. **`budget_tokens` → 400**: reasoning tokens are output tokens under the single `max_tokens` budget and no lever can cap a segment, so accepting it would promise a spend cap this server cannot keep (lane/reasoning-schema-20260823; it used to be accepted, never read, never enforced) |
| `output_config.effort` | the SAME canonical set as `reasoning_effort` on the chat surface and `reasoning.effort` on `/v1/responses` (`none`/`minimal`/`low`/`medium`/`high`; `xhigh`/`max`/`ultra` clamp to `high` — Claude Code sends `xhigh` by default on current models): `none`/`minimal` suppress thinking, invalid values 400. When `thinking.type` is also present it wins the on/off switch (the documented Anthropic lever); the effort is still validated and still supplies the level for level-consuming templates. Before issue #31 this field was silently dropped |
| `timeout_ms` | request deadline in milliseconds, `1000`..=`90000`, default `90000` — the SAME parameter and semantics as the chat surface and `/v1/responses`. Non-streaming it bounds the complete response; streaming it bounds time to first token only. A miss is `408` (`type: "timeout"`, `code: "deadline_exceeded"`) and is **not billed**; out of range or wrong type is a 400 naming the field. See SERVING.md "Request deadlines and the billing promise" |
| `metadata.user_id` | session-affinity nomination (same as `user` on the chat surface) |
| `mcp_servers` | → clear 400 (server-side MCP does not run here) |
| `output_config.*` other than `effort` | → 400 naming the key (previously accepted and ignored) |
| everything else | accepted and ignored (`cache_control` anywhere, `context_management`, `service_tier`, beta-paired fields) |

**Response**: `{"id":"msg_…","type":"message","role":"assistant","content":[…],
"stop_reason":…,"stop_sequence":…,"usage":{…}}`. Content blocks in generation order:
`thinking` (when the model produced separated reasoning; `signature` is honestly empty —
there is no signing key here), `text`, then `tool_use` blocks (`input` is the parsed JSON
object). `stop_reason`: `tool_use` | `stop_sequence` | `max_tokens` | `end_turn`.

**Usage honesty**: `input_tokens` excludes cache reads (Anthropic semantics);
`cache_read_input_tokens` is worker-truth prompt tokens whose KV was resumed from cache;
`cache_creation_input_tokens` is 0 — there is no separate cache-write billing tier.

**Streaming**: `message_start` (real admission-truth input tokens) → `ping` →
`content_block_start` / `content_block_delta` (`text_delta`, `thinking_delta`,
`input_json_delta`, a `signature_delta` before a thinking block closes) /
`content_block_stop` per block → `message_delta` (cumulative usage, final
`stop_reason`/`stop_sequence`) → `message_stop`. Mid-stream faults emit `event: error`
with the Anthropic error body, then close. SSE comment keep-alives cover long prefill.

**Errors** are Anthropic-shaped everywhere on this surface:
`{"type":"error","error":{"type":…,"message":…},"request_id":…}` with the standard
status→type mapping (400 `invalid_request_error`, 401 `authentication_error`, 403
`permission_error`, 404 `not_found_error`, 413 `request_too_large`, 429
`rate_limit_error`, 5xx `api_error`, 503/529 `overloaded_error`). Statuses and retry
headers (`Retry-After`, `retry-after-ms`, `x-should-retry`) are identical to the chat
surface; only the body shape differs. The request id is also exposed as both
`request-id` and `x-request-id` headers.

Not implemented: `/v1/messages/count_tokens` (clients fall back to their own counting)
and server-side tools. Assistant-history `thinking` blocks are accepted and dropped, as
described above.

## `/v1/responses` — OpenAI Responses API (stateless subset)

`POST /v1/responses`. Streaming and blocking. Auth: `Authorization: Bearer <key>`, same
tenant rules as every surface.

This is the **stateless** Responses API: nothing is stored server-side, and the client
resends the full conversation in `input` each turn (`store:false` semantics — exactly
what Responses-only agent CLIs do against custom providers). Stateful features refuse
with a clear 400 naming the field:

- `previous_response_id`
- `store: true` (`false`/absent are fine)
- `conversation`
- `background: true`
- `truncation: "auto"` (no server-side context trimming; the honest value is `"disabled"`)
- `prompt` (stored prompt templates)
- `input` items of type `item_reference`

**Supported request fields**

| Field | Handling |
|---|---|
| `model` | same registry as `/v1/models` |
| `input` | string, or item array: `message` (roles `user`/`assistant`/`system`/`developer`; parts `input_text`/`output_text`/`text`; `input_image` → vision path), `function_call` (`name`, string `arguments`, `call_id`) → assistant tool_calls (consecutive calls merge into one turn), `function_call_output` (`call_id`, string-or-content-array `output`) → tool turn, `reasoning` items (this server's own prior output echoed back) are consumed |
| `instructions` | leading system turn |
| `tools` | flattened `{"type":"function","name",…,"parameters"}` → template tools branch. Non-function tool types (`web_search`, `namespace`, `custom`, …) are **dropped** from the toolset with a server log line: stock Responses clients send some unconditionally, so a 400 would refuse every default-config request; a dropped tool is one the model never sees (it cannot be half-called), not a silent behavior change to anything that is produced |
| `tool_choice` | `"auto"` / `"none"`; forcing forms → clear 400 |
| `max_output_tokens` | native `max_tokens` |
| `temperature`, `top_p`, `user` | native |
| `timeout_ms` | request deadline in milliseconds, `1000`..=`90000`, default `90000` — the SAME parameter and semantics as the chat surface and `/v1/messages`. Non-streaming it bounds the complete response; streaming it bounds time to first token only. A miss is `408` (`type: "timeout"`, `code: "deadline_exceeded"`) and is **not billed**; out of range or wrong type is a 400 naming the param. See SERVING.md "Request deadlines and the billing promise" |
| `reasoning.effort` | mapped to the model's native thinking control through the ONE canonical table every surface shares (`none`/`minimal`/`low`/`medium`/`high`; `xhigh`/`max`/`ultra` clamp to `high`; anything else 400s — identical acceptance on the chat surface's `reasoning_effort` and `/v1/messages`' `output_config.effort`, issue #31). On a model with no graded ladder a graded value translates to reasoning ON — byte-identical to `reasoning:{"enabled":true}`, documented in SERVING.md — so stock codex/Claude Code effort values work on every served model. `reasoning.summary` accepts only `"auto"` — this server returns reasoning verbatim and does not summarise, so a summary mode it cannot perform is a 400; any other `reasoning` key, and a non-object `reasoning`, are 400s matching the chat surface exactly |
| `text.format` | `text` no-op; `json_object` / `json_schema` → the same constrained decoder as the chat surface's `response_format` |
| `prompt_cache_key` | session-affinity nomination. It is deliberately NOT a cache salt: prefix-cache isolation stays tenant-scoped, so a shared instructions prefix keeps its cross-session cache hits |
| everything else | accepted and ignored (`include`, `parallel_tool_calls`, `stream_options`, `client_metadata`, `metadata`, `service_tier`, `text.verbosity`) |

**Response** (blocking): `{"id":"resp_…","object":"response","status":…,"output":[…],
"usage":{"input_tokens","input_tokens_details":{"cached_tokens"},"output_tokens",
"total_tokens"},…}`. Output items in generation order: `reasoning` (with
`summary[].summary_text`; `encrypted_content` is null), `message` with `output_text`
content, `function_call` items (`call_id`, `name`, string `arguments`). A token-budget
cutoff is `status:"incomplete"` with `incomplete_details.reason:"max_output_tokens"`,
never a fake `completed`.

**Streaming** honors the grammar strict clients verify: `response.created` first; every
item opens with `response.output_item.added` **before** any of its deltas
(`response.output_text.delta`, `response.reasoning_summary_text.delta`,
`response.function_call_arguments.delta`); the full final item rides
`response.output_item.done` (the authoritative content for spec-following clients);
exactly one terminal event — `response.completed` (carrying `response.id` + usage),
`response.incomplete`, or `response.failed` (fault, with `error.code`/`message`) — then
the stream closes. Every frame carries its `type` inside the data JSON and a monotonic
`sequence_number`.

**Errors** before the stream commits use the standard OpenAI error body
(`{"error":{"message","type","param","code"}}`), statuses and retry headers identical to
the chat surface.

## Accounting parity (all three dialects)

One admission path (`surfaces::admit_translated` mirrors `chat_completions` exactly):
canonical model resolution → tenant auth → lane → **`timeout_ms` validation** →
prepaid-budget reservation → ledger receipt (`route` records which dialect:
`/v1/messages`, `/v1/responses`) → rate-limit slot → **deadline-aware backpressure** →
meter line → worker submit → deadline-bounded admission wait.
A request deadline, its 408, its zero-debit ledger outcome and the admission sheds are
therefore identical on all four dialects; see SERVING.md "Request deadlines and the
billing promise" for the contract and the outcome→debit census. Receipt discipline on the response side is the chat
surface's: prompt usage recorded at admission, one completion record per token, terminal
complete/reject synced before the response finishes. Per-tenant capture (when armed)
stores the **translated** internal messages array — the exact prompt the template
rendered — with the same consent/trial posture as chat completions.
