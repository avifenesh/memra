# HTTP API reference

This is the request and response contract of `memra-server`: every route, the rules all routes
share, and the fields each surface accepts or refuses. It is for client developers. Operating the
server (models, caching, deployment, supervision) is [Serving](SERVING.md); every `MEMRA_*`
variable named here has its row in [Flags](FLAGS.md).

Four routes generate text over one serving core. `/v1/chat/completions` and `/v1/completions`
are native; `/v1/messages` (Anthropic) and `/v1/responses` (OpenAI Responses) rewrite their body
into the internal chat request and share its auth, admission, deadline and usage path, so only the
rendering differs. A client hard-wired to one dialect points straight at the server: an
Anthropic-format client through its base-URL setting, a Responses-format client through a custom
provider with `wire_api = "responses"`. On every surface a field the engine cannot honor is a 400
naming the field, never a silent downgrade; cosmetic and client-telemetry fields are accepted and
ignored.

## Routes

`router()` in `crates/memra-server/src/lib.rs` is the authority.

| Method | Path | Purpose | Auth |
|---|---|---|---|
| GET | `/health`, `/livez` | Inference liveness (same handler) | open |
| GET | `/readyz` | Readiness for traffic | open |
| GET | `/models` | Model list; `?schema=openai` (default), `openrouter` or `openmodels` | open |
| GET | `/v1/models` | OpenAI model list with per-model capabilities | open |
| GET | `/v1/models/{id}` | One model; the id may contain a raw or `%2F`-encoded slash | open |
| GET | `/v1/auth/check` | 204 when the key is accepted | key (Bearer or `x-api-key`) |
| POST | `/v1/chat/completions` | OpenAI chat completions | key |
| POST | `/v1/completions` | Text completions (native or OpenAI shape) | key |
| POST | `/v1/messages` | Anthropic Messages | key (Bearer or `x-api-key`) |
| POST | `/v1/responses` | OpenAI Responses, stateless | key |
| GET | `/v1/responses/{id}` | Poll a background response | key |
| POST | `/v1/responses/{id}/cancel` | Cancel a background response | key |
| GET | `/v1/jobs/{id}` | Poll a background chat or text job | key |
| POST | `/v1/jobs/{id}/cancel` | Cancel a background chat or text job | key |
| POST | `/v1/embeddings` | Embeddings from a prefill | key |
| POST | `/v1/rerank` | Cohere-shaped rerank from a prefill | key |
| POST | `/v1/tokenize`, `/v1/detokenize` | Token inspection, CPU only | key |
| POST | `/v1/audio/transcriptions` | Batch transcription (503 `engine_unbound` today) | key |
| POST, GET | `/v1/audio/sessions` | Open (503 `engine_unbound` today) or list streaming audio sessions | key |
| POST | `/v1/audio/sessions/{id}/frames`, `/v1/audio/sessions/{id}/close` | Append frames, close a session (404 `unknown_session` today) | key |
| GET | `/metrics` | Serving counters, JSON or Prometheus text | metrics |
| GET | `/yield/metrics` | Per-lane QoS counters and step p50/p99 | metrics |

The stock server has no `/admin/*` route; a deployment binary adds its own through `ServerWiring`.

## Rules every route shares

### Authentication

- `MEMRA_API_KEY` sets one bearer key (tenant `default`). `MEMRA_API_KEYS` sets a keyring (a TOML
  file of SHA-256 hashes, hot-reloaded, or an inline list) with per-key tenant, lane class and
  rate limit. With neither, the server is open; a non-loopback bind then refuses to start unless
  `MEMRA_ALLOW_OPEN_BIND=1`. Key management: [Serving](SERVING.md#api-keys).
- Keys travel as `Authorization: Bearer <key>`. `/v1/messages` and `/v1/auth/check` also take
  `x-api-key`; when both headers are present, each is tried.
- Auth reads headers only and runs before the body is read, so a rejected caller cannot spend the
  body parser.
- Metrics auth is required when the bind is non-loopback, when API keys are configured, or when
  `MEMRA_METRICS_TOKEN` is set. With a metrics token, only that token reads metrics and API keys
  get 403. Without one, an API key reads metrics, scoped to its tenant under a keyring.

### Errors, limits and nulls

- Errors use the OpenAI object `{"error":{"message","type","param","code"}}`, except on
  `/v1/messages`, which uses the Anthropic shape. Responses carry an `x-request-id` header.
- The body ceiling is 192 MiB; a larger body is a 413 (`request_too_large`) before admission.
- An unknown model is a 400 with `param: "model"` and code `model_not_found`. A request may name
  the exact roster id or a unique suffix after the last `/`; an ambiguous suffix is refused.
- A body that does not parse is a 400 `invalid_request_error`. A type error names `param` when the
  parser has a path for it (`stream`, `stop_token_ids[0]`); a syntax error reads
  `invalid JSON: <description>`. A missing or non-JSON content type is a 415. This applies to the
  routes that parse through the server's JSON extractor (chat, completions, embeddings, rerank,
  tokenize, detokenize, audio); `/v1/messages` and `/v1/responses` parse their own bodies.
- On chat and text completions an explicit `null` means absent for every field openai-python
  declares nullable (`stream: null` runs non-streaming, `prompt: null` is an omitted prompt). Null
  is a 400 only where it has no absent meaning: `model`, chat `messages` and `tools`, and memra's
  own `background` and `stop_token_ids`. `max_tokens` and `max_completion_tokens` name one cap; a
  body that sets both to values is a 400.
- While the server drains after SIGTERM, new generation requests get a 503 with `Retry-After`.

### Lanes, concurrency and session affinity

- `x-lane: interactive | judge | harvest` picks the admission lane; an unknown value is a 400.
  Batch-class keys default to `harvest` and get a 403 on `interactive`. Lane policy:
  [Serving](SERVING.md#lanes-x-lane).
- Responses on the four generation surfaces, tokenize and detokenize carry `X-RateLimit-Limit`,
  `-Remaining` and `-Reset` with concurrency-slot meaning: the lane's session cap, the free
  slots, and the estimated seconds until one frees. A request past the waiting-queue bound or
  its lane's SLO, or whose estimated queue wait exceeds its deadline, is shed at admission with a
  429.
- Session affinity uses the first present of `session_id`, `user` and the `x-session-id` header
  (on Messages, `metadata.user_id`; on Responses, `prompt_cache_key`).
- `cache_salt` names a prefix-cache namespace: requests share cached prefixes only with the same
  salt. Under a keyring the namespace is also scoped to the tenant.

### Deadlines: `timeout_ms`

All four generation surfaces, embeddings and rerank take `timeout_ms` in milliseconds.
Non-streaming accepts 1000 through 90000, default 90000, on a discrete card; on a unified-memory
device the maximum and default are derived
([SERVING.md](SERVING.md#unified-memory-devices-gb10-dgx-spark)). Streaming defaults to and caps
at the deployment's first-token ceiling (`MEMRA_STREAM_TTFT_MS_MAX`; 90000 unless the operator
raises it or the device has unified memory). Blocking, it bounds the complete response; streaming, it bounds
the time to the first generated token.

- A non-streaming request whose pessimistic estimate cannot finish inside its deadline is refused
  up front with a 400 (`code: "nonstream_deadline_infeasible"`) that names a `max_tokens` that
  fits. It takes no slot and runs no GPU work.
- A miss before any token is generated, or before stream headers, is a 408 `deadline_exceeded`,
  and no usage is counted. `/v1/messages` and `/v1/responses` answer every non-streaming miss
  this way.
- On chat and text completions, a non-streaming miss after generation started returns the tokens
  produced so far with `finish_reason: "error"`, `native_finish_reason: "deadline_exceeded"` and an
  `error` object (`code: "deadline_exceeded"`, `metadata.error_type: "timeout"`); the native
  completion shape reports `stop_reason: "Deadline"`
  with the same `error`. Those delivered tokens count as usage.
- A stream already committed for prefill keepalives ends with a named error event: an `error`
  event of type `timeout` on `/v1/messages`, `response.failed` with `code: "deadline_exceeded"`
  on `/v1/responses`.

Semantics: [Serving](SERVING.md#request-deadlines-timeout_ms).

<a id="tool-calls-and-structured-output"></a>

### Structured output: `response_format`

- `{"type":"text"}` changes nothing; `{"type":"json_object"}` and
  `{"type":"json_schema","json_schema":{...,"schema":{...}}}` decode under an llguidance grammar.
  On `/v1/responses` the same forms arrive as `text.format`. Compile limits and cost:
  [Serving](SERVING.md#constrained-decoding-response_format).
- On a template with a thinking switch, `response_format` turns thinking off for the request. On
  a template whose think segment always opens, the think segment runs unconstrained and the
  grammar engages at its close; a template with no recognizable close sequence answers 400.
- `response_format` together with a constrained tool request (see [Tool calls](#tool-calls)) is a
  400. The DeepSeek-V4-Flash route refuses `response_format` by name.
- `/v1/models` reports `capabilities.structured_output` per model.

Tools and reasoning have their own sections: [Tool calls](#tool-calls) and
[Reasoning controls](#reasoning-controls).

## Chat and text completions

### Response shape: native or OpenAI

`/v1/chat/completions` always returns OpenAI shapes. `/v1/completions` returns the native memra
shape `{model, text, tokens, stop_reason, n_tokens, prompt_tokens, cached_tokens, elapsed_s}`
unless OpenAI shape is on: `MEMRA_COMPAT=openai` turns it on, any other value keeps the native
shape, and with `MEMRA_COMPAT` unset it is on exactly when `MEMRA_API_KEY` is set. The native
shape is what the validation harnesses read.

### Request fields

| Field | Chat | Completions | Handling |
|---|---|---|---|
| `model`, `max_tokens`, `stream`, `stream_options`, `temperature`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, `stop`, `stop_token_ids`, `n`, `background`, `cache_salt`, `session_id`, `user`, `timeout_ms`, `max_ctx` | yes | yes | supported |
| `messages`, `max_completion_tokens`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, reasoning fields | yes | | supported |
| `prompt` or `prompt_ids`, `chat` (wrap the prompt in the chat template as one user turn) | | yes | supported; `prompt_ids` are checked against the vocabulary |
| `logit_bias` | 400 | 400 | no device-side bias hook |
| `logprobs` | 400 unless `false` | 400 | not implemented |
| `top_logprobs` | 400 | | not implemented |
| `best_of` | 400 unless 1 | 400 unless 1 | use `n` |

Chat messages take `system`, `developer` (treated as system), `user`, `assistant` and `tool`
roles, string or part-array content, and `image_url` parts on vision models (data URIs always;
`http(s)` only with `MEMRA_FETCH_URLS=1`).

### Streaming usage

With `stream_options.include_usage: true` on chat or OpenAI-shape completions, content and finish
chunks carry `usage: null`, then one `choices: []` chunk carries the full usage object (cached
prompt tokens and speculative acceptance fields included) before `[DONE]`. Without it, usage rides
the choices-bearing finish chunk. A failed stream ends with its error and `[DONE]`, never with a
fabricated usage chunk.

### Token-id stops on chat and completions

`stop_token_ids: [u32]` (default `[]`, at most 16) stops on raw sampled or accepted ids, before
detokenization. A longer list or an id outside the vocabulary is a 400 with
`param: "stop_token_ids"`; when the model's vocabulary size is unknown, the vocabulary check is
skipped, as for `prompt_ids`. The first match ends generation with `finish_reason: "stop"`; that id
and any later ids in its speculative burst are neither emitted nor counted in
`usage.completion_tokens`. Listed ids win over EOS; unlisted EOS keeps its usual accounting. String
`stop` works independently, streaming or not.

### `ignore_eos` on completions (deepseek-v4.1 route)

`POST /v1/completions` accepts `ignore_eos: bool` (default `false`), the benchmark field of other
OpenAI-compatible servers: with `true` the model's end-of-sentence token no longer ends generation;
it is detokenized into the stream and fed back like any token, and the request ends at
`max_tokens`, a stop string or a stop id. Only the deepseek-v4.1 route serves it. For any other
model `ignore_eos: true` is a 400 with `param: "ignore_eos"`; `false` is accepted everywhere. The
chat surface has no such field.

### Multiple choices: `n`

`n` defaults to 1 and is bounded at 8; a boot's decode-slot cap can lower it, and `/v1/models`
reports `capabilities.max_choices`. One leader prefills once; its prompt KV and recurrent state
are copied into independent rows, each with its own sampler, cache, parser and stop state. With
`seed`, row i uses `(seed + i) mod 2^64`; without, one master seed is drawn for the group.
`max_tokens` bounds each choice. Admission reserves n times the output bound and n decode slots
under one request-quota entry, and counts the prompt once; `usage.completion_tokens` sums all
rows. Streams may interleave indices, each row has one finish reason, and `[DONE]` ends every
stream, after the last row or after the error event of a failed stream. A worker failure never
becomes a successful partial choice list. With a deployment ledger attached, a receipt failure
returns `request_ledger_unavailable` (HTTP 500 before headers; after them, that error body and
`[DONE]`) and releases every reservation.

The route is text-only and single-device. Speculative, parallel, sliding-window, latent and other
unsupported states refuse `n > 1` by name rather than switching numerical program, and
`background: true` with `n > 1` is refused.

### Background jobs: `background`

`background: true` on chat or text completions returns `{id, status: "queued"}`. Poll with
`GET /v1/jobs/{id}` and cancel with `POST /v1/jobs/{id}/cancel`. Both authenticate the tenant (a
rotated key of the same tenant keeps access), and a foreign or unknown id is the same 404.

- A pending job shows identity and status. A terminal one keeps its original chat, OpenAI-text or
  native-text envelope plus its status: `completed`, `incomplete` (token or context cutoff),
  `cancelled` (with the partial output and a cancelled error) or `failed`. Repeated polls return
  the same body.
- A background request keeps the normal admission and model limits but skips the synchronous
  deadline and feasibility gate, so `timeout_ms` does not bound it. With `stream` it is a 400, and
  a non-boolean value is a 400.
- The stock in-memory store keeps a result for `MEMRA_BACKGROUND_JOB_TTL_SECS` (900 by default)
  after the job ends, under a store cap of `MEMRA_BACKGROUND_JOB_MAX_BYTES` (64 MiB). A deployment
  binary may supply its own store.

Delivery contract: [Serving](SERVING.md#background-delivery-for-long-non-streaming-requests-background-true).

## Tool calls

Tools are client-defined functions. The server renders them through the model's own chat
template and parses the model's tool-call blocks back into the surface's call shape. Chat takes
`tools: [{"type":"function","function":{name, description?, parameters?}}]`; Responses takes
`{"type":"function","name",...,"parameters"}` entries and maps them to the chat form; Messages
takes `{name, description, input_schema}` with its own `tool_choice` forms (see its table).

### Rendering

- Each template renders tools in its own protocol, reproduced byte for byte from the vendor
  template: the Qwen class's `<tool_call>` and `<function=...>` blocks, Hy3's `:opensource`
  tags (`<tool_call:opensource>`), Gemma 4's `<|tool_call>`, GLM-5.3-Flash's own form and
  DeepSeek-V4's DSML.
- A model whose template has no tools branch answers a request carrying `tools` with a 400
  (`chat template has no tools branch`) before admission.
- `tool_choice: "none"` renders the prompt without the tools block, byte-identical to a request
  without tools, and runs no call parser.
- Requests without tools skip the tools renderer and the call parser. The one exception is the
  Qwen3.8 effort-ladder template, where every chat request uses the tools-capable renderer,
  because only that renderer writes the vendor's default effort on an unset request.
- `usage.prompt_tokens` counts the rendered tools block, and a repeated tools block is an ordinary
  cacheable prefix.

### `tool_choice` and `parallel_tool_calls`

| Value | Behavior |
|---|---|
| `tool_choice` omitted, `null` or `"auto"` | the template's native path; the model decides |
| `"none"` | tools are not rendered and no parser runs |
| `"required"` | at least one call, forced through a tool-call grammar; without declared tools it is a 400 |
| `{"type":"function","function":{"name":...}}` | a call to that function, forced through the grammar; an undeclared name is a 400 |
| `parallel_tool_calls` omitted | the template's own behavior |
| `parallel_tool_calls: false` | ordinary content or one complete call, through the grammar |
| `parallel_tool_calls: true` | several calls where the dialect allows them; a 400 on the single-call Gemma 4 template |

A forced or single-call request is a constrained tool request. It runs on the Qwen-class and
Gemma 4 dialects; Hy3, GLM-5.3-Flash and DeepSeek-V4 answer a named 400 (`constrained tool calls
are not supported by this template dialect`). The arguments are one JSON object checked against
the function's `parameters` schema (default `{"type":"object"}`, and the schema must allow an
object). These refuse before generation, with `param` naming `tool_choice` or
`parallel_tool_calls`:

- no declared function, or more than 128;
- a tools array larger than the constrained schema byte limit, a tool that is not a function,
  or a function name that is empty, longer than 64 bytes, outside `[A-Za-z0-9_-]` or repeated;
- a `parameters` schema the grammar compiler refuses;
- an explicit `stop` or `stop_token_ids`, which could cut a call short;
- `response_format` in the same request.

### Tool-call output

- Calls parse from the generated stream into OpenAI `tool_calls`: streaming deltas, or
  `message.tool_calls` with `content: null` on a turn that is only calls. Ids are deterministic,
  `arguments` is a compact JSON string with values coerced to the declared schema types, and
  `finish_reason` is `"tool_calls"`. Messages returns `tool_use` blocks with parsed `input`;
  Responses returns `function_call` items.
- A block that does not parse is returned as content, verbatim and with its tags; an unterminated
  block flushes as raw content when generation ends. A malformed call is never an error and never
  drops bytes: content plus parsed calls reassemble to the generated text.
- Text inside an open reasoning segment is not scanned, so a call the model mentions while
  reasoning is not a call.
- In history, `tool_calls` belong on `assistant` messages (elsewhere they are a 400), and tool
  results arrive as `tool` role turns.
- `/v1/models` reports `capabilities.tools` (the template has a tools branch),
  `forced_tool_calls` (a constrained dialect exists) and `parallel_tool_calls`. These describe the
  API mechanics, not a model's support state.

Receipts: [research/serve-tools-20260802/](../research/serve-tools-20260802/) (round trips, the
streaming schema check, the malformed-block transcript),
[research/tool-choice-20261003/](../research/tool-choice-20261003/) (forced and single-call
requests) and [research/integrate-cache-20260802/](../research/integrate-cache-20260802/) (tools
with the prefix cache).

## Reasoning controls

One effort table serves all three dialects (`canonical_effort_for` and `parse_think` in
`crates/memra-server/src/lib.rs`): the same values are accepted and refused, with the same effect,
on every surface.

| Value | Meaning |
|---|---|
| `none`, `minimal` | thinking off |
| `low`, `medium`, `high` | thinking on; the level reaches only templates that read one |
| `xhigh`, `max`, `ultra` | thinking on at the highest level the template distinguishes: `max` on DeepSeek-V4-Flash and GLM-5.3-Flash, `high` elsewhere |
| no reasoning field | the template's own default, or the `default_reasoning_effort` the model's [metadata](serving/model-metadata.md) declares |
| anything else | 400 |

`minimal` means off on purpose, unlike hosted APIs that map it to a short low-effort think: the
server's promise is that the no-reasoning side is real. On a model that cannot turn reasoning off,
`minimal` is the same 400 as `none`.

The levels save tokens; they are not a quality ladder. On Step-3.7-Flash (NVFP4, 2x RTX PRO 6000,
three prompts, n=8 sampled per level, 2026-08-29) every level was honored, `high` reasoned less
than `medium` on both non-trivial prompts, and the request with no level reasoned longest
([receipt](../research/step37-reasoning-effort-20260829/RESULTS.md)). Why the table reads this
way: [decision](decisions/REASONING-CONTROL.md).

### Fields per surface

| Surface | Fields |
|---|---|
| `/v1/chat/completions` | `reasoning_effort`; a `reasoning` object with exactly `effort`, `enabled` and `exclude` (any other key, `max_tokens` included, is a 400; a `null` key or object is unset); `enable_thinking` and `chat_template_kwargs.enable_thinking`, aliases of `reasoning.enabled`; `include_reasoning: false` and `reasoning.exclude: true`, which turn reasoning off rather than hide it. `chat_template_kwargs` takes `enable_thinking` and `preserve_thinking: true`; `preserve_thinking: false` and any other key are a 400 |
| `/v1/messages` | `thinking.type`: `enabled`, `disabled` or `adaptive` (the model's default); `thinking.budget_tokens` is a 400, since no lever can cap a reasoning segment; `output_config.effort` uses the table, and `output_config` takes no other key |
| `/v1/responses` | `reasoning.effort` uses the table; `reasoning.summary` (or its alias `generate_summary`) accepts only `"auto"`, since reasoning is returned verbatim; other keys are a 400 |

### Precedence and refusals

- An explicit switch (`reasoning.enabled`, either `enable_thinking` spelling, Anthropic
  `thinking.type`) decides on or off over the switch an effort value implies. The effort value is
  still validated and still supplies the level.
- Two switches that disagree are a 400: `reasoning.enabled` against `enable_thinking`, the two
  `enable_thinking` spellings, or reasoning on with `include_reasoning: false` or
  `reasoning.exclude: true`.
- Every off spelling renders the same prompt as `reasoning_effort: "none"`.
- A graded level on a template with only an on/off switch means on, and renders the same prompt
  as `reasoning: {"enabled": true}`. Stock agent clients send `xhigh` on every request, so this
  mapping keeps them working against binary-switch models.
- Step-3.7-Flash and GLM-5.3-Flash open their think segment unconditionally and have no switch.
  An off request from the client is a 400 naming the model (`cannot disable reasoning`). A
  `default_reasoning_effort` of `none` or `minimal` never refuses a client that sent nothing; it
  renders the lowest level instead.
- `max_tokens` is one budget for reasoning and content. Reasoning tokens count as completion
  tokens, there is no separate reasoning budget and no `reasoning_tokens` field, so a small budget
  on a thinking request can end with `finish_reason: "length"` and empty content. To spend less,
  turn reasoning off, lower the level or raise the cap.

### What each template renders

| Template | absent | `none`, `minimal` | `low` | `medium` | `high` | `xhigh`, `max`, `ultra` |
|---|---|---|---|---|---|---|
| Qwen3.5 and 3.6 class, including Ornith and AgentWorld (`enable_thinking` switch) | on: open `<think>` | closed `<think></think>` | on | on | on | on |
| Qwen3.8 (switch plus the vendor's effort sentence) | on, `xhigh` sentence | closed | on, `low` sentence | on, no sentence | on, `xhigh` sentence | on, `xhigh` sentence |
| Gemma 4 (`enable_thinking`, template default off) | off: closed thought channel | off | on: `<\|think\|>` in the system turn | on | on | on |
| Hy3 (`reasoning_effort` header) | `no_think` | `no_think` | `low` | `low` (no medium level) | `high` | `high` |
| Step-3.7-Flash (`Reasoning:` line; think always open) | no line | 400 | `Reasoning: low` | `Reasoning: medium` | `Reasoning: high` | `Reasoning: high` |
| GLM-5.3-Flash (`Reasoning Effort:` line; think always open) | `Max` | 400 | `Low` | `High` (no medium rung) | `High` | `Max` |

DeepSeek-V4-Flash renders its encoding's effort prefix: off is chat mode with no prefix; on the
0731 encoding `high` adds the absolute-max prefix and `xhigh`, `max` and `ultra` the beyond-max
prefix, while absent, `low` and `medium` add none. On Qwen3.5 and 3.6 class and Gemma 4 templates
the three on levels render identical prompts; the level is wired only on templates that read
one. The `/models?schema=openrouter` feed lists the native rungs of the two ladders: `xhigh`,
`medium` and `low` on Qwen3.8, `low`, `high` and `max` on GLM-5.3-Flash.

### Delivery

Reasoning text is always returned on every surface, never stripped server-side, and a request
with reasoning off carries no reasoning field rather than an empty one. To keep it out of view,
drop it client-side; to stop generating it, turn reasoning off.

| Surface | Non-streaming | Streaming |
|---|---|---|
| `/v1/chat/completions` | `message.reasoning` (string) and `message.reasoning_details[]` (`{type: "reasoning.text", text}`) | `delta.reasoning` |
| `/v1/messages` | a `{type: "thinking", thinking, signature: ""}` content block | `content_block_start` of type `thinking`, then `thinking_delta` |
| `/v1/responses` | a `{type: "reasoning", summary: [{type: "summary_text", text}]}` output item | `response.reasoning_summary_text.delta` and `.done` |

Assistant history may carry `reasoning` (alias `reasoning_content`). The Qwen3.8 ladder template
replays it inside a `<think>` block, empty when absent, as its vendor template does by default
(`preserve_thinking`).

## `/v1/tokenize` and `/v1/detokenize`: token inspection

Both are CPU-only POST routes with the same key, body ceiling and per-tenant request slot as chat
completions (the `x-ratelimit-*` trio and a named 429 at the key's concurrency cap). They queue no
GPU work and count no generation usage. Both echo a supplied `x-request-id` header.

- `/v1/tokenize` takes `{"model","prompt"}` or `{"model","messages","tools"?,
  "chat_template_kwargs"?, "add_generation_prompt"?}`, exactly one of `prompt` and `messages`. It
  returns `{"tokens":[...],"count":N,"max_model_len":...}`. Messages render through chat's own
  builder, tools and model metadata defaults included; `add_generation_prompt` defaults to `true`,
  matching chat's `prompt_tokens`. Image and video messages are refused, not partially counted. A
  raw prompt parses literal special tokens; `add_special_tokens` (default `true`, prompt form only)
  adds BOS exactly as `/v1/completions` does, and `false` returns the reversible ids
  `/v1/detokenize` maps back to the exact input. `max_model_len` is the context `/v1/models`
  advertises, or null.
- `/v1/detokenize` takes `{"model","tokens":[...]}` and returns `{"prompt":"text"}`. Control
  tokens decode to their literal strings; an out-of-vocabulary id is a 400 with
  `param: "prompt_ids"`.

Token inspection uses the worker's tokenizer source, including GGUF tokenizer overrides for
HF-directory models. The DeepSeek-V4-Flash route renders with the same plain-versus-tools
predicate, so the prompt it serves is the prompt counted here.

## `/v1/messages`: Anthropic Messages API

`POST /v1/messages`, streaming and blocking. Query strings such as `?beta=true` are accepted, and
`anthropic-version` is accepted and not enforced.

| Field | Handling |
|---|---|
| `model` | same registry as `/v1/models` |
| `max_tokens` | required, at least 1 |
| `messages` | roles `user`, `assistant`, `system`; string or block-array content |
| content blocks | `text`; `image` (base64 source only); `tool_use` on assistant turns becomes tool calls; `tool_result` on user turns (string or text-block content, `is_error` accepted) becomes a tool turn; `mid_conv_system` becomes a system turn; `thinking` and `redacted_thinking` in history are dropped, as templates re-render history without think segments; other block types are a 400 |
| `system` | string or text-block array, concatenated; `cache_control` markers are ignored (prefix caching is automatic) |
| `tools` | client-defined tools (`name`, `description`, `input_schema`); server tool types (`web_search_*`, `bash_*`, `code_execution_*`, ...) are a 400 |
| `tool_choice` | `{"type":"auto"}` or `{"type":"none"}`; `any` and `tool` are a 400 on this surface (use the chat surface's `required` or named form to force a call); `disable_parallel_tool_use` is ignored |
| `stop_sequences` | stop strings; the match is reported as `stop_sequence` |
| `temperature`, `top_p`, `top_k`, `timeout_ms` | as on chat |
| `thinking`, `output_config.effort` | see [Reasoning controls](#reasoning-controls); other `output_config` keys are a 400 |
| `metadata.user_id` | session affinity |
| `mcp_servers` | 400: server-side MCP does not run here |
| anything else | accepted and ignored (`cache_control`, `context_management`, `service_tier`, beta fields) |

**Response**: `{"id":"msg_...","type":"message","role":"assistant","content":[...],
"stop_reason","stop_sequence","usage"}`. Content blocks come in generation order: `thinking`
(when the model separated its reasoning; `signature` is empty, as there is no signing key),
`text`, then `tool_use` blocks with `input` as parsed JSON. `stop_reason` is `tool_use`,
`stop_sequence`, `max_tokens` or `end_turn`.

**Usage**: `input_tokens` excludes cache reads (Anthropic semantics); `cache_read_input_tokens`
counts prompt tokens whose KV was resumed from cache; `cache_creation_input_tokens` is 0.

**Streaming**: `message_start` (with real input token counts), `ping`, then per block
`content_block_start`, `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta`,
and a `signature_delta` before a thinking block closes) and `content_block_stop`, then
`message_delta` (cumulative usage, final `stop_reason` and `stop_sequence`) and `message_stop`. A
mid-stream fault emits `event: error` with the Anthropic error body, then closes. SSE comments keep
the connection alive through long prefill.

**Errors**: `{"type":"error","error":{"type","message"},"request_id"}` with the standard status
mapping (400 `invalid_request_error`, 401 `authentication_error`, 403 `permission_error`, 404
`not_found_error`, 408 `timeout`, 413 `request_too_large`, 429 `rate_limit_error`, 5xx
`api_error`, 503 and 529 `overloaded_error`). Statuses and retry headers (`Retry-After`, `retry-after-ms`,
`x-should-retry`) match the chat surface. The request id is in both `request-id` and
`x-request-id`.

Not implemented: `/v1/messages/count_tokens` and server-side tools.

## `/v1/responses`: OpenAI Responses API (stateless subset)

`POST /v1/responses`, streaming and blocking. Nothing is stored between requests: the client sends
the whole conversation in `input` each turn (`store: false` semantics). These stateful features
are a 400 naming the field: `previous_response_id`, `store: true`, `conversation`,
`truncation: "auto"`, `prompt` (stored prompt templates) and `input` items of type
`item_reference`. A non-boolean `stream`, `store` or `background` is a 400 (`<field> must be a
boolean`); `null` is absent.

| Field | Handling |
|---|---|
| `model` | same registry as `/v1/models` |
| `input` | a string, or items: `message` (roles `user`, `assistant`, `system`, `developer`; parts `input_text`, `output_text`, `text`, `input_image`), `function_call` (`name`, string `arguments`, `call_id`; consecutive calls merge into one assistant turn), `function_call_output` (`call_id`, string or content-array `output`), and `reasoning` items (this server's earlier output echoed back), which are consumed |
| `instructions` | a leading system turn |
| `tools` | `{"type":"function","name",...,"parameters"}` entries render through the template's tools branch. Other tool types (`web_search`, `namespace`, `custom`, ...) are dropped with a server log line: stock clients send some unconditionally, and a dropped tool is one the model never sees |
| `tool_choice` | `"auto"`, `"none"`, `"required"`, or `{"type":"function","name":...}`, mapped to the chat forms |
| `parallel_tool_calls` | boolean (anything else is a 400), passed to the chat rule above |
| `max_output_tokens` | the `max_tokens` cap |
| `temperature`, `top_p`, `user`, `timeout_ms` | as on chat |
| `reasoning` | see [Reasoning controls](#reasoning-controls) |
| `text.format` | `text` is a no-op; `json_object` and `json_schema` use the chat surface's constrained decoder; other types are a 400 |
| `background` | `true` returns `status: "queued"`; poll `GET /v1/responses/{id}`, cancel `POST /v1/responses/{id}/cancel`. The result stays until its TTL passes after the job ends; polls do not consume it. With `stream: true` it is a 400 |
| `prompt_cache_key` | session affinity only; it is not a cache salt, so a shared instructions prefix keeps its cross-session cache hits |
| anything else | accepted and ignored (`include`, `stream_options`, `client_metadata`, `metadata`, `service_tier`, `text.verbosity`) |

**Response** (blocking): `{"id":"resp_...","object":"response","status",...,"output":[...],
"usage":{"input_tokens","input_tokens_details":{"cached_tokens"},"output_tokens",
"total_tokens"}}`. Output items come in generation order: `reasoning` (with
`summary[].summary_text`; `encrypted_content` is null), `message` with `output_text` content, and
`function_call` items (`call_id`, `name`, string `arguments`). A token-budget cutoff is
`status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`.

**Streaming** follows the grammar strict clients check: `response.created` first; every item
opens with `response.output_item.added` before any of its deltas (`response.output_text.delta`,
`response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`); the final
item rides `response.output_item.done`; then exactly one terminal event, `response.completed`
(with `response.id` and usage), `response.incomplete` or `response.failed` (with `error.code` and
`message`), and the stream closes. Every frame carries its `type` in the data JSON and a monotonic
`sequence_number`.

**Errors** before the stream commits use the OpenAI error body, statuses and retry headers.

## Usage accounting across surfaces

Every generation surface enters one admission path (`surfaces::admit_translated` mirrors
`chat_completions`), so a request counts the same prompt and completion tokens whichever dialect
carried it, and the deadline gate, the 408s and the admission sheds behave the same on all four;
only the non-streaming partial on a mid-generation deadline differs by dialect (see
[Deadlines](#deadlines-timeout_ms)). Prompt usage is
recorded at admission and completion tokens as they stream. The stock server ships no accounting:
a deployment binary attaches metering, a request ledger, tenant budgets or capture through
`ServerWiring`, and the stock binary refuses their variables (`MEMRA_REQUEST_LEDGER`,
`MEMRA_TENANT_BUDGETS`, `MEMRA_CAPTURE_DIR`, ...) at boot. The seam and the outcome each terminal
path settles: [deployment binaries](serving/deployment-binary.md#the-metering-seam).

## Embeddings and rerank

Both run a prefill of a causal LM, with no decode step, and read the last prompt position.

- `POST /v1/embeddings`: `{model, input: string | [string] (1 to 32), dimensions?,
  encoding_format?, timeout_ms?}`. Each vector is the last-token hidden state after the final
  norm, L2-normalized; `dimensions` truncates (Matryoshka) and re-normalizes. Only
  `encoding_format: "float"` is served. Returns `{object: "list", data: [{object: "embedding",
  index, embedding}], model, usage: {prompt_tokens, total_tokens}}`.
- `POST /v1/rerank`: `{model, query, documents: [string] (1 to 64), top_n?, instruction?,
  return_documents?, timeout_ms?}`. Scores come from the model's yes/no logits (the
  Qwen3-Reranker convention). Returns `{results: [{index, relevance_score, document?}], usage}`.

Each input is one admitted worker request with the usual auth, lane and rate limits, and its
tokens count as prompt usage. Worker
semantics: [Serving](SERVING.md#embeddings-and-rerank).

## Audio

`/v1/audio/transcriptions` and the `/v1/audio/sessions` lifecycle (open, list, append frames,
close) exist with their session, frame and shed contracts. No speech pack can be resident yet, so
each route answers as follows today:

| Route | Outcome today |
|---|---|
| `POST /v1/audio/transcriptions` | 503 `engine_unbound` once the request passes validation |
| `POST /v1/audio/sessions` | 503 `engine_unbound` once the model and lane resolve, checked before the session cap |
| `GET /v1/audio/sessions` | 200 with the operator view: `drive`, `sessions_live`, `sessions_max`, `queue_frames`, `frame_ms`, `final_slo_ms` and the `scheduler` counters, with shed counts per code |
| `POST /v1/audio/sessions/{id}/frames` | 404 `unknown_session`; `frames` outside 1 to 256 is a 400 first |
| `POST /v1/audio/sessions/{id}/close` | 404 `unknown_session` |

Transcription takes JSON with base64 audio in `file`; a multipart form is refused with a 415 and a
message. `top_p` is a 400, and `temperature` is accepted only as 0 or a rung of the model's
declared ladder. Speech program:
[Speech](SPEECH.md).

## Models, health and metrics

- `GET /v1/models` lists each served model with `context_length`, modalities and a `capabilities`
  block (`streaming`, `max_choices`, `tools`, `forced_tool_calls`, `parallel_tool_calls`,
  `structured_output`, `reasoning`, `prompt_caching`). Pricing, lifecycle and reliability fields
  come from the optional `MEMRA_MODEL_METADATA` file. `GET /models` defaults to
  `{"object":"list","data":[{"id":...,"object":"model"}]}`; `?schema=openrouter` and
  `?schema=openmodels` render the
  listing feeds described in [model metadata](serving/model-metadata.md#gateway-listing-surface).
- `GET /health` and `/livez` report inference liveness: `status` `ok`, `draining` or `unhealthy`,
  with worker phase and GPU probe blocks. `ok` and `draining` answer 200; a dead, faulted, stalled
  or still-loading worker answers 503. `GET /readyz` answers `ready` (200) or `not_ready` (503).
  Supervision: [Serving](SERVING.md#health-supervision-and-faults).
- `GET /metrics` returns JSON, or Prometheus text when `Accept` prefers `text/plain` or
  `application/openmetrics-text`. `GET /yield/metrics` returns per-lane QoS counters and engine
  step p50/p99.

History of this contract (dated changes and the old accounting text):
[history/API-SURFACES.md](history/API-SURFACES.md).
