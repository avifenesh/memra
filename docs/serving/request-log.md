# Per-request timing log

`MEMRA_REQUEST_LOG` makes `memra-server` write one JSON line per generation request: its timing,
token counts, tool calls and a session key that holds across turns, and no prompt or reply text.
This page is the field contract, for operators who collect the log and for code that reads it.
The data is shaped for predicting when a session's next request arrives.

## Enabling it

```bash
MEMRA_REQUEST_LOG=/var/log/memra/requests.jsonl memra-server
```

- The log covers `POST` on the four generation routes: `/v1/completions`,
  `/v1/chat/completions`, `/v1/responses` and `/v1/messages`, streaming and blocking. Refusals
  get a line too, auth refusals included, because the middleware sits outside every other layer.
- Unset or empty, the middleware is not installed and nothing is written. A value that is not
  valid UTF-8, or a path that cannot be opened, stops the boot before any model loads.
- The file is opened for append at boot and created with mode 0640. The boot log names the path
  and the queue bounds.
- Each line is one `O_APPEND` write, so lines never interleave. The server keeps its descriptor:
  rotate with `copytruncate`.
- A dedicated thread hashes, serializes and writes. Its queue holds at most 4096 lines and 64 MiB.
  A line that does not fit is dropped and counted, and serving never waits for it. A failed write
  is counted the same way and logged once.
- At shutdown the writer gets up to 2 s to finish the lines it already accepted. A request still
  running when the process exits (a drain-killed stream, a background job) has no line.
- `memra_request_log_lines_total{outcome}` on the Prometheus scrape counts `written`,
  `dropped_queue_full` and `dropped_write_error` lines. The family is absent when the log is
  unset ([metrics.md](metrics.md)).

The variable's catalog row is in [FLAGS.md](../FLAGS.md#server-and-api).

## Line fields

| field | meaning |
| --- | --- |
| `v` | line schema version, `1` |
| `request_id` | the response's `x-request-id` header (`request-id` when only that is set), which is also the body's `id` |
| `route` | one of the four paths above |
| `model` | the canonical served model name; `null` when the request named no served model or was refused before its handler resolved one |
| `stream` | the response was SSE |
| `session_key` | 64 hex characters, see [The session key](#the-session-key); `null` when the request has neither identity |
| `session_key_source` | `session_id`, `user` or `x-session-id` (the explicit identities, in priority order), `fingerprint`, or `null` |
| `t_arrival` | Unix seconds with microseconds: when the request reached the server |
| `t_first_token` | when the first generated token reached the HTTP layer; `null` when none was generated |
| `t_end` | when the response ended: the last SSE frame, or the blocking body ready to send. A `background` request, or one cut before its terminal event, ends when its generation stopped |
| `end_reason` | `stop`, `length`, `error`, `cancelled` or `deadline`, see [The end reason](#the-end-reason) |
| `http_status` | the response status; `499` when the client left before a response head existed |
| `prompt_tokens`, `cached_tokens` | the worker's prompt accounting, the same numbers as `usage.prompt_tokens` and `usage.prompt_tokens_details.cached_tokens` |
| `generated_tokens` | the worker's completion count; for a request cut before its terminal event, the tokens it had produced |
| `tool_calls` | `[{"name", "arguments_sha256"}]` in emission order; `[]` when the reply had none; `null` on `/v1/completions` (raw text, not parsed) and on a request refused before it was submitted |
| `prompt_sha256` | sha256 of the prompt token ids the model was fed (template and tools rendered), each id as a little-endian u32. Reproduce it from `/v1/tokenize` |

`arguments_sha256` hashes the call's `arguments` string as the OpenAI-shaped surfaces send it. On
`/v1/messages` that string is the compact JSON of the call's `input`, keys in order.

Times come from one wall-clock reading at arrival plus monotonic offsets, so a clock step during
a request does not distort its own durations. A field is `null` when the request never reached
the stage that produces it: a 400 has no token counts, and a request refused before its handler
bound a model has `model: null`.

With `n` above 1 the line describes choice 0: its receiver, its tool calls and the group's
deadline.

## The session key

An explicit identity wins, trimmed, in the order [session affinity](../SERVING.md#session-affinity) uses:
`session_id`, then `user`, then the `x-session-id` header. `/v1/responses` passes
`prompt_cache_key` as `session_id`, and `/v1/messages` passes `metadata.user_id` as `user`.

Without one, the key comes from the implicit affinity tier: the first three entries of the
conversation fingerprint chain over the fed prompt, with the live turn dropped, as little-endian
u64 values (24 bytes). Two requests to one model share this key exactly when the worker's
implicit tier would nominate one's parked session for the other:

- Segments split at the tokenizer's control tokens, and each contributes only its first and last
  8 tokens. Two sessions that open with the same system prompt and the same first user turn share
  a key, as they share a nomination.
- The chain needs three closed segments before the turn being generated. On ChatML templates
  (`<|im_start|>` and `<|im_end|>` both split) a system prompt plus the first user turn already
  make three. A first turn with no system prompt has no key yet (`null`); its later turns do.

The key is the sha256 of the identity bytes (the id string, or the 24 chain bytes). When the
request has a cache namespace, the material is `<namespace> 0x1f <identity>`, so two tenants that
both name a session `1` get different keys. The namespace is the raw `cache_salt` without a
keyring, and `t:<tenant> 0x1f <cache_salt>` with one (the salt may be empty). With no namespace,
a client reproduces its key as `sha256(session_id)`.

## The end reason

Resolved in this order, first match wins:

1. The consumer's own verdict. `deadline` when `timeout_ms` cut the request (a committed
   stream's in-band deadline error, or a blocking partial). `cancelled` when a background request
   was cancelled. `error` when a stream ended with an in-band accounting error.
2. What the client was told. No response head is `cancelled`. An HTTP 408 of any cause (the
   request deadline, or a body that did not arrive in time) is `deadline`. Any other status of
   400 or above is `error`.
3. The worker's terminal event. `length` for `max_tokens` or the context limit. `stop` for end of
   turn, stop strings and ids, and a reply that ends in tool calls. An engine error, or a stream
   the worker closed without a terminal event, is `error`.
4. Otherwise the consumer let go: a success head whose generation stopped without a terminal
   event (the client hung up) is `cancelled`.

## Privacy of the log

- No prompt text, reply text or tool arguments are written. Content appears only as
  `prompt_sha256` and `arguments_sha256`, identities only as `session_key`.
- The digests are unsalted, so the log is pseudonymous, not anonymous. A short or guessable
  `user`, `session_id`, `x-session-id` or argument string can be recovered by hashing guesses,
  and a known prompt can be matched by hashing its token ids. Send opaque identifiers if that
  matters.
- Tool names are written in clear.
- The timing fields describe each client's traffic pattern. Treat the file as usage data: it is
  created 0640, and access to it should follow the same rule as operator metrics.

## Cost and exactness

Every hook observes; none changes an event, a parsed piece or a frame. Generated bytes are
identical with the log set and unset: the CPU test `request_log_on_and_off_serve_identical_bytes`
covers the HTTP side, and a device A/B on the RTX 5090 Laptop (2026-11-09; Qwen3.5-9B, four boots
in the order unset, set, set, unset, eight requests per boot over all four routes, three runs)
found 0 byte differences ([`research/request-log-20261109/`](../../research/request-log-20261109/README.md)).

- The GPU worker copies the prompt ids once per request, only when the log is set.
- Per received token event the HTTP side pays a few branches and one integer add: no lock,
  allocation or I/O.
- A queued line pins its prompt ids (4 bytes per token) until it is written. For short prompts the
  line bound comes first (about 40 s of disk stall at 100 requests per second). For long ones the
  byte bound does (16M prompt tokens, about 160 requests of 100k tokens).

## Example line

From the device A/B: Qwen3.5-9B, the first turn of a chat session with no identity, which called
a tool.

```json
{"v":1,"request_id":"chatcmpl-f946dd69c47663b1a9c6f3ff2cca5cb9","route":"/v1/chat/completions","model":"m","stream":false,"session_key":"d25d7a17bd4bf7772bad1f45675bcb2d66e0b692dc40dedd8048b46d99f16aec","session_key_source":"fingerprint","t_arrival":1794181690.737148,"t_first_token":1794181690.799047,"t_end":1794181691.087214,"end_reason":"stop","http_status":200,"prompt_tokens":295,"cached_tokens":256,"generated_tokens":56,"tool_calls":[{"name":"get_weather","arguments_sha256":"6e1e312d537bc71b5410b0599f5a508142149e13174c6ee0d1671658845bc67d"}],"prompt_sha256":"3cb76adf31b02bf26debfe59be0f8b771e6184d2d1e2100c343b995c22ca4830"}
```

For a per-phase TTFT breakdown (parse, queue, tokenize, prime, first decode, first SSE byte) on
completions and chat completions, `MEMRA_TTFT_TRACE=1` prints one `[ttft]` line per request to the
server log instead.
