# memra#1076 GB10 live check: an n = 2 stream past the SSE prefill commit (0a34ce3444e9)

The check memra#1076 owed: on GB10, an `n = 2` streaming chat request whose prefill runs past the
60 s SSE prefill commit must stream to completion with no 408 or 429. Main is the red arm.

Conditions: one DGX Spark (GB10), Qwen3.8-27B GGUF (NVFP4+Q5_K trunk) on the plain route
(`MEMRA_SERVE_SPEC=0`, no draft; n > 1 shared prefill is offered only there), `MEMRA_CTX=262144`, `MEMRA_MAX_SESSIONS=32`, derived unified-memory defaults (the 60 s SSE
prefill commit and the derived streaming deadline, both named in the boot's `[deadline]` line), one
fresh boot per arm. Request: one user message of about 131k tokens (the gb10-prefill harness
document with a fresh nonce, so no prefix-cache hit), `n: 2`, `stream: true`, temperature 0,
thinking off, `max_tokens: 40`. `harness/n2_stream_check.py` logs every SSE line with its arrival
time (`check/events.jsonl`) and judges `check/result.json`.

Binaries: main = the cell-evidence build of `5d70118b99`; head = the hashed cell binary of `0a34ce3444`
(the same bytes as the sm_121a cell record banked beside this receipt).

| arm | HTTP | headers at | first token (choice 0 / 1) | finish (0 / 1) | error events | `[DONE]` | total | verdict |
|---|---|---|---|---|---|---|---|---|
| 01-main | 408 | 60.016 s | - / - s | - / - | none; body: {"error":{"message":"admission did not complete within the 60000 ms pre-header budget (streaming first-token timeout_ms=997311); generation was cancelled and th | False | 60.016 s | FAIL |
| 02-head | 200 | 60.022 s | 186.023 / 186.023 s | length / length | none | True | 191.083 s | PASS |

Result: on main the stream is refused before its headers, at the 60 s commit, with the pre-header
408 the PR fixes. On the head the headers commit at 60.0 s, keepalives hold the stream through the
prefill, both choices stream their first token at 186.0 s and finish, and the stream ends with
`[DONE]` at 191.1 s (131057 prompt tokens reported). No 408, no 429, no error event.

The first attempt (`spec-route/`) booted the GB10 recommended spec arm (masked MTP draft). There
both binaries answer `n: 2` at once with HTTP 400 ("n-choice shared prefill is unavailable for this
model/boot above n=1; selected spec, vision, parallel or unsupported state routes require n=1"),
the boot's documented capability, not a deadline outcome; that is why the check runs the plain
route.

Server lines per arm:

- `01-main` (memra-server sha256 `fb2649d9c42973d7`): `[deadline] unified memory (48 SMs, served ctx 262144): streaming TTFT default/max 997311 ms (derived), SSE prefill commit 60000 ms; non-streaming default/max 3045311 ms (derived: full-context prefill + 8192 open output t`
  - `[admission] "q38": plain 29696 B/token, spec 31552 B/token; fixed residual learns from effective-free deltas`
  - `[admission] request cost: model="q38" ctx=131161 path=plain = 29696 B/token x ctx + 4563MB prefill-workspace + 0MB fixed = 8458MB`
  - `[admission] fixed residual high-water: model="q38" 551MB (observed 4445MB at ctx 131105 path=plain)`
  - `[admission] request cost: model="q38" ctx=131161 path=plain = 29696 B/token x ctx + 4563MB prefill-workspace + 551MB fixed = 9009MB`
- `02-head` (memra-server sha256 `15267bcb3b529423`): `[deadline] unified memory (48 SMs, served ctx 262144): streaming TTFT default/max 997311 ms (derived), SSE prefill commit 60000 ms; non-streaming default/max 3045311 ms (derived: full-context prefill + 8192 open output t`
  - `[admission] "q38": plain 29696 B/token, spec 31552 B/token; fixed residual learns from effective-free deltas`
  - `[admission] request cost: model="q38" ctx=131161 path=plain = 29696 B/token x ctx + 4563MB prefill-workspace + 0MB fixed = 8458MB`
  - `[admission] fixed residual high-water: model="q38" 603MB (observed 4496MB at ctx 131105 path=plain)`
  - `[admission] request cost: model="q38" ctx=131161 path=plain = 29696 B/token x ctx + 4563MB prefill-workspace + 603MB fixed = 9061MB`
  - `[admission] fixed residual high-water: model="q38" 641MB (observed 4535MB at ctx 131105 path=plain)`

Files: per arm `env.txt`, `server.log`, `gpu.csv` (250 ms), `check.out`, `check/events.jsonl`,
`check/result.json`; `chain.log` is the run's own summary; `harness/` holds the client and runner.
The home directory reads `~`.
