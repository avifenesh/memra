# GLM-5.3-Flash standard-surface receipts

Raw request/response pairs. **Build fingerprints are REDACTED** (`memra-<hex>` ->
`memra-<redacted-build-fingerprint>`): they are build identity, incidental to what these
prove, and `live_fingerprint` is a sev1 pattern in the repo's pre-push boundary scanner.
Redacted rather than allowlisted, so no standing exemption is created on a path whose
contents change on every regeneration.

## `before-*` — the audit, against the pre-fix binary on the bench box (2026-08-28)

Endpoint `http://127.0.0.1:18400`, model `zai/glm-5.3-flash`. Every one of these is a 200 (or
the correct 400) served on a **ChatML-rendered prompt** — the defect the lane closed. See
`../BRINGUP.md` "Standard-surface audit and fix" for the table.

| file | what it shows |
|---|---|
| `before-chat-plain` | `/v1/chat/completions`, no tools |
| `before-chat-tools-call` | tool definition -> `finish_reason: "tool_calls"`. It parsed only because the model obeyed the QWEN `<function=…>` instruction it was handed |
| `before-chat-tools-final` | tool result fed back -> final answer (rendered as a qwen `<tool_response>` user turn, not `<\|observation\|>`) |
| `before-messages-tools` | `/v1/messages` (Anthropic) tool_use, same core |
| `before-responses-effort-low` | `/v1/responses` with `reasoning.effort:"low"`. **`cached_tokens: 288` of 288 input tokens** — a full prefix-cache hit against the identical no-effort request, which is the proof that the effort level rendered ZERO bytes. Response body truncated in the receipt: the sampled answer looped and never called the tool |
| `before-effort-none-refused` | the off-request this template genuinely cannot honour -> named 400. Correct, and kept |
| `before-effort-invalid-refused` | out-of-table level -> named 400 |
| `before-response-format-refused` | `response_format` -> named 400 ... |
| `before-models` | ... while the model row claimed `structured_output: true`. Fixed |

**Provenance, stated rather than assumed.** The first four pairs (`before-chat-plain`,
`before-chat-tools-call`, `before-chat-tools-final`, `before-messages-tools`) are raw curl
output pulled off the box. The remaining five (`before-responses-effort-low`,
`before-effort-none-refused`, `before-effort-invalid-refused`,
`before-response-format-refused`, `before-models`) are **transcribed from the same session's
probe output**: the banking script was still running when the decode-perf lane restarted the
server for its next timing cell, so those curls came back empty and the bodies were written
back from what the earlier probes had already printed. The four refusal bodies are
deterministic server strings and the `before-models` body is the verbatim `/v1/models` payload;
`before-responses-effort-low`'s `output_text` is explicitly marked truncated. Re-runnable
against any pre-fix binary; not re-run, because the box has not been free since.

## `roundtrip-*` — the live agentic round-trip (2026-08-28, bench box, banked)

Two `/v1/completions` calls carrying the VERBATIM bytes of
`../surface-fixtures/{21-roundtrip-turn1-ask,22-roundtrip-turn2-after-result}/expected.txt`.
Those bytes are what the shipped Rust arm renders — asserted, not assumed, by
`glm5_fixtures_match_the_vendor_jinja`, which replays every fixture's `input.json` through the
REAL request pipeline and compares byte-for-byte. So this exercises the exact prompt a tools
request produces on any of the three wire formats, against the live model, without needing the
fixed binary deployed first.

| turn | prompt | the model's answer |
|---|---|---|
| `roundtrip-turn1-call` | tool declared, user asks for Paris weather | `</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`, `finish_reason: "stop"`, 12 completion tokens |
| `roundtrip-turn2-final` | the tool result rendered into an `<\|observation\|>` block | reasoning, then **"The weather in Paris is 21°C and sunny"** — it read the result — `finish_reason: "stop"`, `cached_tokens: 176` (a prefix hit on turn 1's prompt, so turn 2 genuinely extends it) |

Two things this settles that no unit test can. First, the model emits the **GLM** call wire
(`<tool_call>NAME<arg_key>…`), not the qwen `<function=…><parameter=…>` wire the ChatML
fallback used to instruct it into — the old render was a different program wearing the same
200. Second, `finish_reason: "stop"` on both means generation ended on the declared
`<|observation|>` / `<|user|>` eos ids rather than running past them.

Both emissions are pinned VERBATIM through the shipped parser by
`toolcall::tests::glm5_live_emissions_parse_into_the_serve_surface`, which closes the chain
fixture bytes -> live model -> OpenAI-shape `tool_calls` + split reasoning.

**Still owed:** the post-deploy battery on the FIXED binary — boot `glm5=true` caps line, full
chat/`/v1/messages`/`/v1/responses` tool cycles through the server's own rendering, and a
vendor-default sampled probe with a spec-engagement receipt. This round-trip proves the dialect
against the model; it does not exercise the server-side wiring on the deployed build.
`run-roundtrip.sh` re-runs the pair.

## `run-roundtrip.sh` — the driver

Ran once. The bench box has been continuously running the decode-perf lane's chained
timing cells, and any request lands inside a measurement (interleaved-A/B law). The driver is
banked ready to run; it POSTs the fixture-pinned native prompt bytes
(`../surface-fixtures/21-roundtrip-turn1-ask`, `22-roundtrip-turn2-after-result`) to
`/v1/completions`, so it exercises the exact prompt a tools request produces on any of the
three wire formats without needing the fixed binary deployed first. Read its header for what
it does and does not prove.
