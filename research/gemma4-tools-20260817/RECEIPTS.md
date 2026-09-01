# gemma-4 tools — build receipts (lane/gemma4-tools, 2026-08-18)

Branch `lane/gemma4-tools` (worktree `wt-gemma-tools`), based on `main` @ v0.90.0.

The served gemma-4-31b-it REJECTED tool-carrying requests ("chat template has no tools
branch") — measured with real Codex CLI 0.147.0. Owner directive: gemma is agentic; all models
serve tools identically on the OpenAI-standard API. This lane makes the gemma4 tooluse dialect a
first-class tools arm, byte-parity-pinned to the official Google tooluse jinja.

## What was built

- **Renderer** (`crates/memra-tokenizer/src/chat.rs`): a gemma4 tooluse arm in
  `apply_chat_template_tools_ex`, engaging when the template carries `<|turn>` AND `<|tool>`.
  A faithful port of `official-tooluse-template.jinja` (extracted byte-identical from the
  official Q8_0-MTP GGUF — confirmed `== the served trunk's embedded template`):
  - tool DEFINITIONS in the first system turn, compact non-JSON dialect (`declaration:NAME{...}`),
    `dictsort` key order everywhere (case-insensitive/stable), types UPPERCASED, strings wrapped
    in the `<|"|>` special, and the full `format_parameters` macro (enum / array-items /
    nested-object properties+required / nullable);
  - assistant tool CALLS (`<|tool_call>call:NAME{k:v,...}<tool_call|>`), args dictsorted and
    rendered by `format_argument` (bare keys, `<|"|>`-strings, `true`/`false`, bare numbers,
    `None` for null, recursive maps/sequences);
  - tool RESPONSES: OpenAI role:"tool" forward-scan (name resolved via `tool_call_id` ->
    assistant `tool_calls[].id`) rendering `{value:...}`, plus the Google-native `tool_responses`
    mapping/non-mapping path;
  - `strip_thinking` on model content, reasoning re-render (`<|channel>thought`) for a
    tool_calls-carrying assistant after the last user turn, turn continuation (suppressed
    duplicate `<|turn>model`), the dangling open `<|tool_response>`, and the add-generation-prompt
    suppression when the last rendered thing was a call/response.
  - Typed values reach the serde-free tokenizer crate via a new `chat::Val` tree built
    server-side (`json_to_val`).
- **Output parser** (`crates/memra-server/src/toolcall.rs`): a `gemma_tools` mode splitting
  `<|channel>thought…<channel|>` -> `reasoning` and `<|tool_call>call:NAME{…}<tool_call|>` ->
  OpenAI `tool_calls` via a recursive-descent dialect parser (quote-aware `<|"|>` strings that
  may contain braces/commas/colons/newlines; bare tokens -> number/true/false/None else string;
  nested `{}`/`[]`). Spans never leak into content; malformed/unterminated surface verbatim.
- **Stop discipline**: gemma tool requests add `<tool_call|>` to the request stop set (scoped,
  never global) so the model stops at its handoff instead of hallucinating a `<|tool_response>`.
  The stop token stays in the stream (not a silent eos) so the parser closes the span.
- **Capability truth**: `caps.tools_branch` now keys on the shared `template_has_tools_branch`
  (`<tools>` OR `<|turn>`+`<|tool>`, never hy3), so `/v1/models` reports `capabilities.tools:true`
  for the gemma trunk and all three surfaces (chat/completions, /v1/messages, /v1/responses)
  funnel tools through the same path. Non-function tools on /v1/responses are still dropped with
  a log; function tools flow (confirmed live below).

## Fixture coverage (byte-parity oracle)

`gen_fixtures.py` renders the official + QAT jinja under jinja2 (the oracle) for 20 cases,
committed as `fixtures/NN-*/{input.json,expected.txt}`. The Rust test
`gemma4_tools_fixtures_match_the_official_jinja` renders each through the memra arm and asserts
byte equality; `gemma4_tools_flow_through_build_chat_request` proves the REAL serve pipeline
(build_chat_request) renders three OpenAI-reachable cases identically.

Cases: system+tools · no-system+tools · nested-object/array/enum/required/nullable declaration ·
single call cycle · parallel calls · dangling call · multi-cycle agentic history · native mapping
response · native non-mapping responses · content-parts tool result · thinking-on+tools ·
reasoning re-render + history strip · assistant continuation · nested-args call · history-only
(tool_choice none) · content+calls · arg value shapes (string/float/bool/null/multiline) ·
QAT closed-tail · QAT dangling-no-tail · dictsort of arguments.

18 official + 2 QAT fixtures; 20 total (>= 14 required).

## Gate transcript (tools/serve-gemma4-tools-gate.sh, RTX 5090 Laptop 24GB, flock-serialized)

Boot caps (server log):

    [worker] g4: template caps tools=true think=false think_switch=true chat_ok=true
             effort_levels=false gemma_think=true ctx=262144 tok="gemma4" instruct=None
    [responses] dropped non-function tools (model will not see them): namespace:multi_agent_v1, web_search

Verdict tail:

    ok: /v1/models advertises capabilities.tools=true
    ok: boot output-sample non-degenerate (27 distinct words)
    == phase 1: chat/completions tool call ==
    ok: phase 1: finish_reason=tool_calls, well-formed get_weather args (id=call_e874e31718a7db3e)
    == phase 2: tool result -> final answer ==
    ok: phase 2: coherent final answer citing the tool result
    == phase 3: streamed tool_calls deltas ==
    ok: phase 3: stream carried tool_calls deltas + finish_reason + [DONE]
    == phase 4: codex exec (real client, /v1/responses) ==
    note: 31B OOM'd on codex's ~8k prefill (24GB card) — retrying codex on the smaller
          tooluse trunk (identical gemma4 tools arm)
    ok: phase 4: codex round-trip on the tooluse fallback trunk (31B OOMs codex on 24GB)
    gemma4-tools-gate: ALL GREEN

Phase 1 tool call (actual, on the 31B):

    finish_reason: tool_calls
    tool_calls: [{"id":"call_e874e31718a7db3e","type":"function",
                  "function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}]

Phase 2 final answer (31B): `The weather in Paris is currently 21°C with clear skies.`

Phase 3 stream deltas (31B): header `{"index":0,"id":...,"function":{"name":"get_weather","arguments":""}}`
then `{"index":0,"function":{"arguments":"{\"location\":\"Oslo\"}"}}`, `finish_reason:"tool_calls"`, `[DONE]`.

Phase 4 REAL codex 0.147.0 over /v1/responses (12B tooluse trunk — genuine round-trip):

    exec /usr/bin/zsh -lc 'echo GEMMA-TOOLS-GATE —'  succeeded in 0ms:
    GEMMA-TOOLS-GATE —
    codex: The command printed: `GEMMA-TOOLS-GATE —`   (tokens used 15,696)

    (function_call item -> codex ran the shell tool -> function_call_output round-tripped ->
     final assistant message reported the marker. The real client, the real /v1/responses
     surface, the gemma tools arm end to end.)

## Battery verdict

`cargo test` — memra-tokenizer lib 28 passed / 0 failed; memra-server bin 303 passed / 0 failed
(includes the fixture-parity oracle, the pipeline render test, and the gemma parser tests).

`tools/local-ci.sh` (correctness) — ALL GREEN, 0 failures:

    kernel-check: ALL GREEN (106 cells, 1 skipped)
    prime-gate: ALL GREEN
    run-spec K=1..8 self-consistency: PASS (Qwen 35B, 8/8)
    argmax-margin-gate: PASS (31B, calibrated)
    VERIFY-GATE K=7 depth: PASS (31B) · spec self-consistency 64/64: PASS (31B)
    run-gen argmax depth: MATCH (12B) · VERIFY-GATE K=7 depth: PASS (12B)
    decode-batch-gate config/strict B=8/B=4: ALL GREEN (9B NVFP4 + 9B Q8_0)
    graph-warmup-stress: ALL GREEN (10 cycles x 4 arms + overlap + canary)
    correctness stage: GREEN
    serve-smoke: 0 failed  [incl. "gemma4 arm: clean content, stop at turn end",
                 "gemma4 thinking-on: thought/content separated, tags stay syntax",
                 "gemma4 arm: zero panics in the server log"]
    serve-stress-gate: ALL GREEN (c=64)
    accept-gate: 1 pass, 0 fail

The lane touches only chat-template rendering + server-side tool parsing/plumbing — zero
engine/kernel/decode-arithmetic changes — so the exactness/spec/decode gates above are unaffected
by construction and confirm no regression.

## Deviations from the jinja (and why)

1. **Unresolved tool-response name -> "unknown"** instead of the jinja's `str + None` crash
   (`.get('name') | default('unknown')` renders None; concatenation then raises). Unreachable
   from OpenAI histories, where the `tool_call_id` always resolves the name.
2. **QAT-vs-official closed-tail split**: the official served trunk emits a bare `<|turn>model\n`
   on the thinking-off generation prompt; the local QAT q4_0 gate trunk emits the closed thought
   channel. Keyed on the exact gen-prompt literal (present only in the QAT template). Each side
   matches its own template's bytes.
3. **Content-parts on user/system messages**: the serve pipeline flattens parts to one string and
   whole-string trims before dialect dispatch, where the jinja trims each part (and joins system
   parts with ' '). Equal when parts carry no whitespace edges; tool-result parts (raw concat)
   are byte-exact. Not fixture-covered as a divergence (documented in gen_fixtures.py).
4. **developer -> system**: the pipeline normalizes the OpenAI `developer` role to `system` for
   every dialect (OpenAI's own equivalence); the jinja would render a literal `<|turn>developer`.
5. **One call per turn on generation**: per the owner directive, generation stops when
   `<tool_call|>` completes. The renderer AND parser both handle parallel calls for history
   round-trips, but a single generation emits one call before the stop fires.
6. **Number text**: relies on serde_json `Number::to_string()` matching jinja's `{{ number }}`
   (Python `str()`) — verified across the fixtures (`21.5`, `1500.5`, `30`, `5000`, `12`).

## Risks / left undone

- The 31B monolithic gemma4 prefill OOMs codex's ~8k-token agentic prompt on a 24GB card;
  codex acceptance runs on the 12B tooluse trunk (identical arm). A card with more headroom (or
  chunked serve-prefill for gemma4, out of this lane's scope) would run codex on the 31B directly.
  The 31B DID serve the curl tool phases (short prompts) green.
- Perf stage not run: the lane changes no decode arithmetic, so the perf drift-tripwire would
  only measure unrelated engine timing.
