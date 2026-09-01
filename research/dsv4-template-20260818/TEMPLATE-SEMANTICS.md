# encoding_dsv4 census — DeepSeek-V4-Flash template / think-modes arm (lane 5, 2026-08-18)

Semantic source (THE LAW): `encoding_dsv4.py` staged from the hyperscaler box, copied byte-identical
into this dir (`ref/encoding/encoding_dsv4.py` == `ref/artifact-encoding/encoding_dsv4.py`,
sha256 `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0`). Citations `E:N`
are line numbers in that file; `R:N` cites `ref/encoding/README.md`. The NVFP4 artifact's
`encoding/tests/test_{input,output}_{1..4}` are the AUTHORITATIVE rendered fixtures
(the ref repo ships inputs only; both are mirrored under `ref/`).

Token ids verified against the artifact `tokenizer.json` (`ref/tokenizer.json`, 6,367,146
bytes, same file at both roots).

## 1. Special tokens (E:17-35), ids verified

| token | id | special flag | note |
|---|---|---|---|
| `<｜begin▁of▁sentence｜>` BOS | 0 | true (Control) | emitted by the TEMPLATE, not the tokenizer (`add_bos_token: false`, tokenizer_config.json:2) |
| `<｜end▁of▁sentence｜>` EOS | 1 | true | also pad (tokenizer_config.json:23-30) |
| `<｜User｜>` | 128803 | false (UserDefined) | user/developer turn prefix |
| `<｜Assistant｜>` | 128804 | false | assistant turn prefix (transition token) |
| `<think>` | 128821 | false | open think block |
| `</think>` | 128822 | false | close think block |
| `｜DSML｜` | 128825 | false | DSML markup token (no angle brackets of its own) |
| `<｜latest_reminder｜>` | 128828 | false | reminder role prefix |
| `<｜action｜>` | 128829 | false | task token |
| `<｜query｜>` | 128830 | false | task token |
| `<｜authority｜>` | 128831 | false | task token |
| `<｜domain｜>` | 128832 | false | task token |
| `<｜title｜>` | 128836 | false | task token |
| `<｜extracted_url｜>` | 128844 | false | README-only (R:150); not minted by encode_messages |
| `<｜read_url｜>` | 128845 | false | task token |

`<tool_result>` / `</tool_result>` are NOT vocab tokens — plain text, BPE'd (verified:
absent from added_tokens and vocab). `｜` is U+FF5C (fullwidth vertical line, 3 UTF-8
bytes) in every special token.

`tokenizer_config.json`: `add_bos_token: false`, `add_eos_token: false`; the tokenizer.json
`post_processor` is plain ByteLevel — HF encode adds NO special tokens. **BOS is template
text** (E:546). Trap for any future GGUF mint: `tokenizer.ggml.add_bos_token` MUST be false
or the prompt double-BOSes (renderer emits the literal).

Pre-tokenizer (tokenizer.json `pre_tokenizer`): the DEEPSEEK3_LLM three-pass scheme —
`\p{N}{1,3}`, CJK/kana isolation `[一-龥぀-ゟ゠-ヿ]+`, then the
punct/letter/space alternation — byte-for-byte the pattern set memra already ports as
`split_deepseek_v3` (memra-tokenizer/src/unicode.rs:353-497). `from_hf_dir` previously
keyed `pre` only on the qwen regex → this lane adds detection of these Split patterns →
`pre = "deepseek-v3"` (integer-exactness receipt: cross-check gate below).

## 2. Roles (render_message, E:223-394)

| role | render | cite |
|---|---|---|
| `system` | `{content or ""}` bare; `+ "\n\n" + tools block` if msg.tools; `+ "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{json}"` if msg.response_format | E:265-270, E:49-51 |
| `developer` | `<｜User｜>{content}` + same tools/response_format appendix; content REQUIRED (assert) | E:272-283 |
| `user` | `<｜User｜>` + content, or content_blocks joined `"\n\n"` (text blocks raw; tool_result blocks `<tool_result>{content}</tool_result>`; list-form tool content: text parts joined `"\n\n"`, non-text parts `[Unsupported {type}]`) | E:285-311, E:60-62 |
| `latest_reminder` | `<｜latest_reminder｜>{content}` | E:313-314 |
| `tool` | raises NotImplementedError — MUST be pre-merged into user turns (`merge_tool_messages`) | E:316-317 |
| `assistant` | `{reasoning}{content}{tool_calls}` + EOS (or no EOS when `wo_eos`) | E:319-361, E:45-46 |
| anything else | raises NotImplementedError | E:362-363 |

No trimming anywhere — content is embedded verbatim (unlike qwen/gemma arms' `|trim`).

Multiple/late system messages: no position constraint in render_message; a system message
anywhere renders bare content (no `<｜User｜>` prefix, no separator token).

## 3. Transition tokens (E:365-394) — the "generation prompt" is implicit

After rendering message i, transition text is appended IFF i is the last message OR
messages[i+1].role ∈ {assistant, latest_reminder} (E:366; note the inverted guard).
Then:

- msg has `task` (E:369-382): non-action tasks append the task token directly
  (`<｜query｜>` etc.). `action` appends `<｜Assistant｜>` + think-token + `<｜action｜>`,
  where think-token is `<think>` iff thinking mode else `</think>` (E:381).
- else role ∈ {user, developer} (E:384-392): append `<｜Assistant｜>` +
  - `<think>` if thinking && !drop_thinking (E:387-388),
  - `<think>` if thinking && drop_thinking && i >= last_user_idx (E:389-390),
  - `</think>` otherwise (chat mode always; past user turns under drop_thinking) (E:392).
- else (system/assistant/latest_reminder without task): nothing.

Consequence: two consecutive assistant messages render `{a1}<EOS>{a2}<EOS>` with no
second `<｜Assistant｜>` — the prefix only ever comes from a preceding user/developer
transition (title-task continuation shape, R:146, R:156).

`last_user_idx` = index of the LAST message with role user or developer (E:209-216).

## 4. The three think modes (E:241, E:261-263, E:344-348, E:384-392)

`thinking_mode` is a REQUIRED argument, `"chat"` or `"thinking"` (assert E:241). The
knobs and their renders:

- **chat** (Non-think): transitions end `<｜Assistant｜></think>`; assistant bodies never
  render reasoning (thinking_part only set under `thinking_mode == "thinking"`, E:344).
- **thinking** (Think High): transitions end `<｜Assistant｜><think>` for the last user
  turn; assistant bodies at index > last_user_idx (or all, when drop_thinking is off)
  render `{reasoning_content}</think>` before content (E:344-348; strict `>` at E:345 for
  the body vs `>=` at E:389 for the transition).
- **thinking + reasoning_effort="max"** (Think Max): identical to thinking plus
  `REASONING_EFFORT_MAX` (E:64-68, ends `"\n\n"`) prepended at **index 0 of the full
  message list, BEFORE the role dispatch** (E:262-263) — i.e. before the system content;
  with no system message it lands before `<｜User｜>` of the first user turn (verified by
  running the oracle).
- `reasoning_effort` accepted set: {None, "high", "max"} (assert E:261).
  **"high" is a no-op — renders byte-identically to None** (its only use is the assert;
  grep: `reasoning_effort` appears at E:223/235/261-263/512/530/569 only).

> **PREVIEW-ENCODING LAW ONLY (0731 re-gate, 2026-08-18).** The 0731 checkpoint's
> encoding_dsv4.py REMAPS this ladder: accepted set {None→"low", "low", "high", "max"},
> where low = default/no prefix, **high = the OLD "max" text**, and **max = a NEW,
> stronger text**. Everything else in this census is byte-identical across the two
> encodings. Full diff + verdicts + detector (config.json dspark_* census → 
> `chat::Dsv4Encoding`): **ENCODING-DIFF.md**. The E: cites in this section are into
> `ref/encoding/`; the 0731 equivalents are E:67-80/274-278 of `ref-0731/encoding/`.

`drop_thinking` (default True, E:510):
- Auto-disabled when ANY message (context included) defines `tools` (E:549-551) — tool
  conversations keep all think blocks.
- When active (thinking mode only, E:553): `_drop_thinking_messages` (E:575-599) runs on
  the full list BEFORE rendering: messages with role ∈ {user, system, tool,
  latest_reminder, direct_search_results} or index >= last_user_idx are kept; earlier
  assistants lose `reasoning_content`; earlier developer (and any other) messages are
  DROPPED ENTIRELY (E:597 comment). Indices/last_user_idx are then recomputed on the
  filtered list.
- In chat mode no filtering happens at all (E:553 guard) — developer turns stay.

Assistant after a task message (`prev_has_task`, E:341-348): NO thinking part even in
thinking mode (task outputs are bare, fixture 4).

## 5. Tools — declaration, calls, results

### Declaration (E:70-95, E:189-206, E:255-256, E:267-268)
`msg.tools` is OpenAI format; `tools_from_openai_format` extracts each `tool["function"]`
(E:109-111). The block appended to system/developer content after `"\n\n"`:
`TOOLS_TEMPLATE` with `{tool_schemas}` = one `json.dumps(function_obj, ensure_ascii=False)`
per line joined `"\n"` (E:199-206) — **python dumps separators `", "` / `": "`, insertion
key order, non-ASCII raw**. TOOLS_TEMPLATE ends with a trailing `\n` (E:95 — the
closing `"""` sits after a newline).

### Assistant tool_calls (E:52-58, E:139-166, E:323-336)
```
\n\n<｜DSML｜tool_calls>\n
<｜DSML｜invoke name="NAME">\n
<｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>\n   (one per arg)
</｜DSML｜invoke>\n            (invokes joined by "\n")
</｜DSML｜tool_calls>
```
placed after content, before EOS (assistant_msg_template E:45: `{reasoning}{content}{tool_calls}` + EOS).
Params: `arguments` (a JSON string) is parsed; string values render raw with
`string="true"`, everything else `json.dumps(..., ensure_ascii=False)` with
`string="false"` (E:157-164). Unparseable arguments string → single param named
`arguments` carrying the raw string (E:152-155). Values are NOT escaped — a string value
containing `</｜DSML｜parameter>` would corrupt the wire (upstream accepts this).

### Tool results (E:60-62, E:296-306, E:401-457)
No `tool` role on the wire. `merge_tool_messages` (E:401-457) converts every tool message
into a `tool_result` content block on a user message, and EVERY user message into
content_blocks form — so **any consecutive run of user/tool messages fuses into ONE
`<｜User｜>` turn**, blocks joined `"\n\n"` (E:309), tool blocks wrapped
`<tool_result>{content}</tool_result>`. A user message with a `task` does not accept
merges after it (E:441). Only `task`/`wo_eos`/`mask` survive the user-message rewrite
(E:450-452) — see banked finding #4.

`sort_tool_results_by_call_order` (E:460-499): within a user turn holding >1 tool_result
blocks, the tool blocks are reordered by the preceding assistant's `tool_calls[].id`
order (`tool_use_id` ↔ id; unknown ids sort as 0, python sort is stable); non-tool block
positions are preserved.

### Model-emitted wire format + official parser — VERDICT: DEFINED
`parse_message_from_completion_text` (E:687-744) is the official completion parser:
1. thinking mode: reasoning = text until `</think>` (stop set includes
   `\n\n<｜DSML｜tool_calls` but hitting it is an assert failure, E:716-718 — a tool block
   inside the think segment is invalid).
2. content = text until EOS or `\n\n<｜DSML｜tool_calls` (the `\n\n` belongs to the
   SYNTAX, not content, E:710, E:720-721). **No separator newlines after `</think>`** —
   content begins immediately (unlike qwen's `</think>\n\n`).
3. tool calls: `parse_tool_calls` (E:630-684) — strict grammar: `>\n` after the block
   open and after each parameter close (E:648-649, E:678-679); invoke head
   `^\s*name="(.*?)">\n$` (E:659); parameter
   `^ name="(.*?)" string="(true|false)">(.*?)<$` DOTALL (E:668) — the `<` of
   `</｜DSML｜parameter>` terminates the value, values may span lines; duplicate parameter
   names are an error (E:673-674). Ends at `</｜DSML｜tool_calls>` then EOS immediately
   (E:730-731: no content after tool calls). Multiple invokes per block are legal.
4. `decode_dsml_to_arguments` (E:169-186): `string="true"` → value JSON-encoded;
   `string="false"` → value embedded as raw JSON text; arguments string built
   `{"k": v, ...}` with `", "` separators.
5. Tail law: text must end exactly at EOS **or** at end-of-string (stop_token None is
   accepted, E:733) — so stopping generation at `</｜DSML｜tool_calls>` (inclusive)
   parses cleanly. Special tokens inside content/reasoning are asserted absent
   (E:735-737).

**Stop conditions**: EOS (id 1) is the only stop the format needs — the template puts EOS
directly after `</｜DSML｜tool_calls>` (E:45). memra scopes an additional
`</｜DSML｜tool_calls>` stop string to dsv4 tool requests (the gemma `<tool_call|>`
pattern: prevents a run-on into hallucinated tool results; the stop stays in the stream
so the parser closes the span).

## 6. Tasks (quick-instruction heads) (E:28-36, E:369-382; R:139-156)

`task` field values: action | query | authority | domain | title | read_url (E:28-35).
- `action` on a user msg: `...{user}<｜Assistant｜>{<think>|</think>}<｜action｜>` (E:379-382).
- `query`/`authority`/`domain`/`read_url` on a user msg: token appended directly after
  the user content (E:375-377).
- `title` on an assistant msg: appended after that assistant's EOS (fixture: R:146).
- assistant AFTER a task msg renders with no think part (E:341-348).
Not reachable from the OpenAI serve surface (no `task` field) — implemented in the memra
renderer? NO: out of scope for the serve arm, banked as not-rendered (the arm has no
input that can set it); fixtures cover it only via the oracle-vs-oracle artifact test 4,
which the Rust arm reproduces through an explicit `task`-bearing Turn extension — see
IMPLEMENTATION notes below for what was actually wired.

## 7. encode_messages assembly (E:506-572)

- `prompt = BOS` iff `add_default_bos_token` (default True) and `context` empty (E:546).
- Preprocess: `merge_tool_messages(messages)`, then `sort_tool_results_by_call_order`
  over context+messages (E:538-542).
- `effective_drop_thinking = drop_thinking && !any(m.tools)` (E:549-551).
- thinking && drop: filter via `_drop_thinking_messages`, recompute counts (E:553-558).
- Render each message with the transition law of §3.
- `context` param = already-rendered prefix messages (not on the memra surface; ignored).

## 8. Banked ambiguities / findings (refuse-on-ambiguity receipts)

1. **No default thinking_mode exists.** `encode_messages` requires it (E:507, no default;
   assert E:241). README quick-start uses `"thinking"` (R:17). memra's
   `ThinkMode::Default` therefore has NO template-own default to inherit; this arm maps
   **Default → thinking** (the README's own example and the model's agentic positioning)
   and NoThink → chat, Think → thinking. One-line flip if the owner rules otherwise.
2. **"max" is unreachable from the OpenAI `reasoning_effort` surface.** The API ladder is
   none|minimal|low|medium|high (main.rs `parse_think`); dsv4's only distinct levels are
   {chat, thinking, thinking+max} and its own "high" is a no-op alias of None (E:261-263).
   Mapping OpenAI "high" by NAME to dsv4 "high" (= plain thinking) leaves Think Max
   unreachable; mapping "high" → "max" would inflate every high request with the fixed
   prompt paragraph. **Renderer supports all three (effort `Some("max")` renders the
   prefix); the serve-side choice of which OpenAI value (if any) reaches "max" is left to
   the serving lane / owner.**
   *SUPERSEDED IN PART by the 0731 re-gate (ENCODING-DIFF.md): the 0731 encoding gives
   "high" a REAL prompt prefix by NAME, dissolving the preview dilemma for that rung.
   Current wiring: dsv4 models ARE effort-consuming (`ModelCaps::dsv4` forwards the level;
   the renderer resolves it against the artifact's detected `Dsv4Encoding` — preview
   renders the documented no-op, never a corrupt prompt). The native "max" rung stays
   beyond the OpenAI ladder (parse_think unchanged; vLLM precedent: chat_template_kwargs)
   — still an owner/serving-lane call whether to expose it.*
3. **`direct_search_results` role is kept by `_drop_thinking_messages` (E:587) but
   `render_message` raises NotImplementedError for it (E:362-363)** — an upstream
   inconsistency; the role cannot render. Not implemented; a Turn with that role errors.
4. **Tools on user messages are silently destroyed upstream**: `merge_tool_messages`
   rewrites user messages preserving only task/wo_eos/mask (E:450-452), and the
   drop_thinking auto-disable check runs AFTER the merge (E:538 vs E:550) — so
   `tools` on a user message neither renders nor disables drop_thinking. The memra
   surface attaches request-level tools to the leading system turn (creating an
   empty-content system turn when the request has none — matching the oracle's render of
   `{"role":"system","content":"","tools":[...]}`), so this corner is unreachable.
5. **`num_to_render` recount bug-shape (E:557)**: `_drop_thinking_messages(context)`
   recomputes last_user_idx WITHIN the context slice, which can disagree with the
   full-list filter when the context ends in assistant turns. Unreachable for memra
   (context always empty); banked, not worked around.
6. **wo_eos / mask / task / context**: not on the OpenAI serve surface. `wo_eos`
   (E:253, E:350-355) renders an assistant turn without EOS (prefill/continuation);
   `mask` is training-only metadata (preserved by merge, never read by render). The Rust
   arm implements `wo_eos`-equivalent behavior ONLY via the fixture harness (no serve
   input reaches it); `task` rendering IS implemented (Turn.task) because fixture 4 needs
   it for byte parity, but no serve path sets it.
7. **Float-text fidelity**: tool schema/argument JSON re-render uses the client's own
   number text via `Val::Num` (serde_json preserve-order + Number::to_string). Python
   `json.dumps` and serde agree on integers and common floats but can differ on exotic
   floats (e.g. `1e30` → python `1e+30`). Same caveat the gemma arm banked; no fixture
   uses such a float; a real client replaying a dsv4-parsed call round-trips our own text.
8. **`<think>` in the transition vs body**: the `<think>`/`</think>` tokens around an
   assistant turn belong to the PRECEDING transition (E:384-392), while `</think>` after
   reasoning belongs to the assistant body (E:346). Chat-mode assistants render
   reasoning-less even if `reasoning_content` is present (E:344 guard) — reasoning
   supplied by a client in chat mode is dropped, matching the oracle.

## 9. memra implementation map (this lane)

- Renderer: `crates/memra-tokenizer/src/chat.rs` — `apply_dsv4_template` +
  `template_is_dsv4` marker law (`<｜Assistant｜>` AND `｜DSML｜` in the template string;
  the checked-in sentinel template is `research/dsv4-template-20260818/dsv4-chat-template.sentinel.jinja`,
  deliberately free of every other arm's markers: no `add_generation_prompt`,
  `enable_thinking`, `<tools>`, `<|turn>`, `hy_User`, `render_message_content`,
  `<|im_start|>`, `<|channel>`, `reasoning_effort is defined`).
  Dispatch precedes the step35/gemma/qwen marker checks (a faithful dsv4 template
  contains `<think>`, which the qwen tail detection would otherwise claim).
- Parser: `crates/memra-server/src/toolcall.rs` — `ToolStreamParser::dsv4(...)`:
  thinking-open prompts route text to `reasoning` until `</think>` (NO separator-newline
  swallow — dsv4 content starts immediately); `\n\n<｜DSML｜tool_calls>` opens a call
  span buffered to `</｜DSML｜tool_calls>` and parsed by a strict port of
  `parse_tool_calls` (multiple invokes → multiple OpenAI calls; malformed span surfaces
  VERBATIM per house policy — deviation from the oracle's raise, matching the gemma arm);
  non-ASCII tags use char-boundary-safe partial-suffix holdback.
- Caps/wiring: `worker.rs` `ModelCaps::dsv4` keyed on the same marker law (and excluded
  from `qwen_think`); `main.rs` arms the dsv4 parser + adds the scoped
  `</｜DSML｜tool_calls>` stop for dsv4 tool requests.
- Tokenizer: `from_hf_dir` detects the DEEPSEEK3_LLM pre-tokenizer pattern in
  tokenizer.json → `pre = "deepseek-v3"` (was falling back to the qwen35 split with a
  warning — silently wrong ids for this checkpoint).
