# DeepSeek-V4.1-Flash onboarding, step 2: the `encoding_dsv41` chat dialect (2026-11-09)

Verdict: memra renders DeepSeek-V4.1's chat dialect byte for byte as the vendor's
`encoding/encoding.py` does on all 166 render cases, and refuses the 17 inputs the vendor refuses.
Its tokenizer gives the artifact tokenizer's ids for every render (24,521 ids). The serve path's
parser reproduces the vendor's parse of the 11 completions the vendor accepts, and keeps the house
malformed policy on the 4 it refuses. The `deepseek_v41` pack's `encoding_dsv41` now resolves to
this renderer, and the TokenizerTemplate gate exercises it on the artifact's own files. DSv4 0731
renders, receipts and its artifact lock are unchanged. No GPU was used.

## Oracle

`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`. The artifact
ships its dialect as code and no template. `ref/` holds the pinned bytes, listed in
`ref/SHA256SUMS`; `gen_fixtures.py` refuses to run if any of them drift.

| file | sha256 (prefix) | role |
|---|---|---|
| `ref/encoding/encoding.py` | `13231c59` | the oracle |
| `ref/encoding/test_encoding.py`, `ref/encoding/tests/*` | | the vendor's tests and its five golden pairs |
| `ref/inference/examples/*` | | the vendor's example cases |
| `ref/tokenizer.json` | `c90dfa01` | ids (adds `<｜System｜>` 128799 and `<｜deepseek_image｜>` 129264 to the DSv4 vocab) |
| `ref/tokenizer_config.json`, `ref/config.json` | `6ac8c8dc`, `8be45ce0` | `config.json` is the file step 1's lock hashed |
| `ref/LICENSE` | `f2c6c602` | MIT |

The vendor's own suite passes on these files: `vendor-pytest.log`, 55 passed.

## How V4.1 differs from encoding_dsv4

It is its own dialect, not a third DSv4 ladder (`chat::Dsv4Encoding::V41`, census key
`model_type` `deepseek_v41` / `deepseek_v41_text`):

- DSML tags carry a leading space: `<｜DSML｜ calls>`, ` invoke`, ` parameter`.
- Reasoning effort is a numeric budget, 1 to 100 (`low` 50, `high` 75, `max` 100, default
  `high`), rendered once at index 0 in thinking mode behind `<｜System｜>`. A leading system
  message takes the same token.
- A system message may appear mid-conversation. It counts as a user turn for the generation
  header and the drop-thinking boundary.
- A user turn holding only tool results does not start a new turn, so the current turn's
  reasoning survives tool results.
- Image parts render as `<｜deepseek_image｜>`; a message's parts join with `\n\n`. The
  placeholder may not appear in any text.
- Tool names may carry a namespace (`ns::name`); a namespace object's description is prepended
  to the tool's.
- There is no `developer` role.

## What the fixtures cover

`fixtures/<case>/input.json` + `expected.txt`, `fixtures/tokenization.json`, `parse/<case>.json`.
`gen_fixtures.py` documents the input fields.

| source | renders | refusals |
|---|---:|---:|
| every case of `test_encoding.py` (`vt-*`) | 31 | 11 |
| the vendor's `encoding/tests` pairs and `inference/examples` (`vendor-*`) | 17 | 0 |
| the generated matrix (`m-*`) | 118 | 6 |

Branches reached:

- effort: default, `low`, `high`, `max`, 1, 42, 63, 88, 100; chat mode with an effort; -1, 0
  and 101 refused, in chat mode too;
- the System token: thinking at index 0 over a user, system, assistant or latest_reminder
  message, a leading system in chat, none on a plain chat prompt;
- mid-conversation system: last, followed by a user, carrying tools, after tool results;
- tools: one, two, request tools with no system message (a synthetic empty system message, the
  serve law the DSv4 arm uses), namespaces as a string, an object with and without a
  description, a tool with no description of its own, a null namespace, tool level over function
  level, an already qualified name; nested and conflicting namespaces refused;
- `response_format` with and without tools;
- tool calls: typed arguments (int, bool, float, null, list, object, escapes, non-ASCII), empty
  arguments, double-encoded arguments, non-object arguments (a string and an array), two calls,
  namespaced by call, by function and by name;
- tool results: sorted by call id, an unknown id, merged into a user turn, followed by a user
  turn, an image-only result;
- drop-thinking: multi-turn, the current turn kept across tool results, disabled by tools,
  disabled by tools on a user turn;
- tasks: action, query, authority, domain and read_url in both modes, title, a task answered,
  a task followed by a user turn, a task user turn that blocks a merge; an unknown task refused;
- images: interleaved with text, image only, the internal `image` block, in system and assistant
  messages, multi-turn, inside tool results re-sorted by call order (the vendor test's four input
  shapes, proven equal by the generator); the placeholder refused in user, tool, text-part,
  assistant and reasoning text;
- `wo_eos` in both modes, latest_reminder (leading, after a user turn, first), an empty
  conversation, an empty user message, consecutive user messages, `developer` refused.

Four gates read the fixtures (`crates/memra-server/src/dsv41_encoding_tests.rs`):

1. every case through `chat::apply_dsv41_template_media` (bytes and image order) and through the
   template-less dispatch the tokenizer uses; thinking cases also through `ThinkMode::Default`;
   every refusal refused;
2. every render through the `Tokenizer` loaded from `ref/` (census, no template), against the ids
   transformers' `AutoTokenizer.encode` gives (the call the vendor's `inference/generate.py`
   makes; the generator cross-checks the `tokenizers` library), and every dialect marker as one id;
3. the 67 cases the OpenAI surface can express, through `build_chat_request`;
4. the 15 completions through `ToolStreamParser::dsv41`, whole and one char at a time.

The test also checks the five vendor golden files are fixtures verbatim, an agent turn re-renders
to exactly what the model generated (the parked-prefix law the DSv4 arm pins), and the V4 and
V4.1 tag sets never parse each other.

## Results

| command | result |
|---|---|
| `cargo test -p memra-server --lib dsv41` | 9 passed |
| `cargo test -p memra-server --lib` | 1111 passed, 0 failed, 10 ignored; every `dsv4_*` render, wire, tokenization and parser gate included |
| `cargo test -p memra-tokenizer` | 85 lib + 3 parity passed |
| `cargo test -p memra-cli --lib` | 30 passed |

Red arms, each run once and reverted: removing the tool-result-only skip from the last-user rule
(the V4 law) fails gates 1, 2 and 3 on `vt-drop-thinking-without-tools`; removing the parser's
nested-namespace refusal fails gate 4 on `p-rejects-nested-namespace`.

`memra model inspect` against the pinned revision (`inspect-v41.log`) writes every receipt of
step 1 byte-identical except `../inspect/tokenizer-contract.tsv`, which now records the exercised
renderer: `renderer_encoding V41`, a 380-token probe. Against
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` --against
`deepseek_v4_dspark` the lock, census, plan, gates and refusal are byte-identical to step 1
(`inspect-dsv4-0731/artifact-lock-sha256.txt`) and the gate exercises `encoding_dsv4` under the
`V0731` census (`inspect-dsv4-0731/tokenizer-contract.tsv`).

## Where memra departs from the vendor

- A null tool or latest_reminder content: Python formats it as the string `None`; memra carries
  content as a string and renders nothing.
- Tool-call arguments reach the renderer decoded. The vendor also decodes double-encoded strings
  and wraps non-object arguments as `{"arguments": ...}`; the fixture harness applies that
  tolerance, and the serve surface keeps refusing non-object arguments with a 400, as it does on
  every dialect.
- A level name the vendor does not define (`medium`) renders the default budget, 75, where the
  vendor asserts. The DSv4 arm renders its default for the same request.
- An unknown role or task is an `Err` where the vendor raises.
- The OpenAI wire has no namespace field. A call named `ns::name` renders as the vendor renders
  it, and the parser returns `ns::name` as the call name where the vendor returns the name and the
  namespace apart; the renderer accepts that spelling back, so the round trip holds. A
  function-level `namespace` on a request tool renders; a tool-level one is not read.
- The renderer emits image placeholders and their prompt order; the serve surface refuses image
  parts on V4.1 by name until a V4.1 vision program exists.
- Malformed model output: the vendor raises; memra surfaces the block verbatim as content (the
  `\n\n` before a DSML open stays stripped, as on the DSv4 arm) and a never-closed think segment
  is reasoning. Parsed arguments are compact JSON.
- Floats in tool arguments go through serde_json's formatting (`1.5e-7` where Python writes
  `1.5e-07`), as on the DSv4 arm. No fixture carries one.
- The vendor's `context=` argument renders a suffix after an encoded prefix; memra renders whole
  conversations, and the generator proves the vendor's context variant equal.

## Regenerate

From the repository root, CPU only, about two seconds:

```bash
uv run --with 'transformers==5.22.2' --with 'tokenizers==0.23.3' \
    python3 -I -B research/modelplan-onboarding-dsv41-20261109/encoding/gen_fixtures.py
cargo test -p memra-server --lib dsv41
```

`generated.json` records the counts and versions (Python 3.14.4). Two regenerations give
byte-identical fixtures; the one vendor message that lists a Python set is normalized.
