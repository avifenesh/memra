# Inkling 2 Mini: tokenizer, chat dialect and response parser (PR2)

Scope: the text surface of `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`.
CPU-only. No model, numeric program, runtime or serving binary is qualified by anything here, and
no support state moves.

The checkpoint's own files define the program: `tokenizer.json`, `chat_template.jinja` and the
`response_template` in `tokenizer_config.json`, copied unchanged under `artifact/` (Apache-2.0,
`artifact/NOTICE`; sha256 pinned in `gen_fixtures.py`). The libraries that read those files are
what the gates compare against. Other engines are checked against, not followed.

## What the artifact declares

| surface | declaration | engine side |
|---|---|---|
| vocabulary | byte-level BPE, 199,998 pieces, 60 added tokens 199998..200057, all special, no normalizer | `Tokenizer::from_hf_dir` |
| split | one `Split` regex, `Isolated`: the o200k_base pattern | `PreSplit::Gpt4o`, `unicode::split_gpt4o`, pre id `inkling2` |
| whole-word lookup | `model.ignore_merges: true` | read from `tokenizer.json`; by family for a GGUF |
| stop | `eos_token` = `<|content_model_end_sampling|>` = 200006, one id | `eos_id` / `eog_ids` |
| prompt | `chat_template.jinja` | `chat::apply_inkling2_template` (`chat::INKLING2_DIALECT`) |
| response | `response_template` | `ToolStreamParser::inkling2` in memra-server |

## The split and its id

memra had no o200k machinery: the four existing byte-level splits are the qwen35, qwen2,
deepseek-v3 and glm4 patterns. The regex is byte-equal to tiktoken's `o200k_base` `pat_str`.

It needs the cased letter categories (`Lu`, `Lt`, `Lm`, `Lo`, `Ll`), which the llama.cpp-derived
table does not carry. `gen_unicode_classes.py` reads all seven classes of the regex from
Oniguruma itself (the engine under the tokenizers `Split` step), one scalar at a time, and
checks them equal to Unicode 16.0.0. The generated table is
`crates/memra-tokenizer/src/unicode_gpt4o_data.rs`.

The regex names a family of tokenizers, not one. llama.cpp at 29b52d398b29 quotes it under
`LLAMA_VOCAB_PRE_TYPE_GPT4O` (`src/llama-vocab.cpp:455-459`) and maps several `pre` ids to that
type: `gpt-4o` with three other families (line 2665), and `inkling2` (lines 2671-2675), which
its converter writes for this tokenizer (`conversion/base.py:2472`). memra has gated one member
of the family, so it takes one id:

- a `tokenizer.json` with this regex loads as `inkling2` when its vocabulary has Inkling 2's own
  control tokens (`<|content_model_end_sampling|>` and `<|content_invoke_tool_json|>`), and is
  refused otherwise;
- a GGUF loads with `pre = inkling2`; `pre = gpt-4o` is refused like any unported family.

## The whole-word lookup

`ignore_merges` decides 337 vocabulary pieces (`kwargs`, `Aleksandr`, base64 fragments): the
merge table does not build them, and only the lookup returns their id. They are the corpus's
`unmerged-*` cases.

Is the lookup the family's own program, or a quirk of one file? `verify_tiktoken.py` builds a
tiktoken encoding from the vocabulary the artifact also ships in tiktoken's format and encodes
the whole corpus: tiktoken equals tokenizers with the flag as shipped on all 1,013 cases, and
differs from tokenizers with the flag off on exactly the 337 (`verify-tiktoken.txt`). tiktoken
looks a whole piece up before it merges; that is the same law.

A GGUF has no key for the flag. Each split therefore carries its family's declared value
(`PreSplit::gguf_ignore_merges`: true for glm4 and inkling2, false for qwen35, qwen2 and
deepseek-v3), so a GGUF load gives the checkpoint's ids. llama.cpp leaves the lookup off for
`inkling2`; a GGUF tokenized there differs from the checkpoint on those 337 words. The same
flag was wrong for GLM-5.3-Flash on main, on both load paths: that fix and its gate are in
`research/glm53-flash-bringup-20260827/parity-evidence/ignore-merges-20261121/`.

One more finding from the red arms: the split is almost implied by the merge table. Run through
the glm4 split, all but 7 of the first revision's 992 corpus cases still produce the reference
ids; the 7 are the combining-mark and uncased-script cases. The vocabulary was trained behind
this split, so the merge table rarely joins across one of its boundaries.

## The prompt

`chat_template.jinja`, restated:

- Tools first: `<|message_system|>tool_declare<|content_xml|>` + the tools as sorted compact JSON
  (`description`, `name`, `parameters`, `type` only, defaults filled) + `<|end_message|>`.
- `<|message_system|><|content_text|>Thinking effort level: N<|end_message|>` is always written,
  once: before the first message that is not a system message, or last when there is none.
  Default 0.9. Names: none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99.
  A number in [0, 0.99] is written as Python prints it. Anything else is refused.
- A message is `{role marker}{name}{content marker}{body}<|end_message|>`. An assistant turn
  writes its reasoning (`<|content_thinking|>`), its text, then one
  `<|message_model|>{name}<|content_invoke_tool_json|>{"name":..,"args":{..}}` message a call
  (arguments sorted at the top level only), and closes with `<|content_model_end_sampling|>`.
- A message whose content is null or missing writes no text message; `""` writes an empty one.
  `Turn::no_content` carries the difference, and the HTTP layer sets it.
- JSON numbers are re-serialized by Python: `1e-7` in a request is `1e-07` in the prompt, an
  integer keeps every digit. `py_json_number` does the same from the request's spelling. An
  integer past u64 is still lost before it gets there: the HTTP layer reads the request with
  `serde_json`, which turns it into a float. That is the serve step's to fix.
- The template never reads `add_generation_prompt`. The model opens its own message.

Engine choices on top of it:

- Thinking off (`ThinkMode::NoThink`) writes level 0, the template's `none`.
- A numeric effort has its own entry (`inkling2_effort_number` + `apply_inkling2_template`); the
  shared entry takes the OpenAI string, and the template refuses a numeric string.
- Image parts are refused (text-only until a vision lane). The template would write
  `<|content_image|><|unused_200054|>`.

## The response

The model writes messages; the parser reads the content marker of each. Thinking bodies are
`reasoning`, text bodies (and the tool-error and tool-text kinds) are content, a
`<|content_invoke_tool_json|>` body is one call. Text outside a message is dropped. A body ends at
`<|end_message|>` or at the end of the turn; inside a body every other marker is body text. With
the request's tools, a string argument is cast to its parameter's declared types.

The markers are control tokens, and the parser is a machine over marker events with two front
ends (`crates/memra-server/src/toolcall/inkling2.rs`):

- `push_token(markers, id, text)`, keyed by token id. A marker is a marker only when its own id
  arrives; text is never searched. A body that spells `<|end_message|>`, a header and a whole
  call in ordinary tokens is content, and no call. This is the serve path's entry.
- `push(text)`, which finds markers by spelling. transformers' `parse_response` reads text, so
  its fixtures gate this one. It has the vendor's blind spot and is not for serving.

Arguments keep the model's numbers. The call body is read by a JSON reader that holds a number
as its literal: an integer of any size stays exact, `1E400` stays `1E400`.

Departures from transformers' `parse_response`. Each is a fixture in `fixtures/parses.json` with
the vendor's own result beside it, and each is a choice:

| case | vendor | memra | why |
|---|---|---|---|
| call body is not JSON, or lacks `name` or `args`, or is empty | raises | the span, verbatim, as content | a finished generation is never failed on its content |
| `args` is not an object, or `name` not a string | passes it through | the span, verbatim, as content | the OpenAI wire carries a string name and an object |
| unterminated call whose JSON is incomplete, or a turn that ends at the call marker | raises | the span, verbatim, as content | as above |
| `NaN`, `Infinity` in the body | reads floats | the span, verbatim, as content | they are not JSON, and the arguments must be |
| `1E400` | reads `inf` | `1E400`, as written | a JSON number literal; `inf` cannot be sent |
| nesting past 128 levels | parses | the span, verbatim, as content | the reader recurses over model output |
| a header longer than 4,096 bytes on a malformed call | (raises) | surfaced without the header | bound on what is held while waiting for a marker |
| string `"1_000"`, `"１２３"` for an integer or number parameter | casts (Python numerals) | stays a string | JSON Schema has no such numbers |
| digit string `"12345678901234567890"` for a number parameter | `12345678901234567168` (through a float) | `12345678901234567890` | the digits are the value |

Matched to the vendor in the review round, where the first revision had left it without saying
so: integers at or past 9e15 cast exactly; `nullable` is read by truthiness (`0`, `""`, `[]` are
false); the string items of a list argument are cast one by one; a repeated key keeps its
first place and last value.

A lone surrogate escape (`"\ud800"`) in a call body is refused by the reader, where Python
accepts it. There is no fixture: the vendor's result cannot be written to a UTF-8 file. The
reader's own test covers the refusal.

Not done here: the serve route (`ModelCaps`, the parser's wiring through `push_token`, the worker
handing it ids) belongs to the serve PR. `tail_ends_with_tool_call` cannot tell an Inkling 2
call from any other message by its tail.

## Gates

| gate | oracle | size | command |
|---|---|---|---|
| render bytes | transformers 5.23.0 `render_jinja_template`, jinja2 3.1.6 | 70 cases: 56 byte-equal, 10 refusals with the template's message, 1 image refusal, 3 the typed request surface cannot carry | `cargo test -p memra-tokenizer --test inkling2_chat` |
| render ids | tokenizers 0.23.3 | 57 renders | same |
| corpus ids, both modes, `tokenizer.json` path | tokenizers 0.23.3 | 1,013 cases | same |
| corpus ids, both modes, GGUF path | tokenizers 0.23.3 | 1,013 cases | `cargo test -p memra-tokenizer --test whole_word_lookup` |
| whole-word lookup, both paths, both values | by construction (six pieces) | 1 | same |
| every scalar value | tokenizers 0.23.3, SHA-256 of the ids of each 4096-scalar chunk | 272 chunks, 1,112,063 scalars | `cargo test --release -p memra-tokenizer --test inkling2_chat -- --ignored` |
| the lookup is the family's | tiktoken 0.12.0 over `tiktoken/tokenizer.model` | 1,013 cases | `verify_tiktoken.py` |
| parse | transformers 5.23.0 `parse_response` | 46 cases, each five ways: text whole, text one char at a time (twice), ids with whole and one-char text tokens | `cargo test -p memra-server --lib toolcall::inkling2` |
| spelled markers stay text | by construction | 1 | same |

The red arm of each fix is in `receipts/red-arms.txt`.

Regenerate: `gen_fixtures.py` (its docstring has the command). It rewrites `fixtures/`,
`corpus.tsv` and `ref-ids.tsv` and must leave them unchanged.
