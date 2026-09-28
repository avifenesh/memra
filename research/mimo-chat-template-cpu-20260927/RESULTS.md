# MiMo V2.6 Flash RL text chat template CPU parity

Verdict: Memra renders the seven bounded text-only chats in `goldens.tsv` with
the same UTF-8 bytes and token IDs as `transformers==4.57.1`
`AutoTokenizer.apply_chat_template` over the pinned source files. This extends
the raw-tokenizer receipt in `research/mimo-tokenizer-cpu-20260927/RESULTS.md`.

## Source

- Hub artifact: `XiaomiMiMo/MiMo-V2.6-Flash-RL` at
  `3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
- Downloaded with `hf download` at that revision, selecting only
  `tokenizer.json` and `tokenizer_config.json`.
- `tokenizer.json` SHA-256:
  `ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e`.
- `tokenizer_config.json` SHA-256:
  `413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06`.
- The exact `chat_template` string extracted from `tokenizer_config.json` is
  `source-template.jinja` (UTF-8 SHA-256:
  `11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059`).
  Memra's MiMo arm dispatches only on this exact template.
- Reference environment: `transformers==4.57.1`,
  `tokenizers==0.22.1`, `jinja2==3.1.6`, with no model weights or
  accelerator framework installed. All source renders used local files
  with offline flags.

## Cases and exact receipt format

`generate_goldens.py` defines the input conversations and records each result
as `name<TAB>rendered UTF-8 hex<TAB>comma-separated source token IDs` in
`goldens.tsv`. It verifies both downloaded SHA-256 hashes, calls
`AutoTokenizer.apply_chat_template` for text and IDs, and confirms that the
rendered string re-encodes to those IDs with `add_special_tokens=False`.

| Case | Rendered bytes | Source IDs |
| --- | ---: | ---: |
| `system_user_gen` | 122 | 22 |
| `system_user_closed` | 137 | 24 |
| `system_user_final` | 100 | 19 |
| `tool_history_gen` | 552 | 130 |
| `tool_history_closed` | 567 | 132 |
| `tool_history_final` | 530 | 127 |
| `answer_17_plus_25` | 97 | 23 |

The system and user cases preserve leading and trailing user spaces and
non-ASCII text. The tool history includes a source-rendered tool declaration,
an assistant message with reasoning and a typed tool call, a `tool` text
result, and an optional generation prompt. `closed` passes
`enable_thinking=False`; `final` omits the generation prompt.

The original `67695c5` `chat.rs` was compiled in scratch and given the same
pinned template and the two default-generation conversations. Its rendered
strings were encoded with the pinned `tokenizers==0.22.1`; Memra's `tok-parity`
then independently matched those baseline IDs in both special-token modes
(`2/2`). `baseline-goldens.tsv` records the exact original rendered bytes
and IDs for both cases. The measured source mismatch was:

| Case | Original Memra bytes / IDs | Source bytes / IDs | First different byte / ID index |
| --- | ---: | ---: | ---: |
| `system_user_gen` | 128 / 26 | 122 / 22 | 52 / 8 |
| `tool_history_gen` | 1408 / 310 | 552 / 130 | 19 / 3 |

The original generic path trimmed user spaces and added inter-turn newlines.
Its tools path injected Qwen's tool instruction block and grouped tool
responses differently from the source template.

The answer-bearing seed is the exact user content `What is 17 plus 25? Answer
with the number only.` Its source-rendered text is:

```text
<|im_start|>user
What is 17 plus 25? Answer with the number only.<|im_end|><|im_start|>assistant
```

The last byte is a newline. Its exact UTF-8 bytes, in hex, are:

```text
3c7c696d5f73746172747c3e757365720a5768617420697320313720706c75732032353f20416e73776572207769746820746865206e756d626572206f6e6c792e3c7c696d5f656e647c3e3c7c696d5f73746172747c3e617373697374616e740a
```

Its source token IDs are:

```text
151644,872,198,3838,374,220,16,22,5519,220,17,20,30,21806,448,279,1372,1172,13,151645,151644,77091,198
```

## Implementation and checks

The pinned template emits adjacent `<|im_start|>role\n...<|im_end|>` turns,
keeps content whitespace, and includes `<think>{reasoning}</think>` on prior
assistant turns. Tool results keep their own `tool` role. The optional
generation prompt is `<|im_start|>assistant\n`, with `<think></think>`
appended only when `enable_thinking=False`. Memra's generic ChatML path has
different turn spacing, content trimming, assistant history and tool-result
framing, so `crates/memra-tokenizer/src/chat.rs` uses a separate, exact-source
MiMo renderer. `tok-mimo-chat-parity` checks the seven source byte and ID
goldens through `Tokenizer::from_hf_dir`, including the plain and tools-capable
entry points.

The replay uses only the two pinned model files:

```sh
scratch=$(mktemp -d /tmp/mimo-chat-parity.XXXXXX)
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL tokenizer.json tokenizer_config.json \
  --revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4 \
  --local-dir "$scratch/hf" --max-workers 2
sha256sum "$scratch/hf/tokenizer.json" "$scratch/hf/tokenizer_config.json"
uv venv "$scratch/venv"
uv pip install --python "$scratch/venv/bin/python" \
  'transformers==4.57.1' 'tokenizers==0.22.1' 'jinja2==3.1.6'
cpus=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:3])))')
taskset -c "$cpus" env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$scratch/venv/bin/python" \
  research/mimo-chat-template-cpu-20260927/generate_goldens.py "$scratch/hf"
git diff --exit-code -- research/mimo-chat-template-cpu-20260927/goldens.tsv \
  research/mimo-chat-template-cpu-20260927/source-template.jinja
taskset -c "$cpus" env CARGO_TARGET_DIR="$scratch/target" CARGO_BUILD_JOBS=3 \
  cargo run --locked -j 3 -p memra-tokenizer --bin tok-mimo-chat-parity -- \
  "$scratch/hf" research/mimo-chat-template-cpu-20260927/goldens.tsv
taskset -c "$cpus" env CARGO_TARGET_DIR="$scratch/target" CARGO_BUILD_JOBS=3 \
  cargo test --locked -j 3 -p memra-tokenizer --lib --bin tok-mimo-chat-parity \
  -- --test-threads=3
rm -rf "$scratch"
```

Observed Memra gate: `7/7 exact byte and token-ID cases`. The library suite
passed `79/79` tests, and the source-fixture binary passed `2/2` tests.

## Limits

This qualifies only text content, global text tool declarations, assistant
tool calls with object arguments, text tool results, and the pinned source
template. It does not cover image, audio or video content, per-message tool
declarations, `tool_call.input`, string-form tool-call arguments, different
template revisions, generation quality, model loading, or serving behavior.
MiMo's template has no `reasoning_effort` parameter; that input and
per-message tool declarations return an error on the tools-capable path.
