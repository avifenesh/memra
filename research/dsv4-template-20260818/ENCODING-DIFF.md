# encoding_dsv4 preview → 0731: complete behavioral diff + re-gate verdicts (2026-08-18)

0731 publish-gate support-checklist item 3. The lane-5 template arm was gated against the
PREVIEW encoding; 0731 remaps the reasoning-effort ladder, invalidating the effort-level
branches until re-gated. This file is the full diff census with a verdict per difference,
and the record of the re-gate.

## Sources (both mirrored in this research dir; the python IS the law)

| oracle | path | sha256 | bytes |
|---|---|---|---|
| preview | `ref/encoding/encoding_dsv4.py` (== the nvidia NVFP4 artifact's copy, `ref/artifact-encoding/`) | `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0` | 27,908 |
| 0731 | `ref-0731/encoding/encoding_dsv4.py` (fetched read-only from the minted artifact on the rented bench box, `/home/ubuntu/models/dsv4-flash-0731-nvfp4/encoding/` — byte-identical to the 0731 source per 0731-MINT-RECEIPTS.md) | `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6` | 29,001 (+1,093) |

Full `diff -u` over `encoding_dsv4.py`: FOUR hunks — one constant block, one logic block,
two docstring touches. Everything else is byte-identical. File-level census of the rest of
`encoding/`:

| file | preview → 0731 | verdict |
|---|---|---|
| `encoding_dsv4.py` | 4 hunks (below) | see per-difference verdicts |
| `README.md` | +docs for the three-level ladder (table + both prefix texts) | no action (documentation of the same delta) |
| `test_encoding_dsv4.py` | byte-identical | no action |
| `tests/test_input_{1..4}.json` | byte-identical | no action |
| `tests/test_output_{1..4}.txt` | byte-identical to the preview ARTIFACT's authoritative outputs (`ref/artifact-encoding/tests/`; the base preview repo mirror never carried outputs) | no action — the artifact-fixture gate now renders under BOTH encodings and asserts identity (`dsv4_artifact_fixtures_are_byte_identical`) |

## Behavioral differences in encoding_dsv4.py (complete)

### D1 — effort constants: one prompt → a three-level ladder (the remap)

- preview E:64-68: `REASONING_EFFORT_MAX` — the "Reasoning Effort: Absolute maximum…" text.
- 0731 E:67-79: `REASONING_EFFORT_PROMPTS = {"low": "", "high": <the OLD "max" text,
  byte-identical>, "max": <NEW, stronger "Reasoning Effort: Beyond maximum — …" text>}`;
  0731 E:80: `DEFAULT_REASONING_EFFORT = "low"`.

Consequences: `low` = default/no-op (new accepted value; the preview oracle ASSERTS on the
string); `high` = the old "max" text (was a documented NO-OP == None on preview);
`max` = a new, stronger text (was the old text). The dash in the new text is U+2014.

**Verdict: renderer change needed + mapping change needed.** Done:
- `chat::Dsv4Encoding { Preview, V0731 }` + `dsv4_effort_prefix()` (crates/memra-tokenizer/
  src/chat.rs) render the exact per-revision table; both encodings stay supported (the
  preview artifact is still gated in the program).
- Serve mapping: `ModelCaps::dsv4` models now receive the OpenAI `reasoning_effort` level
  (main.rs `build_chat_request` forwarding gate) — under V0731, `"high"` is a REAL prefix;
  under Preview it renders the documented no-op. `"medium"` (defined by neither revision)
  renders as the default level — the hy3 never-corrupt clamp precedent. The native `"max"`
  rung stays beyond the OpenAI ladder (`parse_think` unchanged; vLLM precedent is
  `chat_template_kwargs`) — renderer/fixture-supported, serve surface untouched.

### D2 — effort resolution + injection condition in render_message

- preview E:261-263: `assert reasoning_effort in ['max', None, 'high']`; prefix injected
  only when `index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max'`.
- 0731 E:274-278: `reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT`;
  `assert reasoning_effort in REASONING_EFFORT_PROMPTS`; prefix injected whenever
  `index == 0 and thinking_mode == "thinking"` (the `low` entry is `""`, so low/None add
  nothing). Chat mode never renders a prefix under either revision.

**Verdict: renderer change needed** (same change as D1 — the prefix is resolved once per
render and pushed before the first rendered message, `""` when nothing applies).
Refuse-on-ambiguity law: with an UNKNOWN encoding revision, rendering refuses exactly the
two cells whose bytes differ across revisions (thinking + `"high"`/`"max"`) and stays
infallible everywhere the revisions agree (None/low/chat-mode/every other input).

### D3 — docstrings (render_message E:235→247-248, encode_messages E:530→545-547)

Doc-only restatement of the ladder. **Verdict: no action.**

### Everything else: verified NO delta (grep-level + full-diff evidence)

Roles and their templates, `TOOLS_TEMPLATE` + DSML tool-call/parameter wire format,
`<tool_result>` blocks, merge/sort/drop-thinking preprocessing, transition tokens,
task heads, special tokens (BOS/EOS/`<think>`/`</think>`/`｜DSML｜`/`<｜User｜>` etc.),
stop conditions, and the PARSING functions (`parse_tool_calls`,
`parse_message_from_completion_text`, `_read_until_stop`) are byte-identical — the
memra DSML stream parser (`toolcall.rs`) needs **no action**. Tokenizer files are
byte-identical across preview/0731 source/mint (Gate C receipt), so the deepseek-v3
pre-tokenizer detection and id crosschecks transfer.

## Encoding-revision detector (config-keyed, never filenames)

The 0731 checkpoint added exactly four `dspark_*` keys to config.json in the SAME revision
that remapped the ladder (`dspark_block_size`, `dspark_markov_rank`,
`dspark_noise_token_id`, `dspark_target_layer_ids` — 0731-PREP.md §1.1); tokenizer_config/
tokenizer/generation_config are byte-identical across the two checkpoints, so config.json
is the artifact's ONLY marker. `Tokenizer::from_hf_dir` (memra-tokenizer/src/lib.rs,
`dsv4_encoding_from_config`) runs the census over the dir's config.json CONTENT:

| census (`model_type == "deepseek_v4"`) | result |
|---|---|
| all four keys | `Some(V0731)` |
| none | `Some(Preview)` |
| PARTIAL set | **load refuses** (corrupt/hand-edited config — never guess a ladder) |
| config.json absent/unparseable, or foreign model_type | `None` (unknown) — renders refuse thinking+high/max |

The detected revision rides the `Tokenizer` into every serve render (methods pass it to
`chat::apply_chat_template_tools_ex`); the free-function seam takes it explicitly (fixture
harness / bins). GGUF trap, banked: no dsv4 GGUF lineage exists and no metadata key is
defined — `from_gguf` sets unknown, so a future dsv4 GGUF mint MUST carry an encoding
marker in its metadata or its effort renders refuse (mirror of the GGUF template-mint law).

## Re-gate record (all CPU, rig-side)

| gate | fixtures | result |
|---|---|---|
| `dsv4_0731_fixtures_match_the_oracle` | **55** generated from the official 0731 oracle (`gen_fixtures.py --encoding 0731`): the 25-case base matrix re-rendered under 0731 semantics + a systematic **2 modes × {low,high,max} × 5 shapes {single, multiturn, system, tools, toolresults} = 30**-case effort matrix | byte-identical |
| `dsv4_template_fixtures_match_the_oracle` (regression) | 25 preview fixtures, UNCHANGED bytes (regenerated, 0 file diffs) | still byte-identical under `Preview` |
| cross-encoding fixture delta | base-25 rendered under both oracles | differs on EXACTLY {03,07,12}-max (new text) + 04-high (now prefixed); all 21 others byte-identical — matches the diff prediction |
| `dsv4_artifact_fixtures_are_byte_identical` | 4 authoritative outputs under BOTH `Preview` and `V0731` | byte-identical |
| `dsv4_0731_tokenization_crosscheck_matches_official_ids` | 5 new 0731 fixtures (both new prefix texts, incl. tools + tool-result shapes) → ids vs official HF `tokenizers` 0.23.1 over `ref/tokenizer.json` | memra == HF exactly |
| `dsv4_effort_prefix_law` + `dsv4_effort_renders_per_encoding_through_dispatch` (memra-tokenizer) | the full (thinking, effort, encoding) table incl. both refusal cells | pass |
| `hf_dir_dsv4_encoding_detection` (memra-tokenizer) | detector census: none/all/partial/foreign/missing | pass (partial refuses) |

Suites: `cargo test -p memra-tokenizer --lib` → **32 passed** (was 29);
`cargo test -p memra-server --bin memra-server` → **320 passed** (was 318); clippy over the
lane crates: zero NEW warnings (the 7 memra-tokenizer warnings are pre-existing,
line-shift-identical under `git stash` A/B).

Reproduce: `python3 gen_fixtures.py --encoding 0731`; crosscheck ids banked via HF
`tokenizers` 0.23.1 (scratch venv, cleaned).
