# Lane 5 receipts — encoding_dsv4 template / think-modes + tools arm (2026-08-18)

Branch `lane/dsv4-template` (forked from lane/dsv4-flash-loader @ f78a0cd944), worktree
`~/projects/wt-dsv4-template`, UNMERGED (owner decides merges).

Oracle: `ref/encoding/encoding_dsv4.py` (== the NVFP4 artifact's copy, sha256
`bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0`). The python IS the
law; byte parity is the only acceptance (GGUF template-mint house law).

## What was built

| piece | file |
|---|---|
| Renderer arm | `crates/memra-tokenizer/src/chat.rs` — `apply_dsv4_template` + helpers (`dsv4_merge`, `dsv4_sort_tool_results`, `dsv4_drop_thinking`, `dsv4_render_tools`, `dsv4_render_tool_calls`, `dsv4_json`), `template_is_dsv4` dispatch in both `apply_chat_template_str` and `apply_chat_template_tools_ex`, `template_has_tools_branch` extension |
| Turn struct | `chat.rs` — added `task: Option<String>` + `tools: Vec<Val>` (dsv4-only; other dialects ignore) |
| Parser arm | `crates/memra-server/src/toolcall.rs` — `ToolStreamParser::dsv4` + `Dsv4Call` state + `parse_dsv4_calls`; char-boundary guard on `partial_suffix_len` (multibyte `｜`) |
| Caps + wiring | `worker.rs` — `ModelCaps::dsv4` (+ instruct_type "deepseek"); `main.rs` — dsv4 parser arming + scoped `</｜DSML｜tool_calls>` stop + Turn field plumbing |
| Tokenizer | `crates/memra-tokenizer/src/lib.rs` — deepseek-v3 pre-tokenizer detection in `from_hf_dir` (`pre_tokenizer_is_deepseek_v3`) |
| Census | `TEMPLATE-SEMANTICS.md` (file:line cites, banked ambiguities) |
| Fixtures | `fixtures/` (25 generated) + `gen_fixtures.py` + `dsv4-chat-template.sentinel.jinja` + `ref/artifact-encoding/tests/` (4 authoritative) + `fixtures/tokenization-crosscheck.json` |

## Tools wire format verdict: DEFINED

encoding_dsv4 defines a full DSML tool protocol (TEMPLATE-SEMANTICS.md §5). Renderer arm
AND parser arm both implemented (standard-surface law: a served model needs identical
full API incl. tools). Declaration → system/developer block; assistant calls →
`<｜DSML｜tool_calls>`/`<｜DSML｜invoke>`/`<｜DSML｜parameter … string="true|false">`; results →
user `<tool_result>` blocks (merged, sorted by call order). Model-emitted stop: EOS after
`</｜DSML｜tool_calls>`. Malformed span policy: surfaced VERBATIM (house policy; the oracle
raises) — matches the gemma arm.

## ThinkMode mapping (three modes) — REVISED by the 0731 re-gate (see section below)

| dsv4 mode | wire | memra |
|---|---|---|
| chat (Non-think) | `<｜Assistant｜></think>` | `ThinkMode::NoThink` |
| thinking | `<｜Assistant｜><think>` | `ThinkMode::Default` / `Think` |
| thinking + effort rung | effort prompt prefix before system | `reasoning_effort = Some(level)`, resolved per `Dsv4Encoding` |

Default→thinking (the model has no template-own default; thinking_mode is a required arg,
README example uses thinking, model is agentic). Effort resolution is ENCODING-KEYED
(0731 re-gate 2026-08-18, ENCODING-DIFF.md): preview law = "high" no-op / "max" the
absolute text; 0731 law = low default / "high" the absolute text / "max" a new stronger
text. dsv4 models are now effort-consuming on the serve path (`ModelCaps::dsv4` forwards
the OpenAI level; "medium" clamps to the default rung); the native "max" rung stays beyond
the OpenAI ladder — renderer/fixture-supported, owner call to expose (finding #2).

## Fixture gate table (ALL GREEN)

| gate | fixtures | result |
|---|---|---|
| `dsv4_template_fixtures_match_the_oracle` | 25 generated (3 modes × single/multi/system/tools-decl/tool-cycle/tool-results-sorted/tasks/reminder/typed-args) | byte-identical to `encode_messages` |
| `dsv4_artifact_fixtures_are_byte_identical` | 4 authoritative `test_output_{1..4}` | byte-identical |
| `dsv4_tokenization_crosscheck_matches_official_ids` | 5 rendered prompts → ids | memra == official HF tokenizer (pre = "deepseek-v3") |
| `dsv4_default_thinkmode_renders_thinking` | Default==Think, NoThink==chat | pass |
| parser (`toolcall::tests::dsv4_*`) | 11 tests: call parse, reasoning split, multi-invoke, typed args, malformed-verbatim, char-by-char, holdback, include_reasoning:false | pass |

Cert lines (binary + invocation + banked output):
- `cargo test -p memra-server --bin memra-server dsv4` → `14 passed; 0 failed`
  (4 render/tokenize + 10 parser… note: 11 parser + 3 render-side + tokenize + default = 14).
- `cargo test -p memra-server --bin memra-server toolcall` → `32 passed; 0 failed`.
- `cargo test -p memra-tokenizer --lib` → `28 passed; 0 failed`.
- `cargo test -p memra-server --bin memra-server` → `317 passed; 0 failed` (was 303).
- clippy: zero warnings in dsv4 lane files (chat.rs dsv4 region, lib.rs deepseek
  detection, toolcall.rs, main.rs/worker.rs dsv4 wiring); remaining warnings are all in
  pre-existing gemma/hy3 code untouched by this lane.

Reproduce fixtures: `python3 research/dsv4-template-20260818/gen_fixtures.py` (needs
`ref/encoding/encoding_dsv4.py` on the path); tokenization-crosscheck ids banked via HF
`tokenizers` 0.23.1 (scratch venv, cleaned).

## Banked ambiguities (full list in TEMPLATE-SEMANTICS.md §8)

1. No template-own default thinking_mode (required arg) → Default maps to thinking.
2. Think-Max unreachable from OpenAI reasoning_effort; renderer supports it, serve path
   leaves it None (never corrupts). Serving-lane/owner call whether to expose.
3. `direct_search_results` role kept by drop_thinking but raises in render_message
   (upstream inconsistency) — not implemented.
4. Tools on user messages silently destroyed upstream — unreachable on the memra surface.
5. `num_to_render` context-recount shape — unreachable (context always empty).
6. `wo_eos`/`mask`/`task`/`context` off the OpenAI surface; `task` IS rendered (fixture 4
   needs it), `wo_eos` banked as not-wired (always emit EOS — the only fixture behavior).
7. Float-text fidelity (Val::Num exact text; exotic-float caveat, no fixture hits it).
8. `<think>` transition-vs-body ownership; chat-mode drops client-supplied reasoning.

## Long-run hardening — llama.cpp #26965 shape (2026-08-18)

Recon (darklanes `research/deepseek-flash-20260818/RECON.md`): llama.cpp issue #26965 —
the joyai-llm/deepseek-v3-class pre-tokenizer stack-overflows on long uniform ASCII runs
inside tool results ('Z'×131072). Upstream runs that family through generic backtracking
`std::regex` (ECMAScript), and the alternation `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
recurses per character on a uniform run.

**Verdict: IMMUNE — no fix needed; tests land as the regression guard.**

Why ours cannot share the failure: `unicode::split_deepseek_v3` is not a regex engine.
It is three ordered ITERATIVE passes (`split_pass` closures over collapsed bytes /
codepoints) — plain scan loops, `pos` advances monotonically every iteration, zero
recursion and zero backtracking state. The alt-4/alt-5 "backtrack" is a precomputed
`last_rn` / `run_end - 1` index, not engine backtracking. Downstream, `bpe_merge_word`
is llama's heap-based merge loop (also iterative, O(n log n)); `st_partition` is linear
`str::find` scans. Rust test threads run on 2 MiB stacks, so any recursion proportional
to run length would have aborted the new tests — none did. Measured: the 1M-char case
completes in single-digit seconds in a debug build (linear, not quadratic).

Regression guards (both committed, both in the default suites):

| test | cases | invocation | result |
|---|---|---|---|
| `unicode::tests::deepseek_v3_split_survives_long_uniform_runs` | 'Z'×131072 (the issue's reproducer), 'Z'×1M, ' '×131072, '7'×131072, mixed 256k (Z/space/digit/newline runs), 'é'/'中'/'ア'/'▁'×65536; reassembly + shape spot-checks + 10s bound | `cargo test -p memra-tokenizer --lib deepseek_v3_split_survives_long_uniform_runs` | 1 passed, 0.19s |
| `tests::dsv4_tool_result_long_runs_render_tokenize_roundtrip` | same 7 run shapes carried as a dsv4 tool-RESULT message in a full tool cycle; render (sentinel template) → `encode` via the real ref tokenizer (`pre == "deepseek-v3"`) → `decode(encode(x)) == x` + 60s/case bound | `cargo test -p memra-server --bin memra-server dsv4_tool_result_long_runs` | 1 passed, 3.54s (debug, incl. the 1M case) |

Full suites after: `cargo test -p memra-server --bin memra-server` → `318 passed; 0
failed` (was 317); `cargo test -p memra-tokenizer --lib` → `29 passed; 0 failed` (was 28).

HF cross-check on the 131k reproducer (id parity is a receipt, not a test gate — the
committed gate for giant cases is crash-safety + round-trip): scratch venv, HF
`tokenizers` 0.23.1 over `ref/tokenizer.json`. The rendered 131k-tool-result prompt is
132,518 chars → **65,878 ids, memra == official HF tokenizer EXACTLY**; HF encoded it in
0.03 s and its own decode round-trips — HF did NOT choke (its Rust regex engine is not
`std::regex`, so it does not share upstream's failure either). Bridge: run the
memra-server test with `DSV4_LONGRUN_DUMP_DIR=<dir>` to dump `rendered-131k.txt` +
`memra-ids-131k.csv`, then compare `Tokenizer.from_file(ref).encode(rendered).ids`.
Scratch venv + dumps cleaned.

## 0731 effort-ladder re-gate (2026-08-18) — publish-gate support-checklist item 3

0731's encoding_dsv4.py remaps the effort ladder (low=default no-op / high=the OLD "max"
text / max=a NEW stronger text); everything else in the encoding is byte-identical to the
preview (full diff census + per-difference verdicts: **ENCODING-DIFF.md**). Work landed:

| piece | file |
|---|---|
| Encoding enum + per-revision prefix law + refuse-on-ambiguity | `crates/memra-tokenizer/src/chat.rs` — `Dsv4Encoding`, `dsv4_effort_prefix`, `DS_EFFORT_ABSOLUTE_MAX`/`DS_EFFORT_BEYOND_MAX`; `apply_dsv4_template` + `apply_chat_template_tools_ex` take the revision |
| Config-keyed detector (dspark_* census; partial set refuses the load) | `crates/memra-tokenizer/src/lib.rs` — `dsv4_encoding_from_config` in `from_hf_dir`, carried on `Tokenizer`, auto-passed by its render methods |
| Serve mapping: dsv4 consumes the OpenAI effort level | `crates/memra-server/src/main.rs` — `build_chat_request` forwarding gate (`effort_levels \|\| dsv4`); "medium" renders as the default rung (hy3 clamp precedent); parse_think unchanged ("max" stays beyond the ladder) |
| 0731 oracle mirror | `research/dsv4-template-20260818/ref-0731/encoding/` (sha256 `abc0d261…`, fetched read-only from the minted artifact on the box) |
| 0731 fixture matrix | `fixtures-0731/` — **55** cases (`gen_fixtures.py --encoding 0731`): base-25 under 0731 semantics + 2 modes × {low,high,max} × 5 shapes; differs from preview on exactly {03,07,12}-max + 04-high, as the diff predicts |
| Tokenization crosscheck | `fixtures-0731/tokenization-crosscheck.json` — 5 new fixtures (both new prefix texts) vs official HF tokenizers 0.23.1: memra == HF exactly |

Fixture gate table (ALL GREEN): `dsv4_0731_fixtures_match_the_oracle` (55, V0731) ·
`dsv4_template_fixtures_match_the_oracle` (25 preview, REGRESSION — fixture bytes
unchanged) · `dsv4_artifact_fixtures_are_byte_identical` (4 authoritative, now rendered
under BOTH revisions) · `dsv4_0731_tokenization_crosscheck_matches_official_ids` ·
`dsv4_effort_prefix_law` + `dsv4_effort_renders_per_encoding_through_dispatch` +
`hf_dir_dsv4_encoding_detection` (memra-tokenizer).

Cert lines: `cargo test -p memra-tokenizer --lib` → **32 passed; 0 failed** (was 29).
`cargo test -p memra-server --bin memra-server` → **320 passed; 0 failed** (was 318).
`cargo test -p memra-server --bin memra-server dsv4` → 17 passed. clippy: zero NEW
warnings in lane files (stash A/B identical up to line shifts).

## What remains (needs serving lane / owner)

- ~~Whether OpenAI `reasoning_effort` "high" (or a new knob) should reach dsv4 Think-Max.~~
  RESOLVED by the 0731 re-gate: "high" maps by name onto 0731's real high rung (preview
  keeps its documented no-op); the native "max" rung stays beyond the OpenAI ladder —
  owner call whether to expose it (vLLM precedent: chat_template_kwargs).
- Live real-CLI round-trip gate (needs dsv4 served on the box; PLAN says Flash is
  not-for-serving yet — same gate shape as `serve-gemma4-tools-gate.sh` if/when it serves).
- response_format on a dsv4 turn (encode_messages supports it; memra maps response_format
  to constrained grammar — a different mechanism, banked as not-wired for dsv4).
- A dsv4 GGUF mint (none exists) must define + carry an encoding-revision marker in its
  metadata; `from_gguf` sets unknown today, so effort high/max renders would refuse.
