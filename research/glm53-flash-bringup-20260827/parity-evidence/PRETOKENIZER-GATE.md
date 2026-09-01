# glm4 pre-tokenizer gate — GLM-5.3-Flash (lane/glm53-flash-bringup, 2026-08-28)

Artifact pins (all re-verified locally, not taken from the lane's earlier receipts):

| thing | sha256 | matches |
|---|---|---|
| `tokenizer.json` (zai-org/GLM-5.3-Flash @ 04c4e9e9) | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` | `inspect-receipts/artifact.lock` `tokenizer_sha256` |
| `config.json` (same rev) | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` | `artifact.lock` `config_sha256` |
| `tokenizer_config.json` (same rev) | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` | (not previously banked) |

## The one-atom claim, verified

The checkpoint's `pre_tokenizer` is `Sequence[Split(Regex, Isolated), ByteLevel(use_regex=false)]`
with ONE regex. Character diff against memra's `QWEN2_PRETOKENIZE_REGEX`: a single insert of
`{1,3}` at offset 59, nothing else. `QWEN2.replace(r"|\p{N}|", r"|\p{N}{1,3}|", 1)` is
byte-equal to the checkpoint's string. Pinned as an assertion in
`split_regex_identification_is_exact`.

## Identity of the family

llama.cpp `chkhsh` (sha256 over `str(tokenizer.encode(CHK_TXT))`, the converter's own family
fingerprint) for this tokenizer is
`cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267` — **identical** to the
entry upstream registers for `zai-org/GLM-4.7-Flash` under the name `glm4`
(`convert_hf_to_gguf_update.py:174`), which maps to `LLAMA_VOCAB_PRE_TYPE_CHATGLM4` and to this
same regex. `zai-org/GLM-4.7-Flash`'s `tokenizer.json` downloads byte-identical (same sha256).
So memra's id for the split is `glm4`, and a future GGUF mint through the upstream converter
resolves without a second change. A staged GGUF on the rig
(`/data/ai-ml/hf-models/glm47-flash/GLM-4.7-Flash-Q4_K_M.gguf`) carries
`tokenizer.ggml.pre = 'glm4'`, `tokenizer.ggml.model = 'gpt2'`, 321649 merges — confirming the
metadata id in the wild. (That file is a truncated download — tensor `blk.44.ffn_up_exps.weight`
runs past EOF — so the `from_gguf` end-to-end run could not be executed on it.)

## Oracle and method

Oracle is the CHECKPOINT'S OWN tokenizer through HF `tokenizers` 0.23.1 (the engine
`transformers` delegates to and the one that defined the vocab at training time) — the same
method as the step35 SKU gate (`research/step-sku-20260807/`), which is stronger than
re-executing llama.cpp's algorithm in a second regex engine. Generator:
`../pretok-ref-glm4.py`, which refuses to run unless tokenizer.json hashes to the pin above.

Corpus: 526 cases — digit runs of every length 1..12 plus 14; digits against letters, punct,
space, newline; `\p{N}`'s Nl/No members (Roman numerals, circled digits, fractions,
superscripts, math bold, Arabic-Indic, fullwidth); contractions in both cases plus U+017F;
NFD/NFC accents, leading/interior marks, Arabic harakat, Hebrew niqqud, Devanagari matras;
whitespace and newline runs incl. NBSP / ideographic space / U+2028 and the non-`\s` lookalikes
U+180E and ZWSP; CJK, Japanese, Korean, Cyrillic, Greek, Arabic, Thai; emoji with ZWJ, skin
tone and regional indicators; code/JSON/markdown/HTML/ChatML; llama.cpp's own `CHK_TXT`; three
3000-char uniform runs; a 400-case deterministic fuzz layer (`Random(20260827)`) over an
alphabet containing one member of every class that can start an alternative; and — the branch
the split itself does not own, but every real request takes — 14 cases of GLM's own control
tokens (`[gMASK]<sop>`, `<|system|>/<|user|>/<|assistant|>/<|observation|>`, think, tool_call,
arg_key/arg_value, code-FIM, `/nothink`, box, `[MASK]`, glued/partial literals) plus three
renders of the checkpoint's own `chat_template.jinja` (simple, multi-turn with accents/CJK/
emoji, and a tools render), which exercise `tokenizer_st_partition` rather than the regex.

## Results

| gate | result |
|---|---|
| split-level: `unicode::split_glm4` vs the checkpoint's Split step, 526 cases | **0 mismatches** |
| end-to-end token ids: `tok-parity` (HF dir) vs `tokenizers`, both `add_special` modes | **526/526 identical**, 12838 plain ids — `tok-parity-glm53-hf.txt` |
| resolution: `Tokenizer::from_hf_dir` on the real checkpoint | `pre="glm4"`, `split=Glm4`, vocab 154856 |
| counter-check: `unicode::split_qwen35` vs the same oracle | **337/526 MISMATCH** — the corpus discriminates |

## Class semantics, measured against the real engine (not assumed)

Swept over the full codepoint space with `Split(..., behavior="removed")` and compared to
memra's `unicode_data.rs` flag table:

- `\s` — **exact match**, 25 codepoints, zero difference in either direction.
- `\p{L}` — memra 136726 ⊂ onig 141028 (4302 onig-only).
- `\p{N}` — memra 1831 ⊂ onig 1911 (80 onig-only).
- `\p{M}` — memra 2450 ⊂ onig 2501 (51 onig-only).

The onig-only codepoints are Unicode assignments newer than llama.cpp's generated table
(Garay/Myanmar-Pao digits, Todhri letters, …). This is a PRE-EXISTING vintage gap shared by
every memra pre-tokenizer (qwen35, qwen2, deepseek-v3), not something this change introduces,
and it is the one class of input where memra's split can still differ from the checkpoint's.

`(?i:)` is Unicode simple case folding: U+017F LATIN SMALL LETTER LONG S matches the `'s`
alternative (`'ſx` -> `["'ſ", "x"]`). A full sweep of the codepoint space found U+017F to be
the ONLY codepoint case-equal to any of s/t/r/e/v/m/l/d beyond their ASCII uppercase forms.
llama's `tolower` map leaves it alone, so `unicode::contraction_fold` folds it explicitly.

## qwen35 / qwen2 non-regression

`split_qwen35` is byte-untouched (separate machine). Proven, not asserted:
`cargo test -p memra-tokenizer --test llama_parity` ran for real against the Qwen3.5-9B NVFP4
MTP GGUF and llama.cpp's `llama-tokenize` post-change — **12/12 corpus strings matched
EXACTLY**, plus `golden_pairs` and `round_trip`. Inside the crate,
`glm4_equals_qwen35_off_the_two_divergences` re-uses the deepseek-v3 corpus minus its
digit/mark/U+017F cases and asserts the two machines agree everywhere else.

## Reproduce

```
python3 research/glm53-flash-bringup-20260827/pretok-ref-glm4.py <dir-with-tokenizer.json> --rust
cargo run -p memra-tokenizer --bin tok-parity -- <dir> parity-evidence/corpus.tsv parity-evidence/ref-ids.tsv
cargo test -p memra-tokenizer
```
