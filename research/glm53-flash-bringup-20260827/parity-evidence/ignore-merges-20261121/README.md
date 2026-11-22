# GLM-5.3-Flash: the whole-word lookup (`ignore_merges`), 2026-11-21

Scope: token ids of `zai-org/GLM-5.3-Flash` @ `04c4e9e95c5da8862dced7e5056455116f83a7e0`
(`tokenizer.json` sha256 `19e77364…a82d`, the pin of `../../inspect-receipts/artifact.lock`).
CPU-only. This is a tokenizer fix: it changes prompt ids of a supported model, and qualifies
no numeric program, runtime or serving binary.

## What was wrong

The checkpoint's `tokenizer.json` declares `"ignore_merges": true`. A pre-split word that is
itself a vocabulary piece then encodes as that one id; the merge table is not consulted. memra
did not read the flag. For 893 pieces the merge table builds something else, so memra produced
other ids than the checkpoint's tokenizer. 802 of the 893 are a space and a word, most of them
CJK (`" 参考"`: memra `[26767, 224, 98580]`, the checkpoint `[99855]`).

The gate of 2026-08-28 in the parent directory (526 cases, `PRETOKENIZER-GATE.md`) did not see
it: none of its cases holds one of those 893 pieces as a whole pre-split word. Its result
stands for what it measured, the split.

## The fix

- `tokenizer.json` path: `Tokenizer::from_hf_files` reads `model.ignore_merges`.
- GGUF path: GGUF has no key for the flag. `PreSplit::gguf_ignore_merges` gives each split its
  family's declared value, so `pre = glm4` takes the lookup. llama.cpp sets it for `glm4` too
  (`src/llama-vocab.cpp:2600-2604` at 29b52d398b29).

## Gate

`gen_corpus.py` finds the pieces the flag decides (tokenizers 0.23.3 with the flag as shipped
against the same file with it off), and writes each alone and inside a sentence: 1,786 texts,
of which the ids of 1,695 move with the flag (`gen-corpus.txt`).

| build | path | result | receipt |
|---|---|---|---|
| base `fbe9f28cbd` | `tokenizer.json` | 91 of 1,786 identical, **1,695 differ** | `tok-parity-base-fbe9f28cbd.txt` |
| this change | `tokenizer.json` | **1,786 of 1,786** | `tok-parity-head.txt`, `cargo-test-head.txt` |
| this change | GGUF built from the same file, `pre = glm4` | **1,786 of 1,786** | `cargo-test-head.txt` |
| this change, family value off | GGUF | 1,695 differ | `../../../modelplan-onboarding-inkling2mini-20261121/chat/receipts/red-arms.txt` |
| this change | both paths, the 526 cases of 2026-08-28 | 526 of 526 | `cargo-test-head.txt` |

Reproduce, with the pinned `tokenizer.json`, `tokenizer_config.json` and
`generation_config.json` in a directory:

```
cargo run -p memra-tokenizer --bin tok-parity -- <dir> corpus.tsv ref-ids.tsv
MEMRA_GLM53_TOKENIZER_DIR=<dir> cargo test -p memra-tokenizer --test whole_word_lookup -- --nocapture
```

The 20 MB `tokenizer.json` is not committed, so the cargo test is artifact-gated (declared in
`tools/skip-census.tsv`) and does not run in CI. What CI does run is the same law on a six-piece
vocabulary through both paths, and the Inkling 2 vocabulary, which is committed, through both.

## Did any committed receipt carry the old ids?

No. `scan_committed_ids.py` read every id sequence in the tracked GLM receipts and in every Rust
source (`.ids`, `.tsv`, and integer arrays in `.json`, `.jsonl`, `.txt`, `.rs`): 2,750 sequences
in 6,711 files. It decodes each with the checkpoint's tokenizer and re-encodes the text with the
flag on and off. Not one decodes to a text on which the two differ, so none can hold an id
this fix changes (`scan-committed-ids.txt`). Nothing was rewritten.

A request whose prompt carries one of the 893 words now tokenizes as the checkpoint does.
Measurements taken before this change on such prompts ran a prompt a few tokens longer than the
vendor's; none is among the committed receipts.
