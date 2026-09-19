# Qwen2 tokenizer split — 2026-09-20

Literal Qwen2 split/BPE parity passes in HF and GGUF loaders. Full raw-input parity for a
tokenizer declaring NFC remains blocked by the separately tracked normalization omission,
[issue #554](https://github.com/avifenesh/memra/issues/554).

The change gives `qwen2` its own `PreSplit` variant. It shares the existing literal GLM
character-class scanner with a one-digit bound; GLM retains its three-digit bound and the
Qwen3.5 scanner is unchanged. No runtime dependency or GPU code is added.

## Regression and independent oracle

The checked-in synthetic byte-level BPE fixture was generated with Hugging Face
`tokenizers==0.22.2`, without a normalizer. Its merge chains deliberately cross possible
pretokenizer boundaries. Both real loader paths failed before the fix on `e\u0301`:

```text
left: [257]
right: [101, 274]
test result: FAILED. 1 passed; 2 failed
```

After the fix, all 39 fixture cases match in both encode modes through HF and GGUF, and the
Qwen3.5 combining-mark control retains its merged token. Inputs cover decomposed accents,
Hebrew niqqud, Arabic harakat, Devanagari, non-ASCII number classes, Unicode contraction
folding, whitespace, control bytes and emoji. These three tests always execute without
external artifacts. The complete tokenizer library suite reports 72 passed / 0 failed.
Existing model-dependent tests can skip absent local artifacts; those are not new model proof.

## Real vocabulary check

Pinned source: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
`manifest.json` records the tokenizer/sidecar hashes and oracle version. The corpus contains
39 adversarial inputs plus 512 deterministic mixed-script inputs (seed 540).

| Input/program | Result |
|---|---|
| Raw inputs, complete HF reference including declared NFC, candidate HF loader | 301/551 cases identical in both modes; 250 fail |
| Independently NFC-normalized inputs, candidate HF loader | 551/551 cases identical in both modes |
| Same normalized inputs, metadata-only GGUF wrapper of pinned vocabulary/merges | 551/551 cases identical in both modes |

Exactly 250 raw inputs change under NFC. For example, the full HF reference converts raw
`e\u0301` to `é` and returns `[963]`; the native loader preserves the raw bytes and returns
`[68,53839]`. `from_hf_dir` does not read the declared normalizer on the baseline either.
The normalized-input checks isolate split/BPE behavior; they do not qualify full normalization
or a model weight artifact. The GGUF file is a metadata wrapper, not a published weight mint.

## Reproduce

The ordinary unit suite is offline:

```sh
cargo test --locked -p memra-tokenizer --all-targets
cargo clippy --locked -p memra-tokenizer --all-targets --no-deps -- -D warnings
```

Regenerate the small embedded fixture with
`crates/memra-tokenizer/tests/fixtures/generate_qwen2.py` and its documented pinned command.
To fetch only the pinned real tokenizer and regenerate the raw/normalized corpora:

```sh
uv run --no-project --with tokenizers==0.22.2 python research/qwen2-tokenizer-20260920/capture.py
cargo build --locked -p memra-tokenizer --bin tok-parity
target/debug/tok-parity research/qwen2-tokenizer-20260920/artifacts research/qwen2-tokenizer-20260920/artifacts/corpus-normalized.tsv research/qwen2-tokenizer-20260920/artifacts/reference-normalized.tsv
target/debug/tok-parity research/qwen2-tokenizer-20260920/artifacts/tokenizer.gguf research/qwen2-tokenizer-20260920/artifacts/corpus-normalized.tsv research/qwen2-tokenizer-20260920/artifacts/reference-normalized.tsv
```

Using `corpus.tsv` and `reference.tsv` instead exposes #554 and is expected to fail until
normalization is implemented. Preserve that failure; do not call the normalized check full
tokenizer parity.

The raw corpus/reference TSVs and logs are retained as gzip files beside this record; capture.py regenerates plain TSVs under artifacts/. Source hashes are retained here too. Host: macOS ARM64, Rust 1.97.1.
No model weights, GPU generation, server smoke, throughput measurement, deployment, or
support-state promotion was performed. Artifact-specific end-to-end generation remains a
separate qualification step.
