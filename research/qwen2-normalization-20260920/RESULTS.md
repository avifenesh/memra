# Declared NFC input normalization — 2026-09-20

The HF loader now executes declared NFC around the correct raw/normalized added-token stages. Unsupported normalization programs refuse. A versioned Memra GGUF input-program extension preserves the same declarations; existing artifacts without that key retain identity normalization.

Independent HF tokenizers0.22.2 expectations were captured before the implementation. All315 cases across31 variants pass through HF and the explicit GGUF import in both encode modes. Eleven small special-recognition cases independently cross parse-special and add-special switches;22 declaration cases test required/optional supported programs and explicit refusals. All 78 tokenizer unit tests and 12 CLI unit tests pass after review fixes. Strict Clippy passes. Three legacy model-dependent integration functions skip because their model/reference artifacts are absent; they are not counted as validation passes.

The pinned real Qwen2.5 vocabulary now matches551/551 raw inputs through the HF loader, fixing the250 baseline mismatches caused by missing NFC. The explicit GGUF raw-input check also matches 551/551 cases in both modes; independent review verified all four real-vocabulary special-mode cases; no model weights or support-state promotion follows from a metadata-only tokenizer artifact. Native integration/review is pending.

NFC uses the same native Rust normalization crate/table version as the reference; regex word-boundary tables were checked byte-identical across the reference/runtime regex-syntax versions. See dependency-provenance.json. Execution remains inside memra-tokenizer in Rust.

## Reproduce independent oracles

The baseline Git commit fdb781362c8366e11db492cba14bfe1dc6bb6941 must be available locally. First run the existing pinned tokenizer-only capture, then copy its generated artifacts directory into this record's ignored pinned/ directory. With tokenizers==0.22.2 installed, run generate_expectations.py and verify_expectations.py here. They use the frozen baseline fixtures and HF only, without importing this implementation.

The ordinary offline regression suite uses checked-in fixtures:

```sh
cargo test --locked -p memra-tokenizer --all-targets
cargo clippy --locked -p memra-tokenizer --all-targets --no-deps -- -D warnings
```

export_pinned_gguf.py takes the pinned tokenizer directory and an output path. It checks the exact tokenizer SHA before producing an explicitly labeled tokenizer-only GGUF with input-program version1. Compare raw corpus.tsv against reference.tsv with tok-parity for both the HF directory and that GGUF. The older no-declaration GGUF continues to represent identity normalization.

The baseline-540-native-receipt.json.gz capsule is immutable historical qualification of source `67fce3a9` published in PR558; it is not native qualification of this change.

## Review corrections

Independent test, performance, and architecture reviews identified consumed whitespace spans emitting extra IDs and normalized token collisions being reordered by ID. New HF-only captures cover 48 inputs across 18 variants with all four parse-special/add-special combinations, through both HF and explicit GGUF loading. Both minimal failures reproduced before correction. Empty adjusted spans are discarded, while nonempty overlap is preserved; stable special-class grouping retains declaration order.

The onboarding tokenizer fingerprint now validates and binds the explicit input-program bytes, distinguishing absent, identity, NFC, and changed added-token flags. Existing no-declaration fingerprints retain their previous bytes. A focused CLI test executes both identity and NFC encodings, verifies differing fingerprints, and checks unsupported declarations refuse.

JSON value parsing now rejects nesting beyond 128 levels before building a deeply nested declaration. Accepted 32-level program bounds remain separate and smaller. Boundary tests exercise arrays and objects.

The security reviewer service stopped before issuing a final report. It is recorded as incomplete, not passed. Completed independent domain findings are repaired here; focused re-review and final native/integration gates remain pending.

## External review: inverted whitespace spans

The external review found that a cursor clamp can create a start greater than end, in addition to the previously tested empty span. The native regression first reproduced extra IDs for `<X>` followed by two tabs. The guard now discards both empty and inverted adjusted spans, preserving nonempty overlapping behavior. Tests cover identity/NFC, raw/normalized matching, both loaders and all parse/add-special modes (96 comparisons).

Pinned HF 0.22.2 itself reports `PanicException: AddedVocabulary bad split` for these three inverted-span inputs. The recorded oracle errors are retained; no token-ID parity claim is made for inputs on which the oracle provides no result. The new tests instead require consumed suffixes not to emit another token or replay text. All 79 tokenizer unit tests and strict Clippy pass after this correction; original oracle-defined parity tests remain green.
