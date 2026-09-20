# Tokenizer input programs

The HF byte-level BPE loader preserves `tokenizer.json.normalizer` and added-token matching flags. Missing or null normalization means identity. Supported declarations are NFC and Sequence wrappers containing only NFC or further supported sequences; an empty Sequence is identity. Other programs and malformed declarations refuse at load, including when the experimental unknown-pretokenizer option is enabled. Programs are limited to 32 levels and 256 stages. The JSON parser independently limits value nesting to 128 levels before constructing the program.

NFC uses `unicode-normalization-alignments` 0.1.12, the implementation and Unicode 9.0 data used by the pinned HF `tokenizers` 0.22.2 reference. Added-token word boundaries use Unicode 16.0 regex word categories. The program is retained by `Tokenizer::normalization_program()`; `normalizer::NFC_UNICODE_VERSION` identifies the normalization tables. Updating these interpretations requires fresh parity evidence.

Encoding proceeds in this order:

1. Extract `normalized=false` added tokens from raw input.
2. Normalize each remaining text span independently.
3. Extract `normalized=true` added tokens, using normalized matching content.
4. Apply the declared pre-tokenizer and BPE to remaining text.

A raw added token prevents composition across its span. `special`, `normalized`, `single_word`, `lstrip`, and `rstrip` have independent meanings. Matching is leftmost-longest within each stage. Literal special recognition is controlled by `parse_special`; BOS/postprocessing is separately controlled by `add_special`. The original added-token declaration and ID are retained even when its matching spelling is normalized.

Special declarations precede ordinary additions; declaration order is preserved within each class, including when distinct spellings normalize to the same content. A whitespace-adjusted empty match emits no ID.

## Explicit GGUF import

Existing GGUF artifacts without an input-program declaration use their established GGUF tokenizer behavior and identity normalization. Family names such as `qwen2` never enable NFC.

Memra defines the optional **STRING** metadata key `tokenizer.memra.input_program`. This is a Memra extension, not a general GGUF normalization key. Its JSON object has:

- `version`: integer `1`;
- `normalizer`: the explicit HF declaration, or `null` for identity;
- `added_tokens`: the complete HF added-token declarations with their IDs and flags.

For example, a converter can form the value from validated HF metadata:

```python
program = {
    "version": 1,
    "normalizer": tokenizer_json.get("normalizer"),
    "added_tokens": tokenizer_json.get("added_tokens", []),
}
value = json.dumps(program, ensure_ascii=False)
```

The BPE vocabulary and merges remain in their ordinary GGUF fields. Every imported ID/content must exactly match the GGUF vocabulary. Token classes must also agree: special entries are Control/Unknown, ordinary additions are UserDefined. Every nonempty Control/Unknown/UserDefined entry must be represented, and duplicate imported IDs refuse. A converter must populate `tokenizer.ggml.token_type` consistently rather than discarding ordinary added-token behavior.

Version 1 supports byte-level BPE input programs; SPM imports, unknown versions, unsupported normalizers, incomplete declarations and conflicting vocabulary/classes refuse. A converter promising HF input-ID parity must preserve this declaration and verify raw-input parity with a consumer that understands it. Silently dropping NFC or relying on another reader to interpret an unknown extension is not a parity guarantee.

Onboarding validates an explicit declaration using the runtime tokenizer and includes its exact bytes in a domain-separated tokenizer fingerprint. Normalization and added-token flag changes therefore change the tokenizer identity. Artifacts without the extension retain their existing fingerprint.

## Validation

`cargo test --locked -p memra-tokenizer --all-targets` includes frozen, independently generated HF 0.22.2 expectations: 315 cases across 31 variants through HF and explicit GGUF imports, special-recognition/postprocessing controls, unsupported-import cases and unchanged no-declaration behavior. The pinned real-vocabulary qualification separately exercises 551 raw multilingual/adversarial inputs and the real special-token matrix.

The expected token IDs were captured before the implementation. Decoder strings and offsets in the capture are diagnostics; this change qualifies input normalization/matching and does not add an offset API.
