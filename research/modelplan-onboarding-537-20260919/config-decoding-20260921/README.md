# Canonical config decoding repair — 2026-09-21

The previous config scanner did not decode JSON object keys consistently. A literal
`"hidden_act":"relu"` was refused, while the equivalent `"hidden\u005fact":"relu"`
could disappear and select the default SiLU program. Duplicate keys, malformed JSON,
and wrong field types could also become defaults.

The repair normalizes the single tree returned by `strict_json::parse_object` through
`HfConfig::try_parse`. Source loaders, CLI inspection/verification, the Qwen4Exp loader,
and checkpoint correctness runners propagate parsing errors. The trusted-fixture
`HfConfig::parse` wrapper also uses this decoder and fails closed. The adjacent
`LAYOUT.json` reader now uses the same decoded-key contract and refuses unknown layouts.
Independent repack-manifest consumers remain owned by #541; their integration must take
the helper and its Cargo features together.

## Numeric and schema behavior

- JSON number tokens retain their decimal spelling until the field selects its type.
  Scalars and array elements parse directly to `f32`, preserving rounding, subnormal
  behavior, and negative zero. Non-finite `f32` values refuse.
- Unsigned config fields use exact decimal/exponent arithmetic. Whole spellings such
  as `2.0` and `20e-1` are accepted; fractional values, negative values, overflow, and
  rounded fractions refuse. Manifest consumers can still use `Value::as_u64` to reject
  all float/exponent/string/null spellings, including an integer-valued float.
- Actual objects named `$serde_json::private::Number` stay objects; they cannot forge
  number tokens or bypass decoded duplicate-key checks. Nesting is bounded.
- Null is accepted only for the optional fields identified in `ConfigObject::value`.
  Other consumed declarations are typed. Top/text precedence is retained.
- GLM-DSA's declared scalar `moe_layer_freq: 1` matches its existing every-layer MoE
  schedule after the dense prefix. Other scalar intervals refuse. MiniMax's per-layer
  array is retained. Schema selection follows the effective text model type.

The decoder requires serde_json `raw_value` and `arbitrary_precision` atomically;
`preserve_order` feature unification is covered as a separate CPU test mode. The two
reference-tensor `usize` annotations resolve inference made ambiguous by the new
serde_json dependency; they do not change arithmetic.

## Evidence and limits

Base: `ac3c39712486170ce2beb77527df50eede07070c`. The frozen native16 results remain bound
to their historical source and executables. This patch has no native result and does
not promote support states, formats, hardware defaults, or performance numbers.

`evidence.json` binds changed code, commands, raw-record hashes, public-log hashes, and
the literal-config corpus comparison. `logs/` retains the CPU output. Public copies
replace local workspace paths and Rust executable build-id suffixes, and remove
trailing blank lines for the repository whitespace gate; original
bytes and their hashes remain in the private audit archive.

Five synthetic literal plan hashes are pinned in
`crates/memra-gguf/tests/fixtures/canonical-config-plans.tsv`. Five checked-in model
configs (GLM-5.2, GLM5-Next, Qwen4Exp tiny/full, and Step) produce byte-identical
normalized configs and plan hashes against the base. This is config evidence, not a
checkpoint numerical or serving qualification.

The compiler/CLI tests, both map-order modes, strict Clippy, and Linux-target
engine/server type checks pass. Linux-target checks use `DOCS_RS=1`, so they do not
compile CUDA or execute Linux/GPU binaries. The related CPU suites reproduce the
known macOS Qwen3.5 mixed-GDN reference fixture mismatch, with the exact same bit
vectors as an independently built base control. Every outcome is retained; no
tolerance or expected result was changed. The GGUF skip census distinguishes the
artifact-dependent early-return cases and ignored tests from executed regressions.

Before integration: independently review this source and its shared-helper feature
composition, then run fresh Linux/native qualification on the coordinator-selected
combined source. Do not reuse historical binaries or relabel their receipts.
