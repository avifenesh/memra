# Foundation review repairs

The independent review of `34b083f7f5fd1cee6ed9969764623f70432d0cbd` found four
material API/contract defects, covered by five counterexamples. All five fail on that frozen
production source and pass after this repair. Runtime activation remains pending and this
record conveys no native or merge approval.

| Finding | Repair | Controls |
|---|---|---|
| Source-declared interpretation omitted from digest | Capture the opened source's scale layout, quantization/activation policy and dtype-preservation declarations in the immutable bundle; include them in digest domain v2. | Same bytes with different scale declarations now produce different digests. Correctly swizzled and linear artifacts produce identical decoded scale bytes. Changing a file after opening does not change the captured interpretation. |
| Malformed input-scale shape accepted | Require a floating scalar for a matrix, or a scalar/per-expert vector for a stacked bank, during metadata binding. | `[7]` for a matrix refuses; valid scalar `[]` and `[1]` compile and materialize. |
| Native validation inconsistent with auxiliary reads | Validate folded scale values and effective reciprocal before optional native eligibility. Stream over planes without a full extra scale copy. | Divisor reciprocal overflow and NaN under a non-native transform both error; a valid transformed operand still returns optional native absence. |
| AWQ input-axis scale left in checkpoint column order | Reuse the weight's V-head column permutation for scalar byte offsets, preserving the scale's encoding. | Nonuniform F32 scale follows the permuted weight columns; BF16 bytes preserve dtype/length and follow independently specified head order `[0,2,1,3]`. |

The interpretation digest remains distinct from opened-artifact identity. The later #542
integration must compose its exact opened-byte hash with this digest under a separate domain;
current runtime/environment guards remain separate too.

Validation, local Rust 1.97.1:

- Five frozen-head counterexamples: **5 expected failures**, all intended assertions reached
  (`raw/red.log.gz`).
- Bound-source suite including repairs and positive controls: **21 passed, 0 failed, 0 ignored**
  (`raw/bound-tests.log.gz`).
- Full GGUF/CLI suite: **316 GGUF library test functions completed, 2 explicit ignores; 11 CLI,
  1 Step contract and 7 GGUF-inspect tests passed** (`raw/compiler-suite.log.gz`). As in the
  foundation record, artifact-dependent early returns are not native artifact qualification.
- Clippy all targets for GGUF/CLI with warnings denied: passed (`raw/clippy.log.gz`).
- Linux target engine library/bins/tests and server type check: passed with `DOCS_RS=1` and the
  separate `target/541-linux-typecheck` directory (`raw/linux-typecheck.log.gz`). These are CUDA
  documentation stubs, not native CUDA binaries.
- Formatting and diff whitespace checks passed.

The inherited macOS Qwen3.5 reference-bit mismatch remains recorded in the original
[validation](../VALIDATION.md). Neither its arithmetic nor its tolerances changed.

`raw/regressions.rs` preserves the exact five review repro additions. `raw/sources.sha256.json`
and `raw/files.sha256.json` bind this repair source and its raw evidence. Original foundation
receipts remain unchanged. Independent rereview of the repair is required before approval.
