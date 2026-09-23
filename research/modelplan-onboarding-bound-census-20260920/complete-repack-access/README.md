# Bound access for complete internal repacks

Historical adapter stage at `caf52b441bbf11a5cf4ddf2624ff9c2d99184aea`.
The later [complete-repack identity stage](../complete-repack-identity-20260921/README.md)
implements identity for complete artifacts; the original validation below is preserved.

The existing internal manifest-repack source now implements bound metadata, tensor, optional native
representation and disk access for complete artifacts with no fallback or pruning mask. It uses
exact bound physical names and validates shape, storage, byte length, dtype, transform and auxiliary
accounting before reads. Original encoded bytes and the existing BF16-vector-to-F32 widening are
preserved; GGUF-block NVFP4 is not misrepresented as source-native ModelOpt planes.

Container interpretation is explicit: a complete repack uses GGUF tensor names/layouts but is not
a GGUF file. Runtime GGUF-only accounting and explicit spill admission therefore check the captured
container metadata, not the logical tensor-name dialect. Repack automatic disk-backed expert
selection remains enabled as before; bare GGUF ordinary loading remains pinned/pageable.

Sparse overlays and masks explicitly refuse bound compilation until their composite tensor contract
is implemented. The existing #542 `artifact_sha256` refusal is unchanged: no strict rewrite identity
or qualification is claimed for repacks, including complete artifacts, at this stage. Raw repack and
overlay behavior is unchanged, and these remain internal research artifacts, not a new public format.

Controls cover a complete Qwen MoE repack with Q8_0 expert banks and BF16 norm vectors, byte/dtype/
shape parity with the original source, opaque automatic disk views, retained bytes after source drop
and pathname replacement, container-specific accounting/spill behavior, missing/extra/wrong-shape
refusals, masked and fallback-overlay refusals, and unchanged strict-identity failure.

Full host GGUF/CLI: 360 GGUF passed (2 ignored), 13 CLI, 1 Step integration, 7 inspector, 3 external
preflight and 2 compile-fail doctests passed. All-target GGUF/CLI Clippy with warnings denied passed.
Linux-target engine/server lib/bin/test check passed with `DOCS_RS=1` documentation stubs. Formatting
and full stage whitespace check passed. Raw logs and source hashes are retained here.

This is source-adapter coverage only. Root activation, overlay/pruning composite semantics and
identity, scoped CPU expert access, other draft/repack paths, the latest reviewed upstream stack,
and native qualification remain pending. No GPU run, performance or positive support-state change
is claimed.
