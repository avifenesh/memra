# Opened-source and sealed-program identity composition

This stage composes the reviewed #542 source identity API with #541's sealed catalog. It follows
the actual dependency merge `ce0be1eb9b5186214e84b9f218b3abf5b71b8e7f` and does not activate a
bound eager root, promote support, or claim native qualification.

`BoundTensorSource::artifact_identity()` obtains the digest from the retained source's
`artifact_sha256()`, propagates errors, and rejects malformed SHA-256 strings. The source component
continues to cover every opened shard plus #542's captured/effective source interpretation. It is
never replaced by a pathname, supplied manifest, or the semantic digest.

The second component is the sealed config/plan/contract/binding/full-catalog/interpretation/scope
digest. Its domain is now `memra-bound-tensor-source-v4` because it also includes the captured
runtime source metadata. The composite uses `memra-bound-runtime-artifact-v1` and length-prefixes
both tags and values (`opened-source-sha256`, `semantic-scope-sha256`). Both component digests remain
available for inspection. The bound runtime's `artifact_sha256()` returns this composite.

`RuntimeSourceMetadata` supplies format, activation precision, expert-encoding preservation and
NVFP4 layout tag without file or path access. Binding snapshots it; the bound runtime's metadata
and policy accessors use that snapshot. Rewrite identity capture retains its previous interpretation
fields but obtains them through this handle-free API instead of `gguf()`/`st_dir()`.

`TensorSource::bound_program()` supplies the already sealed config and plan. Source preflight clones
that pair for a bound adapter, retaining normalized Step factors and the selected component plan.
It does not rerun legacy tensor lookup or recompile a different component view. Ordinary raw-source
preflight stays unchanged. #542 final post-load capture, mutation revocation, environment/library
checks and private NVFP4 backing are retained.

Validation:

- Full GGUF/CLI command: 358 GGUF pass (2 ignored), 13 CLI, 1 Step integration, 7 inspector pass.
- New controls separate payload changes from scope changes; retain both component hashes;
  reject unavailable/malformed source identity; survive config/checkpoint pathname replacement;
  cover unselected vision payloads while still denying their materialization; and reuse sealed
  Step and GGUF config/plan/format metadata. This is CPU/fixture evidence, not native qualification.
- GGUF/CLI all-targets Clippy with warnings denied: pass. A test-only redundant string conversion
  was corrected after the recorded first lint failure; final passing lint output is retained.
- Existing actual-source runtime identity/snapshot host harness: 18 pass.
- Linux-target engine/server lib/bin/test Clippy with warnings denied and `DOCS_RS=1`: pass.
  Documentation stubs were used; CUDA code was not executed.
- Formatting and the complete stage diff whitespace check: pass.

Raw commands/results and intermediate failures are preserved as lossless gzip with hashes.
Remaining scope includes GGUF metadata and spill-context consumers, scoped CPU expert access,
other source/derived/standalone draft paths, incoming storage corrections, and final source/native
gates before root activation. Existing native Step vision remains unchanged.
