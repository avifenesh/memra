# Complete internal repack opened-artifact identity

Complete internal repacks now derive artifact identity from the exact opened manifest/config,
all mapped shard bytes, their manifest-relative owners, the effective normalized configuration
(including stripped MTP and full prefill scale bits), and expert activation precision. Config and
activation precision come from one retained config snapshot. The catalog also receives that raw
config snapshot for additional declared inventory. No pathname is reopened while hashing.

The manifest is hashed byte-for-byte: whitespace edits and file renaming conservatively invalidate
identity. Each headerless shard's digest is paired with its declared file name; a sorted multiset
would miss two files swapping payloads while the manifest remained unchanged. Full mappings include
padding and bytes outside tensor windows. The bound adapter combines this opened-source identity
with its existing semantic/scope digest. This does not add an overlay or pruning contract: any
fallback or mask still refuses identity and bound compilation.

Controls exercise path replacement/unlinking of weights, config and manifest; changed tensor names,
offsets, shapes, encodings and expert strides; changed config/activation interpretation and effective
config; swapped shard ownership; modifications outside declared windows in each shard; bound/runtime
identity composition; and continuing mask/fallback refusal. Original Q8 expert bytes, BF16 vector
widening, and opaque retained disk access remain covered by the complete-repack regression.

Validation: 365 GGUF unit tests pass (2 ignored), 13 CLI, 1 Step integration, 7 inspector,
3 external authority controls and 2 compile-fail doctests pass. Four focused repack-identity tests,
18 actual-source runtime identity/snapshot host tests, GGUF/CLI all-target warnings-denied Clippy,
Linux-target engine/server lib/bin/test Clippy with DOCS_RS=1 documentation stubs, formatting and
whitespace checks pass. Exact source hashes, command statuses, test binary hashes and lossless logs
are in evidence.json and raw/. The documentation-stub check is not CUDA execution.

Independent source review passed on exact `82a28b225557f17ff80bb9a8daf08dec33daa752`;
the reviewer independently ran four identity, two bound and 18 runtime identity/snapshot controls.
The lossless report, logs and verification proof are under `review/`. This is a complete internal
repack identity implementation,
not a new public format, model support promotion, native qualification or root-loader activation.
Sparse overlays/pruning, standalone draft composites and canonical native repack consumer boundaries
remain separate acceptance work. The coordinator's final #542 dependency ref has not been substituted.
