# Engine canonical-output consumer

Builds on reviewed `853099a12231bd1bd4b4a9e65afee66a3fd86360`. The TensorSource seam
`try_canonical_nvfp4_bank` returns None for ordinary unbound sources. BoundRuntimeSource uses its
semantic ABI and the retained canonical producer; missing, unsupported and failed bound requests
return errors and cannot fall through to pathname caches.

The production repack module checks the returned extent against the engine's expected byte count.
Both stacked and per-expert NVFP4 disk paths consume that opaque view before constructing any
legacy cache path. Existing macro values, row geometry, pin budgets and pinned-prefix copies are
preserved. Unpinned expert ranges remain bounded subviews. Whole-slab population and access advice
stay behind their existing policies; a partial view is never widened to its containing file.
Memory-gather mode no longer creates an unused cache directory. Raw strict and legacy cache
behavior remains covered by its original nine tests.

A tenth test drives actual fully bound safetensors through the production repack helper. It checks
codec bytes, raw-source None, size mismatch, unknown/non-native bound refusal, detached reads,
bounded tails, and propagation of output-root failures. The CPU harness includes the production
module directly and reuses the existing target with CARGO_INCREMENTAL=0; no broad target copy or
CUDA linking is needed. The exact harness inputs, logs, source hashes and preserved executable
hash are in evidence.json.

Frozen checks pass: production harness10/0ignored, GGUF420/2declared ignores, strict Linux
DOCS_RS engine-library Clippy, format and diff. Inherited artifact-dependent early returns do not
become checkpoint evidence. These are CPU/type checks, not executed pinned CUDA copies or native
qualification.

Dense/hybrid root entrypoints, plan_backend fallback/admission, worker, configuration and shared
decoder seams are unchanged. No #542 WIP or #537 latest-main composition was imported. Accepted
main LRU/removal behavior remains a parent integration obligation outside this delta.
Next acceptance is independent source review; root adapter adoption, external/trimmed drafts and
coordinated native numeric/serving/worker/H2D/I/O qualification remain pending. No root/native/model
support, performance-default or main promotion is asserted.
