# Explicit GGUF spill placement through scoped views

The source API separates explicit GGUF spill requests from automatic disk-backed expert selection.
For raw and bound GGUF sources, ordinary stacked loading retains the existing pinned/pageable
policy. Available physical backing alone no longer selects mmap. Explicit `try_find_gguf_disk`
returns the authorized exact tensor view; unknown/denied bound names still fail, and an unselected
or transformed view cannot be widened into an unrestricted mapping.

`SpillCtx` now owns only the shared pinned budget and placement counters. Tiered expert loading
gets its tensor through the source and creates checked per-expert windows. It pins while the
same budget covers a window, then retains that window as disk backing. The old direct-GGUF entry
point delegates to the same implementation. Counters advance only after allocation/placement
succeeds. The hybrid root no longer requests an unrestricted `GgufFile` for spill setup; its
GGUF-only flag admission and other-format storage policy are unchanged.

Spill windows retain the parsed mapping and opened inode. The configured random/normal mmap
advice is applied when a projection view is selected rather than by mapping entire shards again.
No whole-map population is introduced. This timing/mapping change still needs native spill
correctness and performance qualification; these receipts do not assert equivalent latency.

Validation:

- Full host GGUF/CLI: 358 GGUF pass (2 ignored), 13 CLI, 1 Step integration, 7 inspector,
  3 external preflight controls, and 2 compile-fail doctests pass.
- GGUF/CLI all-target Clippy with warnings denied passes. Bound controls verify automatic GGUF
  disk selection remains absent while explicit reads retain bytes; opaque/raw subrange tests
  verify limits. Existing scale/layout/scope controls remain included.
- Linux-target engine/server lib/bin/test Clippy with `DOCS_RS=1` and warnings denied passes.
  Engine context ownership tests are typechecked, not executed on this host.
- New ignored CUDA gate `model::tests::scoped_gguf_spill_preserves_bytes_budget_and_ordinary_host_policy`
  compares raw/scoped exact encoded expert bytes, qtype/stride, zero/partial/full pin budgets,
  scoped disk windows and ordinary host placement. It has NOT run. Execute only after source and
  rig admission, alongside worker/H2D and existing numeric/serving gates.
- Formatting and full stage whitespace checks pass. Initial typecheck failure and every final
  passing log are retained losslessly with source hashes.

Root activation is still held. CPU expert scoped access, repack/draft/other-source coverage, the
next reviewed #542 stack and final source/native gates remain. No GPU, throughput, support, main
merge, or universal-loader completion is claimed.
