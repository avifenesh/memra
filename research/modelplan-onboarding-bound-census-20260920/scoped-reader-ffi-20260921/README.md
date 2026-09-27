# Retained scoped reader C ABI — primitive only

A prepared `BoundDiskReader` can now issue an owned Rust handle and a borrowed process-native
C ABI descriptor for its exact authorized extent. The descriptor offers retain, release,
range-checked positioned reads and file-generation/cache-key metadata. It exposes no descriptor,
mapping or widening operation. Metadata's absolute offset is informational: reads accept only
relative offsets validated against the sealed range.

A detached native consumer must retain while the original handle is still alive, then release
exactly once after all work drains. Foreign pointers and balanced ownership are the usual unsafe
C ABI contract; safe Rust cannot invoke the callbacks without an unsafe block. Read errors return
errno-style status and a zero count. Null/oversized/out-of-range requests are rejected before
constructing a foreign-memory slice. File generation metadata uses the same device/inode/size/ctime
fields needed by the existing CPU cache; it is not an artifact identity or a substitute for #542.
The ABI is provided on Unix, matching the native CPU backend's platform.

Validation:

- Five disk tests pass, including two new ABI tests. Two independently retained callbacks survive
  Rust owner drop, unlink/path replacement and worker-thread reads, reject range/overflow access,
  and release the last reference. Pointer/error and metadata controls pass.
- `tools/test-scoped-disk-ffi.py` compiles the committed C header and C consumer with warnings
  denied, links it against the actual Rust implementation and passes a cross-language control:
  C retains; every Rust file/source/bundle/view/handle is dropped; the pathname is replaced;
  C reads the original tensor bytes, verifies bounds/overflow failures and releases ownership.
  The final run copies the repository lockfile to preserve dependency versions. Both the first
  offline-resolution run and the lockfile-seeded final run are retained distinctly.
- Full GGUF/CLI tests: 362 GGUF pass (2 ignored), 13 CLI, 1 Step integration, 7 inspector,
  3 external source-authority controls and 2 compile-fail doctests pass.
- Host GGUF/CLI all-target Clippy and Linux-target GGUF all-target Clippy, warnings denied, pass.
  Linux-target code was typechecked, not executed. Formatting and whitespace checks pass.
- CI runs the actual C ABI control. Temporary harness projects are removed automatically.

This is a reader primitive, not completion of the native CPU descriptor migration. CPU numerical
kernels, cache/direct/mirror selection, detached prefetch adoption, error-drain paths and v2/v3
interoperation still need integration and qualification. No native CPU expert or CUDA numerical
execution, throughput result, root activation, support promotion or main merge is claimed.
