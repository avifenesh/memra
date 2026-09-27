# Bounded views through expert loading and spill workers

The fallible expert-disk API now distinguishes raw-source extents from compiler-bound views.
The bound runtime returns only opaque views, and its legacy mmap API refuses raw handle access.
A presence-only probe retains the old-source guard without exporting the mmap.

The engine stores a bound slab in `HostBuf::Bounded`, takes checked per-expert subranges, and
retains the backing until asynchronous transfers complete. An opaque process-local backing token
supports lifetime deduplication; it is not an artifact hash or persistent identity. Uniform banks
coalesce only through an adjacency check that preserves scope and backing. Mixed banks retain
individual windows and existing expert-layout metadata and pruning behavior.

The blocking and worker spill readers accept either source class. A worker request owns a prepared
reader; bounded requests retain the authorized range rather than a file descriptor. Buffered,
random-advice and direct modes preserve their I/O choice. Ticket admission, cancellation, pinned
buffer phases, owner-thread H2D, and publication are unchanged. The mmap fallback sees only the
selected expert window. Raw-source CPU expert ABI v2 remains unchanged. A bounded CPU expert
request explicitly refuses until a scoped CPU read adapter is implemented, rather than exporting
a descriptor or changing the CPU backend to pointer-only I/O.

## Validation and limits

- 43 host bound-filter library tests passed. The Step FP8 test now checks runtime disk access,
  verifies the opaque arm and exact native bank bytes, rejects a read beyond the bank, and checks
  declared absence versus unknown-name error. Disk join tests reject mixing bound and raw arms.
- `cargo clippy --locked -p memra-gguf --all-targets -- -D warnings`: passed.
- `DOCS_RS=1 cargo check --locked --target x86_64-unknown-linux-gnu
  --target-dir target/541-linux-typecheck -p memra-engine --lib --bins --tests
  -p memra-server`: passed (`linux-3` log).
- `DOCS_RS=1 cargo clippy --locked --target x86_64-unknown-linux-gnu
  --target-dir target/541-linux-typecheck -p memra-engine --lib --tests
  -p memra-server --no-deps -- -D warnings`: passed on the final source.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

Linux engine checks use documentation stubs. Existing engine and CUDA worker regressions have been
adapted and typechecked, but were not executed here. Native worker/H2D and direct-I/O gates remain
required before activation. Intermediate compile failures are retained alongside the final passing
logs; all logs are losslessly compressed with byte hashes.

No root loader is activated by this commit. Remaining integration includes the scoped CPU expert
adapter, GGUF metadata/SpillCtx paths, other source/derived-view coverage, standalone MTP/student
paths, source-byte identity composition with #542, and final native/source gates. No GPU work,
performance result, native qualification, merge, or whole-issue completion is claimed. Existing
Step vision implementation and its historical qualifications remain unchanged.
