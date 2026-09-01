# GGUF parser error-discipline progress

## 2026-08-11 — lane start

- Confirmed worktree `/home/avifenesh/projects/wt-cx-ggufhard` is on `lane/cx-ggufhard` and clean.
- Read the lane brief and `/home/avifenesh/projects/bw24/CLAUDE.md` before implementation.
- Scope is CPU-only hardening of `memra-gguf`: corrupt or truncated files must return contextual `io::Error` values instead of panicking, while valid-file behavior remains byte-identical.
- Planned proof: targeted hand-built corrupt/truncated GGUF fixtures, including unwind guards, followed by `cargo test -p memra-gguf`.
- Constraint acknowledged: do not run `cargo fmt`.

## 2026-08-11 — parser hardening implemented

- Baseline `cargo test -p memra-gguf`: 78 passed.
- Pre-fix fixture reproduced the unchecked-read panic at `lib.rs:191`: `range end index 8 out of range for slice of length 4`.
- Added six fixture tests covering truncated fixed fields and strings, impossible counts and arrays, invalid dimensions and tensor sizes, unknown/unsupported types and zero alignment, tensor ranges beyond/overflowing EOF, and split metadata mismatches. Every corrupt open is guarded with `catch_unwind` and must yield `Err`.
- Replaced parser slice indexing, unchecked integer casts/arithmetic, allocation-before-bounds-checking, unknown-type panics, block divisibility assertion, and split merge assertions with contextual `io::Error` returns.
- Tensor ranges are now checked against the owning shard mmap during `parse_one`, before the infallible zero-copy accessors can be reached.
- Focused parser/split gate: 12 passed in debug and release modes.
- Added an exhaustive tiny-fixture truncation gate: every incomplete byte prefix returns `Err` without unwinding; the complete control fixture still opens.
- Final `env RUSTFLAGS='-D warnings' cargo test -p memra-gguf`: 85 passed, 0 failed.
- A lane-scoped Clippy pass is clean after allowing the repository's established lint classes. Unfiltered strict Clippy still reports unrelated pre-existing findings in `dequant.rs`, `config.rs`, `hf_mapping.rs`, `nvfp4_repack.rs`, and `source.rs`; none were changed.
- Final diff/API audit complete: public zero-copy access signatures are unchanged, parser-produced ranges are proven safe at open, and no parser path calls the remaining infallible block-layout panic.
