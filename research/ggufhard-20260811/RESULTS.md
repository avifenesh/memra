# GGUF parser error-discipline results

## Verdict

**PASS.** `memra-gguf` now rejects the requested corrupt and truncated GGUF classes with contextual `io::Error` values. The fixture suite observes no unwind from `GgufFile::open`, and valid single-file, split-file, micro-GGUF, safetensors, and source-path tests remain green.

## What changed

- `Cursor` fixed-width and variable-width reads are bounds checked. Truncation errors quote the byte offset, requested byte count, remaining byte count, and path.
- String, metadata-array, KV-count, tensor-count, and dimension allocations happen only after their declared counts are checked against the remaining file bytes; fallible reserve errors are returned.
- Negative structural counts/extents, impossible `n_dims`, element-count overflow, tensor-byte-size overflow, unknown metadata/tensor types, unsupported block layouts, and zero/overflowing alignment return `InvalidData`.
- Non-block-aligned tensor element counts return an error containing the tensor name, element count, and block size.
- Split version, `split.no`, `split.count`, and `split.tensors.count` mismatches return errors containing both observed and expected values.
- Every tensor's `data_start + offset + n_bytes` arithmetic is checked, then the resulting range is validated against its owning shard mmap during `parse_one`. The public zero-copy `tensor_data` and `tensor_file_range` APIs remain unchanged and only receive validated parser-produced ranges.

## Regression fixtures

The new hand-built fixtures cover:

- truncated fixed fields and a string truncated mid-payload;
- oversized `n_tensors`, negative `n_kv`, and an oversized metadata array;
- oversized `n_dims`, negative extents, non-divisible quant blocks, and element-count overflow;
- unknown metadata/tensor types, a known type without a loader layout, and zero alignment;
- tensor ranges past EOF and offset arithmetic overflow;
- split shard number/count/total mismatches;
- every incomplete byte prefix of a valid tiny GGUF, with the complete file as the positive control.

Each corrupt open is wrapped in `catch_unwind`: `Ok(Err(_))` is the only passing outcome.

## Evidence

- Baseline: `cargo test -p memra-gguf` — 78 passed.
- Pre-fix reproduction: the four-byte prefix panicked at the old unchecked read with `range end index 8 out of range for slice of length 4`.
- Focused debug gate: `cargo test -p memra-gguf split_tests::` — 12 passed.
- Focused optimized gate: `cargo test -p memra-gguf --release split_tests::` — 12 passed.
- Final warning-deny gate: `env RUSTFLAGS='-D warnings' cargo test -p memra-gguf` — 85 passed, 0 failed.
- Lane-scoped Clippy, with only the repository's established lint classes allowed — passed.
- `git diff --check` — passed.

Unfiltered `cargo clippy -p memra-gguf --all-targets -- -D warnings` is not currently a clean repository gate: it reports pre-existing findings in files outside this lane. No unrelated Clippy cleanup was performed.

## Compatibility and scope

- Existing valid-file tests pass unchanged, including byte/range assertions for single and split GGUF fixtures.
- No public API signatures changed.
- This was the briefed CPU-only parser lane. No GPU, model-quality, spill-performance, board, release, merge, or tag work was performed.
- `cargo fmt` was not run.
