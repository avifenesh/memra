# OWED 7: aligned over-read in the direct worker (2026-09-25)

**Landed CPU-verified; GPU-unqualified.** `MEMRA_SPILL_IO=direct` now reads an unaligned expert
extent through its enclosing 4 KiB window instead of refusing it into the mmap fallback. B3's
`direct16` arm stays refused until the GPU gates below pass. Registration: `CPU-PREREG.md`.

## What changed (`crates/memra-engine/src/spill_pread.rs`, final SHA-256 `c528a67f...a7cf6cb0`)

- `direct_window(offset, len)` gives the aligned window and the payload's head; an aligned
  extent is its own window, so aligned direct reads are the old program.
- Direct-mode pinned buffers are `align_up(capacity, 4096) + 4096` bytes (`direct_buffer_bytes`).
- Workers read the window with `pread_window_at`, which accepts EOF only after the payload is
  complete; exact reads (every non-direct read and aligned direct reads) keep `pread_exact_at`.
- `bytes(index, len)` returns the payload at the buffer's head; H2D copies exactly the payload.
- `PreadStats.overread_bytes` counts window bytes beyond the payload on successful reads.
- `docs/FLAGS.md` `MEMRA_SPILL_IO` row corrected. No new env read (flags census clean).

## CPU gates (raw logs in `owed7/cpu/`)

| Gate | Result |
|---|---|
| 1 window arithmetic: every head 0..4095, 8 payload lengths, 4 block offsets; overflow refused | PASS (`direct_window_is_aligned_contains_payload_and_fits_buffer`) |
| 2 O_DIRECT window reads on ext4 vs buffered `pread_exact_at`: GGUF-shaped layout (base 10,991,392, three slice sizes), 2,000 seeded random extents, an extent ending at an unaligned EOF; an EOF-crossing extent is `UnexpectedEof` | PASS (`direct_window_reads_match_buffered_reads`) |
| 3 red control: the same unaligned extent through O_DIRECT without the window is `EINVAL` on ext4 (filesystem checked by statfs magic; the skip branch did not run) | PASS (`unaligned_direct_read_without_window_is_refused`) |
| 4 `cargo test -p memra-engine --lib spill_pread` | `test result: ok. 9 passed; 0 failed; 2 ignored` (before and after `cargo fmt`) |
| 4 `cargo clippy -p memra-engine --lib --tests -- -D warnings` | exit 0 |
| 4 `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh` | exit 0, clean, "no uncovered runtime names" |

Build: `MEMRA_CUDA_ARCH=120a` set explicitly so `build.rs` never probed the down card;
`CARGO_BUILD_JOBS=12` under `CPUQuota=1200% MemoryMax=20G`.

## Census prediction (the exact qualification check)

`m1-prereg/direct-alignment-census-qwen36-35b.json` now also computes each slice's over-read:
all 123 expert tensors, all 31,488 slices, over-read exactly **4,096 bytes** per read
(`len % 4096 = 0`, `head > 0`); one read of every slice over-reads 128,974,848 bytes (0.83% of
the 15,600,713,728-byte bank).

## GPU gates owed (box first, the 5090 after its reset)

1. `cargo test -p memra-engine --lib spill_pread::tests::direct_worker_overread_preserves_exact_bytes -- --ignored --exact`
   under the rig lock: pinned-pool bytes equal the file, `read_errors = 0`, `fallbacks = 0`,
   `overread_bytes` equal to the per-extent prediction.
2. B3 step 1 for `direct16`: `run-gen` `MATCH`, 128 token ids identical to `mmap-random`,
   `fallbacks=0`, and `overread_bytes == 4096 x reads` from the stage counters (OWED 8).
