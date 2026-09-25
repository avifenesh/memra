# Self-review: integ63 (A days 49 to 51: items 7 to 9, design P refuted and reverted; F's M1 build-out and the collector's proof binding)

Author's review of the full diff `main..lane/spill-integ63-20260925`, posted as a PR comment per the owner rule.

## What the diff is
- Server `worker.rs` (A, day 49): log-only split lines for the demote's pre-submit and the hash helper's copy and hash,
  printed only where a contract demote is pending. Design P (item 17) landed and was reverted on its (g) failure, so the
  net crate change from A is the split lines.
- Engine `spill_pread.rs` (F, OWED 7): the `direct` spill mode over-reads unaligned extents to their 4 KiB window
  instead of falling back to mmap. Opt-in; the default stays `mmap`.
- Engine `lib.rs`, `run_gen.rs`, `storage_bench.rs`, server `worker.rs`, `kv_handoff_gate.rs` (F, OWED 8, 9 and the B2
  seam): stage counters and timing, measurement only.
- `tools/tier-battery.py` (F's OWED 14, routed by the lead): NVMe labels only through a PASS proof bound to the live mount.
- `docs/FLAGS.md`: the `MEMRA_SPILL_IO` and `MEMRA_SPILL_STATS` rows. No new `MEMRA_*` name.
- Research: A's DAY49 to DAY51 and F's M1 tooling and receipts; the integ63 record (ruling 58), both batteries, this file.
- Main `cf6b82db4` (#727) is merged in, clean; both batteries ran on the merged tree.

## What I checked
- `direct_window` and `direct_buffer_bytes`: checked arithmetic, `None` on overflow, buffer sized for the worst head
  offset; staging starts at the head so the staged bytes are exactly the extent's, the same bytes mmap would give.
- The split lines: timing and `getrusage(RUSAGE_THREAD)` only, one `unsafe` with its ownership stated, the boxed split
  keeps the `Demoting` variant's size; no branch reads them.
- P's revert returns the crates to `e4de9c804`'s (the diff against main carries no reserve code).
- OWED 14 fails closed on a missing, failing or mount-mismatched proof, including under `--allow-unproven-storage`.
- The verdict lines in the record are copied from A's DAY49 and DAY51 and F's proof receipt.

## Batteries
- CPU battery 15 of 15 on the pre-merge tree, and on the merged tree 15 of 15 after the pytest step's rerun with the
  system interpreter (the post-reboot PATH `python3` lacked pytest): server 930, engine lib 573, portable 388 with 0
  skipped, pytest 87.
- GPU battery on an RTX PRO 6000 box with the same 9B model: every cell green, fault gate 255 ok per arm, the pause gate
  with the 27B `ALL GREEN` (40 ok).

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: opt-in modes and log-only lines. Revuto: if capped or unavailable, this comment
is the review.
