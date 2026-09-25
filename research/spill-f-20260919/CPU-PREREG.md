# CPU-side M1 prerequisites: pre-registration of OWED 7 to 14 (2026-09-25)

Registered before the code, on the lead's order of 2026-09-25 ("work OWED 7 ... and then 8 to
13 as you planned, pre-registered, CPU-only, same rules"; 14 as a patch plus test in these
records). No GPU runs here: the local RTX 5090 is down and the rented box is not handed over.
Every item below lands CPU-verified; the GPU halves are named and stay owed.

## OWED 7: aligned over-read in the direct worker (engine)

Problem (census receipt `m1-prereg/direct-alignment-census-qwen36-35b.json`): `MEMRA_SPILL_IO=direct`
refuses every extent that fails `direct_extent_aligned(offset, len)` and the MoE cache then
serves it from mmap. On the pinned artifact that is all 31,488 slices, so the arm is mmap under a
direct label.

Change (`crates/memra-engine/src/spill_pread.rs`, one caller-visible behavior):

- A direct request for `(offset, len)` reads the enclosing window
  `[align_down(offset, 4096), align_up(offset + len, 4096))` through the O_DIRECT descriptor
  into the pinned buffer; the payload is `buffer[head .. head + len]` with
  `head = offset - align_down(offset)`. An aligned extent has `head = 0` and a window equal to
  the extent, so aligned reads are byte-for-byte the old program.
- Direct-mode pinned buffers are `align_up(capacity, 4096) + 4096` bytes, the proven bound of
  any window whose payload fits `capacity`. The payload capacity check is unchanged.
- The window read accepts EOF only after the payload is complete (a window may extend past the
  end of the file); an EOF inside the payload stays a short read.
- `bytes(index, len)` returns the payload slice at `head`; H2D copies exactly the payload.
- New counter `overread_bytes` (window bytes requested minus payload bytes, successful reads).
  Unaligned direct extents no longer produce a fallback; runtime I/O errors still do.
- No new env read, no new flag; `docs/FLAGS.md`'s `MEMRA_SPILL_IO` row is corrected in the same
  commit (it currently says unaligned direct extents fall back to mmap).

CPU gates (all must pass before commit):

1. Window arithmetic, exhaustively for every `head` in `0..4096` and payload lengths
   `1, 4095, 4096, 4097, 557056, 860160, capacity`: window aligned, contains the payload, and
   fits the buffer bound; overflow near `u64::MAX` refused, not wrapped.
2. Window reads on a real ext4 file opened with O_DIRECT (4096-aligned heap buffer): 2,000
   seeded random unaligned extents plus every extent of a synthetic GGUF-like layout
   (`base % 4096 = 1824`, slices of 450,560 / 557,056 / 860,160 bytes) return bytes identical
   to buffered `pread_exact_at` of the same extent; an extent ending at an unaligned EOF passes;
   an extent crossing EOF fails as `UnexpectedEof`.
3. Red control: the same unaligned extent read with O_DIRECT *without* the window is refused by
   ext4 (`EINVAL`), showing the window is what makes the arm reachable.
4. `cargo test -p memra-engine --lib spill_pread`, `cargo clippy -p memra-engine --lib` and
   `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh`.

GPU gates (owed to the box and to the 5090 after reset; `direct16` stays refused in B3 until they
pass): a new `#[ignore]` CUDA test driving `PreadPool` in `Direct` mode over unaligned extents
(bytes equal the file, `overread_bytes` equal the prediction, `fallbacks = 0`), then B3 step 1:
`run-gen` `MATCH`, 128 token ids identical to `mmap-random`, `fallbacks=0`, and
`overread_bytes == 4096 x reads` (every slice of the pinned artifact has `len % 4096 = 0` and
`head > 0`, so each window is exactly one block longer than its payload).

## OWED 8: per-stage spill counters (engine; no new env read)

`PreadStats` gains: `worker_read_ns` (wall time of positioned reads on worker threads, summed),
`demand_read_ns` (blocking `pread` mode), `wait_ns` (owner thread blocked on a worker completion
or on an H2D event to free a buffer), `h2d_submits`, and `overread_bytes` (from 7). The existing
7-field accessor stays; a new `Engine::moe_pread_stage_stats()` returns the full struct.
`run-gen` prints a `spill stages DECODE-WINDOW:` line beside its existing window line (and the
steady-state twin), and the server's existing `MEMRA_SPILL_STATS` snapshot appends the stage
fields. Gates: unit tests for counter accumulation on the CPU-visible paths, the build, clippy,
fmt, flags census. Clock reads are two `Instant::now()` per read, off the GPU stream.

## OWED 9: `storage-bench` stage timing

The frozen `StorageSample` wire struct is not changed. `io_ns` (today `None`) becomes the summed
wall time of `ExtentStore::read` calls. Write and commit time for `roundtrip`, and verification
time, go to one extra stderr line `[storage-bench] stages put_ns= commit_ns= read_ns= verify_ns=`
that the collector's storage-sample parser does not consume (checked against the parser before
landing). Gates: the existing storage tests, a run of both phases in all three modes on this
rig's ext4 with `io_ns` present, and the collector's parser on the new output.

## OWED 10: host/storage 250 ms sampler (`m1-host-sampler.py`, lane tool)

Samples `/proc/diskstats` rows for named devices, `/proc/meminfo` (MemAvailable, Cached,
Dirty, Writeback), `/proc/stat` aggregate CPU, `/proc/<pid>/io` of a target pid when alive, and
NVMe hwmon temperatures, as JSONL with monotonic and wall timestamps, until SIGTERM. A validator
reports gaps above 500 ms and missing devices. Gates: a 5 s run on this rig, validator PASS, and
red controls (a forced gap, an absent device).

## OWED 11: cache-regime helper (`m1-cache-regime.py`, lane tool)

`cold FILE...`: `POSIX_FADV_DONTNEED` per file, then `mincore` must report 0 resident pages
(retried a bounded number of times, then FAIL). `warm FILE...`: one full buffered read, `mincore`
must report every page resident. `residency FILE...`: report only. `balloon --bytes N
--floor-bytes F`: mmap plus `mlock` N anonymous bytes, refuse if `RLIMIT_MEMLOCK` forbids it or
MemAvailable would fall under the floor (errno quoted), hold until SIGTERM. No global cache drop,
no privilege. Gates: cold then residency 0 and warm then residency 100% on a lane-owned file on
this rig; balloon refusal under a too-small memlock limit (`prlimit` on the child) and a small
successful balloon.

## OWED 12: M1 cell runner (`m1-spill-runner.py`, lane tool)

Executes one regime of B3 under the collector: reads `m1-prereg/b3-arms.lock.json`, checks the
proof identity triple before and after each visit, applies the regime, starts the sampler, runs
the visit binary with the arm's env (common env first, `unset_env` removed) through `tee` into
the visit directory, stops the sampler, and parses only the saved log. Rounds alternate order as
registered. It writes one visit JSON per visit and a summary with the registered verdict rule.
Gates: a full dry run against a stub binary that prints the real `run-gen` line shapes (10
rounds x 6 arms), order and pairing checks, verdict-rule unit tests (winner, loser, flat,
sign-agreement edge), identity-mismatch refusal, and a raw-log-first check (parser never reads a
pipe).

## OWED 13: B2 handoff driver (design first)

Read `tools/kv-host-spill-identity-gate.sh` and the handoff export/import code; register the
exact driver (prompt set, fill sizes, drain signal, restart, import-complete detection, identity
comparison) as an amendment to `M1-PREREG.md` B2 before writing it. Gates: CPU dry run of the
script logic against recorded server log lines.

## OWED 14: collector storage label (patch plus test, routed by the lead)

`tools/tier-battery.py::capture_storage` marks a path NVMe when any `nvmeXnY` name appears in
`lsblk -s` of the findmnt source. Proposed change, delivered as
`owed14/tier-battery-storage-proof.patch` plus `owed14/test_storage_proof.py` (not applied to
the D-owned file here): a new `--storage-proof PROOF.json` argument; `capture_storage` requires
the M1 proof receipt (schema, `verdict=PASS`, `class=nvme-local-direct`, tool hash equal to the
checkout's `m1-nvme-proof.py`) and a live identity equal to the receipt's (device, mount id,
hashed filesystem id), and only then labels the root `nvme-local-direct`. Without a proof the
name match is relabelled `nvme-name-only-unproven` and needs `--allow-unproven-storage`. Gates:
the new test (green proof, stale identity, FAIL receipt, wrong tool hash, missing proof) and the
existing collector tests, on a scratch copy with the patch applied.
