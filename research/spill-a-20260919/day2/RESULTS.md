# WP-A day-2 CPU milestone — 2026-09-19

Repository **avifenesh/memra**, branch `lane/spill-a-20260919`.
Source checked: **`03d15107093d20027e0c441d4af7f85f432597de`**.
Frozen lead `259ff819` merged with `--no-ff` in `97b3caf7`; **no conflicts**.
Local lane commits only: not integrated into main, pushed, deployed, GPU-qualified,
or permission to start model work. This continues, rather than closes, WP-A.

## Delivered

- Actual workspace `memra-tier` replaces the standalone scaffold package; duplicate
  contracts/lib/Cargo files deleted. Day-1 receipts remain intact in their dated files.
  Shared-file changes are exactly four A module exports and one storage test target.
  Shared contracts and all frozen schedule/fixture files are byte-unchanged.
- `object_store` implements the frozen synchronous `ObjectStore`: borrowed store-bound
  transactions, cancel-before/after-publication, canonical `ObjectManifest`, explicit
  governor-issued `ObjectLease`/`ChargedLease`, full manifest and chunk revalidation,
  foreign/released lease refusal and borrowed release. Eviction fails closed while
  leased and is otherwise Unsupported until process-shared GC ownership exists.
- Backend framing retains bounded 4096-byte headers / 1 MiB chunks, but uses distinct
  `extent-header`, `extent`, `valid-bytes`, and canonical object-key domains. Prototype
  binary roots are **not** treated as frozen v1. Padding and metadata canonicality
  remain validated. Metadata growth is bounded before another chunk is accepted.
- `FileBackend::open_mode(..., Persistent)` writes/fsyncs complete staging files,
  atomically publishes with no-clobber hard links, then fsyncs directory changes.
  The existing-file dedup path also syncs existing data and directory. A new store
  directory's parent is synced; its parent must already exist. Root publication is
  last. Reopen tests exercise actual files; simulated crash leaves a partial staging
  file plus unrooted chunks and cannot make an object visible. **This is not a real
  power-loss/faulty-controller qualification.**
- Aligned Linux O_DIRECT read path is cfg-gated for x86_64/aarch64, with fail-closed
  alignment/short-I/O handling and no hidden buffered fallback. Only the macOS path
  ran here. `F_NOCACHE` is in the A-owned CLI native adapter and injected as an opener:
  `memra-tier` retains its frozen `forbid(unsafe_code)` lint. A bare macOS tier-library
  uncached open without the native adapter refuses Unsupported.
- The CPU fake pinned pool implements frozen `PinnedLease`, pins an externally
  charged preallocated **pageable** backing pool once, returns slices only after
  initialization, and preserves demand headroom. CPU slots are not CUDA-pinned.
- Bounded readers now use issuer/sequence and all three epochs. `CpuTransfers`
  implements the shared TransferEngine host-read surface, complete indexed outcomes,
  homogeneous epoch triples, take-once host destinations, consumer lifetime protection,
  cancel-before-publication and acknowledged retirement tombstones. Fixed queue capacity
  pins a caller-provided inflight charge. Its explicit `drive()` is a **synchronous CPU
  work pump**, not something to invoke on a serving scheduler thread. It is not yet the
  async ObjectStore-to-BoundedReader bridge. H2D/D2H/P2P and device-ready views refuse
  Unsupported; no CPU readiness can authorize CUDA consumption.
- `storage_bench.rs` now runs real `roundtrip` and separate-process `restore` modes on
  persistent files, with bounded chunk readback and actual-byte streaming checksums.
  It emits frozen `StorageSample` JSONL; physical traffic/RSS/GPU times remain null.
  Its small ledger is a benchmark-only fixture, not WP-B's production governor.

## Conformance and exact outputs

Reproducer (all actual checks, no skipped gate represented as a pass):

```sh
python3 research/spill-a-20260919/day2/verify.py
```

`commands.json` records every exact command, exit and expected exit. `raw/*.log.gz`
retains unmodified stdout/stderr; `gzip -cd <file>` reads a log. Compression preserves
Cargo's trailing blank lines without whitespace-gate violations. No pipe drops stderr.
`source-manifest.json` binds the tested Rust source files and the actual CLI binary.

| Command | Actual result |
|---|---|
| `cargo fmt --all -- --check` | exit 0; empty output |
| `cargo check -p memra-tier --offline --all-targets` | exit 0; `Finished \`dev\` profile [unoptimized + debuginfo] target(s) in 0.94s` |
| `cargo test -p memra-tier --offline` | exit 0; 36 frozen + 26 storage tests + 3 compile-fail doctests; no ignored tests |
| `git diff --check` | exit 0; empty output |
| `bash tools/check-flags.sh` | exit 0; exact output below |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0; `Finished \`dev\` profile [unoptimized + debuginfo] target(s) in 0.87s` |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | exit 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| `cargo build -p memra-tier --offline` + standalone rustc link of actual engine-local CLI | exit 0; rustc output empty; exact arguments in commands.json |

```text
test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.38s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
check-flags: runtime literal reads=864
check-flags: no uncovered runtime names
check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)
CHARACTERIZATION_PASS: 12 single-run byte-exact rows; development-Mac I/O only; no spill-speed claim
```

`tests/storage/day2.rs` path-imports the unchanged lead `conformance.rs` and runs:

- `object_cancel` against the real persistent filesystem implementation.
- `transfer_cancel` against `CpuTransfers` with actual host leases, explicit drive and
  retirement. Other local tests cover take-once, each stale epoch, partial accepted/rejected
  vectors, missing chunks, post-publish cancellation and live-consumer retirement refusal.
- **There is no reusable PinnedLease schedule in the frozen file.** The concrete pool is
  tested through `&dyn contracts::PinnedLease`, including actual alignment/bytes and one
  governor charge retained until the last slice is released. This is not claimed as a
  nonexistent third shared schedule.

The original 15 storage tests were migrated, not dropped; fake retirement now also
requires graph completion. New tests add persistent reopen/interrupted publication,
canonical-root refusal, foreign transaction/lease checks, charge release after backing
loss, actual F_NOCACHE reads and short EOF, plus CLI positive/red modes.

Development red findings corrected before source commit: unsafe fcntl inside the
unsafe-forbidden tier crate failed compilation; syscall moved to native CLI adapter.
Strict Clippy flagged the large owned `SubmitError`; its explicit lint allowance mirrors
shared owned-rejection semantics rather than adding allocation on rejection. An unused
test import was removed. The receipt staging check caught CRLF in the SSH raw
log; it was losslessly gzip-archived before the receipt commit. Final receipts
contain no build warnings.

## Development-Mac I/O characterization — NOT memra spill speed

Storage inventory from selected `diskutil info -plist /` fields: internal solid-state,
APFS, Apple Fabric. `storage-shape.json` contains no serial/volume/device identities.
These are **12 single runs (N=1 per cell), unoptimized Rust, uncontrolled thermal regime,
no cache flush, no 250 ms telemetry, no interleaved A/B**. Restore is a new process after
roundtrip; it is not a cold-cache guarantee. Timings include hashes, metadata validation,
byte comparisons, syncs and copying. They do not isolate SSD bandwidth. No median,
tail confidence, performance winner/default, or serving throughput is asserted.

| Valid bytes | Read backend | Roundtrip total ms | Restore total ms | Submitted I/O bytes roundtrip / restore |
|---:|---|---:|---:|---:|
| 264 | buffered | 33.916 | 4.869 | 81,920 / 40,960 |
| 264 | macOS F_NOCACHE development fallback | 38.060 | 4.776 | 81,920 / 40,960 |
| 1,048,576 | buffered | 897.717 | 367.185 | 5,304,320 / 2,129,920 |
| 1,048,576 | macOS F_NOCACHE development fallback | 874.922 | 408.814 | 5,304,320 / 2,129,920 |
| 4,194,568 | buffered | 3,599.942 | 1,505.312 | 21,168,128 / 8,495,104 |
| 4,194,568 | macOS F_NOCACHE development fallback | 3,495.316 | 1,487.831 | 21,168,128 / 8,495,104 |

Raw schema rows: `runs.jsonl`, plus one exact stdout log per invocation. Independent
Python reproduces the frozen domain-framed checksum for every output. Six nonempty-dir
roundtrip retries and one unsupported GPU-mode invocation each exited 1 as required.
All characterization scratch was removed. The build products remain in ignored target/,
not a separate scaffold or scratch package.

## Remaining blockers / lead integration decisions

1. **Real hardware/native integration:** no nvcc/full-engine build, CUDA-pinned arena,
   H2D/D2H/peer/fence/address-stable graph, Hy3/PLE/Qwen consumer, Step ladder, PRO or
   four-card battery ran. Only installed target is `aarch64-apple-darwin`; Linux O_DIRECT
   is source-only here and still needs Linux filesystem/alignment qualification.
2. **Async/native adapter:** wire the bounded workers to store transactions and A/D
   owner-thread native transfers, then B/C consumers. CPU `drive` must not become a
   blocking serving-thread implementation. The requested device APIs are consumed but
   unsupported device paths fail closed, not simulated as supported.
3. **Budget/GC integration:** WP-B must supply the real shared governor. Current object
   reservations conservatively cover framed root/chunk lengths per lease; they are not
   deduplicated physical-disk residency accounting. Unreachable chunks are retained and
   eviction remains Unsupported; store-wide disk limits/shared-backing pins/coordinated
   GC and native allocator/metadata overhead accounting remain before production use.
4. **Engine wiring:** lead owns adding memra-tier dependency and explicit `storage-bench`
   bin entry. CPU tests and the rustc reproducer compile the actual CLI source without
   a duplicate contract package; a full engine build has not run on this Mac.
5. **Rig access:** one bounded `ssh -o ConnectTimeout=10 -o BatchMode=yes home hostname`
   attempt this round exited 255: `Connection closed by UNKNOWN port 65535`. No second
   attempt, remote inventory or remote action ran. Hardware cells in CELLS.md stay pending.

No new MEMRA_* read, flag/default, kernel, numeric program, model, public performance
board, lock name, serving instance, or shared definition was changed. Receipt/mesh-reserve/
memory/skill tools were not exposed to this worker; explicit delegated path ownership and
committed command logs are the coordination/evidence records. Read the available sibling
private GPU corpus via the doc router; the older ~/projects path was absent.

Effort: approximately **0.3 agent-hours** in this round (day-1 receipt records 0.4 hours),
against **7 agent-days** for WP-A. GPU time zero. This is a day-2 CPU milestone, not a
statement that the seven-day implementation/qualification budget is exhausted or done.
The lane stays open for the next milestone and lead integration; no worktree deletion yet.
