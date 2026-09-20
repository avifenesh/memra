# WP-A day 5 — Linux direct I/O and pinned-worker development evidence

Repository: **avifenesh/memra**, branch `lane/spill-a-20260919`.
Native source: **`fd49a668ca30a63808c7bad6e7612c38445c3b82`**.
Native storage-bench SHA256:
`916e9168a2c3692767274edad015c86421404bff1c41985becb7d88b5a28f19b`.
Final check base: `2fbf010c` plus the test/receipt additions hash-bound in
[`final-checks/checked-source.json`](final-checks/checked-source.json).
The final `git diff --exit-code fd49a668 -- crates/` passed: native Rust source
has not changed since the box build. This is a pushed lane milestone, **not a
merge, release, model-serving qualification or deployment**.

## Inherited → finished

Resumed five dirty items at `4d56547c`: storage `day4.rs` timing output,
`rig-cells-a.sh`, `box2_cells.py`, `storage_capture.py`, and `day5/`.
Finished all five, with no unrelated changes absorbed:

- Actual monotonic timings for the 200-GB logical metadata fixture.
- One-cell overlay launcher; canonical collector wrapping for collected cells;
  per-cell mount-type, compute-app and **400 W limit / 600 W maximum** records.
- Lane-local, hash/run-ID/command-bound StorageSample/telemetry capture join.
  Explicit `--schema storage-cell` **patch fragment only** for WP-D; shared
  `tools/tier-battery.py` unchanged. See [ADAPTER.md](ADAPTER.md).
- Archived-capture refusal/replay tests, isolated real CLI patch test and eight
  native direct-capture replays: **13 tests pass**. No fabricated token/logit
  pair, tier counter, physical byte count or hardware qualification.

Each box cell was synced, committed and pushed before the next. Two initial
GPU-occupied admission refusals remain archived; they executed no storage
command. After an exclusive slot was granted, all nine collected cells ran
serially under `/tmp/memra-5090.lock`. No serving artifact was accessed.

## O_DIRECT-on-overlayfs verdict

**O_DIRECT read/write was accepted on this overlay filesystem. All eight
roundtrip/restore cells returned byte-exact data with zero fallbacks. No EINVAL
was observed. This does not prove physical NVMe ancestry or spill throughput.**

Verbatim fields present in every sample:

```json
"backend_requested":"direct","backend_actual":"linux-o-direct-read-write","status":"byte-exact"
```

Every sample also reports `"physical_bytes":null` and `"fallbacks":0`.
The executable performs bounded byte-by-byte readback verification and checksum
construction; restore opens the persisted object in a separate process. The
matching roundtrip/restore payload checksums are independently replay-checked.
These are CPU filesystem cells conservatively run through the GPU-lock
collector, **not H2D or GPU-compute benchmarks**.

Raw cells and sample/capture joins:
[`../rented-5090-20260919/day5/`](../rented-5090-20260919/day5/).
All timings below are **N=1 per cell**, raw `total_ns`, not medians. No thermal
steady-state was established. Regime: idle-device admission, 400-W-capped RTX
5090, overlay development characterization; **not spill speed**.

| Valid bytes | Padded bytes | Roundtrip ns | Restore ns | Result |
|---:|---:|---:|---:|---|
| 264 | 4,096 | 5,569,133 | 575,606 | byte-exact, zero fallback |
| 4,097 | 8,192 | 7,380,322 | 980,218 | byte-exact, zero fallback |
| 1,048,576 | 1,048,576 | 50,765,996 | 23,171,543 | byte-exact, zero fallback |
| 4,194,568 | 4,198,400 | 104,761,303 | 58,191,860 | byte-exact, zero fallback |

Five short cells have empty GPU CSVs; three have captured-but-unvalidated CSVs.
Both remain diagnostic, hash-bound evidence, not positive 250-ms tier telemetry.
All eight lane-local joins pass; unknown physical/H2D/D2H/P2P counters stay null.

## Native tests and verbatim tails

Storage-bench release build: exit 0, native CUDA 13.1 / architecture 120a build.
Worker unit-test binary build: exit 0, `--no-run` (execution is separate below).

Linux `cargo test --release -p memra-tier --test storage -j 16`:

```text
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.35s
```

Focused PR519 GC regressions were also re-run, not counted as new distinct tests:

```text
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 47 filtered out; finished in 0.02s
test object_store::tests::review_gc_upgrade_window_keeps_other_stores_lease_fenced ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s
```

Standalone sharded-catalog metadata fixture, single-run timing:

```text
CATALOG_METADATA {"logical_bytes":209715200000,"metadata_bytes":22408192,"touched_metadata_bytes":41408,"payload_reads":0,"install_ns":383600887,"touch_ns":247163,"qualification":false}
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 52 filtered out; finished in 0.39s
```

This fixture represents 209.7 GB **logical metadata only**; actual index bytes
are 22.4 MB. It neither allocates nor reads a 200-GB payload.

Pinned-worker CUDA-required test, collector-locked, native execution:

```text
[spill-worker] WARNING: effective depth 2 < 6; grouped current+next overlap is degraded; continuing with projection concurrency and mmap fallback
[spill-pread] enabled: depth=2 buffer_bytes=64 total_pinned_bytes=128 (bounded worker prefetch, caller-thread compute-stream H2D, mmap error fallback)
[spill-pread] reads=8 bytes=175 errors=1 short_reads=1 fallbacks=0 buffer_waits=5 ring_full=4
test spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 523 filtered out; finished in 0.44s
```

The test intentionally reads eight bytes starting at byte 93 of a 97-byte file,
asserts `UnexpectedEof`, then verifies pool reuse. Thus the captured one error /
one short read is the exercised negative arm, not an inferred production fault.
The depth warning is retained; this test does not qualify grouped overlap or
model-scale H2D behavior.

## Final local checks

Every row below **actually ran** and exited zero. Raw logs, argv, timestamps,
source hashes and results: [`final-checks/`](final-checks/).

| Check | Result / scope |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo check -p memra-tier --offline --all-targets` | PASS, macOS |
| same with `--target x86_64-unknown-linux-gnu` | PASS, cross-compile only |
| `cargo test -p memra-tier --offline --no-fail-fast` | PASS, 171 tests including 4 doctests |
| `cargo clippy -p memra-tier --offline --all-targets --no-deps -- -D warnings` | PASS |
| `python3 research/spill-a-20260919/day5/test_capture.py` | PASS, 13 tests |
| `git diff --check` | PASS |
| `bash tools/check-flags.sh` | PASS, 864 literal reads, no uncovered names |
| `git diff --exit-code fd49a668 -- crates/` | PASS, native Rust source unchanged |

All pushes used `core.hooksPath=tools/hooks`; no skip overrides. Earlier TCP
connection failures were transient and bounded; `fcbc56ae` is already an ancestor
of pushed `d27b99fc`, not an outstanding push. The final report names the exact
remote SHA after the receipt commit (a receipt cannot self-reference its hash).

## Remaining scope / handoff

- **Overlay, unproven**: no exposed block-device evidence establishing the
  physical path. O_DIRECT acceptance is not NVMe qualification.
- No model serving, production battery, balanced A/B, io_uring, actual device
  transfer timing, or PRO-class qualification is claimed by these cells.
- WP-D owns integration/rebasing of the storage-cell dispatch fragment; only
  its isolated patch application and CLI behavior were tested here.
- Owned box scratch was removed after sync. GPU inventory was empty and the
  canonical lock was free at handoff. Existing lane worktrees/branch stay active
  for lead integration, not abandoned scratch.
- Approximately **1.0 agent-hour** in this resumed session, including bounded
  transport retries and coordination, against the stated seven-day / 168-hour
  lane budget (~0.6%). Prior-session hours are not reconstructed or invented.
