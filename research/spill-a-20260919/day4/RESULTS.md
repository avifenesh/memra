# WP-A day 4 — catalog, GC and telemetry CPU milestone

Repository **avifenesh/memra**, lane **`lane/spill-a-20260919`**.
Exact source/test revision: **`87bba112b3f7b2afb5137fe672679fa0da1a796f`**.
`source-manifest.json` also hashes every A storage/telemetry source and test.
This is a **pushed lane milestone**, not main integration, release, deployment,
GPU qualification, live verification, or permission to begin model work.

## Re-entry, commits and push results

- First action merged day-3 integration with `--no-ff`: `4f56b6ad`.
- Immediate required push was refused by the publish census: **`memra-tier absent
  from the crate list in .github/workflows/publish.yml`**. No bypass. Lead supplied
  `200a3c66`; merged it in `e61bf568`. Push then passed, and `git ls-remote` matched
  `e61bf5682f916f14f87e0fcd1f0508cefb4c24c7`.
- `d354cae5`: lazy sharded catalog and lease-safe filesystem GC.
- `c600eb47`: bounded StorageSample-to-D-schema telemetry join.
- `87bba112`: actual child-process termination at the GC crash boundary and
  unknown-shutdown filesystem ownership regression.
- First source push failed to connect to GitHub after 75007 ms. A bounded retry
  after working on verification succeeded; remote matched
  `c600eb47ccadc9857cb9d7b52d10ea2ae7861c2c`. The subsequent source/test milestone
  push advanced origin to `87bba112` and passed every hook.
- One local pre-commit formatting refusal occurred while splitting commits: an
  extra trailing blank line in `tests/storage/mod.rs`. Ran fmt, restaged, committed;
  no hook was bypassed. A temporary compilation error (`ChunkRef` does not implement
  Wire::validate) and Clippy's drop_non_drop test warning were corrected before checks.

The final receipt/runner-only commit and remote SHA are reported in the handoff.
All pushes use `core.hooksPath=tools/hooks`; no override or `--no-verify` was used.

## Implemented and checked

### 1. Model-scale catalog, additive API

`crates/memra-tier/src/object_store/catalog.rs` exports `CatalogStore<G>`,
`CatalogHead`, `CatalogLease` and lookup-touch counters. Frozen `ObjectManifest`,
ObjectStore, contracts, shared tests, root manifests and lib exports are unchanged.
This avoids raising the frozen 1 MiB root limit or giving consumers a second
interpretation of its wire format.

- One small framed/checksummed root, plus one fixed-record index file per configured
  shard directory. Extent `i` maps to shard `i % N`, row `i / N`. Directories can
  name different devices; **no multi-device throughput claim** is made here.
- Each index row is **112 bytes**, binds its checksum to object identity and extent
  index, and contains the original ChunkRef's lengths and checksums. Installation
  streams rows; it never allocates an entire object or whole-object index vector.
- `lookup` reads one root only. `lease_extent` reads that root and exactly one row;
  no payload validation occurs on admission. `read_extent` verifies the complete
  selected framed extent/checksum on its first **and every subsequent** read.
  A missing sibling does not poison a selected read; missing selected bytes fail.
- Metadata-only fixture: **200,000 x 1,048,576 = 209,715,200,000 logical bytes**.
  Actual filesystem content is **22,408,192 metadata bytes**, with zero payload
  files. A lookup plus four leases read exactly **5 roots + 4 index rows = 41,408
  metadata bytes**, zero payload reads. This directly checks O(extents touched)
  access cost after installation, not wall-clock asymptotics or SSD performance.
- `install_index` accepts a trusted immutable census without requiring payloads.
  This makes a sparse *metadata fixture*, not a supported/materialized checkpoint.
  `put_extent` requires the published reference to match exact bytes. There is no
  fabrication, requantization, or fallback numerical program.

Consumer sequence: `open(shard_dirs, same_governor)` -> `install_index(...)` or
`lookup(key)` -> `lease_extent(head, index, request)` -> `read_extent` -> explicit
`release_extent`. After restart, the first lease must re-admit declared backing.
CatalogStore is not yet bound into B/C's native consumers or CpuTransfers' existing
FileBackend-specific background preparation interface.

### 2. Reference protection, budgets and durable GC

**CatalogStore:** admits the caller's conservative backing ceiling through B's
actual shared governor **before index writes**. A resident catalog holds that
single charge until GC or ordinary manager close; repeated extent leases pin it,
not duplicate it. Each explicit release drops one pin. Dropping a lease without
release leaves its active registry entry, pin and charge intact. Unknown manager
shutdown deliberately retains directory ownership and pins until process exit.

GC refuses while any lease remains; `tombstone` durably renames the root, hiding
it from lookup before unlink. `collect` replays missing-file-safe unlinks across
shards and only then returns the charge/removes the tombstone. A process-exit test
terminates a child with code 73 **without destructors**, after tombstone fsync and
before unlink. Parent observes the tombstone plus payload, opens the recovered
store, gets no lookup hit, and collects it. This tests process-crash recovery,
**not power-loss, controller flush correctness or Linux filesystem durability**.

**Existing ExtentStore<FileBackend>:** `ObjectStore::evict` no longer returns
Unsupported for real files. Every FileBackend holds a shared lifetime OS file
lock. Eviction requires exclusive ownership and a sealed permit issued only after
local lease checks. Another store/process's shared ownership makes GC Busy.
Committed roots are the reference-count authority: under the exclusive lock, GC
counts remaining root references, preserves shared chunks, and removes only chunks
with zero live-root references. Root -> tombstone rename is durable before unlink;
corrupt/unreadable roots fail closed instead of being counted as zero. An unknown
store shutdown retains its shared ownership guard and unreleased charges.

The legacy backend still reserves per lease; it is **not** retroactively converted
into a whole-directory persistent-capacity ledger. CatalogStore provides the new
once-per-resident-object charge. Store ownership files are ordinary filesystem
coordination, not new GPU campaign lock names. Non-file fake BlobBackends without
a deletion protocol still explicitly return Unsupported.

### 3. Telemetry join

`src/telemetry/join.rs` exposes bounded `StorageTelemetry::observe/json_line`.
Operation deltas become D-schema JSONL: cumulative submitted read/write bytes,
instantaneous caller-supplied queue/host counters, per-interval observed wait
percentiles, and sticky-null physical bytes if any delta lacked instrumentation.
Caller-supplied device measurements/routes are required for GPU-labeled rows;
CPU fixtures retain null GPU facts. Empty distributions mean no observed waits,
not measured zero latency. Samples must be directional operations, **not mixed
roundtrip aggregates relabeled as reads**.

The nominal interval is 250 ms; monotonic timestamps and <=500 ms actual gaps are
required, matching D's cadence validation. Buffer overflow fails rather than
silently losing samples. Emitting a row resets interval waits but not byte totals.
The dependency-free Python validator reads D's actual checked-in schema; it also
runs missing-required-field red arms. `telemetry-fixture.jsonl` contains two actual
adapter outputs from **synthetic CPU counter inputs**, not an on-rig collection.

### 4. Async follow-up

Inspection confirms day 3 already supplied the requested facade:
`CpuTransfers::enable_background` -> submission -> `TransferEngine::poll`, which
calls bounded nonblocking `progress` and returns indexed outcomes. No `drive`
call is needed. Existing tests for good/corrupt/missing siblings, partial rejection,
quota saturation, cancellation, consumer retirement and take-once publication
all ran again. No duplicate executor or dependency was added. Metadata/open remain
owner-side; this is not a latency-qualified serving integration.

### 5. Runner/cells and io_uring disposition

`CELLS.md` and `rig-cells-a.sh` add sharded-catalog, GC and telemetry CPU conformance
cells. `day3/test_runner.py` now checks **two identical 35-command plans**, retaining
old day-3 receipt outputs as history. Dry-run does not create its fake NVMe path,
take a lock, build, execute GPU work or claim qualification. Real launcher remains
fail-closed on the absent full A2/M1 interfaces and collector.

**io_uring DEFERRED by lead:** no dependency/ring/dispatch added. Current intended
runner commands omit the unapproved uring arm. Proposal remains in its decision
record; bounded pread baseline measurement is the prerequisite to reconsider it.

## Exact checks, actual outputs

Reproducer: `python3 research/spill-a-20260919/day4/verify.py`.
`commands.json` records exact argv/exits; `*.log.gz` retain complete combined output
and raw SHA256. `initial-checks/` retains the earlier passing source checks, rather
than overwriting evidence after adding the two final ownership/crash tests.

| Executed command | Exit / observed result |
|---|---|
| `cargo fmt --all -- --check` | 0; empty output |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | 0; `Finished dev ... in 0.32s` |
| same check + `--target x86_64-unknown-linux-gnu` | 0; `Finished dev ... in 0.48s`; compile only |
| `cargo test -p memra-tier --offline --no-fail-fast` | 0; bank 35, contracts 44, peer 17, placement 6, storage 47, doctests 4; 0 failed/ignored |
| `cargo test -p memra-kv --offline --no-fail-fast` | 0; 51 tests; 0 failed/ignored |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | 0; `Finished dev ... in 0.14s` |
| `cargo test -p memra-tier --offline --test storage telemetry:: -- --nocapture` | 0; schema test and two emitted CPU fixture rows |
| `python3 research/spill-a-20260919/day3/test_runner.py` | 0; output below |
| `bash research/spill-a-20260919/rig-cells-a.sh '/never-created nvme' --dry-run` | 0; complete 35-command plan, no hardware |
| `bash -n research/spill-a-20260919/rig-cells-a.sh` | 0; empty output |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | 0; 8 payload + 3 canonical wire pins match |
| `git diff --check` | 0; empty output |
| `bash tools/check-flags.sh` | 0; exact output below |

The Mac combined check and KV suite retain the existing unrelated memra-gguf
`unused import: std::os::fd::AsRawFd` warning at `source.rs:20`. Tier Clippy is clean;
the unrelated source is unchanged.

```text
RUNNER_DRY_PASS: two identical 35-command plans; 24 filesystem + 3 CPU conformance + existing A2 + 4 blocked probes; no hardware ran
TELEMETRY_SCHEMA_PASS: 2 rows; required-field red arms passed
check-flags: runtime literal reads=864
check-flags: no uncovered runtime names
check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)
```

## Numbered remaining blockers and boundaries

1. **Hardware/native qualification:** no nvcc/GPU here. No full native engine/server
   build, Linux execution, O_DIRECT execution, real pin/DMA, physical-NVMe evidence,
   Hy3/PLE/Qwen/Step serving gate, PRO or four-card battery ran. Cross-check is not
   a replacement for any of these.
2. **Native/catalog bindings:** the additive catalog needs C/B bindings and bounded
   async prepared descriptors before it can replace their frozen small manifests.
   Native allocator/metadata/RSS and host staging budgets still require actual
   adapter accounting and target measurements; bounded CPU heap use is not pin proof.
3. **Collection/recovery scope:** tombstoned committed catalogs/legacy roots recover;
   automatic cleanup of an index installation interrupted *before root publication*
   is not implemented. Such private orphan files refuse overwrite and require a
   future explicit recovery operation. Unknown completion intentionally blocks GC
   until real all-use retirement or process exit, never a timeout-based free.
4. **Telemetry/performance:** adapter schema conformance is complete; an actual
   250 ms collector, directional runtime engagement counters, physical I/O and full
   30-minute A2/M1/native-consumer windows remain implementation/rig work. No winner,
   speedup, hardware default or published board number was selected.
5. **Lead integration:** inspect/replay the exact lane and run integrated/native
   gates before accepting it. No main merge or release is authorized by this receipt.

No runtime env read, dependency, CUDA kernel, external runtime, numeric program,
shared contract/root manifest or generated performance surface changed. No new
FLAGS/KERNELS row is needed. Only A-owned source/tests/receipts were edited; lead's
shared publish fix arrived by merge. Worktree stays open for continued WP-A work,
not a closed lane. All test scratch directories were removed by their owners.

Approximate effort this round: **0.4 agent-hours**, zero GPU time, against WP-A's
**7 agent-day** budget. Prior receipts record 0.4 + 0.3 + 0.5 hours, with earlier
interrupted-agent time explicitly unknown. These are observed session hours, not
an assertion that the implementation/qualification budget has been consumed.
