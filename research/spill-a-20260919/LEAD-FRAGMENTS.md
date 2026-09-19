# Lead-owned integration fragments — NOT applied by A

> Historical day-1 record. Day-2 migration and current limitations are recorded in
> [day2/RESULTS.md](day2/RESULTS.md); prototype contracts/scaffold are superseded.

Do not cherry-pick this lane into the root workspace without reconciling contracts
and wiring. The scaffold tests actual module files and the actual CLI, but no
production engine integration is claimed. `ObjectLease` should become
ObjectManifest + charged lease at freeze; no second global governor is proposed.

## Cargo / module wiring (adapt to frozen crate)

```toml
# root workspace.members: add "crates/memra-tier"
# root workspace.dependencies (use exact integrated workspace version):
memra-tier = { path = "crates/memra-tier", version = "=0.138.0" }

# crates/memra-tier/Cargo.toml dependencies (all already cached for CPU tests)
sha2 = "0.10"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# crates/memra-engine/Cargo.toml dependencies
memra-tier = { workspace = true }

# crates/memra-engine/Cargo.toml explicit bin name, avoid underscore autobin name
[[bin]]
name = "storage-bench"
path = "src/bin/storage_bench.rs"

# crates/memra-tier/Cargo.toml nested integration test entry
[[test]]
name = "storage"
path = "tests/storage/mod.rs"
```

Lead exports, after replacing scaffold definitions with shared frozen types:

```rust
pub mod contracts;
pub mod object_store;
pub mod io;
pub mod pool;
pub mod telemetry;
```

`ObjectStore`, `StoreTxn`, `ObjectLease`, `BlobBackend` currently live in the
A-owned object_store module. Shared traits/types move to contracts or are
re-exported once; do not retain branch-local duplicate definitions. The
scaffold/contracts.rs proposal supplies remaining types used by the CPU modules.
The fake pool concrete lease currently uses CPU backing; real pinned backend
and DeviceLease ownership are intentionally not implemented before freeze.
Delete standalone scaffold after integrated cargo tests run against frozen crate;
retain its source/evidence in Git history, not as a second runtime contract crate.

## FLAGS / KERNELS

No new environment reads, no changed flags, no CUDA/FFI changes. No FLAGS or
KERNELS row needed. Existing spill modes/defaults and expert buffers are unchanged.
If later experimental doors land, actual landing+14 days is the decide-by date;
this CPU milestone creates no dormant runtime door.

## TESTING fragment

> Generic storage CPU fixtures: `cargo test -p memra-tier --test storage` after
> shared crate integration. Covers aligned versioned headers, valid/padded length,
> SHA256 corruption, immutable root-last publication, partial/short I/O, EINTR,
> ENOSPC, lost/cancelled completion retention, explicit retirement, demand-slot
> reservation and pool-smaller-than-object streaming. Fake pinned memory is NOT
> CUDA-pinned; GPU/NVMe/model gates remain pending under
> research/spill-a-20260919/CELLS.md. Before integration, reproducible standalone
> command: `cargo test --manifest-path research/spill-a-20260919/scaffold/Cargo.toml
> -p memra-tier-spill-a-scaffold --offline`.

## INDEX fragment (milestone, not WP closure)

`spill-a-20260919 | CPU extent-store, bounded-reader and fake-pool milestone only; GPU/NVMe/model gates pending, contracts proposed not frozen. | spill-a-20260919/BASELINE.md`

No numbers moved on the performance board and no default selection was made.


## Day-2 handoff (supersedes day-1 wiring/prototype fragments above)

A applied ONLY these allowed shared amendments: four A module exports in
`crates/memra-tier/src/lib.rs` and the `storage` [[test]] entry in its Cargo.toml.
Root manifests/lock and contracts came solely from the conflict-free lead merge.
No additional dependency/lint edits. The standalone scaffold package was deleted.

Lead still owns engine memra-tier dependency and explicit storage-bench bin entry.
The CLI's existing sha2 use is for a bounded streaming actual-byte checksum. Native
macOS F_NOCACHE is in the CLI adapter, not the unsafe-free tier library. CPU linking
and integrated source tests are reproducible via `day2/verify.py` without nvcc.

TESTING fragment: `cargo test -p memra-tier --offline` now executes 36 frozen tests,
26 A storage tests and 3 compile-fail doctests. Only two shared schedules exist for
A (object_cancel, transfer_cancel); pinned behavior is checked through the frozen
trait and common governor in concrete A assertions. These are not GPU/model gates.

INDEX milestone fragment:
`spill-a-20260919 | Frozen-v1 durable filesystem/charged CPU I/O milestone; Mac-only characterization, native/GPU/model qualification pending. | spill-a-20260919/day2/RESULTS.md`

FLAGS/KERNELS/PERFORMANCE: no new MEMRA_* env reads or kernels; no decide-by row
needed and no published number/default changed. Do not put Mac diagnostic timings
on the engine performance board. They are retained with raw rows as development I/O
characterization only. Budget/GC/async/native integration blockers are numbered in
`day2/RESULTS.md`; do not erase them during the merge train.

## Day-3 handoff

Engine dependency/bin fragment was picked up from lead `914229ae`; v1.1 test
revision came from `ce49cd86`. Source slice `3adf6325` and later receipt-only
changes are described in `day3/RESULTS.md`. A's own strict quarantine defect is
fixed with exact red/green logs. Full tier testing still fails D's directed-route
v1.1 schedule; integrate its owner fix before claiming aggregate green.

C can use `ExtentStore::lease_extent/read_extent/release_extent` for selected
payloads, or `CpuTransfers<ExtentStore<FileBackend,G>>::enable_background(workers)`
before submission. Background requests need ceilings for selected framed
chunk+root NVMe bytes and `storage_bytes + 8191` pageable worker scratch; queue
and fake-pool physical backing are independently reserved once through the SAME
injected governor. `poll`/`progress` harvests completions; keep explicit
retire/acknowledge and host-consumer lifetime rules. Existing sync/whole-object
APIs retain frozen semantics. Background metadata/open/admission is still owner-
side; no native serving-latency or GPU-readiness claim.

INDEX/TESTING milestone fragment:
`spill-a-20260919 | Extent-scoped/background CPU I/O and quarantine-shutdown fix; Linux cross-check only, A2/M1 hardware and full trace runner pending. | spill-a-20260919/day3/RESULTS.md`

No new flags, kernels, dependencies or published performance numbers; no FLAGS,
KERNELS or generated-board amendments needed. io_uring remains proposal-only.

## Day-4 handoff (current)

Source/test tip: `87bba112b3f7b2afb5137fe672679fa0da1a796f`, pushed to the A lane.
Full report: `day4/RESULTS.md`; reproducible raw checks and source hashes in day4/.
D's directed-route failure is no longer inherited: integrated CPU suites passed.
Lead publish-census fix `200a3c66` was merged, not independently edited by A.

C/B: additive `object_store::catalog::CatalogStore` exposes a fixed-row sharded
catalog (lookup, lease_extent, read_extent, release_extent, tombstone/collect/evict).
It does NOT change frozen ObjectManifest. Its model-scale binding to C/B and to
background preparation remains work; do not present small-root CpuTransfers as
already accepting CatalogHead. Persistent backing is charged once per resident
catalog; first lease after manager restart must re-admit it. No full payload
allocation/index scan occurs per lookup/lease. See documented orphan/unknown-I/O
recovery limits before production binding.

D: `telemetry::join::StorageTelemetry` converts directional operation deltas into
D-schema JSONL; bounded intervals, cumulative submitted I/O, null physical counters
when unknown. GPU device/routes come from the real collector, never inferred from
I/O times. Two synthetic CPU outputs and schema red arms are persisted in day4/.
This is NOT the full 250ms native collector or a mixed-roundtrip deaggregation.

TESTING/INDEX milestone fragment:
`spill-a-20260919 | Lazy 200GB-class metadata catalog, charged lease-safe GC and telemetry join pass CPU checks; native bindings and hardware gates pending. | spill-a-20260919/day4/RESULTS.md`

No flags, kernels, dependencies, defaults or published numbers changed. io_uring
is DEFERRED pending the rig's bounded-pread baseline. The updated runner adds 3
CPU cells (35 dry commands total) but remains non-green for full A2/M1 qualification.
