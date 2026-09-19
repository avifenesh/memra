# Lead-owned integration fragments — NOT applied by A

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
