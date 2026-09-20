# Native CUDA transfer substrate — days 6–7

**Historical design, superseded in part by [V13-BINDING.md](V13-BINDING.md).**
Day 9 separates source/destination graph retention and allows charged pinned
residency beyond acknowledgement; the old whole-ticket host-lifetime restrictions
below are retained as the original design record, not the current API contract.
Executed evidence is in [day9/RESULTS.md](day9/RESULTS.md).

Status: implementation under qualification; no serving/default/performance promotion.

## Integration fragments (lead-owned files; apply only in native scratch)

`crates/memra-engine/src/lib.rs`:

```rust
pub mod tier_transfer;
```

`crates/memra-engine/Cargo.toml` (no dependency changes):

```toml
[[bin]]
name = "tier-transfer-gate"
path = "src/bin/tier_transfer_gate.rs"
```

## API and ownership

`CudaTransfers::new(owner: Arc<cudarc::driver::CudaStream>, governor:
memra_tier::bank::SharedBudget) -> contracts::Result<Self>` is created on the
existing designated CUDA owner thread. The frozen `TransferEngine` trait is
unchanged. DeviceOwner, the governor handle and host leases are non-Send; runtime
thread checks precede submissions, event observation, allocation and retirement.
No I/O worker submits a CUDA copy or publishes a device pointer.

Associated `Host = CudaPinnedLease` is native CUDA-pinned backing, not the explicitly
pageable CPU `pool::PinnedLease`. `alloc_host(bytes, BudgetRequest)` charges pinned
bytes before allocation, initializes the entire range without forming an
uninitialized Rust slice, and retains its governor pin. `alloc_device` and
`register_device` issue sealed device leases with device-byte charges. Registration
moves an existing `CudaSlice<u8>` on the exact owner stream; it cannot register a
raw pointer. Callers transferring pre-accounted buffers must transfer their prior
admission rather than double-charge the same physical allocation.

`retain_device` creates an owner-checked retained capability. `release_device`
refuses live transfer bindings/leases. `with_destination` provides a ready native
buffer plus its owner stream for a bounded consumer callback. Consumers submit
only to that stream and call `record_consumer(ticket)` after their last use.
A returned `ReadyView` is itself publication, not an advisory poll. Taking the
destination remains exactly-once and keeps the backend's DMA/consumer ownership.

## Events, completion, and retirement

H2D/D2H preflight checks the complete host length, device capacity and generations,
owner identity and context/stream. D2H requires an authentic `record_producer`
event; a structurally plausible `FenceId` alone is rejected. All-rejected/empty
batches return all inputs without submission. Mixed batches enumerate every input
and reject publication if any promised sibling was rejected.

Copies use cudarc pinned-host async copies on the designated owner stream. Each
accepted item records a real CUDA event and installs a separate consumer wait.
`poll` queries that event, distinguishing NOT_READY from driver errors; it never
conflates an installed wait with producer completion or consumer retirement.
Logical checksums describe actual host bytes; the roundtrip gate separately reads
back the real final device allocation and compares its bytes and hash.

Cancellation revokes publication only. Unknown submissions/observations quarantine
owned backing and quota. Drop is not retirement: unretired entries and the sealed
device registry retain their backing/pins rather than freeing potentially-live
DMA allocations. Explicit recovery synchronizes each recorded real event; it
cannot recover an absent event or a failed copy into success.

`retire` requires observed copy completion, all graph pins dropped, all taken host
leases released, and (after any publication) the exact ticket-bound consumer event
observed complete. `retired` is false until this controlled retirement succeeds.
Tombstones remain until `acknowledge`. This whole-ticket meaning is unchanged by
source-only retirement below. Queue occupancy is charged through the same
governor and retained through acknowledgement-independent retirement. Buffers have
explicit device release; host backing frees only after its tracked event completes.

The native lifetime schedule injects loss of *observation*, not fake CUDA execution,
and subsequently re-observes the real event. This is not physical context-loss
recovery evidence. Graph pins require the owner to drop them only after actual
graph execution/destruction retires; the generic substrate does not inspect a
scheduler or CUDA graph object itself.

## Additive day-7 contract — proposed for E's v1.3 schedules

`CudaTransfers::take_device(&DeviceLease) -> Result<CudaSlice<u8>>` synchronizes
the designated owner stream, requires no live transfer/consumer binding or other
retained device lease, removes the sealed registry entry and its governor charge,
and returns the exact native allocation without freeing it or copying bytes.
The caller assumes the allocation's accounting after hand-back. A live binding or
retained lease returns `Busy` without consuming the retry handle; a second take
returns `ForeignLease`. It does not promote an unretired ticket or bypass its
consumer/graph retirement. The frozen `DeviceOwner` and `TransferEngine` are
unchanged; this is an additive native adapter API.

`CudaTransfers::retire_source(ticket) -> Result<()>` retires only that ticket's
source-side ownership, idempotently, after observed producer/DMA completion and
no graph retention or source-side consumer binding. It never marks the whole
ticket `retired`, removes destination bindings, or drops a taken destination.
Unknown completion remains `Quarantined`; pending DMA or live graph/source
consumer binding returns `Busy` without releasing either side.

- D2H: drop the ticket's source `DeviceLease`; the caller may then call
  `release_device(&source)` to free its registry allocation (or `take_device` to
  recover backing). Other retained source leases still prevent that release.
  The destination host lease remains readable, exclusively owned by its tier,
  and governor-charged until that host lease is dropped, independently of source
  residency. `retire_source` alone does not implicitly free a registry allocation.
- H2D: drop the source host lease and its pinned-memory charge after DMA completes;
  the destination device allocation and consumer binding remain live. A published
  destination still needs its authentic consumer fence and graph retirement
  before `retire`; only then can its sole remaining device lease be handed back.
- Frozen `retired(ticket)` continues to mean whole-ticket retirement, including
  destination consumer lifetime. In particular, a live taken host lease still
  makes whole-ticket `retire` return `Busy`, but no longer prevents source release.

Native gate cases exercise a bound-source refusal, graph and retained-lease
refusals, exactly-once native hand-back, D2H source release with a live hash-equal
host image, H2D source retirement, and exact operand readback at 4 KiB–256 MiB.
These changes add no wire fields or runtime-trait methods. E should add these
per-side schedules to v1.3 without redefining v1/v1.1/v1.2 retirement.

## Still Unsupported / pending

- `p2p` and `nvme_read`: fail closed `Unsupported`; single-GPU qualification,
  storage root **overlay, unproven**, no NVMe ancestry claim.
- Partial-host copy lengths: fail closed `InvalidLayout`; use a correctly sized
  host allocation. No scatter/gather, format conversion or numerical changes.
- Scheduler/KvMaterializer and row publication integration belong to B/C; this
  module alone does not establish active KV demote/reload engagement.
- This initial implementation uses charged native pinned allocations, not a
  reusable startup arena. Allocation behavior must not be sold as steady-state
  spill performance. Native qualification is N=1 correctness only.
- Native conformance and 4 KiB–256 MiB roundtrip receipts: day-7 development
  correctness PASS at source `92d332b9`; see `day7/RESULTS.md`. Full native
  workspace clippy remains red on Rust 1.98 dependency lints; no release claim.

No new environment reads, dependencies, flags or kernel entrypoints.
