# WP-C contracts proposal — day 1, pending lead freeze

This document names **compilable** signatures implemented in
`crates/memra-tier/src/bank/{types,residency,rows}.rs`. The isolated no-dependency
`Cargo.toml` here compiles those source files directly; no lead-owned root/lib/contracts
file was modified. Prototype types are provisional: common names migrate to the frozen
`memra-tier::contracts` in the next round, not a permanent branch-local second schema.
No serialization/wire compatibility is claimed yet.

## Bank service and proof types

```rust
// See source for full fields, constructors and concrete error type.
trait BankedResidency {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout>;
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>>;
    fn stage(&mut self, batch: BankBatch) -> Result<BankTicket>;
    fn publish(&mut self, ticket: BankTicket,
               fence: &mut dyn ConsumerFence) -> Result<Vec<BankLease>>;
}
trait ConsumerFence {
    fn ready_on_owner(&mut self, ticket: BankTicket) -> Result<bool>;
}
trait ExactReader {
    fn read_exact(&mut self, tensor: &TensorId, offset: u64,
                  dst: &mut [u8]) -> Result<()>;
}
```

`BankId`: full immutable artifact digest + complete tensor ID + original
`Expert {layer,original_id,projection}` or `Row(u64)` + layout digest.
No expert renumbering, prefix hash or projection-global qtype.
`RecordLayout`: authoritative encoding, row_bytes, ordered exact tensor/offset/len/role
segments; macro scales stay independent segments. Uniform comparison includes encoding,
row_bytes and segment-role/length geometry, but not differing record offsets/scale values.
`Catalog` rejects duplicate IDs, absent retained IDs, masked IDs and mixed uniform declarations.
`BankBatch` has private IDs + demand/prefetch intent, preserves duplicates/order; unique reads.
`BankLease` is immutable Arc-owned bytes + exact layout + governor permit. Cache removal
cannot free a consumer-held lease. `UniformLease::try_new(Vec<BankLease>)` checks the
actual leased source class and compatible programs; private type goes to uniform-only
compute adapters. Compile-fail tests reject ordinary mixed leases/batches at uniform APIs.
This is only the new CPU interface; existing engine uniform guards are not yet migrated.

`BankService::read` progresses accepted CPU requests explicitly; no publication until host
reads AND consumer fence succeed. Tickets include service identity + sequence; replay and
cross-service tickets refuse. Partial read errors retire the whole CPU batch, never a partial
published batch. Queue and batch limits are bounded; refused demands return `Backpressure`.
Caller must retain/retry a mandatory request rather than skip an expert. Cache-pressure hook
`evict_cached` drops cache references, never consumer ownership. `cancel` is safe for this
CPU-only synchronous backend; it is **not** an implementation of asynchronous DMA cancellation.

## Rows and prediction

```rust
trait RowService {
    fn gather(&mut self, batch: RowBatch,
              reader: &mut dyn ExactReader) -> Result<RowLease>;
}
trait Hotness<D> {
    fn demand(&mut self, id: &BankId);
    fn score(&self, id: &BankId) -> u64;
}
trait PrefetchHook<D> {
    type Context;
    fn predict(&self, context: &Self::Context, limit: usize) -> Vec<BankId>;
}
```

`D` is separately `ExpertDomain` or `RowDomain`. `prefetch_batch` enforces the hint
cap, key class and active catalog; no prediction adds demand heat. A router/n-gram adapter
owns context/history; rejected predictions never excuse a later exact demand.
`RowBatch` preserves row order/duplicates. `RowTable::plan` returns merged aligned extents,
logical bytes, unique useful bytes, aligned requested physical bytes, straddles, efficiency
(useful/physical) and amplification (physical/useful). Empty ratios are None. Checked integer
arithmetic, invalid row/granularity/stride and undeclared padded-EOF reads fail closed.
`BoundedRowService` splits coalesced extents into one bounded slot, scatters straddles and
duplicates into admitted exact output. Pool may be smaller than a coalesced extent/batch;
output must fit its explicit quota or refuse. No unbounded whole-table pinned allocation.
Output is **host-ready**, not GPU-ready. Fake ordinary-host and fake NVMe backends exercise
identical bytes. Pinned/UVA and NVMe OS integrations are not implemented.

## Needed from A and B

- **A ObjectStore:** immutable artifact-bound tensor/extent resolution, exact all-segment
  completion/integrity, padded readable range and read granularity; retain original source
  bytes plus scale planes. Short/error reads must never return success. Tail policy explicit.
- **A TransferEngine:** per-entry accepted/rejected outcomes, ticket epoch/service ownership,
  host-ready versus consumer-ready fences, owner-only publication, source/target lifetime
  through copy and consumer completion, cancel/quarantine until both retire. Requests larger
  than pool must progress by chunks. Bank all-or-none acceptance can aggregate these entries
  but cannot interpret a single accepted boolean as all bytes complete.
- **A PinnedLease:** aligned range views, one quota charge and explicit lifetime, reusable slot
  capacity separate from valid bytes. C must not create its own pinned allocator.
- **B shared governor:** atomic admission over pageable/pinned/staging/device/peer/NVMe and
  inflight resources; mandatory-load reserve, deadline/priority/fairness and pressure hooks.
  C uses the **same instance** as KV. Current `Charge {host_bytes,staging_bytes,inflight}`
  and RAII `BudgetPermit` are CPU subset proposal only; governor implementation is test-only.

## 08-sketch ambiguities / proposed resolutions

1. `BankId` says bank but `stage(batch)` mixes original records: define ID as one immutable
   expert projection/row; batch result is **Vec of leases in logical order**, not one lease
   ambiguously representing a whole bank.
2. `publish(ticket)` lacks owner/fence dependency: require transfer-owned ready proof on
   designated CUDA owner, and consumer retirement as well as producer readiness. CPU fake
   trait is not a security boundary or evidence of actual CUDA-thread affinity.
3. Layout requires segment encodings, offsets, valid/padded lengths, identity and macro scale
   ownership. C currently has concatenated exact segments; final shared schema must name
   qtype/row_bytes and scale planes without relying on string conventions.
4. `RowService::gather -> ticket` lacks result/layout/order and host-vs-device readiness.
   Propose ticket plus ordered `RowLease` publish in real asynchronous adapter; day-1 CPU
   implementation deliberately returns a synchronous **host** lease instead.
5. `RowTable` single width/stride cannot represent discontiguous scale-plane rows. Add atomic
   multi-segment row layout to frozen schema; current packed-row fixture must not hide this.
6. Stage all-or-none versus per-entry accepted transfer: C needs per-item completion from A
   and keeps batch publication all-or-none. No short/failed entry becomes a ready sibling.
7. Budget values are not an allocator. Only B supplies the global governor/permit service;
   metadata overhead, actual allocator padding and mandatory scratch headroom need accounting
   in the real adapters. Current limits bound metadata by catalog/batch/queue but charge
   payload bytes only, not allocator overhead.
8. A global heat value is invalid across experts/rows. Keep distinct policies and capped
   predictors; prefetch demand priority/timing and usefulness telemetry remain integration work.
9. Zero-cache mode still needs admitted staging/output bytes. If a record/output cannot fit,
   refuse or use A's defined chunked consumer contract; do not fabricate or silently omit it.
10. Versioning/canonical serialization and digest derivation are not frozen. Use version 1
    only after lead fixtures bind complete key/layout bytes; no persistent storage compatibility
    is implied by the CPU structs here.

## Lead-owned wiring fragments (proposal, do not apply before freeze)

- Root workspace members: add `"crates/memra-tier"`; package/dependency version follows root.
- New `crates/memra-tier/Cargo.toml`: workspace metadata, `edition = "2024"`, no CUDA deps.
- `crates/memra-tier/src/lib.rs`: `pub mod contracts; pub mod bank;` after common-type migration.
- Test registration: `[[test]] name = "bank"`, `path = "tests/bank/main.rs"`; replace standalone
  `memra_bank_prototype` test/doctest imports with `memra_tier::bank` on integration.
- FLAGS/KERNELS: **no fragment**, no new env read, CUDA kernel, FFI or runtime door.
- TESTING: `cargo test -p memra-tier --test bank` plus doctests after integration; CPU tests
  do not satisfy Hy3/PLE GPU cells.
- INDEX provisional row: `spill-c-20260919 | CPU-only bank/row prototype; Hy3/PLE GPU gates pending. | BASELINE.md`
