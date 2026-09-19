# D contract proposal + interface-freeze fixture plan

**Proposal, not frozen shared API.** Day 1, 2026-09-19; baseline memra
`c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`. The coordinator owns freeze,
`contracts.rs`, Cargo/module wiring, merges and shared documentation.

## Compilable signatures

The actual definitions are compiled/tested under `crates/memra-tier/src/peer/mod.rs`,
`peer/topology.rs` and `placement/mod.rs`; no CUDA dependency. This facade is compilable:

```rust
pub use memra_tier::peer::{
    PeerBackend, PeerCapacity, PeerLease, PeerPlan, PeerRegistration,
    ContiguousCopy, ContiguousSpan, PeerProgress, PeerError, CancelState,
    StateBundleAdapter,
};
pub use memra_tier::placement::{
    placement_report, PlacementPlan, PlacementReport, DevicePlan, DeviceBudget,
    RecordOwner, WeightAllocation, RoutePlan, RouteBudget, RouteKind, Endpoint,
    BudgetError,
};

pub fn reserve_peer<C: PeerCapacity>(
    capacity: &mut C, plan: PeerPlan,
) -> Result<C::Lease, PeerError> {
    capacity.reserve(plan)
}
pub fn submit_peer<B: PeerBackend>(
    backend: &mut B, epoch: u64,
    copies: Vec<ContiguousCopy<B::Registration>>,
) -> Vec<Result<B::Ticket, PeerError>> {
    backend.submit(epoch, copies)
}
pub fn report(
    plan: &PlacementPlan, context_tokens: u64, requests: u64,
) -> Result<PlacementReport, BudgetError> {
    placement_report(plan, context_tokens, requests)
}
```

### Capacity: B's governor is the only budget authority

`PeerCapacity::{reserve(PeerPlan)->Result<Lease>, release(&Lease)->Result<()>}`;
`PeerPlan { device, bytes, epoch }`. `PeerLease::plan()` exposes metadata, not a pointer.
Backend-specific associated lease is non-forgeable; common governor permit remains owned
until physical retirement. The fake governor in D tests is test-only, not another runtime
allocator. Production pool allocation must be paired with B's existing/global reservation.

Borrowing on release is deliberate: a `Busy` or driver error must not consume the caller's
retry handle. Drop cannot free physical memory that I/O/DMA/consumer/graph still uses.
Cancellation and rollback invalidate publication, NOT reclamation: old-epoch bytes are
still charged until retired and then must be releasable. Duplicate/foreign release errors
must leave unrelated accounting unchanged.

### Peer backend: local operands only

`PeerRegistration: Clone` keeps allocation ownership, with device/byte/epoch metadata.
`ContiguousSpan` has private fields and checked nonempty/bounded constructor. No stride,
row-index vector, remote-gather kernel or raw pointer appears in the public contract.
Batching means a list of contiguous memcpy descriptors; any selected-row packing happens
locally before/after transfers, never per-row remote attention loads. Source/destination
registrations are retained independently for every accepted descriptor.

`PeerBackend` associated types: `Registration`, `Ticket`, `LocalReady`.
- `submit(epoch, Vec<ContiguousCopy<Registration>>) -> Vec<Result<Ticket, PeerError>>`:
  exactly one result per input, same order; rejects are not silently omitted. Registration
  spans/epochs and producer event owner are validated at submission. Tickets are not permits.
- `poll(&Ticket) -> Result<PeerProgress>`: Pending / Complete(bytes, consumer_fence) /
  CancelPending / Retired / Quarantined. Unknown completion retains both allocations.
- `cancel(&Ticket) -> Result<CancelState>`: PublicationRevoked before publication,
  AlreadyPublished afterward. Copy completion alone is not the publication linearization.
- `materialize_local(&Ticket, current_epoch, consumer_device) -> Result<LocalReady>`:
  atomic epoch/full-byte-count/cancellation check, destination-local check and consumer-stream
  wait installation. It never returns permission to gather remotely.
- `retire_consumer(&LocalReady) -> Result<()>`: records last-use fence, waits for graph pins
  before releasing accounting. Borrowed so a Busy outcome preserves the handle.

A's TransferEngine dispatches this backend on the designated CUDA owner. D does not create a
second H2D/D2H scheduler. Source-free tests must cover caller dropping its original handle
immediately after acceptance. Same-device copies are explicit local routes; P2P unavailable
means peer tier unavailable. Host bounce can be a separately admitted native route, never
an automatic P2P-success rescue.

### StateBundle adapter: B's type, no shadow model state

`StateBundleAdapter` has associated `StateBundle`, `Identity`, `RestoreTicket`;
`capture_committed(epoch)` and `restore_committed(&bundle, &identity, epoch)`.
Concrete StateBundle stays B/lead-owned. D's future Step/Qwen adapter supplies transport
plane descriptors for B's complete envelope, not another prefix cache or model executor.

Required envelope fields at freeze: immutable artifact/plan/numeric-stream/template identity;
logical committed token high-water; transaction generation; required all-pages/trailing-pages
sets; owner aliases; physical plane extents, encoding/layout digests and byte lengths;
SWA/recurrent/history/tail/selection/draft state as required by the existing plan; logits and
hidden anchor where continuation requires them. Aliases name the same physical owner, never
silently materialize a replica. Restore completes only when all mandatory planes/fences agree.
No DSv4 0731 or V4.1 adapter is proposed or executed.

### Placement report

`placement_report(&PlacementPlan, context_tokens, requests) -> Result<PlacementReport>`.
`DeviceBudget` reports weights, global KV, fixed state, staging, loader, scratch, reserves,
total, signed headroom, and estimation annotations. `RecordOwner` specifies explicit physical
replicas and floor(T/ratio), never TP-divided cache. `RouteBudget` reports route class,
endpoints, demand rate and optional <=70%-of-measured-rate decision. Shared uplink contention
still needs the all-route campaign; independently acceptable routes do not prove fabric fit.

An engine adapter to compiled ModelPlan/RecordLayout/tensor inventory is **pending freeze**.
Day-1 inputs are generic normalized descriptors plus 07's arithmetic fixture, not a loader
or an implementation of future model math. Live free-VRAM adapter must avoid subtracting
already-loaded weights twice and include pool/loader/replica peaks.

### Topology

Read-only `TopologyProbe::snapshot_read_only` returns `TopologySnapshot` of devices and
directed edges, with explicit Unknown observations. Current/max link, NUMA affinity,
capability, context grant, pool grant and direct byte evidence are separate. Four devices
need 12 unique directed edges. Unknown, duplicates or failed active link health cannot
approve peer-tier use. No implicit NVLink, peer atomic requirement, device wake or driver edit.

## Common fake-backend conformance suite — coordinator owns final fixtures

Each WP runs the **same deterministic schedules** against its implementation, not a copy of
expected outcomes. Proposed fixture version `tier-contract-v1`. Freeze source hashes with
actual shared definitions; no hashes invented before those files exist. All outcomes below
must be checked with nonzero item counts and at least one deliberately broken backend caught.
The current D tests implement only D's 8 peer and 5 placement cases; other-trait rows are a
fixture plan, not claimed passing consumer tests.

| Shared trait | Per-item acceptance / completeness | Cancel-after-completion | Lease/accounting + progress | Epoch/identity refusal |
|---|---|---|---|---|
| ObjectStore (A) | 3 extents: first succeeds, middle short EOF, last rejected. Root commit must fail; no partial lookup hit. Corrupt chunk and interrupted root publication never expose usable object. | Completed put then cancel before root publication cannot publish; after atomic commit returns AlreadyPublished and existing immutable object remains. | Pool smaller than object progresses in bounded chunks; conflicting puts cannot double-charge/overwrite a retained version; error/ENOSPC returns charges only after retirement. | Wrong artifact/layout/version and colliding key with different canonical identity refuse; old mutable generation cannot replace committed root. |
| TransferEngine (A) | 3 copy/read descriptors yield exactly 3 accepted/rejected statuses with independent valid vs padded byte counts; aggregate completion cannot bless rejected/short entries. | I/O complete but H2D pending remains pinned; DMA complete but not published can revoke publication; published consumer cannot be revoked. | Source drop + poisoned-slot destination reuse; cancel/timeout cannot free while I/O/DMA/consumer pending; unknown completion quarantines. One slot + multi-slot object must progress. | Late ticket from epoch7 after epoch8 allocation cannot write/publish into epoch8; producer/consumer fence owner mismatch refuses. |
| TierStore (B) | Mandatory all-pages and trailing auxiliary groups missing individually reduce/refuse hit; advisory lookup never reserves. One missing owner fails common completeness. | Load complete then purge/cancel before ready cannot become admitted; post-ready cancel follows request retirement. | Governor charges GPU/peer/pinned/NVMe staging once per physical copy; semantic aliases dedup; graph targets keep addresses; mandatory headroom survives optional prefetch flood. | Tenant salt/program/adapter/token-parent mismatch misses; rollback high-water hides rejected rows; old promotion refuses after eviction/reuse. |
| BankedResidency (C) | Mixed 3-expert batch accepts only some descriptors; publish cannot fabricate missing retained projection/scales; pruned original ID never fetched. | Stage complete then cancel before publish yields no cache hit; published bank remains pinned through native GEMM completion. | Zero/tiny/full cache; stable slots cannot evict in-flight source/destination; bank weights and macro scales charged and retired together. | Original expert/projection/tensor/layout identity required; stale predicted stage cannot replace newer owner/slot generation. |
| RowService (C) | IDs [5,2,5,9] preserve order and duplicate values; one rejected physical read cannot become complete logical output. Crossing a 4KiB boundary accounts both physical reads. | Completed row gather before consumer publication can revoke; in-use rows retire only after projection's fence. | Batch larger than pool; dedup physical reads but retain logical multiplicity; bounded cache below batch size; no whole-table pinning. | Artifact/tensor/row/layout namespace required; late prefetch after request history/spec rollback cannot publish wrong rows. |
| PeerCapacity / PeerBackend (D) | Exact per-item acceptance, contiguous spans only; denied directed context/pool grant unavailable; complete bytes checked before local materialization. | Pending cancel blocks reuse; late completion retires without publication; complete-but-unpublished cancel refuses; post-publish cancel AlreadyPublished. | Governor reserve/release, over-capacity refusal, foreign/double release, source lifetime, destination reuse and graph pin; timeout quarantine never credits free bytes. | Reserve with stale epoch refuses; publication wrong epoch refuses; old epoch can reclaim after fences. Foreign ticket/lease cannot release a current allocation. |

Shared schedules: complete-before-cancel, cancel-before-complete, publish-before-cancel,
late-complete-after-reuse-attempt, consumer-fence-delayed, graph-pin-retained, unknown-status,
partial-batch, empty batch, missing-required-plane and collision. Inject corruption into one
item only and assert only its consumer group fails, preserving localization.

## 08-sketch ambiguities for freeze (lead decisions)

1. **Completion aggregation:** one `Completion { accepted: bool }` cannot represent partial
   submission. Return per-item acceptance at submit plus per-ticket item completion; define
   empty-batch semantics and reject a short status vector.
2. **Cancellation linearization:** completed transport versus published consumer are distinct.
   Propose revoked-before-publish / AlreadyPublished-after-publish; retain until all fences.
3. **Opaque handles / issuer identity:** shared ticket IDs need backend/issuer generation to
   prevent cross-engine collision. Epoch alone is not unique. Actual allocations stay engine
   owned; associated handles avoid unsafe raw pointers and duplicate structs before freeze.
4. **Release failure ownership:** consuming `release(lease)` loses retry ownership on Busy.
   Prefer borrowed release (as D proposal) or return lease in the error. Ditto consumer retirement.
5. **Epoch scope:** namespace/block generation, source allocation generation and destination
   slot generation are distinct. Day-1 fake uses equal epoch for both; freeze must split them
   for migrations between existing allocation epochs without treating stale state as current.
6. **State completeness:** B owns semantic bundle; D needs exact required-plane/owner-alias
   manifest, not only raw bytes. Specify all-pages/trailing-pages, commit high-water and rollback
   atomicity. Missing required working set must refuse, never choose another attention program.
7. **Budget authority and graph pins:** B owns global quota permits; D peer reservation wraps
   that permit. Pin lifetime, address-stability duration and UnknownCompletion quarantine
   require governor-visible charge states rather than two independent allocators.
8. **Wire vs in-process ABI:** propose version1, fixed-width LE integers, length-delimited
   ordered fields, SHA-256 digests, no usize/native enum layout, canonical owner/segment order.
   CPU structs currently have no wire serializer; lead must freeze bytes and test vectors.
9. **Route proof expiry:** grants/byte evidence must bind CUDA context lifetime, topology and
   exact runtime/binary; day-1 `Observation` is a design record, not an enduring proof token.
10. **Receipt scope:** positive byte rows are not negative gate verdicts or performance samples.
    GPU collector must add full telemetry/time-window/counter provenance and the campaign
    manifest before any G2/G5/G6 decision. No missing row or unsupported shape counts as PASS.

## Lead-owned landing fragments (not edited by D)

After shared contract freeze, wire `pub mod peer; pub mod placement;` in tier lib, add the
crate to workspace members and pinned workspace dependencies, and explicitly register tests:

```toml
[[test]]
name = "peer"
path = "tests/peer/mod.rs"
[[test]]
name = "placement"
path = "tests/placement/mod.rs"
```

`docs/TESTING.md` proposed paragraph:

> The generic tier battery scaffold is `python3 tools/tier-battery.py --plan`; CPU receipt
> teeth run with `python3 crates/memra-tier/tests/battery/test_receipts.py`. Byte comparison
> requires nonempty same-program ON/OFF state/logit/token evidence and checks the evidence
> files' hashes. These CPU checks do not run GPU transfers or satisfy the four-tier campaign.
> See `research/spill-d-20260919/CELLS.md` for required pair/four-card/Step cells and lock rules.

No new `MEMRA_*` reads, no kernel/FFI edits, and no default changes: FLAGS/KERNELS fragments
are **none**. INDEX/docs landing is lead-only and remains pending rather than a completed
lane verdict. No push, PR, merge or tag from this session.
