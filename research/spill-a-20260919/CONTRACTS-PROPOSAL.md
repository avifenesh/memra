# WP-A concrete contracts proposal — day-1, 2026-09-19

**Proposed, NOT frozen.** CPU implementation and signatures compile together in
`scaffold/Cargo.toml`; it is a standalone package named
`memra-tier-spill-a-scaffold` with library name `memra_tier`. No shared contracts,
workspace wiring, engine library wiring or lead-owned docs have been changed.

## B's cross-session requirements — disposition

1. **Accept: submission Err accepts ZERO operations.** All queue validation and
   capacity failures return every owned input in `Rejected<T>`. The bounded
   reader follows this today: unsuccessful try_send returns its original lease.
   An uncertain CUDA submission must instead return an accepted ticket in
   `Quarantined` state: once work might have been accepted it cannot return Err
   and let a caller release the source. This rule covers submission, not a later
   completion failure. Synchronous `ObjectStore::put` below is a completed write,
   not an async submit; ENOSPC may leave unreachable chunks but never a committed
   root. Literal zero filesystem bytes on a failed write cannot be promised.
2. **Accept: partial batch returns EVERY indexed item.** Proposed
   `BatchSubmission { ticket, items: Vec<ItemAcceptance> }` preserves original
   input order/index and owned rejected inputs. Completion's segment vector
   includes rejects, shorts and failures, not only accepted items. Empty batches
   should be refused; all-rejected batches should return Err with all inputs.
   The final aggregate success requires all requested segments; no filtered
   success vector. Current reader is single-item: tests submit three items to a
   two-item limit and assert the third explicit rejection and both completions.
   Batched TransferEngine implementation remains next milestone.
3. **Accept: explicit release.** `PinnedLease::release(self)` returns the idle
   slot/quota, and `ObjectLease::release(self)` is explicit. Drop remains a safety
   net, not a scheduler signal. In-flight requests own their leases until returned;
   cancellation cannot access or free those leases. Current ObjectLease is an
   immutable metadata snapshot (no deletion/GC exists), NOT B's final charged
   lease; the final lease must own its global-governor permit.
4. **Accept: unknown/lost completion quarantines.** Add
   `TransferEngine::retired(ticket) -> Result<bool>`; only true proves disk + DMA
   + consumer use retired. `Err`, unknown ticket, a cancelled flag, host readiness
   and a timeout are NOT retirement. B keeps charges/object leases until true.
   CPU `FakeTransferLease` explicitly exercises all three independent signals,
   stale epoch rejection and loss-on-Drop quarantine. Quarantined fake slots stay
   unavailable for pool lifetime; real recovery from a successful full owner
   drain is intentionally not implemented here. The production ticket registry
   must retain bounded retirement tombstones until all consumers acknowledge.
5. **Accept: per-segment status/bytes/checksum/epoch.** `SegmentCompletion` has
   original item + segment indexes, status, valid and submitted-I/O bytes,
   `checksum: Option<Digest>`, epoch, `producer_done`, `consumer_fenced`, fence,
   and error. Checksum is SHA256 of VALID bytes and Some only when checked;
   unchecked/pending checksums cannot satisfy verified-read publication.
   Top-level `Completion` repeats producer_done and consumer_fenced as ALL-required
   aggregates, and carries the complete segment vector. HostReady never implies
   DeviceReady. A consumer fence *enqueued* permits ordered consumption, but only
   consumer *completion* permits retirement: those are separate states.

## Ownership / operation rules to freeze

- Submission owns source and destination leases. Device allocation handles are
  owner-resolved IDs, not worker-accessible raw pointers. Proposed DeviceLease
  fields below are metadata-only scaffolding; production construction/validation
  must be sealed by D's device registry and backed by ownership, not forgeable IDs.
- H2D CopyOp host is source, device is destination; D2H reverses their roles.
  P2P has two owner-tagged device leases. Producer fence is mandatory where the
  source has asynchronous writers; None means synchronously host-initialized,
  never "we did not find the event".
- `poll` observes and cannot consume/free ownership. `retire` supplies a
  consumer-done fence; `retired` confirms all physical uses actually ended.
  We still need a frozen `take_host` / `take_device` or leased-ready-view API so
  B/C can receive the destination without copying/forging its ownership.
- CPU ObjectStore handles immutable chunk transactions synchronously; a bounded
  I/O adapter supplies async tickets. This keeps disk-ready separate from
  GPU-ready and avoids ObjectStore pretending to submit CUDA. Root publication
  occurs only after chunk readback verification and complete byte count.
- `lookup` is advisory. The day-1 metadata snapshot cannot reserve residency or
  promise chunks still exist on disk. Read always validates checksum/length;
  B decides optional-prefix miss versus mandatory-active failure.
- Source/input valid length is distinct from aligned stored length, submitted I/O
  length, and actual device-level physical bytes. Last extent's padding is zero
  and is never fed to a model consumer. The fake pool charges full slot backing.

## Extent/root wire v1 (proposal)

All integers little-endian. Alignment/header length 4096. One chunk holds at most
1 MiB valid bytes. Header: magic `MREXT001` at 0..8, version u32 at 8..12,
header length u32 at 12..16, valid u64 at 16..24, padded payload u64 at 24..32,
payload SHA256 at 32..64, zero reserved bytes to 4064, SHA256 of header 0..4064
at 4064..4096. Payload starts at aligned offset 4096; padding is zero.
Chunk file key is SHA256 of the complete encoded extent, not a truncated digest.

Root payload: artifact/semantic/layout 32-byte digests, generation u64, total
valid bytes u64, chunk count u64, then (encoded chunk SHA256, valid u64) entries.
Root uses the same checksummed framing and 1 MiB bound. Root filename is SHA256
of the full 104-byte ObjectKey; its payload repeats the key and lookup checks it
for collisions/wrong identity. Root/chunk namespaces are distinct. Zero-length
objects are valid with zero chunks; empty chunks are refused.

The ordinary filesystem backend atomically publishes completed temporary files
using no-clobber hard links. It is ephemeral: no fsync/restart durability claim.
Persistent mode must add data fsync, publication and parent-directory fsync before
advertising durability, and carry the durability class explicitly in receipts.

## Compilable A-owned API shape

These declarations are in the path-imported production modules (not a second
implementation). Full proposed shared signatures follow below.

```rust
use memra_tier::contracts::{ObjectKey, Result};
use memra_tier::object_store::{ObjectLease, StoreTxn};
use memra_tier::pool::{Admission, FakePinnedPool, PinnedLease, PoolAccounting};

pub trait ProposedObjectStore {
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectLease>>;
    fn begin(&self, key: ObjectKey, valid_bytes: u64) -> Result<StoreTxn>;
    fn put(&mut self, txn: &mut StoreTxn, payload: &[u8]) -> Result<()>;
    fn commit(&mut self, txn: StoreTxn) -> Result<ObjectLease>;
    fn read(&self, object: &ObjectLease, chunk: usize) -> Result<Vec<u8>>;
}
// Actual inherent APIs (signature summary):
// FakePinnedPool::new(slots, slot_bytes, alignment, reserved_demand_slots) -> Result<Self>
// FakePinnedPool::acquire(&self, valid_bytes: usize, Admission) -> Result<PinnedLease>
// FakePinnedPool::accounting(&self) -> PoolAccounting
// PinnedLease::{valid_bytes,storage_bytes,alignment}(&self) -> usize
// PinnedLease::write(&mut self, bytes: &[u8]) -> Result<()>
// PinnedLease::bytes(&self) -> Result<&[u8]>
// PinnedLease::release(self); PinnedLease::quarantine(self)
// ObjectLease::release(self)
```

`PinnedLease` is non-Clone; no public raw pointer and no initialized slice before
complete write. Fake backing is aligned but NOT physically pinned. The production
lease backend must wrap existing PinnedHostArena/cudarc allocation, not allocate
or register memory per request. B supplies one shared budget permit per physical
pool plus per-client reservations; A does not invent another global governor.

## Fixtures requested from B/C/D

- **B:** deterministic opaque heterogeneous segment manifests, all/trailing group
  requirements, parent/salt/program differences, committed high-water/epoch and
  stale-generation completion. Mandatory versus optional error disposition.
  Supply fake global-budget permits and release accounting shared with C.
- **C:** original expert/projection and tensor/row identities; exact valid payload
  + scales, duplicate/reordered rows and pruned-ID rejection; a batch exceeding
  pool capacity. Row tests use opaque 264/288/68-byte records and real PLE byte
  layouts later, never new quantization or altered tensor contents.
- **D:** producer/DMA/consumer fence handles with controlled completion order,
  partial acceptance, unknown submission, late completion and lost peer; sealed
  device/source/destination leases; `PeerBackend` implementation shape.
- All sessions: shared deterministic bytes `((17*i + 3) % 251) as u8`, sizes
  0/1/264/288/4095/4096/4097/1048576, and SHA256 fixture pins in
  `fixtures.json`. Use separate fault injection for header/payload/padding,
  EINTR, EOF, ENOSPC, root publication interruption and stale epoch.

## Ambiguities / decisions for the lead

1. Freeze whether ObjectStore's public API is synchronous byte/chunk storage as
   above, or async read/put tickets atop it. The 08 sketch has no begin transaction,
   no completion polling on store, no destination ownership return and no release.
2. Prefer rename current metadata `ObjectLease` to `ObjectManifest` at integration
   and reserve `ObjectLease` for a non-Clone governor/GC-backed reservation.
   Who constructs shared permits and how does explicit release reach B's governor?
3. Device leases and source ownership: use a sealed backend-associated token or
   registry ID plus retained Arc owner? Metadata-only IDs are not sufficient.
   Freeze ready-view/take ownership and post-consumer release alongside `retired`.
4. Completion scope: original sketch has one accepted bool and bytes count;
   required per-item/per-segment outcomes cannot be inferred from those. Freeze
   vector cardinality, aggregate semantics, tombstone lifetime and checksum enum.
5. How are cancelled transactions' immutable orphan chunks garbage-collected?
   Day-1 retains them, preserving deduplicated/pre-existing chunks. Cleanup must
   never delete another transaction's chunks. No concurrent-process GC here.
6. Accept wire v1 above or change before first durable artifact? Header alignment
   4096 is conservative and needs Linux filesystem/device validation; 1 MiB is a
   bounded test chunk, NOT B's token chunk default or a tuned transfer size.
7. Persistent durability versus ephemeral complete-byte commit needs an explicit
   API class; unsupported durable requests must fail closed, not quietly run here.
8. NUMA, cancellation deadlines, retry/EINTR limits and priority fairness require
   frozen budgets. Day-1 worker is FIFO with caller admission and bounded slots,
   not the final global prioritized tier scheduler.
9. Confirm all fixture checksums and per-segment status schema centrally; actual
   physical bytes must stay unknown unless an instrument measures them.

## Proposed shared signatures (compiled from scaffold/contracts.rs)

The next code block is an exact copy of the compiled proposal, not a frozen ABI.

```rust
// LEAD-OWNED: replace with frozen contracts. PROPOSAL, not a runtime ABI.
use crate::pool::PinnedLease;
pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidLayout,
    Overflow,
    Capacity,
    NotFound,
    Conflict,
    Corrupt,
    UnsupportedVersion(u32),
    ShortIo {
        expected: u64,
        actual: u64,
    },
    Io {
        kind: std::io::ErrorKind,
        os_code: Option<i32>,
    },
    UnknownTicket,
    StaleEpoch,
    NotReady,
    Cancelled,
    Quarantined,
    Unsupported,
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            kind: e.kind(),
            os_code: e.raw_os_error(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub artifact: Digest,
    pub semantic_id: Digest,
    pub layout: Digest,
    pub generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferTicket {
    pub id: u64,
    pub epoch: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    Pending,
    HostReady,
    DeviceReady,
    Failed,
    Cancelled,
    Quarantined,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceId {
    pub owner: u32,
    pub sequence: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub producer_done: bool,
    pub consumer_fenced: bool,
    pub segments: Vec<SegmentCompletion>,
    pub ticket: TransferTicket,
    pub accepted: bool,
    pub status: TransferStatus,
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub consumer_fence: Option<FenceId>,
    pub error: Option<Error>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelState {
    Retiring,
    Retired,
    Unknown,
}
// IDs resolve only inside the designated CUDA owner; never raw pointers.
#[derive(Debug)]
pub struct DeviceLease {
    pub owner: u32,
    pub allocation: u64,
    pub epoch: u64,
    pub bytes: u64,
}
pub struct CopyOp {
    pub host: PinnedLease,
    pub device: DeviceLease,
    pub bytes: u64,
    pub producer_fence: Option<FenceId>,
}
pub struct PeerCopyOp {
    pub source: DeviceLease,
    pub destination: DeviceLease,
    pub bytes: u64,
    pub producer_fence: FenceId,
}
pub struct ReadPlan {
    pub object: ObjectKey,
    pub chunk: u32,
    pub destination: PinnedLease,
    pub epoch: u64,
}
// Rejected ops return ownership, not just an error that drops a live source.
pub struct Rejected<T> {
    pub op: T,
    pub error: Error,
}
pub trait TransferEngine {
    // Err means zero accepted operations and returns every input. If anything was
    // accepted (including an uncertain submit), return a ticket + EVERY item.
    fn submit_batch(
        &mut self,
        ops: Vec<TransferOp>,
    ) -> std::result::Result<BatchSubmission, Rejected<Vec<TransferOp>>>;
    // true only when disk + DMA + all consumer use is proven retired. Err/unknown
    // is NOT retirement permission. cancel/poll never imply retirement.
    fn retired(&mut self, ticket: TransferTicket) -> Result<bool>;
    fn h2d(&mut self, op: CopyOp) -> std::result::Result<TransferTicket, Rejected<CopyOp>>;
    fn d2h(&mut self, op: CopyOp) -> std::result::Result<TransferTicket, Rejected<CopyOp>>;
    fn p2p(&mut self, op: PeerCopyOp) -> std::result::Result<TransferTicket, Rejected<PeerCopyOp>>;
    fn nvme_read(
        &mut self,
        op: ReadPlan,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan>>;
    fn poll(&mut self, ticket: TransferTicket) -> Result<Completion>;
    fn cancel(&mut self, ticket: TransferTicket) -> Result<CancelState>;
    // poll does not retire leases; destination remains protected through consumer use.
    fn retire(&mut self, ticket: TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentCompletion {
    pub item: u32,
    pub segment: u32,
    pub epoch: u64,
    pub status: TransferStatus,
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub checksum: Option<Digest>,
    pub producer_done: bool,
    pub consumer_fenced: bool,
    pub consumer_fence: Option<FenceId>,
    pub error: Option<Error>,
}
pub enum TransferOp {
    H2d(CopyOp),
    D2h(CopyOp),
    P2p(PeerCopyOp),
    NvmeRead(ReadPlan),
}
pub enum ItemAcceptance {
    Accepted {
        item: u32,
    },
    // Rejects preserve source/destination ownership and their input index.
    Rejected {
        item: u32,
        op: TransferOp,
        error: Error,
    },
}
pub struct BatchSubmission {
    pub ticket: TransferTicket,
    pub items: Vec<ItemAcceptance>,
}
```

## StorageSample wire proposal (compiled module)

```rust
//! Versioned JSONL per-operation deltas, not cumulative counters.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageSample {
    pub schema_version: u32,
    pub fixture: String,
    pub backend_requested: String,
    pub backend_actual: String,
    pub status: String,
    pub valid_bytes: u64,
    pub padded_bytes: u64,
    pub io_bytes: u64,
    /// None for buffered filesystem/CPU fake: submitted bytes are not device traffic.
    pub physical_bytes: Option<u64>,
    pub queue_ns: Option<u64>,
    pub io_ns: Option<u64>,
    pub h2d_ns: Option<u64>,
    pub d2h_ns: Option<u64>,
    pub total_ns: u64,
    pub inflight: u32,
    pub pinned_bytes: u64,
    /// None when RSS/backing allocation was not instrumented (never a guessed peak).
    pub pageable_bytes: Option<u64>,
    pub fallbacks: u64,
    pub payload_sha256: String,
}
impl StorageSample {
    pub fn json_line(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}
```
