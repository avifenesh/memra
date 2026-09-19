# WP-B contracts proposal — day 1, NOT FROZEN

Compilable definitions: `crates/memra-kv/src/record.rs`, `src/tiered/mod.rs`,
`src/tiered/integration.rs`. These are the proposed signatures, compiled together with
unit tests; lead moves shared types into memra-tier at freeze. No duplicate authoritative
contracts.rs scaffold exists. No runtime behavior has been enabled.

## Identity, layouts and complete state

```rust
use memra_kv::record::{KvBlockId, ProgramIdentity, ByteSegment, GroupRequirement,
    PageRequirement, RecordLayout, StateBundle, TierError, Digest};
use memra_kv::tiered::{TierStore, TierAdmission, TierReservation, BlockLease,
    TierAdmissionPlan, KvMaterializer, TransferEngine, ObjectStore, TierBudget};
use memra_kv::tiered::integration::{PinnedLease, PeerCapacity, PeerPlan, PeerLease};
```

- ProgramIdentity: ten SHA256 digests: immutable artifact, serialized plan, numeric class,
  stream class, tokenizer, template, adapter, modality, position identity, tenant salt.
  An absent adapter/modality must still have a canonical versioned identity, not omission.
- KvBlockId: namespace, parent hash, token hash, `[start,end)`, group, physical owner, epoch.
  Domain-separated v1 SHA256; integers little endian and token count framed. Empty blocks
  and span overflow refuse. Identity hash is an index; admission rechecks the complete
  returned block/program/layout, not merely the object directory's hash bucket.
- ByteSegment: group/page/owner/role, valid bytes, storage bytes, alignment, encoding digest.
  Valid state and padding are distinct; validate exact storage length and checksum only
  valid bytes. Records never transcode. Layout storage byte accounting has NO TP parameter.
- GroupRequirement: group/owner/role with AllPages or TrailingPages(n). Explicit physical
  owners/replicas plus consumer aliases, never MQA divided by weight TP degree.
- RecordLayout: version=1, page_count, segments, requirements. Refuse missing/duplicate
  segments, undeclared groups/roles, bad alignment, overflow and incomplete trailing pools.
- StateBundle: block id + program + exact layout + committed high-water + owner aliases +
  per-segment valid-byte SHA256. Layout is also supplied independently in TierAdmission
  from the admitted plan; storage cannot substitute its own encoding/layout.

The v1 hash byte encoding is implemented; persistent StateBundle wire serialization is
NOT implemented. Freeze must settle and fixture-pin wire framing before A writes it.

## TierStore state machine (compiled trait)

```rust
// Signatures below use the imported concrete types above and the types in tiered/mod.rs.
// lookup(&self, id: &KvBlockId, eligible_peers: &[u32], local: u32) -> Option<Lookup>
// admit(&mut self, admission: TierAdmission) -> Result<TierReservation, TierError>
// prefetch(&mut self, reservation: &TierReservation) -> Result<(), TierError>
// load(&mut self, reservation: &TierReservation) -> Result<(), TierError>
// advance(&mut self, reservation: &TierReservation) -> Result<Phase, TierError>
// cancel(&mut self, reservation: &TierReservation) -> Result<(), TierError>
// retire(&mut self, reservation: &TierReservation) -> Result<bool, TierError>
// evict(&mut self, id: &KvBlockId) -> Result<(), TierError>
```

Lookup is advisory: local GPU -> eligible peer -> pinned host -> NVMe. The backend lease
rechecks availability atomically. Admission reserves target device, pinned, staging, NVMe
and inflight charges. Optional admission cannot consume mandatory headroom. Device vectors
use actual ordinals. Reject a target smaller than the entire exact operand; a transfer's
staging slot may be smaller than the object. The prototype does not implement A's slot loop.

Reserved -> Prefetching -> HostReady -> Loading -> Ready. All accepted/rejected items,
exact bytes, checksum, epoch and producer completion must agree; GPU readiness also needs
an owner-installed consumer fence. Missing, duplicate, rejected, short, corrupt and stale
items fail the whole promised restore. Failure cancels submitted tickets. Disk-ready is
never GPU-ready. The initial implementation stages even a local/peer source through this
abstract two-phase API; optimized direct local/peer transitions need freeze clarification.

ReadyBlock is privately constructed and only borrowed inside `Hierarchy::with_ready`.
Numeric/stream identity is rechecked there. A callback must attach consumer-use fences to
TransferEngine before return; the public trait cannot prove the correctness of a backend's
CUDA fence implementation. Cancel blocks publication immediately; all budget/leases remain
until **every** submitted ticket reports disk/DMA/consumer retirement. Unknown completion
quarantines. Shutdown cancels and intentionally retains backend owners and leases if any
fence remains unknown, rather than dropping DMA-live memory.

ActiveEpoch uses copy-on-write generation metadata and committed/staged high-water marks.
Active admissions hold the SAME shared guard; fork/rollback changes invalidate pending
loads and ready access, including a late success completion. Old immutable prefix records
remain immutable. Actual byte COW allocation/copy and recurrent replay stay with the native
Cache owner. Owner thread must serialize epoch mutation and compute submission.

## Policies

Compiled pure functions cover committed write-through, selective write-through by reuse
threshold, dirty write-back-before-eviction; best-effort, wait, timeout prefetch; LRU/SLRU
victim ordering. Leased, inflight, dirty or unbacked mandatory state cannot be evicted.
These are primitives, NOT runtime policy/default selections. Global fairness/priority queue,
byte-segment promotion targets and dirty-backlog accounting follow the frozen governor.

Optional pre-admission break-even:
`restore_seconds = restore_bytes / (measured_restore_GB_s * 1e9) + fixed_restore_seconds`;
`recompute_seconds = prefix_tokens / measured_prefill_tokens_s`. Load only if strictly
cheaper; tie recomputes. Invalid/nonfinite calibration refuses. No same-program cold proof
means retain/load; mandatory active state always returns RequireState. Measurements must
bind artifact/plan/binary/numeric class/device/tier and include the complete route. This
linear primitive is not P1's full (prompt,prefix) nonlinear suffix model; add calibrated
saved-prefill curves before making runtime break-even decisions.

## KvMaterializer adapter design (trait only; no GPU code)

`materialize(&mut self, bundle: &StateBundle, program: &ProgramIdentity,
ready: &ReadyBlock, stable_target: u64) -> Result<Self::Operands, TierError>`.

The CUDA owner verifies the ready block/epoch/program and maps semantic roles into the
existing `KvLayer.k/v`, recurrent state and saved logits/hidden operands. Qwen native K/V
bytes retain `[token,kv_head,dim]` order. Gather complete required layer operands into
pre-admitted stable scratch, then call the SAME attention kernel with existing counters.
Do not implement per-chunk softmax/reduction or invent f16/FP8 KV. SWA/recurrent state must
satisfy declared working sets; refuse when scratch cannot fit. Graph address stability and
owner-stream handoff must be tested before graph admission. Pure prefix restore continues
to reconstruct a normal PrefixEntry; active reload uses the consuming attention boundary.
The associated Operands implementation owns/pins the ready resources until consumer fences.

## Exact dependencies requested from A

- ObjectStore: advisory lookup; atomic immutable-generation `lease`; explicit release;
  guarded eviction. Current prototype carries StateBundle for testing; physical shared
  ObjectKey should stay separate from B semantic KvBlockId. A need not understand tokens.
- TransferEngine: prefetch/read and load (H2D/peer), per-item poll, cancel and retired.
  Submission Err MUST mean zero accepted operations. Partial acceptance MUST return a
  ticket with every item including rejects; ticket ID reuse must not alias old epochs.
- Completion: segment index, epoch, status, transferred storage bytes, verified valid-byte
  checksum, producer done, consumer fence installed. Owner-thread submission/publication.
- PinnedLease: opaque allocation id, capacity/alignment/NUMA, quota charge, source/destination
  ownership held across disk and DMA and downstream consumer events. Drop pending => quarantine.
- Storage checksums are verified by A from actual bytes, never echoed from requested metadata.
  FakeTransfer deliberately injects mismatches; a fake's pass is not a data-path proof.

## Exact dependencies requested from D

PeerCapacity reserve(PeerPlan) -> PeerLease; release only after retired ticket. Plan carries
owner/consumer device, bytes, alignment, epoch. Must validate actual peer + mempool grants,
route/topology and governor permit. Unsupported P2P is unavailable, not hidden host bounce.
Capacity report counts full physical aliases/replicas; never divides shared KV by TP.

## 08-sketch ambiguities / lead freeze decisions

1. Shared ownership: should semantic KV types live in memra-tier contracts or remain
   memra-kv with a generic ObjectKey adapter? This slice keeps them in one place until freeze.
2. Physical lease lifetime: consume-on-load BlockLease in sketch cannot independently retain
   host twin + target + active consumer. Proposal borrows lease until explicit retirement.
3. Per-item acceptance/checksums/fences and shutdown ownership are absent in the sketch's
   scalar Completion. Require explicit no-accept error semantics and quarantine contract.
4. Deadline monotonic clock units, fairness priorities, inflight slots, mandatory headroom
   and shared B/C governor ownership need freeze. Prototype is a single isolated ledger,
   not a second production allocator; use AdmissionGovernor injection before integration.
5. Logical page count vs per-group ratios, role vocabulary, ring base/position lineage and
   trailing checkpoint requirements need adapter-derived wire semantics. No V4.1 fields.
6. Prefix hash must not depend on mutable active epoch for cross-request sharing. Proposal
   keeps epoch in block identity and expects sealing to a canonical immutable prefix epoch;
   exact seal/collision/restart framing and golden wire fixtures remain a freeze decision.
7. Active epoch guard mutation and CUDA consumption must have one owner; Arc<Mutex> is only
   a shared metadata handle, not permission for another thread to mutate during consumption.
8. Existing HostPrefixCache GLM handoff/version policy is separate from generic record v1;
   no automatic import migration until complete auxiliary mapping and identity are proven.
9. Local/peer hit optimization, transfer ticket consumer-fence registration API, streaming
   stage minimum alignment and store-vs-GPU valid-byte digests must agree across A/B/D.

## Lead-owned landing fragments

No new MEMRA_* read, kernel, default or published number. FLAGS/KERNELS rows: none.
Root Cargo.toml: no edit. Cargo.lock: add `"sha2",` under package `memra-kv` dependencies
(the version is already resolved to sha2 0.10.9); exact fragment is `Cargo.lock.patch` here.
Shared docs suggested sentence: “Generic active/prefix tier control-plane contracts are
CPU-tested prototypes; HostPrefixCache runtime remains unchanged and real Qwen RAM/SSD
active/prefix qualification remains pending.” No positive support-state promotion.
INDEX proposed row after accepted slice:
`spill-b-20260919 | CPU tier identity/state-machine prototype; real active/prefix GPU gates pending. | BASELINE.md`
