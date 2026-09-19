# Generic spill interface freeze — tier-contract-v1

2026-09-19. Repository **avifenesh/memra**, branch `lane/spill-lead-20260919`,
base `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.

**Local contract/fixture delivery only. Not merged, pushed, deployed, GPU-qualified,
or permission to start model work.** The lead worktree remains open for integration.
No WP implementation was copied or edited; no main, engine, server, CUDA, model,
flag or performance-board changes. The requested shared module slots are documented
in `crates/memra-tier/src/lib.rs`, not stubbed with competing implementations.

## Authority and inputs

`crates/memra-tier/src/contracts.rs` is the single authoritative shared schema.
The original plan's §B “Shared contract package”, §B WP ownership and §C integration
order apply: **contracts → A → D transport → B → C → D final gates**.
Breaking changes require a lead decision, updated wire fixtures and all consumer tests.
The task's ten already-taken lead decisions are applied, not reopened.

Proposal shorthand below refers to each lane's
`research/spill-{a,b,c,d}-20260919/CONTRACTS-PROPOSAL.md`.
Every numbered ambiguity is dispositioned below. Also read: A `LEAD-FRAGMENTS.md`,
A `scaffold/contracts.rs`, B `Cargo.lock.patch`, and the described A object-store,
I/O, pool and telemetry; B record/tiered; C bank; D peer/topology/placement sources.
Observed proposal-lane tips (read-only, not an integration qualification):

| Lane | Tip |
|---|---|
| A | `59f0fd52f4e39f15a216abac007afaac73f29a7e` |
| B | `a4e9abf2086c55bb10995c3fa004f43b472a6f92` |
| C | `17ddbb67f9682ce4a7a54cf9b75e2c2e0da4cd82` |
| D | `682c0b3c83eca6ed59226fef4ddfd12fda610e6f` |

The prescribed private GPU corpus was absent at `~/projects/darklanes/agent-knowledge/gpu/README.md`.
No experiment/default/kernel was designed or promoted here.

## Frozen semantics

### Identity and bytes

- `ProgramIdentity` includes version plus all ten digests: artifact, serialized plan,
  numeric class, stream class, tokenizer, template, adapter, modality, position, tenant salt.
  Absence is represented by the producer's canonical versioned identity, not omitted fields.
- `KvBlockId` contains parent/token hashes, span, group, physical owner, and state epoch.
  Token hashing frames count and little-endian u32 IDs. Index hashes never replace full
  program, layout and block equality at admission/consumption.
- `StateBundle::seal` normalizes immutable-prefix epoch to **0**, after committed-byte
  validation. Active epochs are separate COW generations. Actual byte COW, current-epoch
  authority and atomic mutation/compute submission remain B/native-owner responsibilities.
  Metadata sealing is not proof of a physical immutable copy.
- Each `GroupRequirement` has its **own page_count**; roles are typed, including scales,
  macro scales, history, tails, selection, draft, logits/hidden and adapter extension IDs.
  `ByteSegment` adds original tensor and extent offset, valid/padded lengths and encoding
  (digest + authoritative row_bytes). Never derive KV owner/replica bytes from TP degree.
- All/trailing groups are exact sets: undeclared, duplicate, missing or excess trailing
  segments refuse. Ordered segments preserve the native operand program. Aliases are
  group-scoped and name an existing physical owner without charging another copy.
- `BankId` uses full artifact/tensor/original expert+projection or original row identity.
  Payload and discontiguous scale planes form one atomic record. `UniformLease` requires
  an actually Uniform source class AND equal compute geometry; a homogeneous subset of
  PerRecord metadata is still refused. Bank leases retain actual backend resources and
  the unique governor capability across aliases; explicit retirement invalidates aliases. Refused bank publication returns the original charge and backing for explicit release.

### Submission, ownership and retirement

- `ObjectStore` is synchronous, root-last, immutable chunk storage. Its transaction is
  backend-associated and borrowed for put/commit/cancel. A successful commit yields an
  **ObjectManifest**, not a quota reservation. `lease` rechecks it and returns ObjectLease
  with a non-Clone ChargedLease. A failed synchronous put may leave unreachable immutable
  chunks; it cannot publish a partial root or promise zero filesystem writes.
- Transfer/peer submission `Err` means **zero accepted operations** and returns every
  owned input. A partially accepted submission returns a ticket plus indexed results
  for **every** input. Uncertain submit is accepted/quarantined, never an error that frees
  potentially live inputs. Empty/all-rejected batches refuse.
- Tickets are `(issuer, sequence, Epochs { state, src_gen, dst_gen })`, not consumption
  rights. **One ticket is homogeneous in the epoch triple**; group heterogeneous
  allocation generations into separate tickets. Per-item/per-segment completion still
  repeats all epochs and independent status/valid bytes/I/O bytes/verified checksum.
- `Completion::require` refuses missing/duplicate/reordered/rejected/short/corrupt/stale
  segments even when top-level aggregates claim success. Aggregate producer_done means
  all required producers completed; consumer_fenced means waits were installed, **not**
  downstream compute completed. DeviceOwner additionally checks fence context issuer,
  physical owner and destination generation. Unknown checksums never satisfy ready.
- `DeviceOwner` is a non-Send/non-Sync owner-thread registry. It creates sealed DeviceLease
  handles only with retained backend resource + governor pin; workers get no raw pointer.
  Native adapters must construct it on the actual CUDA owner and validate the real
  allocation/context. CPU construction does not prove CUDA affinity.
- Bind destination ownership at submission. `ready_view` checks complete data and owner
  fences; `take_destination` publishes/takes once while the backend retains physical pins.
  Ready is borrowed/non-Clone. Consumer submission and last-use fence registration stay
  on the owner. Do not use a metadata-only handle to stand in for retained allocation ownership.
- Cancel linearizes against publication, not copy completion: before publish revoke;
  after publish return AlreadyPublished and retire the consumer normally.
- Borrow release/retire handles. Busy never destroys retry ownership. `retired(ticket)`
  must cover disk, DMA, consumers **and graph pins**; false/error/unknown retain memory
  and charges. Unknown shutdown deliberately quarantines retained resources. Tombstones
  last until explicit `acknowledge`; an acknowledged ticket becomes Unknown, never aliases
  a future ticket. Acknowledgement requires all stakeholders to have observed retirement.

### One budget and explicit routes

`BudgetGovernor` is injected once across A/B/C/D. `LeaseIssuer` issues and tracks opaque
capabilities; it is **not an allocator or admission policy**. Production governor is B's
implementation; test ledgers are under `tests/contracts/`, not runtime alternatives.

- Atomically charge physical device, pinned, pageable, NVMe, staging, loader, replica,
  peer and inflight dimensions, including allocator padding/metadata in native adapters.
- Device/pinned/pageable/NVMe fields count physical backing. Peer/replica/staging/loader
  fields are additional quota ceilings/breakdowns, **not bytes added a second time**.
  Fixed pool backing is charged once; individual slices pin it and consume slot/queue
  reservations rather than charging the same backing again. A separately allocated
  scratch/output/replica is an additional physical charge.
- Preserve mandatory-load headroom. Order MandatoryActive > AdmittedRestore > Demand
  (rows/experts by deadline) > OptionalPrefetch > Backup. Deadline units are monotonic
  nanoseconds in one governor clock, not restart-persistent wall time. FIFO/tenant fairness
  breaks equal priority/deadline ties; bound queues and dirty backlog. Continuous mandatory
  demand does not create a service-level guarantee for optional backup.
- `PeerBackend` accepts only checked contiguous memcpy descriptors. Local packing is
  separate; no remote scatter operand API exists. PeerCapacity validates directed context
  and pool grants plus live route proof. Unsupported peer routes are unavailable.
  HostBounce is distinct and explicitly accounted, never a P2P success label.
- `PlacementReport` is metadata, not admission permission. It preserves estimates,
  negative headroom, physical replicas and route demand. Unknown measured rate remains
  None. Per-route 70%-of-measured envelope cannot prove shared-fabric capacity.
- `StorageSample` separates valid, padded, submitted-I/O and instrumented physical bytes.
  Unknown physical/RSS/timing measurements remain None, not guesses or invented zeroes.

## Wire v1 and pins

**Deliberate safer freeze choice, flagged for lead review:** persisted shared metadata
uses compact canonical UTF-8 JSON, not a new hand-written binary serializer. Only
struct declaration order, externally tagged enums, fixed-width integer fields, ordered
vectors and 32-element digest arrays are accepted; no maps/floats/native enum layouts.
Every persisted struct, including nested records, has an explicit `version` field.
`Wire::decode` bounds input at 16 MiB, denies unknown fields, validates nested versions
and semantics, then requires byte-for-byte canonical re-encoding. Reordered keys,
extra whitespace, unknown fields and unsupported versions refuse. Direct serde decoding
is not a validated persistence entry point. Backend limits may be smaller and refuse Capacity.

Digest framing is SHA-256 over:
`b"memra-tier\0v1\0" || LE64(domain length) || domain || LE64(payload length) || payload`.
Shared Wire domain labels are fixed in the trait implementations. Valid payload checksums
use `valid-bytes`; payload padding is zero and excluded from the checksum, but validated.
Metadata does not use raw unframed SHA-256. Fixture file integrity hashes in receipts use
ordinary SHA-256 as file hashes, not as runtime identities.

A keeps its 4096-byte bounded extent framing as a backend format, with actual version
validation, and migrates raw payload/header/content hashes to distinct domains
(`valid-bytes`, `extent-header`, `extent`). Root payload is canonical ObjectManifest
metadata rather than the provisional unversioned 104-byte key layout. There are no
production artifacts to migrate in this lane; **do not read A's prototype roots as frozen v1**.
Its 1 MiB chunk size is a backend bound/fixture, not a token-chunk or transfer-size default.
Persistent mode must add fsync + atomic publication + directory durability or refuse.

`tests/contracts/fixtures.json` pins eight deterministic payload checksums for sizes
0/1/264/288/4095/4096/4097/1048576 and three canonical wire hashes (program/key/bundle).
`fixture_reference.py` independently constructs the wire/hash bytes with Python stdlib;
`--check` compares pins rather than regenerating expected results during tests.

## All proposal ambiguities — decision, rationale, rejected alternative

References in the first column are exact numbered items in each proposal's ambiguity
section; named sections supply additional rationale. No unstated backend policy is promoted.

| Proposal item | Frozen decision and rationale | Rejected alternative / follow-up |
|---|---|---|
| A1 | Synchronous ObjectStore plus explicit transaction lifecycle; keeps byte commit distinct from DMA (A “Ownership / operation rules”). | Async store trait in original sketch; thin bounded async wrapper is WP-A follow-up. |
| A2 | Metadata ObjectManifest, charged ObjectLease, unique ChargedLease + common governor. | Metadata snapshot named lease; Drop-only credit or independent A budget. |
| A3 | Sealed owner registry, retained backing/pins, ready view and take API. | Forgeable metadata-only device IDs; raw-pointer worker API. |
| A4 | Full acceptance and per-segment completion vectors; explicit aggregate verification and acknowledge tombstones. | Scalar accepted bool or silently filtered success list. |
| A5 | Retain unreachable immutable/deduplicated chunks until exclusive-owner or coordinated GC proof. | Deleting chunks on cancelled transaction without reference proof. GC algorithm remains A work. |
| A6 | Versioned canonical shared metadata; keep backend extent framing with domain-separated hash migration. | Adopt the prototype binary root and raw SHA scheme unchanged; see wire review flag. |
| A7 | Explicit Ephemeral/Persistent class; unsupported durable mode fails closed. | Treat a completed ephemeral write as restart durability. |
| A8 | Shared priority order, monotonic ns deadlines, bounded queues and mandatory headroom. | Per-pool FIFO as global scheduler; NUMA/retry/queue numeric limits are configured/qualified by A/B, not invented here. |
| A9 | Shared checksum helper and StorageSample; actual physical bytes Option. | Echo expected checksums or rename submitted bytes as physical SSD traffic. |
| B1 | All shared semantic KV types live in memra-tier; native Cache/HostPrefixCache implementation remains B-owned. | Persistent duplicate contracts in memra-kv. |
| B2 | Borrow reservation/block lifetimes; retain host, device and consumer ownership through retirement. | Consume-on-load dropping host twin or losing Busy retry handle. |
| B3 | Per-item/per-segment exactness, zero-accept Err, unknown quarantine including shutdown. | Scalar completion, cancellation freeing DMA-live memory. |
| B4 | One injected BudgetGovernor, fixed priority enum/ns deadlines and independent quota ceilings. | Prototype Hierarchy's isolated ledger as a second production allocator. Fair queue mechanics remain B implementation. |
| B5 | Per-group page_count, typed roles, explicit extents/encoding/aliases; native adapter supplies ring/position lineage through plan/program identity and required state segments. | Global page_count or V4.1-specific schema fields; omit auxiliary planes. |
| B6 | Immutable prefix epoch 0 after committed sealing; active COW epoch stays distinct; canonical wire pins. | Mutable active epoch polluting cross-request prefix key or sealing uncommitted tails. |
| B7 | Existing native owner serializes epoch mutation and ready/compute submission; methods require current Epochs. | Treat Arc<Mutex> as permission for arbitrary threads to mutate while GPU consumes. |
| B8 | No automatic HostPrefixCache/GLM import migration. Require complete native mapping first. | Reinterpret old handoff version as generic v1. |
| B9 | Local/peer direct paths allowed only through exact owner-ready checks; complete target must fit; staging may chunk. Checksums cover valid bytes and are verified on the actual bytes. | Mandatory host bounce, disk-ready⇒GPU-ready, format conversion while loading. |
| C1 | BankId is one original projection/row; publish ordered Vec<BankLease> with duplicate aliases. | One ambiguous whole-bank lease. |
| C2 | Transfer-owned ready fences + actual retained resources; retire all native consumers. | Host-read completion as GEMM permission or an unbound caller-supplied fence bool. |
| C3 | Shared RecordLayout carries per-segment EncodingId/row_bytes, role, tensor, offset and valid/padded lengths. | Projection-global qtype or strings standing in for scale semantics. |
| C4 | Asynchronous gather ticket plus ordered RowLease publication. | C's synchronous host lease as final GPU readiness API. Host backend remains a valid implementation step. |
| C5 | One multi-segment row layout includes discontiguous scale planes atomically. | Width/stride-only table descriptor hiding missing scale planes. |
| C6 | Transfer partial acceptance is explicit; bank publication is all-or-none. | Publish successfully read siblings of a promised complete batch. |
| C7 | One B governor; physical backing and independent policy ceilings distinguished; native overhead must be charged. | Payload-only independent C allocator or double-counting reusable pool backing. |
| C8 | Separate sealed ExpertDomain/RowDomain hotness/predictor types; prefetch is not demand heat. | One global expert/row heat score. |
| C9 | Zero-cache still reserves bounded staging/output; exact output must fit; larger transfers may progress in slots. | Skip missing expert or pin whole oversized table to avoid admission. |
| C10 | Canonical v1 and independent golden pins, no prototype persistence compatibility. | Unversioned native Rust layouts or local divergent hashes. |
| D1 | Ordered per-item acceptance and per-segment completion; empty/all-rejected refuse. | One accepted bit, short status vectors or implicit missing reject. |
| D2 | Cancel linearizes before/after consumer publication, not DMA completion. | Revoke a published consumer or free complete-but-unretired bytes. |
| D3 | Issuer+sequence tickets and context-bound sealed allocation registry. | Epoch-only ticket uniqueness or cloneable metadata-pointer surrogate. |
| D4 | Borrow every release/retire handle; failure leaves retry ownership. | Consuming Busy release without returning the lease. |
| D5 | Distinct state/src_gen/dst_gen, checked at every publication; homogeneous triple per ticket. | Equate state epoch and allocation generations. Group mixed triples into multiple tickets; flagged below. |
| D6 | One B StateBundle; all/trailing planes, per-group aliases, high-water checks. | Activation-only PP envelope or D-owned duplicate model state. |
| D7 | Common governor pin/charge states visible through quarantine and retirement. | Separate peer free-VRAM ledger or graph pins omitted from retirement. |
| D8 | Canonical JSON shared metadata + domain-separated SHA framing and fixed-width fields. | Hand-written new binary ABI before a complete decoder/fuzz corpus; flagged wire decision above. |
| D9 | Peer route/grant proof must bind live context, topology and exact binary; unknown/expired proof is unavailable. | Persisted capability bool as enduring authorization. Native proof issuance remains D implementation. |
| D10 | Keep CPU contract evidence distinct from byte/model/performance/G2/G5/G6 receipts. | PASS from unsupported/missing rows or allocation-only evidence. Full collectors remain D work. |

## Per-WP migration table

File paths below are relative to each WP root. Shared names must be imports/re-exports,
not another authoritative branch-local definition. Rebase onto this lane before editing.

| WP / provisional type | Frozen replacement | Files to change on resume | Behavior change |
|---|---|---|---|
| A `object_store::ObjectLease`, local ObjectStore/StoreTxn | ObjectManifest; contracts::ObjectStore with associated Transaction; charged ObjectLease | `src/object_store/mod.rs` | Yes: explicit lease/release, borrowed commit/cancel, durability class, key/manifest revalidation. |
| A scaffold ObjectKey/Error/Ticket/Completion | contracts::*; issuer/sequence/Epochs; complete vectors | `research/spill-a-20260919/scaffold/contracts.rs`, `src/io/{mod,retirement}.rs` | Yes: split epochs, reject ownership, tombstones, distinct publication and physical retirement. Remove scaffold after migrated tests pass. |
| A FakePinnedPool/PinnedLease | Implement contracts::PinnedLease; inject shared governor pins/permits | `src/pool/mod.rs`, `src/io/mod.rs` | Yes: no second budget; physical backing charged once; preserve retained ownership and explicit release. Fake memory is still not CUDA-pinned. |
| A device metadata / missing ready API | DeviceOwner/DeviceLease/ReadyView/Destination + TransferEngine | future `crates/memra-engine/src/tier_transfer.rs` | New: retained resources, registered destinations, context checks, take-once and retirement. |
| A extent/root digest/StorageSample | Wire/domain helpers + ObjectManifest; contracts::StorageSample | `src/object_store/mod.rs`, `src/telemetry/mod.rs`, storage tests/CLI | Yes: incompatible with prototype root hashes; explicit versions and unknown physical counters. |
| B record.rs shared types | contracts::{ProgramIdentity,KvBlockId,RecordLayout,StateBundle,...} | `crates/memra-kv/src/record.rs` | Yes: versioned canonical identity, group-specific page_count, Role/EncodingId, explicit extent/tensor, immutable sealing. |
| B local ObjectStore/TransferEngine/Completion | A contracts + adapter for existing HostPrefixCache | `crates/memra-kv/src/tiered/mod.rs` | Yes: synchronous physical store vs semantic hierarchy separated; no independent B transfer schema. |
| B TierBudget/AdmissionGovernor/PeerCapacity/PinnedLease | BudgetGovernor/ChargedLease, shared tier/peer contracts | `src/tiered/{mod,integration}.rs` | Yes: inject ONE governor, borrowed release, live pins, all dimensions, ns deadlines and three epochs. |
| B ReadyBlock/KvMaterializer | BlockLease plus sealed ReadyView and KvMaterializer | `src/tiered/mod.rs`, future engine materializer | Yes: stable local operands and explicit last-use retirement; no numeric crossing. Keep native active guard. |
| B tests/policy | Shared fixture schedules; existing policy primitives remain B-owned | `src/tiered/{tests,policy}.rs`, package dependencies | Adapt hashes/roles/types. No policy default promotion. B's sha2 Cargo.lock edge is **not needed** here and was not applied; resolve it with B's dependency/import rebase. |
| C TensorId/BankId/Segment/RecordLayout | Shared original-ID BankId, ByteSegment/EncodingId/Role | `crates/memra-tier/src/bank/types.rs` | Yes: exact multi-plane layouts; keep catalog masked-ID checks; no string codec conventions. |
| C BankTicket/ConsumerFence/BankLease/UniformLease | TransferTicket/Epochs, transfer-owned ready proof; shared lease/proof types | `src/bank/residency.rs` | Yes: per-transfer failures, all-or-none publication, actual resource/pin retention, explicit retirement invalidating aliases. |
| C Charge/BudgetPermit/SharedBudget | BudgetRequest/BudgetGovernor/ChargedLease | `src/bank/{types,residency,rows}.rs` | Yes: same governor as KV/peer; metadata/padding/scratch accounting, no Drop-only release. |
| C sync RowService/RowBatch/RowLease | gather→ticket→publish ordered RowLease | `src/bank/rows.rs` | Yes: multi-segment rows, exact duplicate order, asynchronous cancellation and consumer retirement. |
| C domain/predictor APIs | shared BankDomain/ExpertDomain/RowDomain/Hotness/PrefetchHook | `src/bank/types.rs`, bank tests | Mostly imports; keep domains separate, no prefetch demand heat. |
| D generic PeerRegistration/ContiguousCopy | sealed DeviceLease + checked shared ContiguousCopy | `crates/memra-tier/src/peer/mod.rs` | Yes: three epochs, owner/context fence checks, retained actual resources, no scatter ABI. |
| D per-copy Vec<Result<Ticket>>/PeerProgress | BatchSubmission/Completion with all indexed outcomes | `src/peer/mod.rs`, peer tests | Yes: unified zero-accept Err, homogeneous epoch triple per ticket, full completion cardinality. |
| D associated PeerLease/PeerPlan | concrete shared PeerLease/PeerPlan wrapping common governor charge | `src/peer/mod.rs` | Yes: explicit bytes/alignment/source/consumer/epochs; borrowed release and graph-safe pins. |
| D StateBundleAdapter | shared concrete StateBundle/ProgramIdentity signature | `src/peer/mod.rs`, future Step/Qwen adapter | No shadow model state. No paused-model execution. |
| D local PlacementReport/DeviceBudget/RouteBudget | shared versioned report metadata | `src/placement/mod.rs`, placement tests | Add versions, replicas breakdown, explicit measured rate/estimate labels; preserve checked arithmetic. |
| D topology proof/receipt harness | D implementation stays D-owned; consume shared contracts | `src/peer/topology.rs`, D battery tests | Bind grants/proofs to live contexts/binary; do not treat CPU observations as permanent authorization. |

All `src/...` shorthand in A/C/D rows is under `crates/memra-tier/`. Lead wires their
actual module exports and integration test targets only when their migrated slices arrive.
No engine dependency edge is added before its owned adapter exists.

## Fixture coverage and limitations

`tests/contracts/conformance.rs` contains reusable, backend-generic schedules for all
six surfaces: ObjectStore, TransferEngine, TierStore, BankedResidency, RowService,
and PeerBackend (with PeerCapacity accounting fixtures). WPs can path-import this exact
file into their own backend test targets; do not copy expected outcomes into new tests.

Additional concrete fake tests exercise:

- Full acceptance cardinality and deliberately broken aggregate/short-vector rejection;
  partial short/rejected entries, corrupted valid bytes and missing/duplicate planes.
- Cancel-before-publish after completed I/O; publish-before-cancel; late completion;
  independent disk/DMA/consumer/graph retirement; unknown-status/shutdown quarantine.
- Shared governor headroom, foreign/double release, explicit pins, busy borrowed handles,
  retained bank alias resources and invalidation after retirement.
- All three epochs independently; program numeric/stream and all ten namespace digests;
  parent-key differences, full-key/manifest collision defenses, immutable seal/high-water.
- Local-only peer publication, context issuer refusal, contiguous bounds and overflow.
- Original masked expert rejection before reads, missing macro-scale planes, PerRecord
  uniform refusal; rows [5,2,5,9], physical read deduplication, real opaque row bytes,
  4KiB-straddling 264-byte rows, and 32-row output larger than one 4KiB staging slot.
- Negative placement headroom/estimate retention and host-bounce mislabel refusal.
- Three compile-fail doctests: scatter list cannot be ContiguousCopy, BankLease cannot
  be UniformLease, ChargedLease cannot be cloned.

The suite is **36 integration tests + 3 compile-fail doctests**, no ignored tests.
Fake producers/fences are controlled CPU schedules; this does **not** implement or
qualify disk fsync/GC, EINTR/ENOSPC handling, actual pinned memory/DMA, address-stable
CUDA graphs, scheduler fairness under service load, native KV materialization, Hy3/PLE,
Step, or four-card topology. A's OS-fault tests and B/C/D model/backpressure batteries
remain required on their implementations. No actual GPU or full engine build was attempted.

## Open integration items for the lead

1. **Review the canonical-JSON choice** versus D/A's proposed all-binary metadata.
   Implemented safety choice is frozen here; changing it requires version/fixture migration
   before any A durable artifacts. This is not a silent claim that the prototype wire matches.
2. **Review homogeneous epoch triples per ticket.** Safe first boundary groups different
   allocation generations into separate tickets rather than weakening stale checks. If a
   backend requires heterogeneous tickets, redesign the ticket/per-item epoch association
   centrally and rerun all six consumers; do not simply relax comparison.
3. **Production governor implementation/queues remain B-owned**, including tenant fairness,
   dirty-backlog backpressure, metadata/allocator overhead and actual configured caps. The
   enum/order/accounting semantics are frozen; no hardware budget or queue timing was invented.
4. **Native authority binding remains A/D-owned**: register actual CUDA allocations on the
   existing designated owner; observe real fence/context retirement; coordinate all-client
   acknowledgement before tombstone deletion. The helper types cannot prove driver truth.
5. **Persistent durability and orphan GC remain A follow-ups**, fail closed or retain safely.
   The thin async ObjectStore wrapper is explicitly deferred. No default or new flag is needed.
6. **B physical COW/continuation adapters and C/D consumers remain unqualified.** Keep
   Qwen native KV, Hy3/PLE and Step generic PP as the existing-model qualification surfaces.
   Paused-model work and new model code are outside this lane.

These items do not reopen the owner's ten decisions. CPU schema/tests are available for
WP rebases; they are not a substitute for the later integrated native/hardware gates.

## Verification receipt

Code and fixture tip checked: **`20cce6e99345bb71c6c244ba4ecfbc374ec3f215`**.
Subsequent decision/receipt documentation does not alter crate source or fixture bytes.
Raw output: [`verification.log`](verification.log). Reproducer:
`python3 research/spill-lead-20260919/verify-freeze.py` (bounded subprocess commands,
no shell evaluation, no env dump, no GPU builds). Toolchain: rustc/cargo 1.97.1 on macOS.

| Exact command | Actually ran | Result |
|---|---|---|
| `cargo check -p memra-tier --offline` | Yes, initial workspace wiring | PASS; Cargo itself added only memra-tier's lock entry. |
| `cargo fmt --all -- --check` | Yes | PASS |
| `cargo check -p memra-tier --offline --all-targets` | Yes | PASS |
| `cargo test -p memra-tier --offline` | Yes | PASS: 36 integration + 3 compile-fail doctests |
| `cargo check -p memra-kv --offline` | Yes | PASS; pre-existing macOS unused AsRawFd import warning in `memra-gguf/src/source.rs:20`, untouched. |
| `git diff --check` | Yes | PASS |
| `bash tools/check-flags.sh` | Yes | PASS: 864 runtime literal reads, none uncovered; no new reads in this lane. |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | Yes, extra check | PASS |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | Yes | PASS: 8 payload and 3 canonical-wire pins |
| Full engine/nvcc/GPU/model battery | **No** | Out of this CPU freeze scope; native CUDA toolchain/hardware absent. |

Observed development red checks were repaired before the receipt: an initial workspace
text replacement also matched a dependency table (fixed before first successful Cargo
lock generation); Rust test borrow checking caught overlapping fake-governor borrows;
strengthened context-issuer validation caught the fake peer completion's old constant
issuer; strict Clippy caught style/type-complexity lints (owned rejection results retain
a documented large-error allowance to avoid allocating when submission is rejected).
Two shell-log wrapper attempts were refused by the harness before execution; the direct
subprocess verifier above avoids shell evaluation/environment inspection entirely.

SHA-256 file pins from the passing receipt:

- contracts.rs: `8d65f2ae27f481f82b52bcac0e8c53ce8c5d929e1c6f735ae26241aa97bef7e6`
- fixtures.json: `1b722a5eb6022b60bf7c62817c3ce773ea6459131af395a7df73dcff41aff18b`

No `thiserror` was added: it was absent from the pinned lock. sha2 0.10.9,
serde 1.0.228 and serde_json 1.0.150 were already locked; no network resolution required.
B's separate memra-kv→sha2 edge was not needed and was deliberately left for B's rebase.
No push, PR, tag, merge, deployment or support-state promotion occurred.
