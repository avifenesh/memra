# Freeze v1.3 — additive ownership contracts

Repository: **avifenesh/memra**. Base: Lane E integration `e589c96b` incorporating
`origin/lane/spill-integ4-20260920` (`5ffe935c`). Lead accepted the two additive
seams below after A's real-backend findings and B's materializer review.

**Frozen contract text, not native qualification.** This document supersedes the
ambiguity/approval wording for these two seams in [the draft](FREEZE-V1.3-DRAFT.md).
The draft's required native binding matrix, event-truth rules and held hardware
surfaces remain required; no earlier schedule or assertion is removed. A owns
CUDA implementation, B native materialization, C device rows, D receipt admission.
E changes the shared contract and CPU conformance only. Native runs remain HELD.

## 1. Synchronous allocation hand-back

The concrete CUDA owner exposes the fallible operation
`take_device(&DeviceLease) -> Result<CudaSlice<u8>>`. `CudaSlice<u8>` denotes the
**original allocation**, moved to the caller, not copied, freed, reallocated,
requantized or replaced. The method is not a CUDA type in `memra-tier` and is not
an associated type/member added to `TransferEngine`. A implements the concrete
owner seam; E introduces no CUDA dependency or runtime backend substitute.

Invariants:

1. Validate owner thread/context/issuer, allocation identity and generation before
   mutating ownership. Foreign, stale, already returned or already released leases
   refuse. A failed attempt retains a usable retry handle and its charge.
2. Hand-back is synchronous and possible only with **no live pins**: source DMA,
   producer, bound consumer, graph, borrowed/retained alias and uncertain work all
   prevent hand-back. A registered wait is not an observed completion. Unknown
   completion remains quarantined, never silently completed by Drop.
3. Atomically remove the allocation from the owner's registry and remove its tier
   governor charge exactly once; return ownership without freeing its bytes.
   Do not unregister first and discover a Busy accounting release afterward.
   No tombstone/alias can subsequently resolve it as an owned tier allocation.
4. Hand-back transfers accounting responsibility to the receiving allocator; it
   does **not** claim the physical allocation disappeared. If the returned slice
   remains tier-resident, the receiving owner must have admitted the new lifetime
   before dropping the old charge. No uncharged tier memory window is permitted.
   This seam is also the path back to a pre-existing caller-owned allocation.
5. Duplicate take cannot double-uncharge or yield another owning slice. A valid
   caller-owned allocation id/address/length and its full bytes are unchanged.
   The caller ultimately owns its free; the tier owner no longer does.

`DeviceOwner::release` keeps its existing destruction meaning. It is **not**
redefined as native hand-back. The CPU fixture combines that existing registry
with a retained resource cell, returning the original Vec allocation and checking
its address, bytes, invalidation, charge, foreign-owner and duplicate-take arms.
That demonstrates the ownership schedule, not native `CudaSlice` extraction.

## 2. Per-side transfer retirement

Add to the frozen trait:

```rust
fn retire_source(&mut self, _ticket: &TransferTicket) -> Result<()> {
    Err(Error::Unsupported)
}
```

The **default body returns `Unsupported` without mutation**. Every existing
backend continues compiling without implementing the method and retains the
whole-ticket path. Unsupported is a capability refusal, not permission to drop
source ownership early. No backend or caller may interpret it as success.

Invariants:

1. Source ownership is independently retirable once the owner has observed the
   producer event complete **and no live consumer is bound to that source**.
   Include source graph captures, borrowed aliases and earlier source users in
   this condition. For composed I/O, every producer that can still read/write
   that source must have completed. Unknown submission/query status retains the
   source allocation, pins and charge.
2. Success releases the ticket's source holds only. The source's independent
   owning lease may still hold its charge; release that lease explicitly through
   the governor. A destination lease keeps its **own governor-charged lifetime**,
   including a destination already moved by `take_destination`.
3. Destination consumer waits, last-use events and graph pins remain in force.
   Source retirement neither publishes bytes nor revokes prior publication,
   takes/frees the destination, signals consumer completion, nor acknowledges
   the ticket. Cancel still has its existing publication-linearization meaning.
4. `retired(ticket)` retains its **whole-ticket** meaning: every disk/DMA,
   source/destination consumer and graph use has retired. Source retirement alone
   cannot make it true or permit acknowledgement. Whole-ticket retirement need
   not mean an independently held destination allocation has been freed.
5. Success is idempotent until acknowledgement, with no second release. A foreign
   or stale ticket refuses without changing either side; after acknowledgement
   the ticket is unknown. Pending/uncertain source retirement refuses Busy or
   Quarantined, retaining retry ownership. A backend unable to separate sides
   returns Unsupported and retains the original whole-ticket lifecycle.
6. For a multi-item ticket, success covers all accepted sources. Preflight all
   sources before mutating ownership; a failed attempt must not partially retire
   them. Rejected inputs are already returned and cannot be freed by this method.
   The H2D CPU binding is one-item; multi-item and other routes require their own
   native binding and remain held rather than inheriting that scope.

## 3. Additive shared schedules and current bindings

Source: `crates/memra-tier/tests/contracts/revision_v13.rs`, re-exported from
`conformance.rs`. Import those exact schedules into native tests; do not fork or
relax them. Fixture hooks arrange/observe actual native events for a native run;
CPU hooks only model ownership. All v1/v1.1/v1.2 schedules remain enabled.

| Schedule / test | Checks | CPU binding | Native disposition |
| --- | --- | --- | --- |
| `device_hand_back` | producer pending, unknown even after producer completes, consumer pending, graph pending; charge/registry survive each refusal; original allocation/bytes returned once, invalidated handle, no double release; foreign owner refuses | `v13_bindings.rs`, existing `DeviceOwner` and shared `Governor`, original Vec backing | A HELD: real allocation, event, graph and accounting binding required |
| `transfer_source_retirement` | foreign/stale source ticket; producer/unknown/source-consumer/source-graph refusal; original source charge Busy until success; idempotence; destination charge and bytes survive source release; whole ticket waits destination consumer/graph; destination survives acknowledgement; post-ack source call refuses | existing `transfer.rs::Transfers` H2D fixture and shared `Governor` | A/B HELD: actual H2D/D2H lifetimes and any used composed route |
| `revision_v13_unmodified_backend_defaults_to_unsupported` | unchanged CPU production backend uses the trait default; no charge mutation | `services.rs`, existing `Objects` fake with production `CpuTransfers` | Compilation compatibility only, not support |

The H2D test uses the existing synchronous CPU publication fixture and separate
lifetime flags. It does not claim a pending CUDA producer published successfully.
Native bindings must physically complete the producer before taking its destination;
the source-pending red arm can be exercised before native publication. Fixture
adapters may retain the native destination after readiness within their observation
hook, but may not weaken any ownership, byte or charge assertion.

Additional required native cells from the draft remain held: zero/partial acceptance,
delayed last-use and graph pins, take twice/after cancel, early taken-destination drop,
unknown shutdown, multi-item source preflight, composed I/O, and each actual route.
Passing this CPU slice is not a waiver of those cells or serving-shape correctness.

## 4. Wire and evidence

**WIRE_VERSION remains 1.** No persisted struct, wire fixture, canonical hash,
ProgramIdentity field or allocation serialization changes. v1.3 is an additive
API/ownership contract revision, not a persisted-format migration.

Native evidence is separately bound to `native-conformance.schema.json` and the
validator proposal in `NATIVE-CONFORMANCE-VALIDATION.md`. Binary SHA-256, exact
source, artifact-or-fixture identity, collector CELL, power limit/max and per-
schedule verdicts are mandatory fields. Unknown data is explicit, never guessed;
HELD/refused/failed schedules cannot become qualification by schema validity.
No new model support, V4.1 surface, performance default, environment read, GPU
execution, deployment or release claim is made here.
