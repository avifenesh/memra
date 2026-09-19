# Day-2 conformance review — Session D, not integration lead

Frozen input: `lane/spill-lead-20260919` at `259ff819` (canonical JSON v1 and homogeneous
`Epochs {state,src_gen,dst_gen}` accepted). Reviewed the entire shared contract and
`tests/contracts/{conformance,peer,transfer,services,rows,wire,support}.rs`.
No shared schedule or contract was edited. This reviews coverage, not hardware readiness.

## What actually runs on D

`tests/peer/mod.rs` path-imports the **unchanged** `../contracts/conformance.rs`.
`fake::shared_schedule_runs_on_d_byte_fake` runs its `peer_cancel` against D's own byte-moving
fake, using sealed DeviceOwner registrations, distinct allocation generations 19/31 and
state epoch 7. D then drains DMA/consumer/graph, acknowledges and explicitly releases both
charges to zero. No fake is exported as a native CUDA backend.

There is **no reusable PeerCapacity schedule** in the freeze. The concrete shared
`peer_capacity_grants_budget_source_lifetime_and_explicit_release` test runs in the contracts
suite, but its private fake is not D's implementation. D independently exercises charge
validation, a single injected shared governor, denied grants, invalid alignment, undercharged
peer bytes, borrowed Busy release, foreign/double release, late old-state reclamation and
quarantine/graph retention. We do not claim to have run a nonexistent generic capacity schedule.
The lead approved proposing that schedule for the next freeze revision, not editing it here.

## Six reusable schedules versus the day-one fixture plan

Legend: **yes** = explicit assertion in this reusable function; **partial** = narrower assertion
or dependent on implicit fake timing; **no** = absent. N/A means another trait owns the surface.
The full concrete freeze suite has substantially more coverage than these six functions.

| Schedule | Per-item acceptance | Cancel after completion | Lease accounting / successful explicit release | Three epochs | Unknown quarantine + retired | Program identity refusal | Scatter compile-fail | Uniform/mixed refusal |
|---|---|---|---|---|---|---|---|---|
| `object_cancel` | no: synchronous one-byte put only | yes for put-complete / root-unpublished; not post-commit cancel | no | no | N/A sync; no retirement case | no | N/A | N/A |
| `transfer_cancel` | no | no: cancel then supplied completion callback | partial: retired/ack assertions, no byte ledger or successful lease release | no | partial: unknown only **after acknowledge**, not unknown-status quarantine | no | outside schedule | N/A |
| `tier_cancel` | no: no missing group/owner injection | yes: Loading→Ready phase, then cancel before ready publication | partial: premature release refuses; no successful release or exact charges | no: uses only original epochs | partial: not-retired, no explicit unknown producer/graph schedule | no: original program only | N/A | N/A |
| `bank_cancel` | no | partial: stage→cancel without a completion hook; concrete fake completes at stage, asynchronous backend may not | partial: not-retired, no release/ledger assertion | no | no explicit unknown case | no | N/A | no |
| `rows_order` | partial: logical IDs/order/duplicates, not rejected read/byte values | partial: cancel after **publication** returns AlreadyPublished; no completed-but-unpublished cancel | partial: release Busy, no successful retirement/release or dedup charge assertion | no | no | no | N/A | no |
| `peer_cancel` | partial: cardinality/order checked by submission.validate, but no predetermined rejected item | no explicit completion hook; D fake is pending, lead fake reports completion immediately | partial: not-retired and premature ack refusal; **no capacity API** | no | no explicit quarantine/recovery | N/A byte transport (StateBundle owns identity) | outside schedule | N/A |

The distinction matters: a fake which completes synchronously on `stage` or `submit` cannot
make the same reusable schedule prove completion-before-cancel for a genuinely asynchronous
backend. The schedule must drive and observe that state explicitly.

## Coverage outside the reusable functions

The unchanged concrete freeze suite exercises partial/short/rejected vectors and deliberately
broken aggregate completion, wrong checksums, missing planes, all three epochs, full program
namespace changes, source ownership and owner-fence issuer checks. It covers Busy then explicit
release, unknown shutdown retention, consumer/graph delay, row dedup/4KiB straddles, masked
original expert IDs, macro-scale completeness, and actual Uniform-vs-PerRecord refusal.
Three compile-fail doctests separately reject scatter→ContiguousCopy, BankLease→UniformLease,
and ChargedLease cloning. Those doctests run with `cargo test -p memra-tier --offline`.

D's additional tests cover nonempty actual copied bytes, accepted/rejected original item
indices, all-rejected input ownership return, empty refusal, wrong local consumer, wrong fence
issuer, short/corrupt completion, all three independently stale epochs, and heterogeneous epoch
triples rejected **without accepting any input**. A caller can drop its original source handle;
the accepted copy and owner registry retain backing/charge until acknowledged retirement.
Topology tests keep all twelve fake directed edges distinct and invalidate observations on
context, binary or topology changes. They are not live grants or route measurements.

## Proposed additional generic cases for the lead (text only)

1. Add `peer_capacity` factory/hooks schedule: use ONE injected governor shared with a non-peer
   reservation; assert physical device and overlapping peer counters, over-capacity/undercharge,
   directed denied context and pool grants, stale state, foreign and double release, Busy retry,
   pin/graph refusal, successful old-state release after physical retirement, and exact zero
   final usage. Do not add a second allocator to the schedule.
2. Add explicit completion hook and poll assertion to each async schedule. Run pending cancel,
   producer-complete but unpublished cancel, publish then cancel, delayed consumer and delayed
   graph fence, and unknown-status quarantine. Unknown cannot credit quota; recovery requires
   all physical uses retired before acknowledge. Verify failed acknowledge preserves retry.
3. Standardize three-item acceptance `[accepted, short, rejected]` with full original indices,
   byte counts/checksums and ownership of rejected inputs; test empty and all-rejected batches.
   Also reject index 0 and index 1 separately, testing the last original item ID so compacting
   accepted descriptors cannot accidentally redefine the API's item indices.
4. Parameterize **each** of state/src_gen/dst_gen stale at submit and publication. Test heterogeneous
   epoch triples return all owned inputs with zero accepted work; splitting into homogeneous
   tickets must succeed. Include foreign issuer and tombstone lifetime through explicit ack.
5. For TierStore/StateBundle adapters, vary each ProgramIdentity digest separately, plus parent,
   salt and committed high-water. For banks/rows vary artifact/tensor/original ID/layout. Byte
   PeerBackend has no program argument; do not invent a shadow program schema to test it.
6. Promote all/trailing/missing-auxiliary and corrupt-sibling fixtures into reusable consumer
   schedules. Check same-program optional miss separately from fatal sole-active-backing loss.
7. Extend rows_order to compare byte payloads/scales (not only returned IDs), exact physical-read
   dedup counts/straddles, complete-before-publish cancellation, and post-retirement release.
   Exercise a batch larger than the staging slot with bounded memory and failure localization.
8. Run UniformLease acceptance and homogeneous-subset-of-PerRecord refusal against C's real
   catalog adapter, preserving original rejection ownership and macro-scale completeness.
   Keep compile-fail cases at shared API level; add source-bound fixtures rather than copies.
9. Shared-governor fairness, dirty-backlog bounds and actual graph-address stability require
   B/native owner integration and timed service cells. A sorted Priority vector or fake graph
   boolean does not prove those properties under contention.

## Qualification boundary

No CUDA, NVMe performance, model execution, kernel-check, argmax/spec battery, official Step
FP8 PP ladder, two-PRO or four-card gate ran in this CPU migration. Same-device/fake byte
identity does not prove PCIe P2P. G0–G7 remain pending for the integrated program.
