# Freeze v1.3 draft — real-backend conformance bindings

> Historical draft. [FREEZE-V1.3.md](FREEZE-V1.3.md) freezes the two accepted
> additive ownership seams. This draft's native binding matrix remains required;
> unresolved native qualification cells are not cleared by the CPU freeze.

Repository: **avifenesh/memra**. Planning baseline: PR #560 head
**dcbc1bc089a6f95ad7f5e4bc7e90aa658dadd69d**; Lane E review merge
**d3d054a8e2a177c10e5e923614945eba96c1d069**.

**DRAFT / text only.** No trait, runtime, wire format, frozen fixture, test assertion,
or native capability is changed or approved here. **WIRE_VERSION stays 1.**
A owns real `CudaTransfers`, B native `KvMaterializer`, C device-published rows;
lead owns cross-lane integration and any decision that would change the contract.
Their implementations are planned inputs, not evidence that these bindings exist.

## Binding rule

Re-use the exact source-shared schedules exported from
`crates/memra-tier/tests/contracts/conformance.rs` (v1, `revision_v11.rs`,
`revision_v12.rs`). Name both the schedule and the **concrete actual backend** in
each test/receipt. A real CUDA allocation underneath a CPU fake is not a native
TransferEngine binding. Mock snapshots test metadata, not event truth or DMA.

Do not relax assertions, fast-forward real completion, or inject fabricated
FenceIds to make an asynchronous backend fit. Deterministic test-only fixtures
may arrange a pending producer/consumer/graph use and then physically complete
it through the backend's real owner. They must record that distinction. A required
schedule with no safe native fixture is **HELD**, never silently skipped or PASS.

## Required schedules by trait / owner

All existing CPU bindings remain enabled; the following are additional native
bindings or integration reruns, not replacements. Physical routes apply only
where implemented and provisioned; unsupported routes must refuse, not bounce
through another route and inherit its result.

| Trait / seam | Exact shared schedules to rerun | Real fixture and observations |
| --- | --- | --- |
| `TransferEngine` — A `CudaTransfers` | `transfer_cancel`, `transfer_complete_cancel`, `transfer_zero_accept`, `transfer_lifetime`, `transfer_completion_bytes` (uses `framed_logical_bytes`) | Real H2D and D2H, framed NVMe→host→device composition where supported; each accepted/rejected item retained in order, zero-accept returns original owned inputs, producer-complete/pre-publication cancel, stale ticket/epochs, unknown completion quarantine, consumer and graph pins, tombstone acknowledgement. P2P is separately held for actual directed peer hardware; it cannot be claimed from H2D. Add native take-once/destination-lifetime red/green cells around the existing schedules. |
| `PinnedLease` / `PinnedFixture` — A | `pinned_lease`, `pinned_quarantine` | Real CUDA-pinned pool, uninitialized read refusal, optional-slot reserve for demand, moved destination ownership, pending DMA and consumer reuse exclusion, last-pool drop while quarantined. Charged backing stays Busy until physical retirement or remains quarantined on unknown shutdown. Real storage/transfer owner must retain the same lease, not copy into an uncharged substitute. |
| `BudgetGovernor` — shared A/B/C owner | `budget_governor`, `governor_priority` | Same injected governor used by native pools, transfer tickets, destination allocations and materializer/row consumers. Observe each dimension before allocation, during pending work and after explicit release; no double-charge/double-release or private unaccounted allocator. Priority schedule must drive the actual queue dispatcher; an enum sort is not evidence. |
| `ObjectStore` — A | `object_cancel`, `object_publish_release`, `object_logical_bytes` | Real framed ExtentStore/FileBackend feeding the native transfer, cancelled publication, committed manifest/lease, exact logical bytes vs physical framing, source-generation identity. Rerun existing short/corrupt/missing-sibling, persistence/reopen and GC ownership tests. Buffered and O_DIRECT are named independent arms; neither proves NVMe ancestry. |
| `TierStore` — B | `tier_cancel`, `tier_identity`, `tier_unknown`, `tier_miss`, `tier_logical_bytes` | Bind to the actual store used by native materialization/scheduler, with the real transfer owner. HostReady must not authorize attention. Observe reserve→prefetch→load→publish, cancellation and unknown retirement; retain sole active backing and graph addresses. A reference control-flow store calling a real copy is not an integrated native TierStore receipt. |
| `KvMaterializer` — B | `kv_materializer` | Real pre-admitted stable local operands. Wrong values for all ten ProgramIdentity fields; independent state/src/dst epochs; missing and corrupt siblings; exact native byte/segment order; foreign consumer fence issuer/device/generation; valid retirement, double-retire and post-retire refusal. Capture device bytes/addresses through a native observation hook, not host expected-byte echo. Native q8_0 K/q5_1 V and opaque packed-record fixtures stay separate; packed records do not qualify any model's math. |
| `StateBundleAdapter` — B | `bundle_integrity`, `bundle_planes` | Bind the adapter actually capturing/restoring complete native state, including required recurrent/scale/position planes and committed high-water. Exercise all identity mutations. There is no cancellation method on this trait; cancellation belongs to store/transfer/consumer lifetime. No invented trait member or dropped plane. |
| `BankedResidency` — C | `bank_cancel`, `bank_complete_cancel`, `bank_identity`, `bank_uniform`, `bank_failed_sibling`, `bank_lifetime`, `bank_completion_bytes`; `bank_source_install` at the installation seam | Actual device publication binding, checksum-bound source records, mixed-layout rejection at uniform-only consumers, original router ids/masks, required scale siblings, cancel before/after publication, unknown/consumer/graph retirement. Native HostExps compile probe alone is insufficient; qualify only the consumers actually wired. |
| `RowService` — C | `rows_order`, `rows_complete_cancel`, `rows_bytes`, `rows_identity`, `rows_failed_sibling`, `rows_lifetime`, `row_completion_bytes` | Actual device row publication, repeated/out-of-order ids preserved, exact expansion program, missing/corrupt sibling failure, live row leases held through the existing compute consumer. Retain the same-program PLE OFF/ON tiny gate and add continuation/churn coverage before broadening its scope. Do not relabel host-only RowService as device-ready. |
| `DeviceOwner` / ReadyView — A/B/C | `ready_owner` plus per-trait completion schedules above | Real registered CUDA allocations and actual producer/wait/consumer observations. Existing schedule checks identity, binding and Busy accounting; separately demonstrate real pending-event refusal and delayed last-use/graph retirement. See the synchronous-fixture caveat below. |
| `PeerBackend`, `PeerCapacity` — D / later multi-card qualification | `peer_cancel`, `peer_complete_cancel`, `peer_lifetime`, `peer_zero_accept`, `peer_submit_epochs`, `peer_completion_bytes`, `peer_capacity`, `peer_directed_grants` | Actual directed context/pool grants, both directions, topology-valid local destination, source/destination owner generations, real producer and consumer fences. Single-card fake capacity does not discharge these gates. Keep held until the required hardware/route exists. |

`acceptance` and `framed_logical_bytes` are shared helpers, not additional device
traits. Exercise their indexed outcomes/logical-length cases through the real
TransferEngine/BankedResidency/RowService/PeerBackend bindings; label snapshot
mutation tests distinctly from real short-I/O/copy fault injection. Neither
ObjectStore nor TierStore has a completion-byte poll API to invent.

## Contract ambiguities native implementations must resolve explicitly

1. **Three different event facts.** `producer_done` means producer completion has
   been observed; `consumer_fenced` means the dependency/wait was installed, not
   consumer completion. `consumer_done` must map to a registered actual last-use
   event on the correct CUDA owner/device/generation. Document event registry,
   context ownership, event reuse rules, observation/query failure behavior and
   exactly which stream has the wait. A numerical FenceId cannot prove ordering.
2. **Retirement request versus physical retirement.** `retire(..., done)` may
   register a completion event; it cannot alone make allocation reuse safe while
   that event is pending. TransferEngine exposes `retired()`; KvMaterializer does
   not. B must state how operand invalidation is separated from backing release,
   and who polls/drains pending consumer/graph uses after logical retire. Unknown
   events retain pins/charge; they cannot become success on Drop. If the existing
   API cannot express this safely, stop for a lead decision rather than change it.
3. **`take_destination` is publication and ownership transfer, once.** Document
   whether the moved public Host/Device lease shares retained internal backing,
   how dropping it before DMA/consumer completion behaves, and who eventually
   releases its budget. Backend must keep pins after take; caller cannot reclaim
   by acknowledging or cancelling the ticket. Test take twice, take after cancel,
   drop taken destination early, retire while destination lives, and acknowledge
   only after all outstanding uses retire. D2H host byte validity is separate
   from slot-reuse permission. Bind ReadyView lifetime to the real destination.
4. **Asynchronous fixture compatibility.** `tier_cancel` expects deterministic
   `advance` phases; `rows_order` publishes immediately. Fixtures must arrange
   actual work readiness before those assertions without changing the assertions
   or production scheduling. `ready_owner` ends with direct `retire_binding` and
   has no completion callback: use an already physically drained identity fixture
   there, and a separate native pending-lifetime cell. `kv_materializer` expects
   successful retirement with its supplied `done`; supply an actually completed
   event for that case, plus a separate delayed-event cell. Otherwise these tests
   would merely authorize metadata to pretend CUDA finished.
5. **Cancellation linearization / partial acceptance.** Distinguish queued,
   submitted-but-unknown, producer-complete, published, consumer-pending and graph-
   pinned states. Once any operation might have been accepted, return its ticket
   and quarantine uncertainty, never return all inputs as zero accepted. Complete
   the real producer before asserting post-completion/pre-publication cancel.
6. **Native byte truth.** Device checksums and valid bytes must describe the bytes
   consumed, including every scale/auxiliary plane. Framed I/O length is not valid
   payload length. Copying bytes into stable scratch must preserve layout and
   numerical class, not requantize or alter attention to fit a smaller window.
7. **Errors and shutdown.** Specify owner-thread drop/drain behavior, event-query
   failure, context loss, worker panic and partial batch submission. Quarantine
   retains accounting even if recovery requires shutdown; do not intentionally
   poison a production context to exercise a test. Keep live graph addresses and
   the only active copy pinned until a proven safe terminal state.

These are implementation/fixture questions against the frozen contract, not
permission to revise it. Each native lane reports its answer and evidence or
marks the question held; lead adjudicates cross-lane disagreements.

## Native conformance receipt — minimum contents

- **Source/binary:** full repository commit; complete dirty/source-fragment manifest
  if any; SHA-256 of executed test/gate binary and loaded kernel/module artifacts;
  native backend type and trait; imported schedule source SHA-256; toolchain,
  CUDA driver/runtime, build target/features. An unrelated green CI binary is not
  the tested binary. Name filtered and ignored tests, not just exit status.
- **Artifact lock:** immutable source revision plus file/byte manifest hash,
  serialized plan hash and full ProgramIdentity (artifact, plan, numeric, stream,
  tokenizer, template, adapter, modality, position, tenant salt). For fixture-only
  tests, publish deterministic fixture generator/input hashes and explicitly mark
  no model artifact; never substitute a synthetic lock for checkpoint evidence.
- **Collector:** stable **CELL run_id**, start/end UTC and monotonic times,
  exact argv, exit/timeout/refusal/failure text, hash-bound raw stdout/stderr,
  collector capture/lock proof and integrity result. Keep interrupted and failed
  attempts immutable. Resolve PR560 collector findings before relying on their
  affected classifications/path provenance. A capture remains qualification=false
  until the separately scoped gate passes; integrity is not numerical equality.
- **Hardware/regime:** hardware shape, device ordinal/count, CUDA capability,
  directed route if used, observed **power.limit and power.max_limit**, and actual
  clock/temperature/VRAM telemetry at 250 ms; idle compute-app snapshots inside
  the canonical campaign lock. Unknown fields remain unknown. Private machine ids,
  addresses, locations and rates do not enter public receipts.
- **Storage:** actual tested path's filesystem/device ancestry, explicit requested
  and actual backend, logical/storage/I/O bytes, fallbacks and retained errors.
  Overlay O_DIRECT acceptance stays overlay; no NVMe claim without actual ancestry.
- **Event/lifetime trace:** ordered ticket/item/segment ids, epochs, allocation ids,
  event issuer/device/generation, producer query, installed wait, observed consumer
  completion, graph pin release, publication/cancel result, take destination,
  retirement/acknowledgement. Snapshots must show charges and pins remain live
  until the last use, then drain exactly once (or remain quarantined if unknown).
- **Numerical result:** full native output/state byte hashes and independently
  obtained expected hashes, checked counts/ordering, program/shape, negative-arm
  outcomes. Serving claims additionally require same-program continuation under
  demote/reload, churn and concurrency, not just memory copy equality. Performance
  decisions require the separate balanced same-window AB/BA N>=5 protocol and
  serving percentile/throughput evidence; conformance timings are not that study.
- **Disposition:** PASS/FAIL/REFUSED/HELD **per trait × schedule × backend × route**,
  raw pointers and precise scope. Separate CPU, Linux execution, real CUDA,
  checkpoint and serving gates. No promotion for required unrun/ignored cells.

## Proposed freeze exit

Lead independently reruns the integrated CPU/compiler suites and relevant native
bindings at the exact merged lane revision, validates receipt hashes and public
contract, then decides whether v1.3 is frozen. A/B/C native conformance does not by
itself qualify NVMe, multi-card transport, model serving, or a new model family.
No V4.1 code/work, dependency, env read, frozen-file edit or new support claim is
part of this draft. Unresolved real-backend ambiguities remain explicit blockers
for the affected surface, not excuses to substitute a different program.
