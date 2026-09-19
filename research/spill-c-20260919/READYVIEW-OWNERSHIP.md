# ReadyView, destination and consumer-fence ownership — day 4

Read-only references at memra integration `020d2047`; no CUDA implementation or
native compile was performed. This document defines the exact rig seam, not a
CPU stand-in for a GPU receipt. `consumer_fenced` means an installed consumer
ordering dependency, **not consumer completion**.

## Existing owner and handoff truth

| Existing code | Requirement for the native bank binding |
|---|---|
| `memra-server/src/worker.rs:1–9,12786–12830` | Dedicated CUDA-affine OS worker creates Engine and models; HTTP tasks send commands only. Create DeviceOwner/allocation and do H2D/event/publication on this owner. I/O workers return owned bytes/tickets, never CUDA objects/pointers. |
| `memra-engine/src/lib.rs:3696–3713` | Implicit cudarc event tracking is deliberately elided. No safety argument may depend on automatic cross-stream tracking. |
| `memra-engine/src/lib.rs:6465–6482` | Existing `with_moe_cache` serializes mutation of the one cache; avoid holding its lock during disk/worker waits. Retain current dispatch numeric selection. |
| `memra-engine/src/moe_cache.rs:389–415` | Before overwriting a victim, record prior compute event and make copy stream wait. If H2D/event setup becomes uncertain, drain; failed drain quarantines both slot and source. |
| `memra-engine/src/lib.rs:7996–8014` | `stage_expert_async` records copy event; `compute_wait` installs the consumer stream wait. The owner maps these actual event objects to FenceId, not a fabricated counter-only proof. |
| `memra-engine/src/moe_cache.rs:1107–1126` | Demand for pending prefetch installs compute wait **before** residency publication; wait failure preserves pending state and source ownership. |
| `memra-engine/src/moe_cache.rs:1197–1237` | Pending prefetch slots are absent from free and both SLRU queues; failed unknown copy retains source keepalives and quarantines slot. |
| `memra-engine/src/moe_cache.rs:829–839` | Current pointer rows rely on single-stream ordering for invalidation. New lease ownership must retain all row/slot allocations through last-use, not simply copy this assumption across streams. |
| `memra-engine/src/moe_cache.rs:1899–1958` | Shutdown explicitly drains compute/copy/read work; when CUDA cannot prove safety, backing leaks instead of being freed. New governor pins must survive the same unknown state. |

Paths in the table are relative to `crates/`.

## Frozen API mapping (no contract amendment)

1. **Bind exact destination:** owner allocates/adopts stable backing with device,
   generation, actual governor charge. `DeviceOwner::bind_destination` at
   `memra-tier/src/contracts.rs:1200–1211` binds ticket to exact allocation and
   checks destination generation. Full bank/source identity stays alongside it.
2. **Accept submission, then retain:** `TransferEngine` at `contracts.rs:1440–1489`
   returns all inputs only on zero acceptance. Partial or uncertain submission
   retains every accepted source/destination, per-segment status and charge.
   Pending bank slots never become reusable because poll/cancel returned.
3. **Producer done is insufficient:** validate all segment lengths/checksums/
   epochs/status with `Completion::require` (`contracts.rs:587–652`). On the
   owner, install compute wait for each copy; only then set each segment's
   consumer_fence and consumer_fenced plus aggregate consumer_fenced. A failed
   sibling prevents publishing the full logical bank batch, even if another
   segment was individually ready. Do not set device=true for CpuTransfers.
4. **ReadyView:** `TransferEngine::ready_view` delegates exact ownership checks
   to `DeviceOwner::ready_view` (`contracts.rs:1212–1245`): bound destination,
   current state/src/dst epochs, complete outcomes, owner/issuer/generation of
   consumer fences. The borrowed ReadyView authorizes the existing exact-byte
   consumer wrapper. `resolve` downcast alone is not readiness permission.
5. **Take once:** `take_destination(ticket,item,current)` moves the destination
   capability once; it does not retire backend pins. Save a ticket→BankId→slot→
   allocation map on the owner. Repeated take, stale epoch or foreign allocation
   must refuse. A typed UniformLease requires the *actual represented source*
   and all layouts, not `layouts.is_none()` copied beside arbitrary pointers.
6. **Consume unchanged math:** cached q8/f32 wrappers at
   `hybrid_forward.rs:19826–19900` and grouped/staged wrappers keep original
   lengths, qtype, row_bytes, macros and accumulation order. PLE uses original
   F32/BF16 gather/expansion then existing projection. No copy policy can switch
   a request between numerical classes.
7. **Last-use fence:** record actual compute completion **after the final
   consumer launch**, including all request aliases and graph uses. Supply this
   owner-issued fence to `retire(ticket, Some(consumer_done))`; `retired` must
   prove disk + DMA + all consumers + graph pins, then `acknowledge` removes the
   tombstone. `DeviceOwner::retire_binding` (`contracts.rs:1247–1254`) is only
   legal after that proof. Cancel before publication revokes publication, not
   the physical resources. Cancel after take is AlreadyPublished.

## Exact unknowns / rig acceptance

- A currently exposes CPU host transfers here; real pinned/CUDA destination
  adoption, FenceId↔CudaEvent registry and event destruction rules are unimplemented.
- Multiple devices/streams: per-owner event grants, wait failure/retry, directed
  transfer routes and rollback generations need D+A ownership review. Never
  transpose a same-device wait proof to a peer route.
- Graph capture/replay needs stable allocation pins through graph destruction;
  native cache pointer-row lifetime and multiple consumer tickets must be audited.
- Need non-vacuous delayed-copy, delayed-consumer, partial-submit, event-create
  failure, wait failure and failed-drain red arms. Retained resources must stay
  charged on unknown shutdown; actual pin release must be observed, not inferred
  from CPU drop.
- Native immutable loader installation and fixed-slot SLRU integration remain
  uncompiled. CPU BankService deliberately publishes host vectors only; its
  finish_host_use is not a CUDA last-use fence.

Run PLE fitting cells first on approved 5090, then full Hy3 mixed/pruned and PLE
on the designated non-serving PRO pair under the canonical whole-box locks.
Require original byte + logit + token identity and serving-boundary evidence;
CPU conformance and `git apply --check` cannot close prerequisite (c).

**Verdict: mapping complete for handoff; implementation/GPU ownership NO-GO.**
