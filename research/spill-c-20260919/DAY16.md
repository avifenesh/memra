# Session C day sixteen: Option C, the promote H2D through the TransferEngine (lead ruling 15, after B's receipts)

Scope: ruling 15, "Option C after B's receipts", the promote's H2D of contract-routed planes through the
same engine behind the same door, and the write-combined finding of day 15 with its first measured cell.
Tree: lane merge of main `1b354be59` (`48ec3f0c5`, PR #599 integ17 plus #601), census `e7d70a77a`, code
`25891baa9`, WC doc and replay `5b03a0a47`, docs `4e7a0a744`, gate matcher fix `75573cad3`. Target-card
cells ran on the `25891baa9` binary (sha256
`cd8e9c11493ff7538c2251af4d7dc2d1be153c95e427d77cf131f09aee808aa8`, release build 38 s on the cached
engine plus the test binary), one RTX PRO 6000 Blackwell Server Edition at its 600 W limit
(`pro-single-day16/card.csv`), through the canonical collector (`tools/tier-battery.py --rig pro-single`,
lock `/tmp/memra-gpu.lock`, inherited by every gate), N=1 (the WC pair N=5 per arm per order),
`executed-not-qualified`. Nothing here is a support state. Door doc: `HOSTPREFIX-DOOR.md` ("Option C",
the census written before code and the landed section); the WC finding: `WC-DESTINATIONS.md`.

## The census, in one paragraph (`HOSTPREFIX-DOOR.md` "Option C", commit `e7d70a77a`)

The promote was eleven steps of `host_promote_prefix_hit` over a host entry held BY REFERENCE and kept
resident afterwards: candidate, class and identity lease, the restore `tier_charge` on the device
dimension before any allocation, `device_entry_from_host` (per plane `plane_up`: `alloc_u8` then
`htod_u8_into` from `HostPlaneBytes::bytes()`, one `cuMemcpyHtoDAsync` on the owner stream with no
completion observation), the verify digest over the new device entry, the identity `require`, then
PUBLICATION at `insert_pinned_demoting` (the one point where the planes become the live entry) and the
`promote:` line. Two things the census settled before code: the contract's H2D CONSUMES its source
lease (`retire_source` drops it; `recover_source` returns it only for a cancelled ticket), while every gate
line depends on the host twin staying resident (the identity gate's r4 second attempt, lane B's typed
refusal in both arms, the tenant gate's re-demote), so moving the entry's lease into the op would empty
the twin on every ON promote; the engine already accepts a shared H2D source but the server had no way
to mint a second owner after the D2H ticket was acknowledged, hence the one engine addition
`CudaTransfers::retain_host`, the host mirror of `retain_device`. And the day-15 promote delta (+131 ms
ON) needed no new mechanism: the promote's timing window (`t0` before `device_entry_from_host`, `ms`
after `insert_pinned_demoting`) contains the inline demote of the entry the insert evicts, and under ON
that demote carries bind's SHA-256 over write-combined destinations (+132 ms on the demote line), so the
same hash was printed twice.

## What changed (commit `25891baa9`)

| Piece | What it does |
|---|---|
| `CudaTransfers::retain_host(&CudaPinnedLease) -> CudaPinnedLease` (`crates/memra-engine/src/tier_transfer.rs`, after `retain_device`) | The one engine addition: a second OWNED handle on one pinned allocation and its governor charge (the private `Rc` cloned, exactly what `take_destination` does for a D2H destination); owner thread and owner context only, `AlreadyReleased` on a released backing. While a twin lives, `write` on either handle and a D2H into either refuse `Busy`; the last owner dropped releases the charge. Not a borrowed-source seam: the op owns a lease, the trait is unchanged, no v1.4. |
| `host_kv_planes_from_contract(engine, tier, src: &HostPrefixEntry, class)` (`worker.rs`) | The route, one batch per promote, the `kv_tier_gate/active.rs restore` sequence applied to the server's entry, in this order: the plane list (every KV plane and the draft plane must be `HostPlaneBytes::Contract` with a lease exactly the plane's size; the source pointers for the unwind's identity check); fresh destination planes from the OFF allocator (`engine.alloc_u8`, K then V per plane; a refusal is the OFF failure `promote failed (device alloc of N B failed: ..)`); `register_device` at `dst_gen` and `retain_device` twins per fresh plane; `retain_host` twins per source lease; `record_producer`; ONE `submit_batch` of `TransferOp::H2d(CopyOp { host: twin, device, bytes, epochs, producer_fence })`, K then V per plane, the draft pair last; `synchronize(&ticket)`; `poll`; `Completion::require(&ticket, &receipts, true)` with `SegmentExpectation { valid_bytes, io_bytes, checksum: the plane's D2H receipt }` BEFORE publication (the H2D's completion checksum is the hash of the host source after the copy, so this holds only if the copy read exactly the bytes the demote wrote); `ready_view` per item (publication in the contract's sense; the engine's own `require` with the consumer fences); `record_consumer`; an owner-stream drain; `retire_source` (the twins drop: the entry's handles are sole owners again); `release_producer`; `retire(&ticket, Some(consumer))`; `acknowledge`; `take_plane` and `into_pooled` per fresh plane into `PrefixPlane`s in their slots; the receipt line. |
| `host_promote_contract_abort` / `_with`, `host_promote_release_fresh` | The unwind: a submitted ticket settles first (event sync; unknown completion is `Latched`); an UNPUBLISHED ticket is `cancel`led (`PublicationRevoked`) and every source twin `recover_source`d exactly once (lane A's rule 1), its `bytes().as_ptr()` checked against the entry's lease, then dropped; a PUBLISHED ticket records its consumer fence, drains, retires its sources; then `retire` (against the consumer fence only if published), `acknowledge`, the producer fence released after a drain, every fresh plane taken back through its retained twin and dropped. Every pre-submit arm drops the originals and the twins first (review finding 1); nothing is discarded (review finding 2): `Refused` when everything came back, `Latched` otherwise. |
| `HostPromoteFailure { Failed, Refused, ReceiptMismatch, Latched }`; `device_entry_from_host(engine, src, tier: Option<(&HostTierContext, HostTierEntryClass)>) -> Result<PrefixEntry, HostPromoteFailure>` | Typed all the way up. `Failed` is the OFF meaning (`rejected_allocs`, `promote failed (..)`); `Refused` prints `promote refused (contracts door): ..; serving without the host entry`; `ReceiptMismatch` drops the host entry (`digest_mismatches`, `remove_at`, `..; host entry dropped, cold path serves`) exactly as `VERIFY FAILED` does; `Latched` is one `TIER DISABLED` line (`host.disable`). The route is selected under `Some((tier, class)) if src.glm.is_none() && host_entry_has_contract_plane(src)`; everything else keeps `plane_up` (both `htod_u8_into` calls, statement for statement); a mixed entry is refused by name inside the route. The ten `host_glm.rs` test call sites pass `None`. |
| `HostContractFault::{PromotePreSubmit, PromotePostPublish}`; `HostTierContext::take_fault(demote)` | `MEMRA_KV_HOST_FAULT=contract-promote-presubmit` (the producer fence refused before any op) and `contract-promote-postpublish` (a refusal after `ready_view` published every item), one-shot. Each route takes only its own side, so one boot arms exactly one route; a demote-side fault stays armed for the demote route while a promote runs (the GPU cell asserts it). |
| `host_tier_governor`: `capacity.device[device] = thrice(device_budget)` | Under C the device dimension carries promoted residents' restore charges, the incoming entry's restore charge (taken before the copy, unchanged order) and its registered fresh planes at once: at most one budget each, so three. Pinned and pageable stay at twice. The boot line prints the device ledger at three budgets and ends `(Option B); KV plane H2D through the same engine on promote (Option C)`. |
| `flip-demote` at promote | The day-15 precedent at bind: when the receipt check refuses `Corrupt` under `MEMRA_KV_HOST_FAULT=flip-demote` (the gate-box diagnostic that corrupts the image after the D2H receipt by design), the door prints `contracts door H2D receipt: <slot> <role> plane host bytes differ from the D2H receipt as injected (..); the verify arm catches it at promote` and requires against the completion's own checksums (the copy completed with exactly the bytes on the host), so the failure gate's digest cell keeps its OFF shape (`FAULT`, `demote:`, `VERIFY FAILED`). Without the fault a difference is `ReceiptMismatch`. |
| Receipt line, ON only, before `promote:` | `[prefix-host] contracts door H2D receipt: ticket issuer=2 seq=3 epochs=0/1/1 items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=df514e08...6d8a5d71 published retired acknowledged` (verbatim from `pro-single-day16/faultgate/ev/promote-presubmit-server.log`). The digest is over the same ordered item checksums under the same domain as the D2H line, so for one entry the two lines carry ONE digest: in that log the H2D `seq=3` digest equals the D2H `seq=2` digest of the same entry's demote. |
| `tools/kv-host-contract-fault-gate.sh` | Two promote cells (`promote-presubmit`, `promote-postpublish`): r1 seeds, r2 evicts into a clean demote, r3 re-asks r1 so the promote takes the injected refusal and the cold path serves (its insert evicts into a clean demote), r4 re-asks r2 so the next promote must complete with an H2D receipt and a `promote:` line; no `TIER DISABLED`, no drop, no `Capacity`, no leaked wording, no refusal beyond the injected one. `75573cad3` fixed the cell's own matcher (see the card table). |
| Tests | CPU: `option_c_contract_route_is_door_only_and_keeps_the_frozen_promote_order` (order, receipt check before the first `ready_view`, allocations before registrations, planes leave after acknowledgement, no `htod_u8_into`/`memcpy_htod`/`clone_dtoh`/`alloc_host` in the route, `plane_up` intact, the abort's cancel-then-recover-then-retire with nothing discarded, the caller's drop and latch, the fault sides, the device ledger); `host_contract_fault_sides_are_taken_by_their_own_route_only`; the ledger test now also admits a third whole-budget device charge and refuses a fourth. GPU (`#[ignore]`): `option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin`, `option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin`, `option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin`, `option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source`. |

OFF is byte-identical by construction: `device_entry_from_host` gains a parameter that is `None` without
the door, and with `None` the `plane_up` loop and the draft `plane_up` run as before; `retain_host` is
reachable only from the route; `host_tier_governor`'s new size is read only by the ledger under the door;
the fault sides change one line of the demote route (`take_fault(true)`), door-only. No new `MEMRA_*`
read (`tools/check-flags.sh` 865, none uncovered). Not done, stated: no `KvMaterializer` type in the
server (the gate's typed operand hand-off for one plane; the server's operands are the `PrefixEntry`
planes published at `insert_pinned_demoting`, and its equivalent checks are the identity lease `require`
and the receipt expectation); the arena path and the DFlash tail slice are unchanged; the allocation flag
is unchanged (`WC-DESTINATIONS.md`).
