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

## Target-card receipts (`pro-single-day16/`, replay `verify-day16.py`)

Every gate cell: door OFF then ON, same binary, same prompts, same environment otherwise; `default` = the
gate's own environment (this MTP artifact serves speculatively, every prefix insert is a spec-boundary
capture with a draft plane, 34 items per batch); `plain` = `MEMRA_SERVE_SPEC=0` (32 items).
`MEMRA_HOSTGATE_CACHE_MB=256` for the host-spill gates. The card was free when the driver started (lane
B's earlier server had exited); lane B's next server took the card between my driver's `done` and the WC
pair cell, which waited on the rig lock through the runner's bounded retries. Verdict lines verbatim.
Collector `--validate` exit 0 on every cell (`validate.log`).

| Cell | OFF | ON | Equal |
|---|---|---|---|
| GPU unit cells, `cargo test --release -- --ignored` under the collector lock (`gputests/`) | | `test result: ok. 6 passed; 0 failed` in 0.25 s: `option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on`, `option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on`, `option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin`, `option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin`, `option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin`, `option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source`, each `... ok`. The first real CUDA exercise of the route, `retain_host`, `cancel` plus `recover_source` and the receipt mismatch path. | |
| `tools/kv-host-contract-fault-gate.sh`, four cells (`faultgate-fix/`; tree `75573cad3`, same binary) | | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 40 `ok:`. Demote side as on day 15 (`seq=1`, `seq=2`). Promote side, verbatim sequences: `promote-presubmit`: D2H `seq=1`, `promote refused (contracts door): tier H2D producer fence refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-presubmit); serving without the host entry`, D2H `seq=2` (the cold path's insert evicted), H2D `seq=3`, D2H `seq=4`, `promote: 86 tokens, 159.6MB`; `promote-postpublish`: D2H `seq=1`, `promote refused (contracts door): tier H2D publication refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-postpublish); serving without the host entry`, D2H `seq=3`, H2D `seq=4`, D2H `seq=5`, `promote:`: the aborted ticket's `seq=2` is consumed and never leaked (in-flight is one batch; a leak would have refused every later transfer with `Capacity`). No `TIER DISABLED`, no `host entry dropped`, no `Capacity`, no `leaked`, no refusal beyond the injected one, in both. The first sitting (`faultgate/`, tree `25891baa9`) ran the same four cells and printed `2 FAILURE(S)` on the gate's own matcher (it compared the FIRST D2H receipt in the log, the r2 one before the refusal, with the refusal's position); every unwind assertion was `ok`; the matcher fix is `75573cad3`, kept as the record. | |
| `tools/kv-host-spill-identity-gate.sh`, **default** | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 13 verdict lines | the same 13 lines | yes |
| same, `[prefix-host] demote:` byte counts | `89 tokens, 160.7MB`, `86 tokens, 160.6MB` | identical | yes |
| same, promote, verify, lane B's device line | `promote: 89 tokens, 159.7MB`, 1 `verify ok`, 1 `[prefix-cache] insert refused` (r4, both arms) | identical | yes |
| same, prefix event sequence (timings stripped, door lines excluded) | 16 events | the same 16 | yes |
| same, contract receipts ON (verbatim order) | none | D2H `seq=1 ... items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=435f0da4...`, then `demote: 89`, then **H2D `seq=2 ... items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=435f0da4... published retired acknowledged`** (the same digest as the D2H of the same entry), `verify ok`, D2H `seq=3 ... df514e08...` (the inline demote of the other entry), `demote: 86`, `promote: 89` | as required |
| same, r1/r3/r4 texts and `cached_tokens` | r3 and r4 `cached 89 / prompt 102`, identical texts | identical texts and counts | yes |
| same, promote and demote wall times (N=1, not a claim; the WC cell measures) | promote `266.1 ms`, demotes `149.6 / 150.6 ms` | promote `423.0 ms`, demotes `277.6 / 278.9 ms`; the promote's window contains the second demote, so the promote itself is `144.1` against `115.5` ms | |
| `tools/kv-host-spill-identity-gate.sh`, plain | `ALL GREEN`, 13 lines, demotes `89 / 160.5MB`, `86 / 160.4MB`, 1 promote, 1 `verify ok`, 15 events | identical; D2H `seq=1 ... items=32 (16 KV planes) ... c90ed5ec...`, H2D `seq=2 ... items=32 ... c90ed5ec... published retired acknowledged`, D2H `seq=3 ... 892a1252...`; promote `417.8` against `267.5` ms, demotes `273.4 / 275.2` against `146.9 / 151.5` | yes |
| `tools/kv-host-spill-failure-gate.sh`, **default** | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`, pre-existing since day 13), 15 lines | the same 15 lines | yes |
| same, digest cell ON (verbatim order) | `FAULT: flipped one demoted K byte`, `demote:`, `VERIFY FAILED: promoted digest aaec09b2... != demote digest 7674c8a1...` | D2H receipt `seq=1`, `FAULT: flipped one demoted K byte (MEMRA_KV_HOST_FAULT=flip-demote)`, `contracts door D2H receipt: Key plane image checksum differs from its D2H receipt as injected (...); the verify arm catches it at promote`, `demote: 89`, then **`contracts door H2D receipt: Kv(3) K plane host bytes differ from the D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote); the verify arm catches it at promote`**, the H2D receipt `seq=2 ... require=ok` (against the completion's own checksums), then `VERIFY FAILED: promoted digest aaec09b2... != demote digest 7674c8a1...`: the OFF shape held, the door named the difference at both ends, no `tier H2D receipt refused` | yes |
| same, alloc and pool-full cells | `TIER DISABLED: pinned host alloc ... (MEMRA_KV_HOST_FAULT=alloc-fail)`; `demote evaporated at the tenant share cap` | identical tier event lines | yes |
| lane A `tools/kv-host-tenant-reclaim-gate.sh fix` | `GATE: kv-host-tenant-reclaim (fix arm) PASS`, 29 verdict lines, 8 demotes, 2 promotes, 3 `evict (tenant share)` | the same 29 lines and sequence; 8 D2H receipts (7 distinct digests: beta's entry demoted twice with one digest) and 2 H2D receipts whose digests (`79f0c67d...`, `394f1089...`) are both among the D2H digests of the same log | yes |
| `tools/serve-smoke.sh` plain + cache-metering | 34 lines, `ok: cache-metering accounting exact (per-request + /metrics + economics)`, `serve-smoke: 0 failed` | the same 34 lines (no host tier on these boots: one `[kv-host-contracts] ... nothing to route` line) | yes |
| lane B `tools/prefix-evict-reclaim-gate.py` | `PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 pool_retained_bytes=140549120 p2=admit-same-tick busy_overlap_s=21.659 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS` | the identical line except `busy_overlap_s=21.661` (the gate's own wall-clock overlap window, 2 ms apart; no host tier on these boots; day 15's equality to the millisecond was chance) | yes, timing stripped |
| lane B `tools/prefix-newest-turn-fits-gate.py` | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS` | the identical line | yes |

The bytes-unchanged chain, one link longer than day 15: `verify ok` in the ON arm on every promote (the
promoted device entry's digest, over planes that crossed BOTH ways through the contract, equals the
pre-demote device digest), the same `verify ok` in OFF, equal demote byte counts, byte-identical r1, r3
and r4 texts and `cached_tokens` across arms, the receipt's per-plane checksum equal to bind's bundle
checksum at demote (day 15), and, new, `Completion::require` at promote holding against those same
receipts before publication, visible as the H2D line's digest equal to the D2H line's digest for every
entry in every ON log (identity default and plain, the tenant gate's two promotes, the fault gate's four
cells), with the one difference in the whole day the injected flip, named at both ends.

### Findings

1. **The day-15 promote delta was the inline demote.** With the promote's own share separated (the
   promote line minus the demote line printed inside its window), OFF promotes the entry in `115.5 ms`
   and days 13 to 15's ON in `116.6 ms` (day 15's `397.7 - 281.1`); Option C's ON promote is `144.1 ms`
   (N=1), the engine's SHA-256 of the write-combined source at H2D completion plus the ticket lifecycle.
   The WC cell (below) is the N=5 measurement of both lines.
2. **The receipt digest chains the two directions.** The H2D receipt of every promote carries the D2H
   digest of the same entry's demote: `435f0da4...`, `c90ed5ec...` (identity default and plain), beta's
   `79f0c67d...` and acme's `394f1089...` (tenant), `df514e08...` (the fault gate's r4). A grep pairs
   every promote with the demote whose bytes it restored, across boots.
3. **A consumed ticket sequence is the leak proof.** The post-publish promote fault's aborted ticket took
   `seq=2` and the next transfers ran `seq=3, 4, 5`: `retire` released its in-flight charge (the whole
   dimension) and `acknowledge` dropped it; the demote-side cells show the same (day 15).
4. **The flip lands on `Kv(3)`.** The first KV plane of this hybrid model sits at layer 3 (layers 0 to 2
   are recurrent), so the door's named difference reads `Kv(3) K plane`; the OFF `FAULT` line does not
   name the slot.
5. **Two gate scripts had matcher bugs of mine, both fixed on the day and kept as records:** the fault
   gate's promote cells compared the first receipt in the log with the refusal (`75573cad3`, `after_any`),
   and the WC cell script tripped `set -u` on its own `local` line (`label: unbound variable`) before any
   boot (`wc-pair/`, rerun as `wc-pair2`).

## The WC pair (`WC-DESTINATIONS.md` "Results"; `pro-single-day16/wc-pair2-retry3/`, replay `wc-pair.py` PASS)

One collector lock hold (64 s, after three 120 s waits behind lane B), four boots OFF, ON, ON, OFF, seven
requests each over the two identity-gate prompts, `MEMRA_KV_HOST_VERIFY` unset, default spec environment.
Regime: 37 to 51 C, at most 492 W under the 600 W cap, 257 samples at 250 ms. Medians (ms): demote OFF
**37.8** against ON **169.2** (N=10 pooled, each order N=5 agreeing within 0.5 ms per position); promote
line OFF **12.2** against ON **171.7**; promote minus its inline demote OFF **4.5** against ON **33.2**.
Every boot shows a first-touch step of about 35 ms on the first three demotes in BOTH arms (fresh pinned
regions page-locked and zero-filled; from r5 a dropped twin's region is reused), so the steady-state pair
is r5..r7: demote 6 to 8 against 136 to 140 ms. The door's cost at demote is therefore about 130 ms per
160 MB entry on this card in this window (two CPU hashes over write-combined memory plus the ticket
lifecycle; the split needs the hash-speed micro-cell), and about 29 ms on the promote's own share (the
observed H2D completion, one hash over the WC source, the ticket), with OFF's 4.5 ms being an
asynchronous launch whose DMA completes inside the next demote's window. Reported as the first cell of the
decide-by review, not a verdict; the flag is the engine's (`tier_transfer.rs`) and is unchanged.

## Local battery (this rig, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, logs `day16-local/`)

On `25891baa9`: `cargo fmt --all -- --check` clean; `cargo test -p memra-server -p memra-kv -p memra-engine
--offline` (under `flock -n /tmp/memra-5090.lock`, no bare GPU run): memra-server 758 passed, 0 failed, 12
ignored (the six GPU cells of B and C plus the six pre-existing), memra-kv 71 passed, memra-engine 515
passed, 24 ignored; 265 suites, 1562 `ok`, 0 failed; `cargo clippy -p memra-server -p memra-kv -p
memra-engine --offline --all-targets -- -D warnings` clean on the first pass; `tools/check-flags.sh` 865
runtime names, none uncovered; `tools/docs-registry-census.sh` clean (58 tables, 901 rows); `git diff
--check` clean. The source-text cell failed once on its own list (the first `take_plane(` in the route is
the registration unwind, not the final take; the trailing `rfind` assertion covers the final one), fixed in
the test. No `DOCS_RS=1` command ran. The final tree's fmt, flags census, docs census and diff-check are in
`day16-local/final-*.log`.

## Push

`git push origin lane/spill-c-20260919` at `a902fd2e9` was refused by the pre-push hook's perf-ci
freshness arm, verbatim: `pre-push: engine files touched after the last perf-ci battery. base (merge-base
with refs/remotes/origin/lane/spill-c-20260919): 64ee65c9c69fd497cf8ac2e35621b2fa248f7f99 engine files this
branch changes: crates/memra-engine/src/tier_transfer.rs Run: tools/local-ci.sh --perf (or --perf-quick for
the 31B subset) Override knowingly with MEMRA_SKIP_PERF_CI=1.` Every other arm passed (perf board,
flags census, releasability censuses, docs-registry census, workflow-file census). No skip variable was
used and none will be by this lane; `tools/local-ci.sh --perf` is the local 5090's multi-model perf
battery, outside today's budget and blocked on this rig by the recorded hit-gate identity regression at
main (memory note 2026-09-20). The lane's tip stays unpushed with this record; the lead decides the
perf-ci run or the receipt.

## What remains

- **The WC decision** (`WC-DESTINATIONS.md`): the engine's allocation flag; the hash-speed micro-cell
  (cached against write-combined SHA-256 GB/s on this host); a cached-pinned `alloc_host` arm measured the
  same way on both rigs; or one hash per plane at demote; or the documented cost.
- **The arena path** (`MEMRA_GLM5_TP_KV_HOST`, refused with the door at boot): its fixed backing is not
  governor-charged and its slices are not leases.
- **The DFlash tail slice**: no drafter artifact identity is derivable from a GGUF digest and no gate boots a
  DFlash drafter on the card.
- **Verify digest v3** (the draft plane in `MEMRA_KV_HOST_VERIFY`), **the pool-full failure-gate line**:
  unchanged from days 14 and 15.


## Review round (PR #605, integ19: revuto findings 1 and 2 on `host_kv_planes_from_contract`'s unwind; both held)

Tree after the fixes: merge of main `435a57a75` (`2ad0be159`), fix `4467131f6`. Both findings were in the
abort's CLASSIFICATION, not in the engine, and both would have latched the tier over a state the engine
could unwind cleanly.

**Finding 1 (partial acceptance).** `host_promote_contract_abort_with`'s unpublished arm looped
`recover_source` over every entry of `sources`, but a rejected op's slot is `None` in the engine
(`submit_batch` pushes `entry.items.push(None)`) and `recover_source` answers `Rejected` for it, so every
rejected index pushed a leak line, the abort returned `Latched` and the caller `host.disable`d the whole
tier, although the rejected ops' twins and device handles had already dropped with the returned batch and
`retire`, `acknowledge` and `take_plane` would have succeeded. Fix: `sources` carries `(item index, the
entry's lease pointer)` per op, and the partial-acceptance arm hands the abort the ACCEPTED items only
(`ItemAcceptance::Accepted { item } => sources.get(item)`), so the abort recovers exactly those and ends
`Refused` as its comment says.

**Finding 2 (an inferred published flag).** The `ready_view` loop passed `published = item > 0` to the
abort. Whether the ticket is published is the engine's state: `CudaTransfers::ready_view` sets it after
`owner.ready_view` succeeds (`tier_transfer.rs:1052`), `with_destination` before its fallible steps
(`:552`); a caller-side flag can disagree with either, and when it does the abort took the cancel arm on
a published ticket (`AlreadyPublished` pushed as a leak), `retire(ticket, None)` hit the `e.published`
`Busy` guard, `retire_binding` never ran and every `take_plane` was refused: a latched tier plus leaked
destinations where the consumer-fence path would have unwound cleanly. Fix: the abort has no `published`
parameter; it asks the engine through `cancel`, the contract's own question ("Revoke only before
publication; after publication return AlreadyPublished"): `PublicationRevoked` recovers the accepted
sources (rule 1), `AlreadyPublished` records the consumer fence, drains and retires the sources, and then
both arms retire (against the fence if published) and acknowledge.

**Injectable, one-shot** (`MEMRA_KV_HOST_FAULT`, promote side, taken by the promote route only):
`contract-promote-reject` (the last op of the batch asks for one byte more than its source holds, so the
engine's own `CopyOp::validate` rejects exactly that item: a real partial acceptance) and
`contract-promote-readyview` (the first `ready_view` runs and publishes in the engine; the route is handed
an injected error for it). FLAGS row updated; `tools/kv-host-contract-fault-gate.sh` gains the cells
`promote-reject` and `promote-readyview` (six cells now). The source-text cell pins the abort's order
(`synchronize`, `cancel`, `recover_source(ticket, *item)`, `record_consumer`, `retire_source`, `retire`),
the two `CancelState` arms, the accepted-only hand-off, and the absence of `published: bool` and `item > 0`.


| Cell (target card, `4467131f6`, binary `b5ea1c25...`, 600 W; `pro-single-day16-review/`, replay `verify-day16-review.py`: `DAY16 REVIEW REPLAY: PASS`, 42 checks; collector `--validate` rc=0 on all four) | Verbatim |
|---|---|
| GPU unit cells (`cargo test --release -p memra-server -- --ignored`, under the collector lock) | `test result: ok. 8 passed; 0 failed` in 0.25 s: the six of the day plus `option_c_partial_acceptance_unwinds_refused_with_every_destination_released ... ok` and `option_c_first_ready_view_failure_unwinds_through_the_published_arm ... ok`. Each: a real `CudaTransfers`, a host image built by the B route, the injected shape, `Refused` with no leak wording, the twins dropped (the image's leases writable again), the ledger back to `(pinned = image, 0, 0)`, then a clean promote. |
| `tools/kv-host-contract-fault-gate.sh`, six cells | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 62 `ok:`. `promote-reject`: D2H `seq=1`, `promote refused (contracts door): tier H2D batch partially refused: 1 of 34 items (injected failure (MEMRA_KV_HOST_FAULT=contract-promote-reject)); serving without the host entry`, D2H `seq=3`, H2D `seq=4` (digest `df514e08...`, the D2H `seq=3` digest), D2H `seq=5`, `promote: 86 tokens, 159.6MB`. `promote-readyview`: D2H `seq=1`, `promote refused (contracts door): tier H2D destination 0 not publishable: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-readyview); serving without the host entry`, D2H `seq=3`, H2D `seq=4`, D2H `seq=5`, `promote:`. In both the aborted ticket's `seq=2` is consumed and never leaked (retired and acknowledged), no `TIER DISABLED`, no `host entry dropped`, no `Capacity`, no `leaked`, no `already published`, no refusal beyond the injected one. The four earlier cells unchanged. |
| `tools/kv-host-spill-identity-gate.sh`, default, OFF vs ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` both arms, verdict lines identical, equal demote bytes, 16 events equal, `verify ok`, one H2D receipt for the one promote with its D2H's digest, no refusal line. |

Local battery on `4467131f6` (`day16-local-review/`, the same quota scope and lock): fmt clean; memra-server
758 passed, 0 failed, 14 ignored (eight GPU cells plus six pre-existing), memra-kv 71, memra-engine 515 with
24 ignored, 0 failed, exit 0; clippy `-D warnings` on the three crates clean; check-flags 865, none uncovered;
docs-registry census clean (58 tables, 905 rows after main's merge); diff --check clean. Final-tree logs
`day16-local-review/final-*.log`.

Effort: approximately 6.5 agent-hours for the day plus approximately 1.5 for the review round (budget 8).
