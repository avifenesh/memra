# Session C day fifteen: Option B, the pageable host-tier D2H through the TransferEngine (lead rulings 14 and 15)

Scope: ruling 15, "Option B (D2H through `TransferEngine` on the pageable path) only after A's
receipts", and ruling 14, "Planes move out of `PrefixEntry` as owned `KvPlane`s for the D2H; no
borrowed-source seam in `CudaTransfers`". Tree: lane merge of main `70038ed01` (`ac3789ca6`), lane
merge of `origin/lane/spill-a-20260919` `8302f6b0a` (`4cc2e92a6`; lane A's day 12 sits on PR #597,
not on main, and the reclaim gate needs its code), census commit `16272e9de`, code commit `5f8d327e2`.
Target-card cells ran on the `5f8d327e2` tree (binary sha256
`3d5386e6c2e1c3d4809390e5661dc050144a6dd4e31eb53682633381558ec674`, release build 44 s on the
cached engine, niced with six jobs while lane B's A/B held the card), one RTX PRO 6000 Blackwell
Server Edition at its 600 W limit (`pro-single-day15/card.csv`), through the canonical collector
(`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`, inherited by every gate),
N=1, `executed-not-qualified`. Nothing here is a support state. Door doc: `HOSTPREFIX-DOOR.md`
("Option B", the census written before code, and this day's receipts).

## The census, in one paragraph (`HOSTPREFIX-DOOR.md` "Option B", commit `16272e9de`)

The pageable demote was twelve steps of `host_demote_prefix_ref` by reference: armed check, GLM
refusals, `host_image_bytes`, lane A's pre-copy reclaim plan, the door's class and
`tier_charge` (pinned = every KV plane), the verify digest over the device entry,
`host_entry_from_device` (per plane `host_plane_from_device`: `PinnedHostBuf::new` then
`dtoh_u8_into_pinned` = `memcpy_dtoh` plus `synchronize`, twice per plane, no ticket, no fence, no
checksum), the charge onto the image, `bind_tier_image` (the only hash of the host bytes), lane A's
reclaim, `insert`. The single point where owned planes can leave the entry is the plane loop of
`host_entry_from_device` and its draft call: `register_device` takes `impl Into<KvPlane>` by value,
`From<CudaSlice<u8>>`, so the K and V slices must be owned; `dead.kv[i].take()` on an `&mut
PrefixEntry` moves the whole `PrefixPlane` out of its `Option` slot without a placeholder and without
a device byte (a `CudaSlice::clone` would be a `cuMemcpyDtoD`), and the same two slices return
through `take_plane` and `into_pooled` into the same slot. Two things the census found that code had
to account for: `CudaTransfers` wants an `Rc<RefCell<dyn BudgetGovernor>>` while the server holds an
`Arc<Mutex<..>>` (one ledger, two handles, an adapter), and the day-13 ledger sized `inflight` at
zero, which would have refused the first contract-routed `submit_batch` with `Capacity`.

## What changed (`crates/memra-server/src/worker.rs`, commit `5f8d327e2`; `host_glm.rs` GPU tests follow the borrow)

| Piece | What it does |
|---|---|
| `HostPlaneBytes { Pinned(PinnedHostBuf), Contract { lease: CudaPinnedLease, receipt: Digest } }`; `HostPlane.k/.v` | The byte owner of a demoted plane. `Pinned` is the OFF arm and every handoff import. `Contract` is the destination a `TransferEngine` D2H delivered, with its own pinned charge and the engine's completion checksum of exactly those bytes. `bytes()` is fallible (a lease read is a driver call), `receipt()`, `flip_first_byte()` (the `flip-demote` fault; a lease has whole-buffer `write` only, so the fault reads, flips and rewrites the plane). Readers changed: `bind_tier_image`, the promote `plane_up` (reads a `&[u8]`, program untouched), the handoff export view (now fallible), the fault. |
| `HostTierLedger(SharedGovernor)`: `impl BudgetGovernor` | The transfer engine's handle on the server's governor: locks the one mutex per call (`Quarantined` on poison; `used()` reads through poison). Every `CudaPinnedLease` and every ticket releases through it. |
| `host_tier_governor(.., inflight)`; `HostTierContext { transfers: Option<RefCell<CudaTransfers>>, inflight }` | The ledger gains `inflight = 2 x max layers + 2` over the loaded models (130 on this artifact: 64 layers). `host_tier_context` builds `CudaTransfers::new(engine.stream(), ledger)` on the worker thread; `None` only in CPU tests, where a demote refuses by name. The boot line ends `in-flight 130; host tier armed; KV plane D2H through the transfer engine on the pageable tier (Option B)`. |
| `HOST_TIER_TRANSFER_EPOCHS = Epochs { state: 0, src_gen: 1, dst_gen: 1 }` | State 0 is the immutable-prefix epoch; the generations are the single worker owner's for the process. The model INSTANCE identity stays with `bind_tier_image`'s generation `Arc` and the identity lease. |
| `host_kv_planes_through_contract(engine, tier, dead: &mut PrefixEntry, class)` | The route, one batch per demote, in this order: the plane list (geometry, non-empty, owner-stream pre-checks: a refused `register_device` drops the plane it was handed, so nothing refusable is left to it); `alloc_host` per K and V in the OFF order (a refusal here leaves the entry untouched; the `alloc-fail` fault and the latch keep the OFF line); a device admission probe on the same ledger for the planes' bytes; `dead.kv[i].take()` / `dead.draft.take()`, `register_device` K and V, `retain_device` twins; `record_producer`; `submit_batch` of `TransferOp::D2h(CopyOp { host, device, bytes, epochs, producer_fence })`, K then V per plane, the draft pair last; `synchronize(&ticket)`; `poll`; `take_destination` per item; `Completion::require(&ticket, &expected, false)` with `valid_bytes` from the server's own geometry; `record_consumer`; `retire_source`; `take_plane` and `into_pooled` back into the same slots; `release_producer`; `retire(&ticket, Some(consumer))`; `acknowledge`; the `HostPlane`s with their receipts; one receipt line. |
| `host_contract_planes_back`, `host_contract_abort` | The unwind: a submitted ticket settles (event sync, sources retired) before any plane can come back; planes return to their slots; the producer fence releases; the ticket retires (against a consumer fence if a destination was taken) and is acknowledged. `Refused` when every plane came back (the entry is whole), `SourceQuarantined` when the engine kept one. |
| `HostContractFailure { Alloc, Refused, SourceQuarantined }`, `HostImageFailure { Failed, SourceQuarantined }`, `HostDemoteOutcome::SourceQuarantined` | Typed all the way up. `Alloc` latches the tier (the OFF posture). `SourceQuarantined` (a quarantined completion: the frozen rule never returns potentially-live inputs) latches the tier and tells the one live-entry caller, the pause sweep's resident-entry half, to DROP the entry rather than keep a device entry with holes; the SLRU sink, the admission flush and the export drain drop theirs anyway; the pause sweep's park half keeps its park (the snapshot was a copy). |
| `host_demote_prefix_ref(.., dead: &mut PrefixEntry)`, `host_entry_from_device(.., dead: &mut PrefixEntry, ..) -> Result<_, HostImageFailure>`; five callers | The exclusive borrow (ruling 14). `evict_all_demoting` and the pause sweep reach their live entries through `get_mut`; the sink, the export drain and the boundary snapshot own theirs. OFF passes the same entry through the same statements. |
| `tier_charge(.., pinned = 0, pageable, ..)` at demote | Under Option B the leases carry the pinned bytes on the same ledger, so the residency charge takes pinned zero and the pageable remainder: the same total, charged once; the twice-the-budget argument of day 13 is unchanged. |
| `bind_tier_image` `add(.., receipt)` | The bundle checksum of a contract-routed plane must equal the plane's receipt: a typed refusal `tier image {role} plane checksum differs from its D2H contract receipt`, except under `MEMRA_KV_HOST_FAULT=flip-demote`, which corrupts the image after the receipt exactly as it corrupts after the verify digest, and prints the injected difference by name. |
| Receipt line, ON only, before `demote:` | `[prefix-host] contracts door D2H receipt: ticket issuer=2 seq=1 epochs=0/1/1 items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=<hex over the ordered item checksums> retired acknowledged` (verbatim from `pro-single-day15/hostgate-identity-on-default/ev/host-on-server.log`). |

OFF is byte-identical by construction: the route is reached only from the `Some(tier) if
host.arena.is_none() && !is_glm` arm of `host_entry_from_device`, `host_plane_from_device` keeps both
by-reference copies (trunk loop and draft) statement for statement, the borrow becomes exclusive with no
statement change, and `host_tier_governor`'s new dimension is read only by the transfer engine (the
source-text test `option_b_contract_route_is_door_only_and_keeps_the_frozen_demote_order` pins the
route's order, its door-only call site, the absence of `dtoh_u8_into_pinned` / `memcpy_dtoh` / a plane
clone in the route, the pinned-zero charge and the sweep's drop on `SourceQuarantined`). The promote
path, the arena path and the f32 planes (`CudaSlice<f32>`, not `KvPlane`s; `HostF32::down`) are untouched.
No new `MEMRA_*` read (`tools/check-flags.sh` 867, none uncovered). Lane A's four `waste_pending_reclaim`
sites stay four: the quarantined arm shares the copy-failed booking.

## Target-card receipts (`pro-single-day15/`, replay `verify-day15.py`: `DAY15 REPLAY: PASS`, 134 checks)

Every cell: door OFF then ON, same binary, same prompts, same environment otherwise; `default` = the
gate's own environment (this MTP artifact serves speculatively, every prefix insert is a spec-boundary
capture with a draft plane); `plain` = `MEMRA_SERVE_SPEC=0`. `MEMRA_HOSTGATE_CACHE_MB=256` for the
host-spill gates. The first cell waited on the rig lock through two 120 s retries while lane B's
`prefix-policy-ab.py` campaign finished (the collector directory for that cell is
`hostgate-identity-off-default-retry2/`; the gate's `ev/` is under the base name); no holder was
touched. Verdict lines verbatim. Collector `--validate` exit 0 on all fourteen cells (`validate.log`).

| Gate | OFF | ON | Equal |
|---|---|---|---|
| `tools/kv-host-spill-identity-gate.sh`, **default** | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 `ok:` | the same 13 verdict lines | yes |
| same, `[prefix-host] demote:` byte counts | `89 tokens, 160.7MB`, `86 tokens, 160.6MB` | identical | yes |
| same, promote and verify | `promote: 89 tokens, 159.7MB`, 1 `verify ok`, then lane B's `[prefix-cache] insert refused: entry 160131968 cannot fit beside 159721792 leased bytes (budget 268435456, spec-boundary, model gate)` | identical (the device cache's typed line, day-14 finding 2 in its new form, present in both arms) | yes |
| same, prefix event sequence (timings stripped, door lines excluded) | 16 events | the same 16 | yes |
| same, D2H contract receipts | none (the route is door-only) | 2 receipts, one before each demote, `items=34 (16 KV planes, draft) complete=34 require=ok`, `epochs=0/1/1`, ticket `seq=1`, `seq=2`, distinct checksum digests | as required |
| same, r1/r3 texts and `cached_tokens` | r3 `0 < cached < prompt` | identical texts and counts | yes |
| same, host-tier refusal lines ON | | `0` (`demote refused`, `promote refused`, `REFUSED`, `TIER DISABLED`, `(contracts door):`, `demote failed`) | |
| `tools/kv-host-spill-identity-gate.sh`, plain | `ALL GREEN`, demotes `89 tokens, 160.5MB`, `86 tokens, 160.4MB`, 1 promote, 1 `verify ok` | identical; 2 receipts `items=32 (16 KV planes)` (no draft plane on the plain surface) | yes, 13 lines, 15 events |
| `tools/kv-host-spill-failure-gate.sh`, **default** | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`, pre-existing since day 13) | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)`, the same 15 lines; poolfull (2), digest (5), alloc (2) tier event lines identical | yes |
| same, digest cell ON (verbatim order) | `FAULT: flipped one demoted K byte`, `demote:`, `VERIFY FAILED` | `contracts door D2H receipt: ticket issuer=2 seq=1 ... require=ok`, then `FAULT: flipped one demoted K byte (MEMRA_KV_HOST_FAULT=flip-demote)`, then `contracts door D2H receipt: Key plane image checksum differs from its D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote); the verify arm catches it at promote`, then `demote: 89 tokens, 160.7MB`, then `VERIFY FAILED: promoted digest aaec09b2... != demote digest 7674c8a1... (89 tokens); host entry dropped, cold path serves` | the OFF shape held; the receipt precedes the flip and the bind names it |
| same, alloc cell | `TIER DISABLED: pinned host alloc of 96832 B failed: injected failure (MEMRA_KV_HOST_FAULT=alloc-fail)` | the same line (the first plane's K, the same byte count), no receipt, no quarantine: the fault fires before any plane leaves the entry | yes |
| same, pool-full cell | `demote evaporated at the tenant share cap before the D2H copy ... reclaim refused: the image alone exceeds the share` (lane A's suffix) | identical | yes |
| lane A `tools/kv-host-tenant-reclaim-gate.sh fix` | `GATE: kv-host-tenant-reclaim (fix arm) PASS`, 29 verdict lines, 8 demotes (`83/160.5`, `89/160.7`, `86/160.6`, `93/160.8`, `93/160.8`, `95/160.9`, `96/160.9`, `83/160.5` tokens/MB), 3 `evict (tenant share)` of acme's own entries, 2 promotes | the same 29 lines, the same 8 demotes, the same evict and promote sequence; 8 receipts, one before each demote and before each reclaim eviction (the reclaim runs after the transfer, before insert); 7 distinct digests over 8 demotes (beta's P_D demoted twice, identical bytes, identical digest) | yes |
| `tools/serve-smoke.sh` plain + cache-metering | 31 `ok:`, `ok: cache-metering accounting exact (per-request + /metrics + economics)`, `serve-smoke: 0 failed` | the same 33 lines; ON boot prints one `[kv-host-contracts] MEMRA_KV_HOST_CONTRACTS=1 with no host tier on this boot (MEMRA_KV_HOST_MB=0): nothing to route` and no `[prefix-host]` line | yes |
| lane B `tools/prefix-evict-reclaim-gate.py` | `PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 pool_retained_bytes=140549120 p2=admit-same-tick busy_overlap_s=21.659 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS` | the identical line, the same identity digest | yes |
| lane B `tools/prefix-newest-turn-fits-gate.py` | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 protected_evictions=2 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS` | the identical line | yes |

The bytes-unchanged chain, as ruling 15 asked: `verify ok` in the ON arm on the promote (the promoted
device entry's digest equals the pre-demote device digest, over planes that crossed as contract
destinations), the same `verify ok` in the OFF arm over the same prompts, equal demote byte counts,
byte-identical r1 and r3 texts across arms, and, new, the receipt's per-plane checksum equal to the
bundle checksum `bind_tier_image` computes over the stored image on every demote (a difference is a
typed refusal; zero refusals in every ON cell; the one difference in the whole day is the injected flip,
named as such).

### Findings

1. **The contract's pinned destinations are write-combined.** `CudaTransfers::alloc_host` allocates
   through cudarc's `alloc_pinned`, which is `cuMemHostAlloc(.., CU_MEMHOSTALLOC_WRITECOMBINED)`
   (`cudarc-0.19.8/src/driver/safe/core.rs:1417-1420`); the pre-door `PinnedHostBuf::new` is
   `cuMemHostAlloc(.., 0)`, cached. CPU reads over write-combined memory are uncached, so the one CPU
   read the door adds at demote, `bind_tier_image`'s SHA-256 over ~160 MB of planes, runs at WC speed.
   N=1 timings, the same binary, never compared as a claim: demote `148.6 / 151.6 ms` OFF against
   `281.1 / 281.1 ms` ON (default), `147.7 / 152.9` against `274.0 / 277.1` (plain); promote `267.4 ms`
   OFF against `397.7 ms` ON (default), `269.2` against `393.1` (plain). The demote delta is the
   bind hash over WC memory plus the ticket lifecycle; the promote delta (the same `memcpy_htod` from a
   `&[u8]` in both arms, the source allocation the only difference) is observed, not explained, and is
   Option C's to measure. The engine's allocation flag is `crates/memra-engine/src/tier_transfer.rs`
   territory, not this lane's; recorded for the lead.
2. **Two hashes per plane under ON, not three.** The engine hashes the destination at completion
   (`progress`) and `bind_tier_image` hashes the image once (unchanged code); the `SegmentExpectation`
   handed to `Completion::require` carries the completion's own checksum, so `require` proves status,
   epochs, exact lengths and producer completion, and the independent byte check is bind's equality
   with the receipt. A take-time third hash was considered and dropped.
3. **The receipt digest names the entry's bytes.** Beta's 83-token entry demoted twice in the tenant
   gate (r1's seed, r7's re-eviction) and carried the same `checksums_sha256=79f0c67d...` both times; the
   identity gate's two entries carried the same two digests in the default and the failure cells
   (`435f0da4...`, `df514e08...`), across boots. Same prompt, same bytes, same receipt.
4. **Lane B's typed refusal replaced day-14 finding 2's line.** The r4 second promote is now declined
   by `[prefix-cache] insert refused: entry ... cannot fit beside ... leased bytes` (lane B's day 14)
   instead of `skip pinned host-promote insert`; identical in both arms, not a door effect; the replay
   compares those lines across arms.
5. **The route engaged everywhere the door was armed.** 2 + 2 + 2 + 8 receipts across the identity
   (default, plain), failure digest and tenant cells, `items=34` with the draft plane and `items=32`
   without, `complete = items` and `require=ok` on all 14, ticket sequences 1..n per boot from one
   issuer.

## Local battery (this rig, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, logs `day15-local/`)

On `5f8d327e2`: `cargo fmt --all -- --check` clean; `cargo test -p memra-server -p memra-kv -p
memra-engine --offline` (under `flock -n /tmp/memra-5090.lock`, no bare GPU run): memra-server 761
passed, 0 failed, 6 ignored (GPU); memra-kv 71 passed; memra-engine 515 passed, 24 ignored; 1565 `ok`
in total; the two new cells `host_tier_ledger_handle_charges_and_releases_the_servers_one_ledger` and
`option_b_contract_route_is_door_only_and_keeps_the_frozen_demote_order` `ok`, lane A's
`tenant_share_reclaim_is_wired_into_the_demote_hook_and_the_metrics` still `ok`; `cargo clippy -p
memra-server -p memra-kv -p memra-engine --offline --all-targets -- -D warnings` clean;
`tools/check-flags.sh` 867 runtime names, none uncovered; `tools/docs-registry-census.sh` clean (58
tables, 903 rows); `git diff --check` clean. Before the commit the order test failed twice on its own
sloppiness (a `.clone()` guard that caught `error.clone()` and a comment, then a 9000-character window
shorter than the hook) and once on a needle that also matched the register-failure arm; each fix is in
the test, not the code. No `DOCS_RS=1` command ran.

## Review fixes (PR #599, integ17: revuto findings 1 and 2 on the unwind; both held)

Tree after the fixes: `30704905f` (the fix), `da52fa885` (the fault gate's own matcher bugs), `8f8f05a19`
(docs), then the merge of main `0175e39d4` (lane B's SLRU removal in `worker.rs`; `worker.rs` auto-merged,
my GPU builder dropped the removed `segment` field, `docs/FLAGS.md` resolved row by row against the base)
at `aefb89d89`. Card cells on `aefb89d89` (binary `0e95bf16607f3b758e000041caf7794b8ecd4e584b7e8b6e0f76df70e102ee62`,
`pro-single-day15-review/`, replay `verify-day15-review.py`: `DAY15 REVIEW REPLAY: PASS`, 46 checks). The
first sitting (`sitting1/`) is kept: the GPU cell was refused by the collector because the driver had
pre-created its `--out` (`[Errno 17] File exists`), and the fault gate printed `6 FAILURE(S)` on two shell
bugs of mine (an order check that read the refusal line as a regex with unescaped parentheses, and
`bash -c` subshells that could not see the script's functions) while its server logs already showed the
unwind working; both fixed in `da52fa885`, no engine or server change.

**Finding 1 (`worker.rs` pre-submit arms).** After registration the ops' original `DeviceLease` handles
sat in `originals` while the retained twins sat in `registered`, so the registry `Rc` count was three;
`DeviceOwner::release` (`contracts.rs:1264-1277`) accepts exactly two, so every `take_plane` in the unwind
refused `Busy`, `host_contract_abort` escalated a recoverable refusal to `SourceQuarantined`,
`host_entry_from_device` latched the tier off for the process and the live-entry callers dropped a whole
device prefix entry with its planes intact and no DMA ever submitted. Fix: every pre-submit arm drops or
clears `originals` before the unwind (the four in-loop arms `originals.clear()`, the `record_producer` arm
`drop(originals)`); the success path was never affected because `retire_source` takes `item.device` first.

**Finding 2 (`host_contract_abort`).** The abort recorded the consumer fence and called `retire` at once;
`retire` reads the consumer event and a `CUDA_ERROR_NOT_READY` is `Busy`; the result was discarded (`let _
= t.retire(..).and_then(acknowledge)`), which leaked the ticket forever: `retire` is the only release of the
batch's in-flight charge, `acknowledge` the only drop of the entry and its host destinations, and the
ledger's in-flight dimension is exactly one batch, so the next `submit_batch` would refuse `Capacity`,
classified `Refused`, and every later demote would silently refuse for the life of the process; the same
for a dropped `release_producer` error. Fix: the abort drains the owner stream before `release_producer`
and again after `record_consumer` (an unpublished ticket refuses `NotReady` there and retires against
`None`), and no result is discarded: a refusal is the new typed `TicketLeaked` outcome, which
`host_entry_from_device` turns into one `TIER DISABLED` line (the latch exists for exactly this) with the
entry whole; the success path's settle drains before `retire` and maps a failure to the same latch.

**Injectable, one-shot.** `MEMRA_KV_HOST_FAULT=contract-presubmit` (the producer fence refused before any
op is submitted) and `contract-postpublish` (the receipt check refused after every destination was taken),
armed once at boot into `HostTierContext::fault` (a `Cell`, taken by the first demote that runs the route;
the GPU unit cells set it directly). `docs/FLAGS.md` row updated.

| Cell (target card, `aefb89d89`) | Verbatim |
|---|---|
| GPU unit cells (`cargo test --release -p memra-server -- --ignored`, under the collector lock) | `test worker::tests::option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on ... ok`, `test worker::tests::option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on ... ok`, `test result: ok. 2 passed; 0 failed`. Each: a real `CudaTransfers` on the engine's stream over a ledger whose in-flight dimension is exactly one batch (three planes, six ops); the injected refusal is `HostImageFailure::Failed` (never `SourceQuarantined`), `host.disabled == false`, every plane back in its slot with the bytes it was built with, the ledger at `(pinned, inflight, device) == (0, 0, 0)`, then a clean demote completes with receipts and holds `pinned == 1392` until the image drops. |
| `tools/kv-host-contract-fault-gate.sh` presubmit | `demote failed (tier D2H producer fence refused: injected failure (MEMRA_KV_HOST_FAULT=contract-presubmit)); nothing demoted`, then `contracts door D2H receipt: ticket issuer=2 seq=1 epochs=0/1/1 items=34 (16 KV planes, draft) complete=34 require=ok ... retired acknowledged`, then `demote: 86 tokens, 160.6MB`; no `TIER DISABLED`, no quarantine, no `Capacity`; `seq=1`: no ticket had been issued before |
| same, postpublish | `demote failed (tier D2H receipt refused: injected failure (MEMRA_KV_HOST_FAULT=contract-postpublish)); nothing demoted`, then the receipt with `seq=2` (the aborted ticket retired and was acknowledged; in-flight is one batch, a leak would have refused this demote with `Capacity`), then `demote: 86 tokens, 160.6MB`; `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` |
| `kv-host-spill-identity-gate.sh` default, OFF vs ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` both arms, equal demote bytes, `verify ok`, one receipt per ON demote |
| `kv-host-spill-failure-gate.sh` default, OFF vs ON | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` both arms (the pre-existing pool-full line); digest cell ON: receipt, flip, named injected difference, `VERIFY FAILED` |

Not observable from the server and stated as such: the transfer engine's `producers` map has no
accessor, so its emptiness after an abort is evidenced by the clean second demote (a new producer fence
is recorded and released) rather than read directly.

Local battery on `aefb89d89` (`day15-local-merge/`, `systemd-run --user --scope -p CPUQuota=1200% -p
MemoryMax=28G`, the test arm under `flock -n /tmp/memra-5090.lock` with a bounded wait: six 90 s waits
while lane B's probe held the lock, then it ran): `cargo fmt --all -- --check` clean; `cargo test -p
memra-server -p memra-kv -p memra-engine --offline` memra-server 756 passed, 0 failed, 8 ignored (the six
GPU cells plus the two new ones), memra-kv 71 passed, memra-engine 515 passed, 24 ignored, 1560 `ok`;
`cargo clippy` on the three crates `-D warnings` clean; `tools/check-flags.sh` every runtime name
resolves; `tools/docs-registry-census.sh` clean (58 tables, 901 rows after lane B's two removed door
rows); `git diff --check` clean. The earlier review-commit battery (`day15-local-review/`) has an empty
test log: `flock -n` refused while the probe held the lock, so its test arm never ran; kept as the record.

## What remains

- **Option C** (ruling 15, after these receipts): the promote H2D through `h2d`, `ready_view` and a
  `KvMaterializer` (`kv_tier_gate/active.rs restore`), reading the `CudaPinnedLease` as the H2D
  source; it touches the restore program directly (census Part C item 2) and must be bit-identical at
  `prefix_restore_at`. Finding 1's promote delta is its first measurement.
- **The write-combined destination** (finding 1): a cached pinned arm of `alloc_host`, or a documented
  cost, is the engine's decision.
- **DFlash tail slice**, **verify digest v3**, **pool-full failure-gate line**: unchanged from day 14.

Effort: approximately 4 agent-hours for the day, plus approximately 2 for the review round (budget 8).
