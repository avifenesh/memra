# Census: the two native owners against the `memra_tier` contracts

Lane `lane/spill-c-20260919`, day twelve. Tree `e9c31d740` (lane merge of main `a1bd0be62`);
every `file:line` below is in that tree. This is the decision input for issue #552
acceptance criterion 1 ("Integrate the existing native HostPrefix and bank dispatch owners
through the shared contracts, preserving packed bytes, scale planes, tenant/program/epoch
identity and CUDA ownership"). It records what each owner carries today, where it drops a
contract field, and what a contract-routed path would need. No code was changed to write it;
the one bank-side step it names as landable is recorded in `DAY12.md` with its commit.

The shared contracts are `crates/memra-tier/src/contracts.rs`: `ProgramIdentity` (123, with
`tenant_salt` at 134), `KvBlockId` (164, `epoch` at 173), `StateBundle` (420) and
`StateKind` (414), `Epochs { state, src_gen, dst_gen }` (523), `TransferTicket` (538),
`FenceId` (544), `TierBudget` (713, `pinned` at 718), `BudgetRequest` (827, `tenant: Digest`
at 831), `BudgetGovernor` (991), `CopyOp` (1289; a D2H requires a `producer_fence`, 1322),
`TransferEngine` (1457; `d2h` at 1463), `BankId`/`RecordId`/`TensorId` (1603 to 1655),
`BankLease` (1682), `BankedResidency` (1808). The native transfer backend is
`crates/memra-engine/src/tier_transfer.rs` (`CudaTransfers` 159; owner thread recorded at
construction 173 to 190 and checked by `check_thread` 192; `alloc_host` 210 charges
`TierBudget.pinned`; `register_device` 273 takes an owned `KvPlane` on the owner stream;
`validate` 598 refuses a foreign stream or context with `WrongOwner`).

## Part A. Bank dispatch owner (`--experts-via-tier`)

Owner side: `crates/memra-engine/src/banked_residency/native.rs` (`install_expert_bank_gate`
140). Proxy side: `crates/memra-tier/src/bank/owner_proxy.rs` (`ExpertBankOwner` 36,
`ExpertBankProxy` 41 holding a `ThreadId` at 42, `ExpertLeaseToken` 48). Dispatch adapter:
`crates/memra-tier/src/bank/expert_dispatch.rs` (`ExpertDispatchId = (u16, u8, u16)` 8,
`SlruExpertDispatch` 24). Consumer: `crates/memra-engine/src/moe_cache.rs` (`banked` and
`banked_pending` fields 255 to 256, `install_banked` 943, `admit_banked` 962, `admit_native`
993).

| Contract field | Carried today (file:line) | Dropped or bypassed (file:line) | Contract-routed path needs |
|---|---|---|---|
| Program identity (artifact) | The installer hashes the open inode (`native.rs:153-168`) and locks it to `APPROVED_SHA` (29). Every `TensorId.artifact` is that digest (`native.rs:419-423`), so every `BankId` and `BankLease` carries it (`contracts.rs:1636-1641`). | The value never crosses the proxy seam: `ExpertLeaseToken` is two `u64`s (`owner_proxy.rs:48-51`); `MoeSlotCache` stores only the proxy (`moe_cache.rs:255`). The `installed` line prints the SHA from the installer's local (`native.rs:364-366`), not from anything the cache holds. | The token that leaves the owner thread carries the `BankId` identity the registry already holds (artifact, record, ticket epochs), and the cache asserts the record it demanded is the record it was leased. Landed today, see `DAY12.md`. |
| Tenant | `BudgetRequest.tenant = artifact` (`native.rs:334`): the gate is single-tenant and the artifact digest stands in for the tenant salt. The gate-local `Governor` keys fairness on it (`crates/memra-tier/src/tier/governor.rs:94-100, 176`). | No `ProgramIdentity` and no `tenant_salt` exist on the bank side at all; the request's tenant is not on the `BankLease` (`contracts.rs:1682-1688`) and not on the token. | A serving-shape bank (door doc item 6) would take its `ProgramIdentity` from the loaded model and put `tenant_salt` in the request; today's gate has nothing to carry. Not a day-twelve step. |
| Epoch | `SlruExpertDispatch` holds one constant `Epochs { 0, 0, 0 }` (`native.rs:344-348`), stamps it on every `BankBatch` (`expert_dispatch.rs:94`) and `BankService::publish` requires `ticket.epochs == current` (`crates/memra-tier/src/bank/residency.rs:556`). | The cache states no epoch; the ticket's epochs are not on the token. A reloaded model behind the same proxy would be indistinguishable to the cache (the proxy `id` changes only because a new owner registers). | Token carries `ticket.epochs` (landed today). An epoch the cache owns (model generation) is item 6 territory. |
| Record identity | `SlruExpertDispatch::new` requires `ids[(layer, proj, expert)].record == RecordId::Expert { layer, projection, original_id }` (`expert_dispatch.rs:37-54`); a masked (pruned) expert has no record (`crates/memra-tier/src/bank/adapters.rs:122-128`). | The MTP block is keyed `layer = u16::MAX` on the cache side (`native.rs:215`) and stored as `RecordId::Expert { layer: 65535 }`; the plan's semantic id is `ExpertBankBlock::Mtp { depth }` (`crates/memra-gguf/src/expert_banks.rs`). A positional convention, not the semantic id. | Keep the convention (it is the native cache's key) but name it once; the token now carries the `(layer, proj, expert)` triple so the cache can check it. |
| Scale planes | Refused, never dropped: `.scale` / `.input_scale` census rows are a catalog refusal (`expert_banks.rs`, day eleven), loaded `macros` / `fp8_blk` refuse at `native.rs:399-404`, and `SlruExpertDispatch::new` refuses any layout that is not exactly one `Role::Payload` segment (`expert_dispatch.rs:58-65`). `map_host_exps` checksums macro and block scales when present (`crates/memra-engine/src/banked_residency.rs:71-120`). | `bank_projection` writes `scales: vec![]` into every `ExpertSource` (`native.rs:467`), so the tier-side scale checksum path is never fed. | Scale admission (door doc item 3, second half): a multi-plane dispatch adapter and an `admit_native` that stages scale planes with the payload. New program, lead decision. |
| CUDA ownership | Typed as a thread: the owner registers on the CUDA thread (`owner_proxy.rs:53-79`) and every proxy call refuses `WrongOwner` off that thread (`owner_proxy.rs:104-107`). The H2D stays the native `admit_native` on the compute stream (`moe_cache.rs:1008`) followed by an explicit `stream().synchronize()` before `finish` (`moe_cache.rs:987-988`; `admit_native` also drains before publish when banked, 1028-1032). `Engine: Send + Sync`, `MoeSlotCache: Send` and the token/proxy `Send + Sync` are compile-asserted (`moe_cache.rs:2122-2130`). | `install_banked` does not check the installing thread (`moe_cache.rs:943-959`); a cache installed off the owner thread fails at the first `admit_banked` (`WrongOwner` from `validate`, 973), after every dispatch path has already been routed to the bank (1227-1232, 1430-1431). `Engine::with_moe_cache` takes a `Mutex` from any thread (`crates/memra-engine/src/lib.rs:6510-6524`); the pp2 stage scope at `crates/memra-engine/src/hybrid_forward.rs:4713` shows the shape that would enter it from a worker. | Door doc item 1 (owner per stage or a typed hand-off) is the real fix and is the lead's PP placement decision. A fail-early install-time owner check needs an `Engine` to test and is not landed today. |
| Transfer engine | Not used. The bank path moves bytes with the native `stage_expert` (`moe_cache.rs:1008`) and borrows the host payload through `with_bytes` (`owner_proxy.rs:134-144`). | No `TransferTicket`, no `Completion`, no `ready_view`; the bank's "completion" is the compute-stream drain. | Door doc item 4 (overlap on copy-stream events) is where a ticketed H2D belongs; performance item with the publish-before-completion hazard `d0acf6f03` closed. |
| Budget | One gate-local `Governor` (`native.rs:302-313`): `pageable` 512 MiB, `staging = max_bytes`, `inflight = 1`; SLRU metadata charged (`native.rs:336-338`); host bank bytes bounded by `BankLimits.cache_bytes` (`native.rs:324`). Host budget refusals are typed (`banked_residency.rs:128-139, 234-248`), GPU slot budget refusals too (262-299). | GPU slots are sized by `build_moe_cache_exact` / native sizing (`native.rs:358-360`) and are not charged to the governor (`TierBudget.device` stays zero). The host backing is `Vec<u8>` (pageable), so `TierBudget.pinned` is unused. The governor is not the server's (there is none on the bank side). | A shared governor is a serving-shape concern (item 6). For the gate the split is documented and correct. |
| Byte provenance | Every retained record is compared byte-for-byte with the loaded `HostExps` (`native.rs:444-446`) and its checksum chained into `records_sha256` (`native.rs:459-461`); the catalog is plan-derived (`native.rs:104-135`). | None. | None. |

What the bank side still bypasses, in one list: (1) no identity on the lease token (fixed
today); (2) tenant is the artifact digest by construction, no `ProgramIdentity`; (3) install
does not check the owner thread (fail-late `WrongOwner`); (4) no `TransferEngine`,
completion is a stream drain; (5) GPU slots outside the governor; (6) scale planes refused,
not carried; (7) the MTP key is positional.

## Part B. HostPrefix owner (`memra-server` host tier)

Owner: `crates/memra-server/src/worker.rs`. Entry types `PrefixEntry` (6125), `PrefixPlane`
(6109: two `CudaSlice<u8>` plus `len`, `k_tok_bytes`, `v_tok_bytes`), `HostPrefixEntry`
(7428), `HostPlane` (7354: two `memra_engine::PinnedHostBuf`), `HostPrefixCache` (7462).
Demotion: sink wrappers `insert_demoting` / `insert_pinned_demoting` (6951, 6976) call
`host_demote_prefix_entry` (8346) which calls the by-reference body `host_demote_prefix_ref`
(8412); planes are copied by `host_plane_from_device` (8085) inside `host_entry_from_device`
(8135). Promotion: `host_promote_prefix_hit` (8872) rebuilds a device entry through
`device_entry_from_host` (8720). The bridge already in the tree is lane B's day-two
sidecar (`research/spill-b-20260919/HOSTPREFIX-EXTENSION.md`): `HostTierContext` (7869),
`tier_charge` (7875), `bind_tier_image` (7927), `IdentitySlot` and `ResidentCharge` in
`crates/memra-kv/src/tiered/hostprefix.rs` (12, 123), reached through the
`memra_engine::cache` re-export (`crates/memra-engine/src/lib.rs:76-78`); `memra-server`
has no direct `memra-tier` dependency (`crates/memra-server/Cargo.toml:18-26`).

Headline finding: `HostTierContext` is defined at `worker.rs:7869` and named only by the
field `tier: Option<HostTierContext>` at 7463. No constructor site exists in the crate
(`grep -rn "HostTierContext\|tier: Some\|\.tier = " crates/memra-server/src/` returns the
two lines above and nothing else); `HostPrefixCache::new` (7526) goes through `Default`, so
`tier` is `None` in every boot (14087). Consequently `tier_charge` returns `Ok(None)`
(7883-7885), `bind_tier_image` returns `Ok(())` (7929-7931), the insert identity check is
skipped (7736-7745), and the promote lease is `None` (8891-8895). The contract route in the
server is compiled, unit-tested on the memra-kv side, and never taken. Everything below that
says "when `tier` is `Some`" describes code no runtime path executes today.

| Contract field | Carried today (file:line) | Dropped or bypassed (file:line) | Contract-routed path needs |
|---|---|---|---|
| Program identity | `PoolKey = (model, PC-ISO namespace)` strings (`worker.rs:3060`) key both caches; `PREFIX_ENTRY_LAYOUT_VERSION = 5` (3611) travels with each entry and is refused on mismatch at demote (8142-8144) and promote (8721-8726); a model-generation `Arc` guards GLM entries (`generation_current`, 7534-7540). When `tier` is `Some`: `HostTierContext.programs: HashMap<PoolKey, (ProgramIdentity, Arc<()>)>` (7871) supplies a full `ProgramIdentity` per pool key; `bind_tier_image` builds a `StateBundle` with `program.clone()` (8043-8052) and `IdentitySlot::bind` requires `bundle.require(program, expected_layout, 0, tokens)` (`hostprefix.rs:30`). | Nothing populates `programs`; the strings in `PoolKey` are not derivable into `ProgramIdentity` digests (artifact, serialized plan, numeric, stream, tokenizer, template, adapter, modality, position, tenant salt: `contracts.rs:123-135`). The server does not hold a `ProgramIdentity` for a loaded model anywhere; the only producer of one in the engine tree is the gate's own construction (`crates/memra-engine/src/bin/kv_tier_gate.rs`). | A bootstrap that builds `ProgramIdentity` from the loaded model (artifact lock, serialized plan digest, numeric/stream class, tokenizer, template, cache salt) and inserts it into `programs` under the pool key at model load. That is the precondition for every other row here. |
| Tenant | Two disjoint tenant identities. Share-cap accounting keys on `auth::meter_key(&key.1)` (`worker.rs:7722, 7776, 7798`; `crates/memra-server/src/auth.rs:585-593`), a string row `t:<tenant>`. The tier request uses `program.tenant_salt` (7910) and `ResidentCharge::reserve_for_program` refuses `request.tenant != program.tenant_salt` (`hostprefix.rs:148-150`). | The string row and the salt digest are never related to each other in code; a purge (`purge_tenant`, 7838-7860) removes by the string row and would not touch a governor charge keyed by salt. | Define the mapping once: `ProgramIdentity.tenant_salt` derived from the same `scope_namespace` the metering row comes from, so `meter_key` and `tenant_salt` name the same tenant. Decision needed on which side owns the derivation. |
| Epoch | Host entries are immutable prefixes: `StateKind::ImmutablePrefix` with `KvBlockId.epoch = 0` (8041, 8048) and `committed_high_water = entry.pos` (8049); `StateBundle::validate` enforces epoch 0 for immutable prefixes (`contracts.rs:445-447`). | The D2H and H2D copies carry no `Epochs { src_gen, dst_gen }` at all: `dtoh_u8_into_pinned` (`lib.rs:13537-13558`) and `PinnedHostBuf::copy_from_device_u8` (`crates/memra-engine/src/pinned_host.rs:218-252`) are plain copies. A `CopyOp` needs `epochs` and a `DeviceLease` whose generation equals `src_gen` (`contracts.rs:1302-1318`). | The demote path registers the plane with a generation and stamps `Epochs` on the `CopyOp`, exactly as `kv_tier_gate/active.rs:165-181` does (`EPOCHS` const at 11, `register_device(backing, EPOCHS.src_gen, ...)`). |
| Transfer engine and completion | Demotion: synchronous `memcpy_dtoh` plus `synchronize` on the ambient stream (`lib.rs:13556-13557`), or `cuMemcpyDtoHAsync_v2` plus `stream.synchronize()` on the plane's stream for arena planes (`pinned_host.rs:240-251`); the arena copy fences even on enqueue error (248-250). Promotion: `htod_u8_into` (8744, 8749). Both run on the worker's CUDA owner thread (7349-7351, 8695, 10272). | No `TransferTicket`, no `producer_fence`, no `Completion::require`, no `ready_view` / `take_destination`, no `retire` / `acknowledge`. Completion is "the synchronize returned". | `CudaTransfers::new(e.stream(), governor)` on the worker thread (`active.rs:399`), `alloc_host` for the destination (`tier_transfer.rs:210`), `record_producer` then `d2h(CopyOp { .. })`, `synchronize(&ticket)`, `take_destination`, `retire_source`, `release_device_observed`, `release_producer` (the sequence at `active.rs:165-219`). Restore is the mirror (`active.rs:220-296`). |
| Device ownership of the source | The demoted entry's planes stay owned by the `PrefixEntry`; the demote is by reference (`host_demote_prefix_ref(&PrefixEntry)`, 8412) so a caller whose device state is live (pause sweep 18267, 18307; admission flush `evict_all_demoting` 8371-8399) removes it only after the host copy publishes. | `CudaTransfers::register_device` takes an owned `KvPlane` (`tier_transfer.rs:273-285`, `impl Into<KvPlane>`) and hands it back only through `take_plane` (366-386). A borrowed `&CudaSlice<u8>` cannot be registered. | Either the demote path takes the planes out of an `&mut PrefixEntry` (moving each `CudaSlice<u8>` into the registry and back through `take_plane`, changing the entry's borrow discipline), or `CudaTransfers` gains a borrowed-source D2H seam. The first keeps the contract as frozen; the second is a v1.4 contract question. |
| Host destination type | `HostPlane { k, v: PinnedHostBuf }` (7354-7360); pageable-path buffers come from `PinnedHostBuf::new` (`pinned_host.rs:186`, `malloc_host`), arena planes from `PinnedHostArena::try_reserve_planes` (`pinned_host.rs:119`; `HostPlaneLeases::take` 8067-8081). | A `TransferEngine` D2H returns `Destination::Host(CudaPinnedLease)` (`tier_transfer.rs:31`, `alloc_pinned` on the context at 225), a different type with a governor charge and a `Rc<PinnedAllocation>` owner. Converting it into a `PinnedHostBuf` would be a second copy. | `HostPlane` gains a second arm (`CudaPinnedLease`) or the promote path reads through `PinnedLease::bytes()` (`contracts.rs:1075`). Server change, lead ruling. |
| Budget | Host budget `MEMRA_KV_HOST_MB` (4224-4260, `docs/FLAGS.md:1425`), plain byte LRU (`insert` 7735-7827), per-tenant share cap `MEMRA_KV_HOST_TENANT_PCT` (7628-7638, checked before the copy at 8452). When `tier` is `Some`: `tier_charge` reserves `pinned` for KV planes and `pageable` for the rest at demote (8467-8496), device bytes at promote (8897-8913), through `reserve_tier_image` (`crates/memra-server/src/admit_memory.rs:358-366`) with `Priority::Backup` / `AdmittedRestore` (7902-7908). | `tier_charge` refuses whenever the startup arena exists ("tier fixed-arena lease handoff pending", 7888-7890), so the `MEMRA_GLM5_TP_KV_HOST` arena path (14088-14112) can never take the contract route as written. The arena itself is not governor-charged. The device-side `PrefixCache` budget (`kv_flex_effective_budget`, `prefix_cache_protected_pct`, 6964-6966) is a separate ledger. | One injected governor (`SharedGovernor`, `hostprefix.rs:119`) with the arena's fixed backing charged once at reserve and slices pinned, as the sidecar comment at 7886-7887 already states. |
| Byte checksums | When `tier` is `Some`: per-segment `checksum(bytes)` over the host copy after the D2H (7997) into `StateBundle.checksums`; `Role::Key`/`Value` with `q8_0` / `q5_1` encodings and a 34 / 24 byte row check (8015-8022), `Recurrent`, `Logits`, `Hidden`, `Transaction` (metadata) segments (8031-8040). Always: `MEMRA_KV_HOST_VERIFY=1` computes `prefix_entry_state_digest` (10798) on the device entry before the copy and re-checks at promote (8916-8946). | The contract checksums attest the host image, not the device source: nothing hashes the device bytes before the copy except the verify arm, and that digest (`memra-prefix-split-state-v2`, 10817) is a different hash program from `contracts::checksum`, so the two are not comparable. | A contract-routed D2H puts the expected checksum on the `SegmentExpectation` (`Completion::require`) so the transfer proves what it wrote; the verify digest stays as the independent receipt. |
| Identity lease and invalidation | When `tier` is `Some`: `IdentitySlot::bind` after the copy (8056-8058), `lease(program, generation)` at insert (7741) and promote (8892), `require` again after the H2D (8954), `Drop` invalidates (`hostprefix.rs:69-73`). | Handoff imports stay unbound by design (`host_entry_from_owned`, 9982; FREEZE B8). The GLM arm (`host_glm::HostGlmState`) is refused by `bind_tier_image` (7944-7956). | Nothing beyond the bootstrap; the sidecar is complete for the plain KV plus recurrent surface. |
| Scale planes | Not applicable to the host tier: KV planes are `q8_0` / `q5_1` blocks whose scales live inside the block bytes (8015-8022 checks the row multiples). Latent (MLA/DSA) planes are refused by the tier (8432-8441) and by `bind_tier_image`. | None. | None for the first slice. |

## Part C. Where a second numeric program or a byte reinterpretation would be risked

1. **Host destination conversion.** Turning a `CudaPinnedLease` into a `PinnedHostBuf` (or back) by copying is a second host copy of the same bytes; not a numeric program, but a second byte owner. Refuse; add the arm to `HostPlane` instead (Part B, host destination row).
2. **Promote path.** `device_entry_from_host` (8720-8830) is the restore program: `htod_u8_into` of exactly `len * k_tok_bytes` bytes (8744-8750), `htod` of the f32 planes. Routing the H2D through `TransferEngine` must copy the same byte ranges into `CudaSlice<u8>` planes of the same lengths; any change to `k_tok_bytes` handling, `len`, or the `.max(1)` allocation floor (8737-8742) changes the restored entry. The one numeric program per request rule applies: a promoted entry and a resident entry must be byte-identical inputs to `prefix_restore_at`.
3. **Arena versus pageable split.** `host_plane_from_device` has two copy programs (arena `copy_from_device_u8` 8108-8113, pageable `dtoh_u8_into_pinned` 8114-8123). A contract-routed D2H is a third. The first slice must be one of them, not a mix within one entry.
4. **Verify digest boundary.** `prefix_entry_state_digest` hashes `at * k_tok_bytes` bytes per plane (10827-10829); the contract checksum hashes `valid_bytes` of the segment (7997 uses the whole `p.k.as_slice()`). These agree only because `p.len == entry.pos` is required (8019); keep that requirement or the two receipts diverge.
5. **Tenant identity.** Deriving `tenant_salt` from a different string than `meter_key` reads would let one tenant's charges land on another's governor row. Derivation must be a single function used by both.
6. **Epoch zero.** Immutable prefixes carry epoch 0 (`contracts.rs:445-447`). Any D2H that stamps `Epochs.state != 0` on a prefix bundle fails `StateBundle::validate`; conversely stamping `src_gen` on a plane whose `DeviceLease` was registered with another generation fails `CopyOp::validate` (`contracts.rs:1314-1316`; the D2H fence check at 1322-1329). Both are refusals, not silent reinterpretation, which is what the contract is for.
7. **Bank MTP key.** `layer = u16::MAX` (`native.rs:215`) is the cache's own convention; a plan with more than one MTP depth would collide. The installer already refuses depth > 0 (`native.rs:217-221`).
8. **Bank scale planes.** Banking a scale-bearing record payload-only would change the dequant program. Refused at three places (Part A, scale planes row); keep refusing until the multi-plane adapter exists.

## Part D. Bank side, the smallest true step (landed today)

The one gap that is a pure identity carry with no byte change: `ExpertLeaseToken` gains
the record identity the registry already holds for the lease (`BankId.tensor.artifact`, the
`(layer, proj, expert)` dispatch id derived from `BankId.record`, and the ticket's `Epochs`),
`with_bytes` / `finish` refuse a token whose identity does not match the pending lease, and
`admit_banked` asserts the lease names the expert it demanded before the H2D. The proxy and
token stay `Send + Sync` (the compile assertion at `moe_cache.rs:2122-2130` still holds),
the H2D program is unchanged, and the mapping from `RecordId::Expert` to
`ExpertDispatchId` is the one `SlruExpertDispatch::new` already enforces
(`expert_dispatch.rs:37-54`), now a single function. Tests and receipts in `DAY12.md`.

Not landed, and why: an install-time owner-thread check needs an `Engine` to exercise
(door doc: no CUDA-free test can construct `MoeSlotCache`) and belongs with item 1; scale
admission is a new program (item 3, second half); a shared governor and `ProgramIdentity` on
the bank are serving-shape (item 6).

## Part E. HostPrefix side, first contract-routed slice: options and recommendation

No server code changed today. Three candidate first slices, each behind `tier: Some`, which
does not exist yet:

**Option A, bootstrap only (identity route, no transfer change).** Construct
`HostTierContext` at boot behind a default-OFF door: build `ProgramIdentity` for each loaded
model, derive `tenant_salt` from the same `scope_namespace` as `meter_key`, inject a
`SharedGovernor` (the server's, not a prefix-only one, per `hostprefix.rs:118`). The
existing sidecar code (`tier_charge`, `bind_tier_image`, insert and promote leases) starts
executing; the D2H and H2D programs are untouched. Gate: with the door OFF every line below
is unchanged by construction; with it ON, the same lines are unchanged and the only new
output is a refusal when identity does not bind. Risk: the arena refusal (7888-7890) means
the door is pageable-path only until the arena lease handoff exists.

**Option B, D2H demotion through `TransferEngine` (the transfer route).** In
`host_plane_from_device`, on the pageable path only, replace the two `dtoh_u8_into_pinned`
calls with the `active.rs:165-219` sequence: `register_device` (requires taking the plane
out of an `&mut PrefixEntry`, Part B device ownership row), `alloc_host`, `record_producer`,
`d2h`, `synchronize`, `take_destination`, `retire_source`, `release_device_observed`,
`release_producer`, then `take_plane` back into the entry. `HostPlane` gains a
`CudaPinnedLease` arm. The `StateBundle` checksums move onto `SegmentExpectation` so
`Completion::require` proves the bytes. Depends on Option A for the identity fields of the
bundle (without a `ProgramIdentity` the `KvBlockId` cannot be built, 8041).

**Option C, promotion H2D first.** Route `device_entry_from_host` through `h2d` plus
`ready_view` and a `KvMaterializer` (the `active.rs:220-296` restore). Touches the restore
program directly (Part C item 2); highest risk, least value first.

Admissibility gate for whichever slice the lead picks, all on one card class through the
collector, door OFF and door ON, N=1 executed-not-qualified:

- `tools/serve-smoke.sh` prints the same `PASS` lines in both arms, including
  `cache-metering accounting exact (per-request + /metrics + economics)`
  (`tools/serve-smoke.sh:159-167`); `tools/cache-meter-gate.py` prints every `ok:` line and
  `cache-meter-gate: 0 failed` (`tools/cache-meter-gate.py:56, 158-167, 204`), with
  identical `prefix_cache_{hits,misses,inserts,hit_tokens}` values in both arms.
- Lane B's `tools/prefix-evict-reclaim-gate.py` (on `origin/lane/spill-b-20260919`, not on
  main) prints the same `V1=ok V2=ok V3=ok V4=ok -> PASS` verdict line in both arms; V4 is
  the sha256 over the three greedy texts and is the bit-identity receipt for the request
  path (`prefix-evict-reclaim-gate.py:40-42, 656-657`).
- `tools/kv-host-spill-identity-gate.sh` and `tools/kv-host-spill-failure-gate.sh` print the
  same verdicts; the failure gate's `alloc-fail` and `flip-demote` arms (`docs/FLAGS.md:1251`)
  must still latch and still catch the flipped byte.
- Bytes unchanged receipt: `MEMRA_KV_HOST_VERIFY=1` prints `[prefix-host] verify ok: promoted
  state digest matches demote digest (N tokens)` (8933-8935) for every promote in the ON arm,
  the `[prefix-host] demote: N tokens, X MB` byte counts (8525-8531) are equal across arms,
  and for Option B the `StateBundle::verify` per-segment checksums equal the
  `Completion` checksums the transfer reported. The `/metrics` counters
  `prefix_host_{entries,bytes,demotions,promotions}` (`worker.rs:2053-2056`) are equal across
  arms for the same workload.

**Recommendation: Option A first, then B on the pageable path.** A is the precondition for
every contract field in Part B (program, tenant, epoch and checksums all hang off
`ProgramIdentity`), it executes code that is already in the tree and unit-tested but has
never run, it changes no copy program, and its OFF arm is byte-identical by construction.
B is the first real transfer-contract crossing and needs the borrow-discipline decision
(move planes out of `&mut PrefixEntry`, or a borrowed-source seam in `CudaTransfers`) that
only the lead can make against the frozen v1.3 contract. C waits for B's receipts. Lead
rulings needed before any code: (1) who derives `tenant_salt` (server `auth` or a memra-kv
helper); (2) the borrow discipline for Option B; (3) whether the arena path is out of scope
for the first two slices (I say yes, per the refusal at 7888-7890).
