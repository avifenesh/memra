# Session C day twelve: contract census, and the lease token carries its identity

Scope: issue #552 acceptance criterion 1 ("Integrate the existing native HostPrefix and bank
dispatch owners through the shared contracts, preserving packed bytes, scale planes,
tenant/program/epoch identity and CUDA ownership"), per the lead's day-twelve order: census
first, then the smallest bank-side identity carry, no server change. No GPU cell ran today;
nothing here is a support state. Tree: lane merge `e9c31d740` of main `a1bd0be62`, then
`468d21016` (census) and `69905776f` (code).

## Census (`HOSTPREFIX-CONTRACT-CENSUS.md`)

Both native owners are mapped field by field against `crates/memra-tier/src/contracts.rs`
(program, tenant, epoch, record identity, scale planes, CUDA ownership, transfer engine,
budget, byte provenance), each cell with the `file:line` where the owner carries the field
today, where it drops it, and what a contract-routed path needs. Headlines:

- Bank side: the artifact digest is on every `BankId` but never crossed the proxy seam (the
  token was two `u64`s); the tenant is the artifact digest by construction (`native.rs:334`);
  the epoch is a constant `{0, 0, 0}` on the dispatch adapter; `install_banked` does not check
  the owner thread (fail-late `WrongOwner` at the first miss); no `TransferEngine` (completion
  is a compute-stream drain); GPU slots sit outside the governor; scale planes are refused in
  three places, never carried; the MTP key `layer = u16::MAX` is positional.
- HostPrefix side: `HostTierContext` (`worker.rs:7869`) has no constructor anywhere in the
  crate, so lane B's sidecar route (`tier_charge`, `bind_tier_image`, insert and promote
  identity leases) is compiled and unit-tested but never executed; the server holds no
  `ProgramIdentity` for a loaded model; the share-cap tenant (`auth::meter_key` string row)
  and the contract tenant (`ProgramIdentity.tenant_salt`) are unrelated; the D2H is a plain
  `memcpy_dtoh` plus `synchronize` (or the arena's `cuMemcpyDtoHAsync_v2` plus fence) with no
  ticket, fence, epochs or `Completion`; the by-reference demote (`host_demote_prefix_ref`,
  `&PrefixEntry`) conflicts with `CudaTransfers::register_device`'s owned `KvPlane`; the host
  destination type (`PinnedHostBuf`) is not the contract's (`CudaPinnedLease`); `tier_charge`
  refuses whenever the startup arena exists.
- Eight named places where a second numeric program or a byte reinterpretation would be risked
  (Part C), all refusals or "do not convert", none silent.
- Proposal (Part E): Option A (construct `HostTierContext` behind a default-OFF door: build
  `ProgramIdentity` at model load, derive `tenant_salt` from the same `scope_namespace` as
  `meter_key`, inject the server's governor; no copy program changes) first, then Option B
  (D2H demotion through `TransferEngine` on the pageable path, the `active.rs:165-219`
  sequence, `HostPlane` gains a `CudaPinnedLease` arm). Admissibility gate: `serve-smoke.sh`
  and `cache-meter-gate.py` lines unchanged across OFF/ON, lane B's
  `prefix-evict-reclaim-gate.py` `V1..V4 -> PASS` verdict unchanged (V4 is the text sha256),
  `kv-host-spill-{identity,failure}-gate.sh` verdicts unchanged, `MEMRA_KV_HOST_VERIFY=1`
  `verify ok` on every promote, equal `[prefix-host] demote:` byte counts and
  `prefix_host_*` counters. Three lead rulings are needed before any server code: who derives
  `tenant_salt`, the borrow discipline for Option B, and arena-path scope.

## Code (`69905776f`): the lease token carries the identity the registry holds

The one bank-side gap that is a pure identity carry with no byte change.

- `crates/memra-tier/src/bank/expert_dispatch.rs:14-36` `pub fn dispatch_id(&RecordId) ->
  Result<ExpertDispatchId>`: the single mapping from `RecordId::Expert { layer, projection,
  original_id }` to the native `(layer, proj, expert)` key (`proj` 0/1/2 Gate/Up/Down; a
  `Row`, an `Other` projection, or a layer or id outside `u16` is `InvalidLayout`; the native
  MTP key `u16::MAX` is the largest layer it names). `SlruExpertDispatch::new` now uses it
  (`expert_dispatch.rs:67-70`) in place of its inline match, same refusals.
- `crates/memra-tier/src/bank/owner_proxy.rs:55-75` `ExpertLeaseToken { owner, lease,
  record: ExpertDispatchId, artifact: Digest, epochs: Epochs }` with `record()`, `artifact()`,
  `epochs()`; all `Copy`, still `Send + Sync` (the compile assertion at
  `moe_cache.rs:2122-2130` is unchanged and still compiles). `identity(&ExpertDemand)`
  (77-85) reads the leased `BankId` and ticket; `require(demand, token)` (87-92) refuses a
  token whose identity does not match the pending lease with `ForeignLease`.
- `demand` (`owner_proxy.rs:165-177`): after the bank publishes, the lease identity must name
  the demanded key; otherwise the demand is retired through the bank's own `finish` and the
  call refuses (`ProgramMismatch`, or the `dispatch_id` error). No token exists for a lease
  the caller did not ask for. `with_bytes` (197) and `finish` (210) call `require` after the
  `UnknownTicket` lookup.
- `crates/memra-engine/src/moe_cache.rs:981-987` `admit_banked`: after `demand`, if
  `token.record() != (layer, proj, expert)` the lease is finished and the admission refuses
  with `banked lease names another expert than the one demanded`, before any H2D. The proxy
  already guarantees the equality; this is the consumer's assertion at the seam. `admit_native`
  and every byte the door moves are unchanged.
- `research/spill-c-20260919/fixtures/slru-synthetic.json`: re-pinned to the new
  `moe_cache.rs` hash (`072cad5dff268819c14919ffea60cdaafeec953eec697edc52cc386ae9b9a180`) by
  `slru-trace.py`; a `diff` of both files without the `source_sha256` line is empty (every one
  of the 2013 rows identical), and `slru-trace.py --check` prints `source/trace matched`.

Tests: `crates/memra-tier/src/bank/owner_proxy.rs` unit tests (`tests` module, 223-...) with a
`Serving` fake bank that leases the record it was built to serve whatever was demanded:
`token_carries_the_identity_the_registry_holds_for_its_lease`,
`a_lease_for_another_record_is_refused_before_a_token_exists_and_retired` (red arm:
`ProgramMismatch`, the bank's `finish` ran once, `close` succeeds with nothing pending),
`a_lease_whose_record_has_no_dispatch_id_is_refused_and_retired` (`RecordId::Row`,
`InvalidLayout`), `a_forged_token_with_the_right_numbers_and_the_wrong_identity_is_foreign`
(three forged tokens, wrong record, wrong artifact, wrong epochs, each `ForeignLease` on
`with_bytes` and `finish`; the true token still completes),
`dispatch_id_is_the_native_key_and_refuses_what_it_cannot_name`; and
`crates/memra-tier/tests/bank/owner_proxy.rs:124` `token_identity_names_the_fixture_lease`
on the real `SlruExpertDispatch` fixture (record `(2, 0, 9)`, artifact `[7; 32]`, epochs
`{7, 19, 31}`). Documented in `docs/TESTING.md` §Experts-via-tier and the door doc's Owner
registry and Admission rows.

CPU gate on this rig (`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`):
`cargo fmt --all -- --check` clean; `cargo test -p memra-tier --offline` 223 pass across eight
targets (lib 7 incl. the 5 new, bank 62 incl. the 1 new, contracts 55, 18, 6, 18, 53, 4),
0 failed; `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` clean;
`DOCS_RS=1 cargo clippy -p memra-engine --offline --lib --bin run-gen --bin run-spec --target
x86_64-unknown-linux-gnu -- -D warnings` clean; `tools/check-flags.sh` 864 runtime names, no
uncovered (no `MEMRA_*` read added); `tools/docs-registry-census.sh` clean; `git diff --check`
clean. The first tier run failed exactly one test, `day4::slru_matches_recorded_native_
semantics_synthetic_trace`, on the `moe_cache.rs` source hash; the re-pin above is the fix and
the rows did not move.

## Boundaries

- No native cell ran: `admit_banked`'s new assertion is compile-checked (DOCS_RS clippy) and
  cannot fire while the proxy holds (the proxy refuses first); the day-eleven target-card
  receipts remain the last native evidence for the door. A native rerun is owed before any
  support-state claim, as before.
- Not landed, by design: an install-time owner-thread check (needs an `Engine`; door doc item
  1), scale admission (item 3, second half), a `ProgramIdentity` or shared governor on the
  bank (item 6). The census names each with its file:line.
- No server code changed. `tools/prefix-evict-reclaim-gate.py` exists only on
  `origin/lane/spill-b-20260919`; the census cites it from there.

Effort: approximately 4 agent-hours.
