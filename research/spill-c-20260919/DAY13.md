# Session C day thirteen: Option A, the HostPrefix contracts door

Scope: lead rulings 13 to 15 on the day-twelve census (`HOSTPREFIX-CONTRACT-CENSUS.md` Part E).
Option A: construct `HostTierContext` behind a default-OFF door so lane B's sidecar route runs,
with `ProgramIdentity` built at model load, one `tenant_salt` owner, and the server's governor
injected; no copy program changes; the startup arena out of scope. Tree: lane merges of main
`5ecfd262c` (`c3154752d`) and `0e7741b76` (`6ec4d29a6`), code commits `ff46abc75`, `e7e23dcf4`,
`b81881dad`. Target-card cells ran on the `b81881dad` tree (binary sha256
`9ed0249500d24bb128d59dcc8ddd2c1df08790b81af30c36cb06a7069448a88f`), one RTX PRO 6000 Blackwell
Server Edition at its 600 W limit (`pro-single-day13/card.csv`), through the canonical collector
(`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`), N=1,
`executed-not-qualified`. Nothing here is a support state. Door doc: `HOSTPREFIX-DOOR.md`.

## The door (`MEMRA_KV_HOST_CONTRACTS`, default OFF, decide-by 2026-10-05)

`crates/memra-server/src/worker.rs`: `kv_host_contracts_door` / `parse_kv_host_contracts`
(4227, 4239; strict `1`/`0`/unset, every other value refuses the boot), `host_tier_arena_refusal`
(4255), `sha256_file_hex` (4271), `host_tier_numeric_class` (4290), `host_tier_program_base`
(4308), `host_tier_governor` (4342), `host_tier_context` (4376), `HostTierContext` with
`program(key)` (8102, 8116), boot wiring (14374 parse and arena check before the arena reserve,
14419 construct after `model_generations`). `crates/memra-kv/src/tiered/hostprefix.rs`:
`tenant_salt` (129, ruling 13) and `shared_governor` (137). `docs/FLAGS.md` row in the same
commit; `tools/check-flags.sh` covers the new read (867 runtime names, none uncovered).

Boot refusals, typed, never a silent OFF: a junk value; `MEMRA_GLM5_TP_KV_HOST=1` (the arena is
charged by its owner, its lease handoff is out of scope, `tier_charge` would refuse every demote
with `tier fixed-arena lease handoff pending`; checked before the arena is pinned); a
checkpoint-directory model; a loaded vision tower. With `MEMRA_KV_HOST_MB=0` no host tier exists,
so the door constructs nothing and prints one `[kv-host-contracts] ... nothing to route` line
under its own tag (`b81881dad`; the identity gate's OFF twin boot asserts `[prefix-host]` stays
silent, and attempt 1 caught the door speaking under that tag).

### `ProgramIdentity` fields and where each comes from (`host_tier_context`, per loaded model)

| Field | Source at model load | Domain |
|---|---|---|
| `artifact` | streaming SHA-256 of the GGUF file (8 MiB reads); on the target card 7,837 ms for the 15.7 GB artifact | `artifact-sha256` over the hex, the `kv_tier_gate` convention |
| `serialized_plan` | `format!("{:?}", HybridModel.plan)`, the compiled `ModelPlan` (`kv_tier_gate` convention) | `plan-debug` |
| `numeric` | `server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B`: `PREFIX_ENTRY_LAYOUT_VERSION` and `memra_kv::kv_blk_bytes()`; one class because the prefix cache is shared by the prime, batched and spec paths that the serving gates hold bit-identical | `numeric` |
| `stream` | the single worker owner thread | `stream` over `server-worker-single-owner-thread` |
| `tokenizer` | the artifact (a GGUF tokenizer is the artifact's) | `artifact-tokenizer` over the hex |
| `template` | `Tokenizer::chat_template()` text, or the ChatML fallback name | `template-jinja` / `template` |
| `adapter` | none served | `adapter` over `none` |
| `modality` | text (a vision tower refuses the door) | `modality` over `text` |
| `position` | prefixes start at token zero (`IdentitySlot::bind` enforces `start == 0`) | `position` over `prefix-from-token-zero` |
| `tenant_salt` | `memra_kv::tiered::hostprefix::tenant_salt(&pool_key.1)`, stamped per pool key by `HostTierContext::program`; the argument is the exact string `auth::meter_key(&key.1)` reads for the share-cap row | `tenant-salt` over the namespace |

Target-card boot line (ON arms): `[prefix-host] contracts door: model gate program identity
artifact_sha256=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
plan_debug_sha256=8fc1154223424f168e69fdf10f4befec7a9af75544e385bb773759954ff427e5
numeric=server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B template=gguf-jinja device=0 (7837ms to hash
...)`, then `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): 1 model program
identities, tenant salt per pool namespace, server governor ledger pinned/pageable 17180MB
device 537MB; host tier armed`.

`programs` is keyed by model name, not by `PoolKey`, because pool namespaces arrive per request
and cannot be enumerated at boot; `program(key)` completes the identity deterministically, so the
charge taken at demote and the lease checked at insert and promote name one program (unit test
`host_tier_context_program_stamps_the_pool_namespace_salt_once`).

### The governor (ruling 15: the server's, injected)

The server had none; `hostprefix::shared_governor` builds one `Arc<Mutex<dyn BudgetGovernor>>`
over `memra_tier::tier::Governor` at boot and `HostTierContext.governor` is the injection point.
Sized as a LEDGER for Option A: twice the host budget on `pinned` and `pageable`, twice the device
prefix budget on `device[ordinal]`, zero headroom, no queue. Residents are at most one budget (the
host LRU invariant), one incoming image is at most one budget (`insert` refuses larger), and the
charge is taken before the LRU makes room, so the ledger must never be what refuses a demote the
OFF arm would have made; the LRU decides, the ledger records tenant, priority and bytes. An image
above the whole budget takes no charge at all (`e7e23dcf4`), so the OFF arm's `skip demote: entry
N MB > host budget M MB` line stays the line. Option B may tighten this once eviction-to-fit runs
before the charge. Unit test `host_tier_governor_ledger_admits_what_the_lru_would_and_binds_at_
twice_the_budget`.

### OFF is byte-identical by construction

Every removed line of the worker diff (`git diff 7efedd13d..b81881dad -- crates/memra-server`) is
inside a `tier`-Some block (`if let Some(tier) = &self.tier`, `let Some(tier) = &self.tier else
return`, `if host.tier.is_some()`, `if let Some(tier) = &host.tier`) or in the `HostTierContext`
definition. Two OFF-path statements are touched, neither changes an OFF value: (a)
`host_entry_from_device` gains an `else if host.tier.is_some()` arm for `model_generation`
(8403); the OFF arm is the same `None`. (b) `insert`'s tier block moves below the whole-budget
refusal (7975), a no-op check on the OFF path moved past another check. The boot wiring adds
statements that evaluate the door and return early on refusal; with the door unset they set one
`bool` to false. Lane B's silent tier refusals at demote, insert and promote now print a
`(contracts door)` line; all inside tier-Some blocks.

## Target-card receipts (`pro-single-day13/`, replay `verify-day13.py`: `DAY13 REPLAY: PASS`)

Every cell: door OFF then ON, same binary, same prompts, same env otherwise. `plain` =
`MEMRA_SERVE_SPEC=0` added to the gate's environment; `spec` = the gate's default environment,
which serves this MTP artifact speculatively. `MEMRA_HOSTGATE_CACHE_MB=256` for the two host
gates (finding 1). Verdict lines verbatim.

| Gate | OFF | ON | Equal |
|---|---|---|---|
| `tools/serve-smoke.sh` plain + cache-metering arms | 31 `ok:` lines, `ok: cache-metering accounting exact (per-request + /metrics + economics)`, `serve-smoke: 0 failed` | the same 33 verdict lines, byte-equal | yes (`verify-day13.py` diff) |
| `tools/kv-host-spill-identity-gate.sh`, plain | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 `ok:` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, the same 12 `ok:` | yes |
| same, `[prefix-host] demote:` byte counts | `demote: 89 tokens, 160.5MB`, `demote: 86 tokens, 160.4MB` | `demote: 89 tokens, 160.5MB`, `demote: 86 tokens, 160.4MB` | yes |
| same, promote and verify | `verify ok: promoted state digest matches demote digest (89 tokens)`, `promote: 89 tokens, 159.5MB` | `verify ok ... (89 tokens)`, `promote: 89 tokens, 159.5MB`: 1 promote, 1 `verify ok` | yes |
| same, r1/r3 texts and cached tokens | r3 `cached_tokens=89` of `prompt_tokens=102` | identical texts, `cached_tokens=89` of 102 | yes |
| same, OFF twin boot (`MEMRA_KV_HOST_MB=0`) | 0 `[prefix-host]` lines | 0 `[prefix-host]` lines, 1 `[kv-host-contracts]` line | gate line `ok: OFF boot never touches the tier` in both |
| `tools/kv-host-spill-identity-gate.sh`, spec | `ALL GREEN` | `KV-HOST-SPILL IDENTITY GATE: 5 FAILURE(S) (teeth=0)`: the feed assertions fail, `ok: r1 ON == OFF byte identity` and `ok: r3 ON == OFF byte identity` hold | NO, finding 2 |
| `tools/kv-host-spill-failure-gate.sh`, plain | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`) | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)`, the same 15 lines; poolfull, digest and alloc cell tier lines identical with timings stripped (`FAULT: flipped one demoted K byte`, `demote: 89 tokens, 160.5MB`, `VERIFY FAILED: promoted digest aaec09b2... != demote digest 7674c8a1...`, `TIER DISABLED: pinned host alloc of 96832 B failed`) | yes; the one FAIL is finding 3, in both arms |
| `tools/kv-host-spill-failure-gate.sh`, spec | `1 FAILURE(S)` (the same pool-full line) | `6 FAILURE(S)` | NO, finding 2 |
| lane B `tools/prefix-evict-reclaim-gate.py` (`50e91d23...` copy, `tools.sha256`) | `PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=0 driver_free_delta_bytes=none trim_released_bytes=none pool_retained_bytes=none p2=admit-same-tick busy_overlap_s=21.654 identity=aa6cc3291b981646 V1=FAIL V2=ok V3=FAIL V4=ok -> FAIL` | the identical line, `identity=aa6cc3291b981646` | yes; red in both arms, finding 4 |

Collector: every cell `capture-integrity`, `qualification: false`, `gpu_power_limits` 600.00 W
(`validate.log`, `--validate` per cell; `failed_commands: 1` is the collector's word for a red
gate's nonzero exit, asserted per gate above).

### Findings

1. **Entry size on this artifact.** Every prefix entry is about 160 MB regardless of length
   (89 tokens 159.5 MB, 256 tokens 164.5 MB): the recurrent conv/ssm state of the hybrid trunk
   dominates, the KV rows add 29,696 B/token. The gates' default `MEMRA_HOSTGATE_CACHE_MB=1024`
   holds six entries and never evicts; 128 (attempt 1, `attempt1-cache128/`) holds none, every
   host gate cell failed its feed assertion in both arms, and the identity gate printed
   `FAIL: no device eviction fired`. 256 holds one and not two, which is the gate's contract.
   A gate input, not a gate change.
2. **Spec serving is outside lane B's surface.** With the gate's default environment this MTP
   artifact serves speculatively and every prefix insert is a `spec-boundary` capture with a
   draft plane. The OFF arm demotes and promotes them (`ALL GREEN`); under the door each demote
   is refused by name: `[prefix-host] demote refused (contracts door): entry carries TP, latent
   or draft planes outside the contract-routed surface (89 tokens, model gate)` (2 lines in the
   identity ON arm, 4 across the failure ON arm's cells), so the feed assertions fail and the
   byte-identity assertions still hold. Ruling 15's equal-count requirement is met on the surface
   lane B qualified (plain KV plus recurrent continuation, `bind_tier_image`), measured with
   `MEMRA_SERVE_SPEC=0`. Whether the surface grows to draft planes (a `Role` for the MTP scratch
   rows and boundary hidden) is lane B's and the lead's call, before Option B or with it.
3. **Failure gate pool-full cell is red on main, both arms.** With `MEMRA_KV_HOST_MB=1` the tenant
   share cap (`MEMRA_KV_HOST_TENANT_PCT`, default 50, lane/kv-tenancy-compaction) evaporates the
   demote before the D2H: `[prefix-host] demote evaporated at the tenant share cap before the D2H
   copy: 89 tokens, 160.5MB (50% of 1MB, ...)`; the gate asserts the older `skip demote: entry
   ... > host budget` line. Identical in OFF and ON; not a door effect; the gate's assertion is
   stale relative to main and is not touched by this lane.
4. **Lane B's reclaim gate is red on this tree, both arms.** V1/V3 fail because lane B's fix is
   not on this branch (the gate's own docstring: red on main). The verdict line and the V4
   identity digest `aa6cc3291b981646` are byte-equal across arms, which is what ruling 15 asks of
   this gate here.
5. **The door must be silent without a tier.** Attempt 1 ran the identity gate ON arm with the
   door announcing itself under `[prefix-host]` in the gate's OFF twin boot (`MEMRA_KV_HOST_MB=0`)
   and the gate printed `FAIL: OFF boot never touches the tier (no [prefix-host] line at all)`.
   `b81881dad`: no tier, no construction, one line under `[kv-host-contracts]`. Pass 2 is green
   on that assertion in both arms.

### On the empty namespace (deviation from the task's test list, surfaced)

The task asked the `tenant_salt` helper to refuse an empty string. A server without a keyring
keys every request's pool on `""` (`lib.rs tenant_namespace`, `auth.rs meter_key`: `""` is the
default single-tenant namespace), which is how every gate above boots. A refusing helper would
have failed every ON-arm demote with `tier program identity missing` and broken the equal-count
requirement. The helper derives a salt for `""`; the test
`empty_namespace_is_the_default_single_tenant_namespace_not_a_refusal` pins that it is distinct
from every keyring namespace. Flipping it to a refusal needs a lead ruling and a keyring on the
gates.

## Local battery (this rig, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`)

On `b81881dad`: `cargo fmt --all -- --check` clean; `cargo test -p memra-server -p memra-kv
--offline` 739 + 67 + 2 passed, 0 failed, 6 ignored (GPU); the ten new tests by name
(`kv_host_contracts_door_parse_is_strict_and_never_falls_back_to_off`,
`host_tier_arena_refusal_names_the_arena_and_passes_without_it`,
`host_tier_program_base_is_a_pure_function_of_its_sources`,
`host_tier_context_program_stamps_the_pool_namespace_salt_once`,
`host_tier_governor_ledger_admits_what_the_lru_would_and_binds_at_twice_the_budget`,
`host_cache_with_contracts_door_refuses_an_unbound_image_and_admits_it_with_the_door_off`,
`tenant_salt_is_one_derivation_of_the_namespace_string`,
`empty_namespace_is_the_default_single_tenant_namespace_not_a_refusal`,
`shared_governor_is_the_injected_trait_object_and_charges_through_it`) all `ok`; `cargo clippy
-p memra-server -p memra-kv --offline --all-targets -- -D warnings` clean; `tools/check-flags.sh`
867 runtime names, none uncovered; `git diff --check` clean; `tools/docs-registry-census.sh`
clean (58 tables, 902 rows). No `DOCS_RS=1` command ran in this target dir today.

## What remains for Option B (ruling 14, after these receipts)

D2H demotion through `TransferEngine` on the pageable path: move each `CudaSlice<u8>` plane out
of `&mut PrefixEntry` as an owned `KvPlane` into `CudaTransfers::register_device` (no
borrowed-source seam), `alloc_host`, `record_producer`, `d2h(CopyOp)` with `Epochs`, `synchronize`,
`take_destination`, `retire_source`, `release_device_observed`, `release_producer`, `take_plane`
back (the `kv_tier_gate/active.rs:165-219` sequence); `HostPlane` gains a `CudaPinnedLease` arm;
the `StateBundle` checksums move onto `SegmentExpectation` so `Completion::require` proves the
bytes; the governor becomes binding once eviction-to-fit runs before the charge. Its
admissibility gate is this day's table plus per-segment `Completion` checksums equal to
`StateBundle::verify`. The spec-surface decision (finding 2) and the arena lease handoff stay
outside B.

Effort: approximately 2.5 agent-hours (budget 8).
