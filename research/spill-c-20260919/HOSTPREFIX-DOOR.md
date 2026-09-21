# HostPrefix contracts door: `MEMRA_KV_HOST_CONTRACTS` (lead ruling 15, Option A)

Status: landed on `lane/spill-c-20260919` day 13 (`ff46abc75`, `e7e23dcf4`, `b81881dad`), default OFF, env door
with a `docs/FLAGS.md` row. Decide-by: **2026-10-05** (14 days after landing, 2026-09-21). Every cell behind it is
`executed-not-qualified` development evidence on one card class; nothing here is a support
state. Rulings applied: 13 (one `tenant_salt` owner in `memra-kv`), 14 (no borrowed-source
seam, no v1.4; not exercised by A, which moves no bytes through `TransferEngine`), 15 (Option
A first, OFF byte-identical by construction, ON leaves the gate lines unchanged, the startup
arena is out of scope). Census that motivated it: `HOSTPREFIX-CONTRACT-CENSUS.md` Part E.

## What the door does

`MEMRA_KV_HOST_CONTRACTS=1` constructs `HostTierContext` (`crates/memra-server/src/worker.rs`,
struct at 8102) once at boot, after every model has loaded and `model_generations` exists, and
stores it in `HostPrefixCache::tier`. That field existed since lane B's day-two sidecar with no
constructor in the crate (census, Part B headline), so the sidecar route (`tier_charge` 8127,
`bind_tier_image` 8176, the insert identity check 7975, the promote leases 9139) was compiled
and unit-tested but never executed. With the door ON it executes; the D2H and H2D programs are
the pre-door `dtoh_u8_into_pinned` / `htod_u8_into` calls, untouched.

| Surface | Where | What it does |
|---|---|---|
| Door read | `worker.rs:4227` `kv_host_contracts_door`, `4239` `parse_kv_host_contracts` | Strict: unset or `0` OFF, `1` ON, anything else (bare `=`, `on`, `true`, `11`, non-UTF-8) refuses the boot with a message naming the variable. No fallback to OFF: a door whose OFF arm must be byte-identical cannot file a typo under the ON arm. |
| Arena refusal | `worker.rs:4255` `host_tier_arena_refusal`; checked at `14374` before the arena reserve and again in `host_tier_context` | `MEMRA_GLM5_TP_KV_HOST=1` with the door refuses at boot: the arena is charged by its owner and `tier_charge` would refuse every demote with `tier fixed-arena lease handoff pending` (8141). The refusal names the arena and both variables; nothing is pinned first. |
| Program identity | `worker.rs:4308` `host_tier_program_base` (pure), `4376` `host_tier_context` (sources) | One `ProgramIdentity` per loaded GGUF model with a zero tenant salt; field sources in the table below. A checkpoint-directory model or a loaded vision tower refuses the boot (typed). |
| Tenant salt | `crates/memra-kv/src/tiered/hostprefix.rs:129` `tenant_salt`; stamped by `HostTierContext::program` (`worker.rs:8116`) | The ONE derivation (ruling 13): `digest("tenant-salt", namespace)` over the exact `PoolKey.1` string the server feeds `auth::meter_key` for the share-cap row (`worker.rs` insert 7975 and remove paths). `programs` is keyed by model name; the pool namespace completes the identity per key, deterministically, so the demote charge and the insert or promote lease name one program. |
| Governor | `hostprefix.rs:137` `shared_governor`; `worker.rs:4342` `host_tier_governor` | The server's governor (ruling 15), one `Arc<Mutex<dyn BudgetGovernor>>` built at boot over `memra_tier::tier::Governor`. Ledger sizing for A: twice the host budget on `pinned` and `pageable`, twice the device prefix budget on `device[ordinal]`, zero headroom, no queue. Why twice: residents are at most one budget (the host LRU invariant), one incoming image is at most one budget (`insert` refuses larger), and the charge is taken before the LRU makes room, so the ledger must never be what refuses a demote the OFF arm would have made (equal `[prefix-host] demote:` counts are the ruling 15 gate). The ledger records tenant, priority and bytes; the LRU decides. Option B may tighten it once eviction-to-fit runs before the charge. |
| Generation | `worker.rs:8403` (`host_entry_from_device`) | Under the door every host image carries the model generation `Arc` (`bind_tier_image` refuses an image without one, 8189). OFF arm unchanged: the same `None`. This is the one OFF-path statement the door touches, an added `else if` arm. |
| Over-budget image | `worker.rs:8734` (demote), `7975` (insert order) | An image above the whole host budget takes no ledger charge and `insert` refuses it by name after the copy, exactly as the OFF arm (`skip demote: entry N MB > host budget M MB`, the failure gate's pool-full cell asserts that line); the door's identity check in `insert` runs after that refusal and before residency. Commit `e7e23dcf4`. |
| Loud refusals | `worker.rs:8726` demote surface, `7979`/`7989` insert, `9163`/`9232` promote | Lane B's tier blocks returned `Failed`/`false`/`None` silently; each now prints a `(contracts door)` line naming the refusal. All inside `tier`-Some blocks. |
| Boot wiring | `worker.rs:14374` parse and arena check, `14419` construct and announce (`b81881dad`: with `MEMRA_KV_HOST_MB=0` no tier exists, so the door constructs nothing and prints one `[kv-host-contracts] ... nothing to route` line under its own tag; the identity gate's OFF twin boot asserts `[prefix-host]` stays silent) | `[prefix-host] contracts door: model <name> program identity artifact_sha256=<hex> plan_debug_sha256=<hex> numeric=<class> template=<gguf-jinja or chatml-fallback> device=<n> (<ms> to hash <path>)` per model, then `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): ...`. |

## `ProgramIdentity` fields and their sources (`host_tier_program_base`, `host_tier_context`)

| Field | Source | Domain string |
|---|---|---|
| `version` | `WIRE_VERSION` (1) | |
| `artifact` | streaming SHA-256 of the GGUF file the model loaded from (`sha256_file_hex`, 8 MiB reads, never the whole file in memory; ~15 GB in a few seconds on the target card's NVMe) | `artifact-sha256` over the hex string, the `kv_tier_gate` convention |
| `serialized_plan` | `format!("{:?}", HybridModel.plan)`: the compiled `ModelPlan` in its Debug form, the same convention as `kv_tier_gate` (`plan-debug`) | `plan-debug` |
| `numeric` | `host_tier_numeric_class()`: `server-prefix-entry-v<PREFIX_ENTRY_LAYOUT_VERSION>-kv-q8_0-<k>B-q5_1-<v>B` from the in-process prefix-entry layout version (5) and `memra_kv::kv_blk_bytes()` (34, 24) | `numeric` |
| `stream` | the single worker owner thread that runs every copy and kernel | `stream` over `server-worker-single-owner-thread` |
| `tokenizer` | the artifact hex (a GGUF tokenizer is parsed from the artifact; `kv_tier_gate` convention) | `artifact-tokenizer` |
| `template` | `Tokenizer::chat_template()`: the GGUF `tokenizer.chat_template` text, or the ChatML fallback name when the GGUF has none | `template-jinja` over the text, or `template` over `chatml-fallback` |
| `adapter` | none (the server serves no adapter) | `adapter` over `none` |
| `modality` | text (a loaded vision tower refuses the door) | `modality` over `text` |
| `position` | immutable prefixes start at token zero (`IdentitySlot::bind` enforces `id.start == 0`) | `position` over `prefix-from-token-zero` |
| `tenant_salt` | `memra_kv::tiered::hostprefix::tenant_salt(&pool_key.1)` per pool key | `tenant-salt` over the namespace string |

Not derived, and refused rather than invented: a directory checkpoint's artifact digest (this
slice is GGUF only), a modality for image-conditioned entries (vision tower loaded).

## What the ON arm routes, and what it refuses by name

Lane B's first-slice surface (`bind_tier_image`): plain KV planes (`q8_0` K rows of 34 B,
`q5_1` V rows of 24 B, `len == pos`) plus recurrent continuation planes, logits, hidden and
the shape metadata. Under the door:

- an entry with TP shards, latent planes or a draft plane (spec-published boundary entries)
  is refused at demote, `[prefix-host] demote refused (contracts door): entry carries TP,
  latent or draft planes outside the contract-routed surface`; the OFF arm demotes it. So with
  spec serving ON, demote counts differ between arms by exactly those entries; the ruling 15
  equal-count gate is measured with `MEMRA_SERVE_SPEC=0` where the gate does not set it;
- a handoff import (`host_entry_from_owned`, unbound by design, FREEZE B8) is refused at
  insert, `[prefix-host] REFUSED demote insert (contracts door)`;
- a GLM (`HostGlmState`) image is refused by `bind_tier_image` (unchanged lane B rule).

## Tests

`crates/memra-kv/src/tiered/hostprefix.rs` `tests`:
`tenant_salt_is_one_derivation_of_the_namespace_string`,
`empty_namespace_is_the_default_single_tenant_namespace_not_a_refusal`,
`shared_governor_is_the_injected_trait_object_and_charges_through_it`.
`crates/memra-server/src/worker.rs` `tests`:
`kv_host_contracts_door_parse_is_strict_and_never_falls_back_to_off` (bare, junk, doubled
values, non-UTF-8), `host_tier_arena_refusal_names_the_arena_and_passes_without_it`,
`host_tier_program_base_is_a_pure_function_of_its_sources`,
`host_tier_context_program_stamps_the_pool_namespace_salt_once`,
`host_tier_governor_ledger_admits_what_the_lru_would_and_binds_at_twice_the_budget`,
`host_cache_with_contracts_door_refuses_an_unbound_image_and_admits_it_with_the_door_off`.

On the empty namespace: the task list asked for "empty refuses". A server without a keyring
keys every request's pool on `""` (`lib.rs tenant_namespace`, `auth.rs meter_key`: `""` is the
default single-tenant namespace), which is how every gate in ruling 15 boots. A refusing helper
would fail every ON-arm demote under those gates and break the equal-count requirement, so the
helper derives a salt for `""` and the test pins that it is distinct from every keyring
namespace. Flipping it to a refusal is a lead decision that also requires a keyring on the
gates.

## Target-card receipts

`DAY13.md`, `pro-single-day13/`. OFF and ON on the same binary, same prompts, N=1,
`executed-not-qualified`, one RTX PRO 6000 Blackwell at its 600 W limit, through the
canonical collector (`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`).

## Removal rule

A negative, flat or superseded receipt deletes the door in the same PR: the env read, the
boot wiring, `host_tier_context` and its helpers, the FLAGS.md row, this doc's status, and
the tests; the verdict moves to the "Removed doors" ledger. The sidecar route it exercises
(lane B's) is the lead's to keep or drop. If Option B lands and the door has served two weeks
with no rollback, the seam is deleted and the constructor becomes the naked default.
