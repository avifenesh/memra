# HostPrefix contracts door: `MEMRA_KV_HOST_CONTRACTS` (lead ruling 15, Option A)

Status: landed on `lane/spill-c-20260919` day 13 (`ff46abc75`, `e7e23dcf4`, `b81881dad`), default OFF, env door
with a `docs/FLAGS.md` row; surface grown to MTP draft-bearing entries on day 14 (lead ruling 16, `DAY14.md`). Decide-by: **2026-10-05** (14 days after landing, 2026-09-21). Every cell behind it is
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

Surface after day 14 (lead ruling 16): lane B's first slice (plain KV planes, `q8_0` K rows of
34 B and `q5_1` V rows of 24 B, `len == pos`, plus recurrent continuation planes, logits, hidden
and the shape metadata) AND MTP draft-bearing entries. Two entry classes (`HostTierEntryClass`,
pure `host_tier_entry_class(glm, mtp_draft, dflash_tail)`), one program each:

| Class | Planes bound (`bind_tier_image`) | Program (`HostTierContext::program(key, class)`) |
|---|---|---|
| `Plain` | `Role::Key`/`Value` per layer (`q8_0`/`q5_1`), `Recurrent` (conv, ssm), `Logits`, `Hidden` (empty on plain-published entries, so no segment), `Transaction` = `host-prefix-shape-v2` | the day-13 identity (`host_tier_program_base`), tenant salt per pool key |
| `MtpDraft` (`entry.draft.is_some()`) | the plain segments plus `Role::Draft` K (`mtp-draft-q8_0`, row bytes `k_tok_bytes`) and `Role::Draft` V (`mtp-draft-q5_1`, `v_tok_bytes`), each checksummed, geometry rule identical to the trunk planes (34/24 multiples, `len == pos`); the shape blob frames draft presence and `(len, k_tok_bytes, v_tok_bytes)` | `host_tier_draft_program`: the plain base with `artifact = digest("artifact-sha256+mtp-draft", framed(trunk hex, draft source))`, `serialized_plan = digest("plan-debug+mtp-draft", framed(plan debug, draft source))`, `numeric = digest("numeric", host_tier_draft_numeric_class())` (`<plain class>+mtp-draft-kv-q8_0-34B-q5_1-24B`); stream, tokenizer, template, adapter, modality, position unchanged; tenant salt per pool key. Draft source: `embedded` (head in the trunk GGUF), or `external:<sha256>` for the per-model `+draft` attach or `MEMRA_MTP_DRAFT` (precedence: per-model, then env, then embedded, the loader's own) |

The class is computed from the same fields at every site (demote refusal and charge, `bind_tier_image`,
insert lease, promote lease and the post-H2D `require`), so one entry names one program. The
`pinned` charge at demote sums the trunk KV planes and the draft plane (both are `HostPlane`s).

Refused by name, typed, nothing skipped silently:

- GLM state (`tp` shards, `latent` planes, `glm` image): `[prefix-host] demote refused (contracts
  door): entry carries TP or latent (GLM) planes outside the contract-routed surface`; also at
  insert (`REFUSED demote insert (contracts door)`) and promote (`promote refused (contracts door)`).
  Unchanged rule; the arena path is out of scope for A and B.
- DFlash draft tail (`dspark_draft`): `... entry carries a DFlash draft tail outside the
  contract-routed surface (no drafter artifact identity in this slice)`. Its slice needs a byte
  manifest of the `MEMRA_DSPARK_DRAFT` export directory as the artifact, `DflashCfg` as the plan, an
  f32 tail numeric class, `Role::Tail` segments per draft layer, and a gate that boots a DFlash
  drafter on the target card (none does today).
- A draft-bearing entry on a model with no MTP head: `tier draft program identity missing: the
  model has no MTP head, so a draft-bearing entry cannot name its program` (never bound to the
  plain program).
- A handoff import (`host_entry_from_owned`, unbound by design, FREEZE B8): refused at insert.
- An unknown model: `tier program identity missing`.

Boot receipt per model: the day-13 program-identity line, then `[prefix-host] contracts door:
model <name> draft program identity source=<embedded|external:hex> numeric=<draft class> (<ms>)`,
or `... has no MTP head: no draft program, draft-bearing entries are refused by name`.

## Draft planes: what a spec-served entry carries (day 14 census, lead ruling 16, before code)

Tree `fbe6c1635` (lane merge of main `30e433c4c`); every `file:line` is in that tree, `worker.rs`
is `crates/memra-server/src/worker.rs`. Under the gates' default environment the MTP artifact
serves speculatively and every prefix insert is `prefix_insert_from_spec_boundary`
(`worker.rs:11609`, publisher `16875-16879`, `why = "spec-boundary"`). Beyond the plain surface
(`kv`, `conv`, `ssm`, `last_logits`, shape) such an entry carries:

| Plane | Device field | Host twin | Byte layout owner | Published by | D2H / H2D program (unchanged by this slice) | `MEMRA_KV_HOST_VERIFY` digest covers it | Identity field that must cover it |
|---|---|---|---|---|---|---|---|
| MTP draft-scratch K/V rows `[0..pos)` | `PrefixEntry.draft: Option<PrefixPlane>` (`worker.rs:6467`) | `HostPrefixEntry.draft: Option<HostPlane>` (`7757`) | `memra_kv::KvLayer` (`crates/memra-kv/src/lib.rs:470`: `k` q8_0 packed, 34 B per 32 elements; `v` q5_1 packed, 24 B per 32); row bytes from `mtp_scratch_layout` (`crates/memra-engine/src/spec.rs:2266-2285`): `(kv_dim/32) * kv_blk_bytes()` with `n_head_kv` taken from the draft geom when the head is a narrower student (`DraftGeom`, `hybrid.rs:3490`) | `SpecSession::draft_plane_ref` (`spec.rs:1582-1592`; returns `None` for a ring-backed step35 scratch), sliced by `copy_u8_into` at `worker.rs:11753-11779` into a `PrefixPlane { len: pos, k_tok_bytes, v_tok_bytes }` | `host_plane_from_device` (`8598-8601`, the same `dtoh_u8_into_pinned` as the trunk planes); `plane_up` (`9145-9148`, the same `htod_u8_into`) | NO: `prefix_entry_state_digest` (`11178-11262`) hashes `kv`, `conv`/`ssm` and `latent` per layer and nothing else | `artifact` (the MTP head's bytes: embedded in the trunk GGUF when `nextn_predict_layers > 0`, or an external GGUF through the per-model `+draft` attach `MtpHead::load_draft` at `13745` / `models[i].2`, or the global `MEMRA_MTP_DRAFT` at `hybrid.rs:5048-5054`); `serialized_plan` (`plan.mtp_blocks`, `model_plan.rs:33`; a `+draft` plan is attached at `13759` by `attach_external_draft` `model_plan.rs:1015`, a `MEMRA_MTP_DRAFT` head is NOT attached to the plan, so the artifact digest of the draft file is the only thing that names it); `numeric` (the draft rows' q8_0/q5_1 block encodings under `PREFIX_ENTRY_LAYOUT_VERSION`) |
| Boundary trunk hidden `last_h` (row `pos-1`, pre-output_norm) | `PrefixEntry.last_h: Vec<f32>` (`6484`) | `HostPrefixEntry.last_h: HostF32` (`7759`) | `n_embd` f32, `capture_boundary_hidden` (`spec.rs:1732`) | `SpecBoundaryCapture.last_h` (`spec.rs:1719`); empty on plain-published entries | `HostF32::from_slice` / `to_vec` | NO | already bound as `Role::Hidden` (`bind_tier_image`); a plain entry has no Hidden segment because `add` skips empty bytes, so the layout already differs by class |
| Boundary logits | `last_logits` (`6465`) | `last_logits: HostF32` (`7756`) | `n_vocab` f32 | both publishers | `HostF32` | NO | already bound as `Role::Logits` |
| DFlash (DSPARK) draft KV tail | `PrefixEntry.dspark_draft: Option<DflashKvTail>` (`6480`) | `HostPrefixEntry.dspark_draft: Option<HostDflashTail>` (`7758`, struct `7710`) | `DflashKvTail` (`crates/memra-engine/src/dflash.rs:4779-4795`): per draft layer an f32 `(k, v)` pair plus `base`, `rows`, `len`, `row_bytes`, `floor` | `drain_dspark_prefix_capture` (`worker.rs:13050`), only under `MEMRA_DSPARK_SPEC=1` with `MEMRA_DSPARK_DRAFT=<export dir>` (`14146-14171`, `DflashDraft::load` `dflash.rs:1531`) | `HostF32::down` per layer (`8602-8620`); `engine.htod` per layer (`9149-9166`) | NO | would need the drafter's artifact identity (a byte manifest of the export directory: `config.json` plus safetensors), `DflashCfg` (`dflash.rs:17-44`) as the plan, and an f32 numeric class; none of that is derivable from a GGUF file digest, and no gate on the target card boots a DFlash drafter. NOT bound in this slice: refused by name |
| GLM state (`glm`, `tp`, `latent`) | `latent` (`6454`), `tp` (`6472`) | `glm: Option<HostGlmState>` (`7748`) | `host_glm` | glm5 publishers (`13068`) | `host_glm` | `host_glm::digest` | unchanged: refused by lane B's rule and by the arena boot refusal (the arena path is out of scope for A and B) |

Consumer of the draft plane: a spec restore (`worker.rs:21305-21372`) requires `entry.draft.is_some()`
(`spec_restore_refusal`, `336-350`: "entry carries no draft plane (plain-published)") and hands
`&draft.k, &draft.v, k_tok_bytes, v_tok_bytes, len, &entry.last_h` to
`spec_session_from_restored_deferred` (`spec.rs:9147`), which re-arms the MTP head's scratch from
those bytes. So a plane written by one MTP head and read by another is exactly the byte
reinterpretation the identity must forbid, and a plain-published entry and a spec-published entry
of the same prompt are two different serving programs (one restores a `SpecSession`, the other
serves plain) that must never share a `ProgramIdentity`.

Findings that shape the slice:

1. **Encodings.** The draft rows use the trunk's block encodings (`KvLayer`, `kv_blk_bytes()`),
   so the geometry rule `bind_tier_image` applies to trunk planes (34 B and 24 B row multiples,
   `len == pos`) applies unchanged to the draft plane; only its row width differs (the MTP head's
   `n_head_kv`).
2. **The verify arm is blind to the draft plane.** `MEMRA_KV_HOST_VERIFY=1`'s `verify ok` attests
   trunk planes only (`11178-11262`); under the door the contract checksums this slice adds are the
   only byte attestation of the draft plane. Extending the digest (`memra-prefix-split-state-v3`)
   would change the digest strings the OFF arm prints in `VERIFY FAILED` lines, so it is not part
   of this slice; recorded here as the follow-up it is.
3. **Chain width and trim do not enter the plane.** With `MEMRA_MTP_HEADS > 1` only head 0's
   scratch plane is published (`draft_plane_ref` returns `scratch.kv`; `mtp_extra`,
   `hybrid.rs:3884`, is not part of the entry). `MEMRA_FRSPEC_TRIM` changes the draft lm_head rows
   (`d2t`), not the draft K/V bytes.
4. **Draft source is knowable at boot for GGUF heads only.** Embedded (`lm.model.mtp.is_some()`
   with no external path), per-model `+draft` (`models[i].2`, hashed like the trunk file), or
   `MEMRA_MTP_DRAFT` (hashed the same way). A model with no MTP head gets no draft program, and a
   draft-bearing entry for it is refused by name rather than bound to the plain program.
5. **Charge split.** At demote the door charges `pinned` for `kv` planes only (`8834-8850`); the
   draft plane is a pinned `HostPlane` too and must join the pinned sum (a tier-Some statement, OFF
   untouched).

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
