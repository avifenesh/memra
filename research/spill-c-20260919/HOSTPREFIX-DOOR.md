# HostPrefix contracts door: `MEMRA_KV_HOST_CONTRACTS` (lead ruling 15, Options A, B and C)

Status: landed on `lane/spill-c-20260919` day 13 (`ff46abc75`, `e7e23dcf4`, `b81881dad`), default OFF, env door
with a `docs/FLAGS.md` row; surface grown to MTP draft-bearing entries on day 14 (lead ruling 16, `DAY14.md`);
Option B landed on day 15 (`5f8d327e2`, `DAY15.md`): under the door the pageable-tier D2H of every KV plane
goes through the native `TransferEngine`, the sequence in the "Option B" section below; Option C landed
on day 16 (`25891baa9`, `DAY16.md`): under the door the promote H2D of every contract-routed KV plane goes
through the same engine, the sequence in the "Option C" section below. What remains before the decide-by:
the arena path (`MEMRA_GLM5_TP_KV_HOST`, refused with the door at boot; its fixed backing is not
governor-charged and its slices are not leases), the DFlash tail slice (no drafter artifact identity
derivable from a GGUF digest; no gate boots a DFlash drafter on the card), and the door's cost on the
target card with CACHED destinations. The write-combined destination decision itself is CLOSED BY POINTER
(day 17): lane A decided it in `docs/decisions/PINNED-DESTINATIONS.md` (lead ruling 23, `PinnedKind::
for_device` in `tier_transfer.rs`: `Cached` on the RTX PRO 6000 Blackwell class, `WriteCombined` on the RTX
5090 class and every unrecognized name, resolved once in `CudaTransfers::new`, no environment variable),
so this lane owns no allocation-flag item and adds no engine flag; the day-16 WC pair
(`WC-DESTINATIONS.md`) measured write-combined destinations and is superseded on that card class. What
the decide-by review still owes is the door's cost read again on the target card with cached
destinations: the same pair cell (demote and promote lines OFF against ON, N=5 per arm per order, both
orders, one lock hold) on a binary carrying `for_device`, so the two SHA-256 passes at demote and the one
at promote run over cacheable memory, and the hash-speed micro-cell that splits the remaining delta
between the hashes and the ticket lifecycle; both are banked (lane A day 15 and this lane's day 18) and listed with every other owed cell in the "Review table for the decide-by" section at the end of this file. Day 17's arena cell (`DAY17.md`) is a separate item and
measured the arena against the pageable tier with the door OFF in both arms. Decide-by: **2026-10-05**
(14 days after landing, 2026-09-21). Every cell behind it is `executed-not-qualified` development
evidence on one card class;
nothing here is a support state. Rulings applied: 13 (one `tenant_salt` owner in `memra-kv`), 14
(planes leave `PrefixEntry` as owned `KvPlane`s, no borrowed-source seam in `CudaTransfers`, no v1.4:
exercised by Option B), 15 (Option A first, OFF byte-identical by construction, ON leaves the gate lines
unchanged, the startup arena is out of scope; B after A's receipts, C after B's). Census that motivated
it: `HOSTPREFIX-CONTRACT-CENSUS.md` Part E.

## What the door does

`MEMRA_KV_HOST_CONTRACTS=1` constructs `HostTierContext` (`crates/memra-server/src/worker.rs`,
struct at 8102) once at boot, after every model has loaded and `model_generations` exists, and
stores it in `HostPrefixCache::tier`. That field existed since lane B's day-two sidecar with no
constructor in the crate (census, Part B headline), so the sidecar route (`tier_charge` 8127,
`bind_tier_image` 8176, the insert identity check 7975, the promote leases 9139) was compiled
and unit-tested but never executed. With the door ON it executes. Days 13 and 14 left the D2H and
H2D programs as the pre-door `dtoh_u8_into_pinned` / `htod_u8_into` calls; since day 15 (Option B)
the D2H of every KV plane on the pageable tier is the transfer engine's `memcpy_dtoh` behind a
producer fence on the owner stream, one batch per demote, and the H2D stays `htod_u8_into`.

| Surface | Where | What it does |
|---|---|---|
| Door read | `worker.rs:4227` `kv_host_contracts_door`, `4239` `parse_kv_host_contracts` | Strict: unset or `0` OFF, `1` ON, anything else (bare `=`, `on`, `true`, `11`, non-UTF-8) refuses the boot with a message naming the variable. No fallback to OFF: a door whose OFF arm must be byte-identical cannot file a typo under the ON arm. |
| Arena refusal | `worker.rs:4255` `host_tier_arena_refusal`; checked at `14374` before the arena reserve and again in `host_tier_context` | `MEMRA_GLM5_TP_KV_HOST=1` with the door refuses at boot: the arena is charged by its owner and `tier_charge` would refuse every demote with `tier fixed-arena lease handoff pending` (8141). The refusal names the arena and both variables; nothing is pinned first. |
| Program identity | `worker.rs:4308` `host_tier_program_base` (pure), `4376` `host_tier_context` (sources) | One `ProgramIdentity` per loaded GGUF model with a zero tenant salt; field sources in the table below. A checkpoint-directory model or a loaded vision tower refuses the boot (typed). |
| Tenant salt | `crates/memra-kv/src/tiered/hostprefix.rs:129` `tenant_salt`; stamped by `HostTierContext::program` (`worker.rs:8116`) | The ONE derivation (ruling 13): `digest("tenant-salt", namespace)` over the exact `PoolKey.1` string the server feeds `auth::meter_key` for the share-cap row (`worker.rs` insert 7975 and remove paths). `programs` is keyed by model name; the pool namespace completes the identity per key, deterministically, so the demote charge and the insert or promote lease name one program. |
| Governor | `hostprefix.rs:137` `shared_governor`; `worker.rs:4342` `host_tier_governor` | The server's governor (ruling 15), one `Arc<Mutex<dyn BudgetGovernor>>` built at boot over `memra_tier::tier::Governor`. Ledger sizing for A: twice the host budget on `pinned` and `pageable`, twice the device prefix budget on `device[ordinal]`, zero headroom, no queue. Why twice: residents are at most one budget (the host LRU invariant), one incoming image is at most one budget (`insert` refuses larger), and the charge is taken before the LRU makes room, so the ledger must never be what refuses a demote the OFF arm would have made (equal `[prefix-host] demote:` counts are the ruling 15 gate). The ledger records tenant, priority and bytes; the LRU decides. Option B may tighten it once eviction-to-fit runs before the charge. |
| Generation | `worker.rs:8403` (`host_entry_from_device`) | Under the door every host image carries the model generation `Arc` (`bind_tier_image` refuses an image without one, 8189). OFF arm unchanged: the same `None`. This is the one OFF-path statement the door touches, an added `else if` arm. |
| Over-budget image | `worker.rs:8734` (demote), `7975` (insert order) | An image above the whole host budget takes no ledger charge and `insert` refuses it by name after the copy, exactly as the OFF arm (`skip demote: entry N MB > host budget M MB`; production demotes under the default `MEMRA_KV_HOST_TENANT_PCT=50` refuse earlier, before the copy, at the share cap, and since day 21 the failure gate's pool-full cell asserts whichever of the two the effective cap produces); the door's identity check in `insert` runs after that refusal and before residency. Commit `e7e23dcf4`. |
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

## Option B: the pageable D2H through `TransferEngine` (lead rulings 14 and 15; day 15 census, before code)

Tree `4cc2e92a6` (lane merge of main `70038ed01` and lane A's `8302f6b0a`, whose day-12 tenant-share
reclaim runs after the image is built and bound, before insert); every `file:line` is in that tree.
`worker.rs` is `crates/memra-server/src/worker.rs`, `tier_transfer.rs` is
`crates/memra-engine/src/tier_transfer.rs`, `contracts.rs` is `crates/memra-tier/src/contracts.rs`,
`active.rs` is `crates/memra-engine/src/bin/kv_tier_gate/active.rs`.

### The pageable demote as it is today (door ON, arena absent), in order

| Step | Where (`worker.rs`) | What happens |
|---|---|---|
| 1 | `host_demote_prefix_ref(engine, host, dead: &PrefixEntry)` 9499 | Entry point, BY REFERENCE. Callers: the SLRU sink `host_demote_prefix_entry` 9433 (owns `dead`, passes `&dead` 9434), the admission flush `evict_all_demoting` 9467-9470 (`&px.entries[&key][i]`, then `remove_at`), the handoff export drain 11242 (owns `dead`), the pause sweep 19522 (an owned boundary snapshot) and 19562 (`&px.entries[..][ei]`, a LIVE resident entry the sweep removes only after `Demoted` or `Evaporated`). |
| 2 | 9504-9528 | `armed()`; the GLM refusals (`tp`, `latent`), unchanged. |
| 3 | 9529 | `host_bytes = host_image_bytes(dead.bytes, toks, logits)`. |
| 4 | 9553 (lane A day 12) | `tenant_share_reclaim_plan` BEFORE the copy: a refusal prints the evaporation line and returns `Evaporated`. |
| 5 | 9569-9625 | Door: `host_tier_entry_class` (refuses by name); an over-budget image takes no charge; else `pinned` = every KV plane plus the draft plane (`len * (k_tok_bytes + v_tok_bytes)`), `pageable` = the rest, `tier_charge(key, class, pinned, pageable, false)` 9615 (a `ResidentCharge`, `Priority::Backup`, the pool's tenant salt). |
| 6 | 9627-9636 | `MEMRA_KV_HOST_VERIFY=1`: `host_roundtrip_digest` 9491 = `prefix_entry_state_digest` 11999 over the DEVICE entry (its own reads), recorded before the copy. |
| 7 | 9638 `host_entry_from_device(engine, host, dead, verify_digest)` 9218 | Layout version; `model_generation` (the tier-Some arm); the byte census `sizes` against `dead.bytes`; `reserve_image` 9282 returns `HostPlaneLeases(None)` on the pageable tier; per layer `host_plane_from_device` 9291; the `flip-demote` fault 9298-9306 (after the trunk planes, before conv/ssm); conv/ssm `HostF32::down` (`Heap`, a pageable `Vec<f32>` on this tier); the draft plane 9336 through the same `host_plane_from_device`; the DFlash tail; entry assembly with `_tier_identity` unbound. |
| 8 | `host_plane_from_device(engine, host, p: &PrefixPlane, planes)` 9168 | `alloc(kb)` 9190 and `alloc(vb)` 9191 through `planes.take(n)` = `PinnedHostBuf::new` (`crates/memra-engine/src/pinned_host.rs:186`, `malloc_host`, zero fill); the `alloc-fail` fault and the latch (`host.disable`, `rejected_allocs`) 9177-9188; the pageable COPY PROGRAM: `engine.dtoh_u8_into_pinned(&p.k, &mut k, kb)` 9199 and `(&p.v, &mut v, vb)` 9204, each `stream.memcpy_dtoh(&d.slice(0..n), host)` then `stream.synchronize()` (`crates/memra-engine/src/lib.rs:13556-13557`), synchronous on the owner thread, no ticket, no fence, no checksum. The arena arm 9193 (`copy_from_device_u8`) is unreachable under the door (boot refusal `host_tier_arena_refusal` 4351). |
| 9 | 9641 | `e._tier_charge = tier_charge`. |
| 10 | 9642 `bind_tier_image(&mut e)` 8975 | Class again, program and generation, surface check; per segment `checksum(bytes)` 9048 over the HOST copy: `Role::Key`/`Value` 9094-9095 (`p.k.as_slice()`), `Recurrent`, `Logits`, `Hidden`, `Role::Draft` 9110-9122, `Transaction` 9123; `KvBlockId::new`, the `StateBundle`, the metadata `tier_charge` 9136, `IdentitySlot::bind` 9140. This is the only place the host bytes are hashed. |
| 11 | 9649-9665 (lane A day 12) | `reclaim_tenant_share`: this tenant's own oldest unleased entries leave, after bind, before insert. |
| 12 | 9669 `insert` 8709 | Version and key; the over-budget `skip demote` line; the door identity check (`_tier_charge.program()` and `_tier_identity.lease` 8762); the cap check (unreachable after step 11); twin replacement; residency; `[prefix-host] demote:` 9674. |

Nothing attests the device source bytes except the verify digest (step 6), which is a different hash
program (`memra-prefix-split-state-v2`) from `contracts::checksum`; the contract checksums (step 10)
attest the host image only, after the fact.

### The contract sequence that replaces step 8 (`active.rs:243-305` `demote`, native `CudaTransfers`)

`register_device(plane, src_gen, request)` `tier_transfer.rs:273` (an OWNED `KvPlane`; the plane's
stream must be the owner stream, 284; `device[ordinal]` charged for `physical_bytes`, 286) then
`retain_device` 302, `alloc_host(bytes, request)` 210 (`pinned` charged 222, `alloc_pinned` plus zero
fill), `record_producer(src_gen)` 548 (a CUDA event on the owner stream), `d2h(CopyOp { host, device,
bytes, epochs, producer_fence })` 779 into `submit_batch` 811: `CopyOp::validate` (`contracts.rs:1306`;
a D2H requires the producer fence 1322-1329), the owner stream and context check 664-676, the whole
initialized host range 661, the `inflight` charge 843, the wait on the producer event 913,
`memcpy_dtoh(&backing.slice(..bytes), pinned)` 941-946 on the owner stream, the item event 949 and its
consumer wait 951. Then `synchronize(&ticket)` 640 (event sync per item), `progress` 688 (status
`Complete`, `valid_bytes`, `checksum = checksum(host.bytes())` 722-724: the ENGINE's hash of the
destination, its own receipt, `producer_done`), `take_destination(&ticket, item, epochs)` 1031
(`publishable` 741 runs `Completion::require(.., &expected, true)`; returns `Destination::Host
(CudaPinnedLease)` sharing the allocation `Rc`, marks the ticket published), `retire_source(&ticket)`
388 (producer done and source idle; the D2H items drop their device leases 430), `take_plane(&lease)`
366 (`require_unbound`, `stream.synchronize()`, registry release, device charge released,
`Rc::try_unwrap` into the `KvPlane`), `KvPlane::into_pooled` (`crates/memra-kv/src/plane.rs:242`) back
to the `CudaSlice<u8>`, `record_consumer(&ticket)` 572 (a published ticket retires only against a
consumer fence, 1080-1088), `release_producer` 555, `retire(&ticket, Some(consumer))` 1071 (releases
the `inflight` charge 1130), `acknowledge(&ticket)` 1140 (drops the entry and the engine's half of the
destination `Rc`, leaving the server's lease the sole owner).

### The single point where owned `KvPlane`s leave `PrefixEntry`

`host_entry_from_device`'s plane loop 9288-9294 (`for plane in &dead.kv { .. host_plane_from_device(
engine, host, p, ..) }`) and the draft call 9336, which today borrow `p: &PrefixPlane`. `register_device`
takes `impl Into<KvPlane>` BY VALUE (`From<CudaSlice<u8>>`, `plane.rs:182`), so the K and V slices must be
owned: on an `&mut PrefixEntry`, `dead.kv[i].take()` moves the whole `PrefixPlane` out of its `Option`
slot (no placeholder value and no device byte: `CudaSlice::clone` is a `cuMemcpyDtoD`, the second copy
program rulings 14 and 15 forbid), its `k` and `v` register, and after `take_plane` and `into_pooled`
the SAME two slices go back into a rebuilt `PrefixPlane` in the same slot with the same `len`,
`k_tok_bytes`, `v_tok_bytes`. The draft plane is the same move on `dead.draft`. So
`host_demote_prefix_ref` and `host_entry_from_device` take `&mut PrefixEntry`; every caller owns its
entry or reaches it mutably (`evict_all_demoting` and the pause sweep's live entry through `get_mut`).
OFF passes the same entry through the same statements; only the borrow is exclusive.

The borrow change has one failure shape the by-reference copy did not: once a plane is registered and a
ticket submitted, a quarantined completion (a CUDA error on the owner stream) keeps the source inside the
engine by the frozen rule (`Entry::drop` `tier_transfer.rs:143-152` forgets live inputs; `take_plane`
refuses `Busy`), and the device entry is no longer whole. OFF's `Failed` means "a caller holding live
device state keeps it"; that cannot hold here, so a new typed outcome `SourceQuarantined` tells the one
live-entry caller (the pause sweep 19562) to drop the entry, and the tier latches off (`disable`, the
same posture as a pinned alloc failure). Every other caller drops the entry anyway. Before submission
(a refused `alloc_host`, `register_device` or `submit_batch`) the registered planes are taken back and
the entry is whole: plain `Failed`.

### Host destination type

`take_destination` returns a `CudaPinnedLease` (`tier_transfer.rs:31`), not a `PinnedHostBuf`; turning
one into the other is a second host copy (census Part C item 1, refused). `HostPlane { k, v:
PinnedHostBuf }` 8049 gains a byte-owner enum with a lease arm. Readers: `bind_tier_image` 9094-9095,
9114, 9120 (`as_slice`); the flip fault 9301 (`as_mut_slice`; a lease has only `write` (74), so under
the door the fault reads, flips and rewrites the plane: a diagnostic, gate-box only); the promote
`plane_up` 9886-9903 (`p.k.as_slice()[..kb]` into `htod_u8_into`: a `&[u8]`, so the promote PROGRAM is
untouched, Option C later); the handoff export 10611-10612 (`as_slice`); the handoff import
constructor 11084 stays on the `PinnedHostBuf` arm.

### Governor: one ledger, two handles

`CudaTransfers::new(owner, governor: SharedBudget)` 173 takes `Rc<RefCell<dyn BudgetGovernor>>`
(`crates/memra-tier/src/bank/types.rs:9`); the server's governor is `Arc<Mutex<dyn BudgetGovernor +
Send>>` (`hostprefix.rs:126`). A `BudgetGovernor` adapter that locks the server's mutex per call, held in
an `Rc<RefCell<..>>` for the transfer engine, keeps ONE ledger (ruling 15): every `CudaPinnedLease`
and every ticket releases through it. Charges the route takes: `pinned` per plane at `alloc_host`, so
the demote's `ResidentCharge` 9615 charges `pinned = 0` under Option B (the leases carry the plane
bytes; the same total, so the twice-the-budget argument in the Governor row above is unchanged);
`device[ordinal]` per registered plane, released at `take_plane` (residents promoted under the door
plus one entry's planes stay within twice the device prefix budget); `inflight` = ops per batch, a
dimension the day-13 ledger sized at ZERO (`host_tier_governor` 4580-4600 sets pinned, pageable and
device only), so `submit_batch` 843 would refuse `Capacity` on the first demote: the ledger gains
`inflight = 2 x max layers + 2` over the loaded models (one batch per demote on the single owner, one
K and one V op per KV plane plus the draft pair).

### Epochs

`Epochs { state: 0, src_gen: 1, dst_gen: 1 }`: state 0 is the immutable-prefix epoch (`KvBlockId.epoch`
0, `StateBundle::validate` `contracts.rs:445-447`); the source and destination generations are the
single worker owner's, constant for the process (the model INSTANCE identity is enforced by
`bind_tier_image`'s `Arc::ptr_eq` generation and the identity lease, not by this number).
`register_device(.., 1, ..)`, `record_producer(1)`, `CopyOp.epochs` and `take_destination(.., epochs)`
all name it; the gate uses `1/1/1` (`active.rs:11-15`).

### Checksums and the receipt

Three hashes over the same host bytes per plane under ON: the engine's at completion (`progress` 722,
the contract's own receipt), the server's at take (the `SegmentExpectation.checksum` handed to
`Completion::require(&ticket, &expected, false)` `contracts.rs:602`: two independent parties agree
before the plane is used), and `bind_tier_image`'s 9048 (the `StateBundle` checksum, code unchanged).
The bound image's `Key`/`Value`/`Draft` checksums must equal the transfer's per-item checksums; a
mismatch is a typed refusal, EXCEPT under `MEMRA_KV_HOST_FAULT=flip-demote`, which corrupts the image
after the receipt exactly as it corrupts after the verify digest today (9298: after the copy, before
bind), so the failure gate's digest cell keeps its OFF shape (`FAULT`, `demote:`, `VERIFY FAILED` at
promote) and the receipt names the injected difference. `alloc-fail` fires at the first plane's pinned
request, before any plane leaves the entry, with the same latch line.

Receipt line per demote (ON only, printed before `demote:`): `[prefix-host] contracts door D2H
receipt: ticket issuer=<n> seq=<n> epochs=0/1/1 items=<n> (<k> KV planes[, draft]) complete=<n>
require=ok checksums_sha256=<hex over the ordered item checksums> retired acknowledged`.

### Receipts and findings (day 15, `DAY15.md`)

Landed as designed at `5f8d327e2`; `pro-single-day15/`, replay `verify-day15.py` `DAY15 REPLAY: PASS`
(134 checks): identity gate `ALL GREEN` OFF and ON under the default and plain environments with equal
demote bytes, `verify ok`, equal event sequences and one receipt before every ON demote (`items=34 (16 KV
planes, draft)` with the draft plane, `items=32` without, `complete = items`, `require=ok`); failure gate
`1 FAILURE(S)` both arms (the pre-existing pool-full line), the digest cell's receipt preceding the flip
and the bind naming the injected difference before `VERIFY FAILED` at promote; lane A's tenant reclaim
fix arm `PASS` both arms with eight receipts; serve-smoke and lane B's two gates line-identical. Finding
to weigh at decide-by: the contract's destinations are write-combined (`cudarc alloc_pinned` =
`CU_MEMHOSTALLOC_WRITECOMBINED`, `PinnedHostBuf::new` is cached), so the bind hash over them runs at WC
speed (N=1: demote 149 vs 281 ms, promote 267 vs 398 ms OFF vs ON; not a claim). Option C (the promote
H2D) is next.

### Review fixes (PR #599, findings 1 and 2; `DAY15.md` review section)

Two unwind defects, both real: before submission the ops' original `DeviceLease` handles held the
registry `Rc` above the count `take_plane` accepts, so every pre-submit refusal escalated to
`SourceQuarantined`, latched the tier and dropped a whole device entry over intact planes (fixed: the
originals are dropped before any pre-submit unwind); and the abort recorded the consumer fence and
retired against it without observing it, discarding the result, which leaked the ticket, its
in-flight charge (the whole dimension) and its destinations, so every later demote would refuse
`Capacity` (fixed: drain, then retire; nothing discarded; a refusal there is the typed `TicketLeaked`
outcome and one `TIER DISABLED` line). Both paths are injectable one-shot faults,
`MEMRA_KV_HOST_FAULT=contract-presubmit` and `contract-postpublish`, exercised by the GPU unit cells
and `tools/kv-host-contract-fault-gate.sh`.

### What stays as it is

OFF: every statement outside the door's `tier`-Some arms; the borrow becomes exclusive with no
statement change. The promote path `device_entry_from_host` 9873 (Option C). The arena path (boot
refusal). The recurrent, logits and hidden f32 planes: they are `CudaSlice<f32>` and `Vec<f32>`, not
`KvPlane`s (`From<CudaSlice<u8>>` only), so they keep `HostF32::down` (`Heap` on this tier) rather
than being reinterpreted as u8 planes; they are not pinned on the pageable tier today and take no
`pinned` charge. `dtoh_u8_into_pinned` stays the OFF program for every KV plane and the arena arm's
neighbour; under ON no KV plane reaches it.

## Option C: the promote H2D through `TransferEngine` (lead ruling 15, after B's receipts; day 16 census, before code)

Tree `48ec3f0c5` (lane merge of main `1b354be59`, PR #599 integ17 plus #601); every `file:line` is in that
tree. `worker.rs` is `crates/memra-server/src/worker.rs`, `tier_transfer.rs` is
`crates/memra-engine/src/tier_transfer.rs`, `contracts.rs` is `crates/memra-tier/src/contracts.rs`,
`active.rs` is `crates/memra-engine/src/bin/kv_tier_gate/active.rs`, `core.rs` is cudarc 0.19.8
`src/driver/safe/core.rs`.

### The promote as it is today (door ON, arena absent), in order

| Step | Where (`worker.rs`) | What happens |
|---|---|---|
| 1 | `host_promote_prefix_hit(engine, px, host, pool_key, prompt, device_best_len)` 10612 | Entry point, from the admission probe on a device miss or a device hit shallower than the host's best (`host_promote_candidate` 10592: `armed()`, `lookup`, deeper-than-device). `generation_current` 10621 (a stale GLM instance drops the entry). |
| 2 | 10631-10672 | Door: `host_tier_entry_class` on the candidate's fields; `tier.program(pool_key, class)`; `candidate._tier_identity.lease(&program, generation)` (an unbound handoff import refuses here). Each refusal prints `promote refused (contracts door): ...` and serves without the host entry. |
| 3 | 10675-10693 | Door: `tier_charge(pool_key, class, device_bytes, pageable, true)`: a `ResidentCharge` with `Priority::AdmittedRestore` on the DEVICE dimension for the entry's device bytes (KV planes and f32 planes) and `pageable` for tokens, logits and hidden. Taken BEFORE any allocation; a refusal is `promote refused (tier admission refused ...)`. |
| 4 | 10694-10697 | `host_len`, the demote's `verify_digest` (`MEMRA_KV_HOST_VERIFY=1`), `t0 = Instant::now()`. |
| 5 | 10699 `device_entry_from_host(engine, src: &HostPrefixEntry)` 10461 | BY REFERENCE, the host entry stays resident. Layout version 10462; the GLM door and owner check 10468-10473; `plane_up` 10474-10501 per KV plane (step 6); `for plane in &src.kv` 10502; conv and ssm `engine.htod(&HostF32)` 10509-10521 (a `Heap` `Vec<f32>` on this tier, `memcpy_htod` from pageable memory); the draft plane 10522 through the same `plane_up`; the DFlash tail `engine.htod` per layer 10526-10545; entry assembly 10546-10590 with `_tier_charge: None`, `pos`, `last_logits.to_vec()`, `last_h.to_vec()`, `bytes: src.device_bytes`, `id: 0`, `pins: 0`. A failure is `promote failed ({err}); serving without the host entry` and `rejected_allocs += 1` (10700-10704); the tier does not latch. |
| 6 | `plane_up` 10474-10501 | `kb = len * k_tok_bytes`, `vb = len * v_tok_bytes`; `engine.alloc_u8(kb.max(1))` 10478 then `alloc_u8(vb.max(1))` 10481 (`crates/memra-engine/src/lib.rs:7463`: `stream.alloc_zeros::<u8>(n)`, the owner stream's pool allocator plus a memset, `keep_if_capturing`); the H2D COPY PROGRAM: `engine.htod_u8_into(&mut k, 0, &p.k.bytes()?[..kb])` 10485 and `(&mut v, 0, &p.v.bytes()?[..vb])` 10490 (`lib.rs:6853`: `stream.memcpy_htod(src: &[u8], &mut dst.slice_mut(off..off+n))`, one `cuMemcpyHtoDAsync` on the owner stream, `core.rs:1602-1611`; a `[T]` source takes `SyncOnDrop::Sync(None)`, `core.rs:1345-1360`, so no synchronize follows: the copy is ordered by the stream); `PrefixPlane { k, v, len, k_tok_bytes, v_tok_bytes }`. The host bytes come from `HostPlaneBytes::bytes()` 7770: `Pinned` is `PinnedHostBuf::as_slice()` (`cuMemHostAlloc(.., 0)`, `pinned_host.rs:190`); `Contract` is `CudaPinnedLease::bytes()` (`tier_transfer.rs:63-71`: `PinnedHostSlice::as_slice()`, which first synchronizes the slice's tracking event, `core.rs:1468-1471`, then the whole initialized range; `cuMemHostAlloc(.., CU_MEMHOSTALLOC_WRITECOMBINED)`, `core.rs:1412-1420`). Both are page-locked driver allocations, so the driver runs the same DMA from either; under ON every KV plane of a resident entry is `Contract` (the D2H route built it) and no `Pinned` KV plane exists on the pageable tier (the one `Pinned` producer left, the handoff import `host_entry_from_owned`, is refused at insert under the door). |
| 7 | 10705-10727 | `MEMRA_KV_HOST_VERIFY=1`: `host_roundtrip_digest(engine, &e)` 10063 = `prefix_entry_state_digest` over the NEW device entry (its own D2H reads), compared with the demote's digest: `verify ok` 10711, or `VERIFY FAILED ... host entry dropped, cold path serves` 10719 (`digest_mismatches += 1`, `remove_at`), or `verify digest failed (..); promote refused`. |
| 8 | 10728-10736 | Door: `identity.require(program, generation)` after the H2D: the identity lease must still hold (the model instance did not change under the copy). |
| 9 | 10739-10744 | `e._tier_charge = tier_charge` (the residency charge rides the device entry); `host.touch(pool_key, hi)` (recency; `hi` is dead after this). |
| 10 | 10746 `px.insert_pinned_demoting(pool_key, e, "host-promote", 1, engine, host)` 7431 | **PUBLICATION.** The one point where the device planes become the live entry: `insert_with_budget_pins` 7463 (version and key, `key_index` twin, `refuse_oversize`, newest-turn preflight, the evictable LRU) with the demote sink `host_demote_prefix_entry` for what the insert evicts (an inline D2H of the evicted twin, under ON the Option B route). The pin keeps the entry until the hit path pins it for the session. |
| 11 | 10747-10758 | `id_index`, `ms = t0.elapsed()`, `promotions += 1`, `promote_ms_total`, `[prefix-host] promote: {host_len} tokens, {MB} in {ms}ms (model ..)`. |

What must hold before step 10, and holds today: the bytes are complete on the device (the H2D is
stream-ordered on the owner stream and every later kernel runs on that stream; there is no explicit
completion observation in the OFF program), the verify digest matched when armed (step 7, the OFF
check), and under the door the identity lease still holds (step 8). Nothing in the OFF promote
attests the host source bytes: the D2H receipt each `Contract` plane carries (Option B) is read by
nobody at promote.

### The promote's timing window (day-15 finding 1, the promote delta, explained by the census)

`t0` 10697 precedes `device_entry_from_host`; `ms` 10748 follows `insert_pinned_demoting` 10746,
whose sink demotes the entry the insert evicts INLINE, so a promote that evicts (every r3 in the
identity gate: the device budget holds one entry) counts the evicted twin's whole demote in its wall
time. Under ON that demote carries bind's SHA-256 over write-combined destinations (day 15: +132 ms
on the demote line), so the promote delta on day 15 (+131 ms) is the same hash, printed twice. The
H2D program is identical in both arms (step 6). The WC cell (`WC-DESTINATIONS.md`) reports the
promote line with and without the inline demote.

### The contract sequence that replaces step 6 (`active.rs:306-382` `restore`, native `CudaTransfers`)

`alloc_device(capacity, dst_gen, request)` or `register_device(plane, dst_gen, request)` (`tier_transfer.rs:246`,
`273`: `device[ordinal]` charged for `physical_bytes`), `retain_device` 302, `h2d(CopyOp { host, device,
bytes, epochs, producer_fence: None })` 765 into `submit_batch` 811: `CopyOp::validate` (`contracts.rs:1302`;
an H2D requires `host.bytes()` readable, the destination generation `dst_gen`, no fence), the owner stream
and context check 660-676, the whole initialized host range 661, the `inflight` charge 837, the optional
producer wait 921, `bind_destination` 924-927 (the destination is bound to the ticket: `ready_view` later
requires it), `memcpy_htod(backing (a PinnedHostSlice), &mut device.slice_mut(..bytes))` 937-944 on the owner
stream (`core.rs:1602`: `stream.wait(pinned.event)` before, the pinned slice's event recorded after, the same
`cuMemcpyHtoDAsync`), the item event 949 and its consumer wait 951. Then `synchronize(&ticket)` 640 (event
sync per item, `progress` 688: status `Complete`, `valid_bytes`, `checksum = checksum(item.host.bytes())`
722-724: for an H2D the completion checksum is the hash of the HOST SOURCE after the copy, `producer_done`);
`ready_view(&ticket, item, epochs)` 1001 (`publishable` 741: `Completion::require(.., &expected, true)`; a
borrowed `ReadyView`, no handle; **this IS publication**, 1027-1028) or `with_destination` 504 (the same, with a
stream-scoped read of the destination: the gate's own D2H readback and checksum, a second copy the server
does not make; the verify digest is its check); `record_consumer(&ticket)` 572 (requires `published`);
`retire_source(&ticket)` 388 (the H2D items DROP their host source, `item.host.take()` 429: an H2D
consumes its source lease); `retire(&ticket, Some(consumer))` 1071 (`retire_binding`, the items dropped,
the `inflight` charge released 1121); `acknowledge` 1140; `take_plane(&keep)` 366 (registry release,
`Rc::try_unwrap` into the `KvPlane`), `into_pooled` (`plane.rs:242`). The `KvMaterializer` (`active.rs:38-97`
`NativeMaterializer`) is the gate's typed operand hand-off for one plane: `materialize` re-checks program,
bundle, ticket and allocation identity, epochs, device and capacity against the `ReadyView`, and `retire`
retires the ticket against the consumer fence. The server's operands are the `PrefixEntry` planes published
at step 10, and its equivalent checks are the identity lease `require` (step 8) and the receipt expectation
below; no materializer type is added.

### The host source: an H2D consumes its lease, the promote keeps its twin

The contract's H2D takes `CopyOp.host: CudaPinnedLease` BY VALUE and `retire_source` drops it (`tier_transfer.rs:429`,
"H2D host source backing and its pinned-budget charge are dropped here"); `recover_source` 452 hands it
back only for a CANCELLED restore (lane A's rule 1, `contracts.rs:1502-1515`). The OFF promote keeps the
host twin resident (`host.touch` 10744; "The host twin is kept (recency-touched)", 10604-10607), and every
gate line depends on it (the identity gate's r4 second promote attempt, lane B's typed `insert refused ...
cannot fit beside ... leased bytes` line in both arms, the tenant gate's re-demote of beta's entry).
Moving the entry's lease into the op would empty the host twin on every successful ON promote: a visible
ON/OFF divergence, refused. `CudaPinnedLease` has no public clone (`tier_transfer.rs:31-33`, the `Rc` is
private). The engine already accepts a SHARED H2D source: `validate` counts owners only for a D2H
destination (658, "A taken host destination may be used as an immutable H2D source before the earlier
ticket is acknowledged", 934-935), `take_destination` mints a second owner for a D2H destination (1057-1058,
"Both owners share ONE physical allocation and governor charge"), `write` refuses `Busy` while an owner is
shared (74-89), `PinnedAllocation::drop` releases the charge on the last owner (90-108). What the server
cannot do today is reach that state after the D2H ticket was acknowledged (Option B acknowledges at demote).
**The one engine addition: `CudaTransfers::retain_host(&CudaPinnedLease) -> CudaPinnedLease`**, the host
mirror of `retain_device` (302): a second owned handle on the same allocation and charge, on the owner
thread, same context. The H2D op takes the twin; the entry's handle never moves; `retire_source` drops the
twin and the entry's handle is the sole owner again (the flip fault's `write` works again, which the GPU cell
asserts). Not a borrowed-source seam (ruling 14): the op owns a lease, the trait is unchanged, no v1.4.

### Charges: the device dimension under Option C

`tier_charge(.., device=true)` at step 3 charges the entry's device bytes as a `ResidentCharge` BEFORE the
copy; `register_device` charges `device[ordinal]` again for each destination plane during the copy and
releases it at `take_plane` (`register_device` doc, 271-272: "Do not double-charge a buffer already
accounted elsewhere; callers must transfer its admission first"; there is no transfer, every registry entry
carries its own `ChargedLease`). Bound: residents' restore charges (promoted entries only; a request-inserted
entry carries `_tier_charge: None`) at most one device prefix budget, one incoming entry's restore charge at
most one budget, its registered planes at most one budget. The day-13 ledger sizes `device` at twice the
budget (`host_tier_governor` 4569-4590), which covers B (residents plus one entry's planes) and every gate
cell of C (identity 256 MB: 160 + 160 at r3, 160 + 160 + 160 at r4 against 512; tenant 384 MB: at most
480 against 768) but not the general bound, so under C the device dimension is sized at three times the
device prefix budget; pinned and pageable stay at twice. The ledger must never be what refuses a promote
the OFF arm would have made (the ruling-15 gate); the LRU decides.

### Publication and what must hold before it, under Option C

Publication stays step 10 (`insert_pinned_demoting` 10746), and every plane reaches it OUT of the engine:
before it, in this order, (a) `synchronize(&ticket)` observed every item's event, (b) the server's
`Completion::require(&ticket, &expected, true)` held with `expected[i] = SegmentExpectation { valid_bytes:
kb_or_vb, io_bytes: same, checksum: the plane's D2H receipt }`: the H2D read exactly the bytes the D2H wrote
(the completion checksum is the hash of the host source after the copy, 722-724), (c) the engine's own
`require` in `ready_view` (publication in the contract's sense), (d) the consumer fence recorded and the owner
stream drained, (e) `retire_source`, `retire`, `acknowledge`, (f) `take_plane` into `PrefixPlane`s, (g) the
verify digest (step 7, unchanged) and the identity lease (step 8, unchanged). A refusal at (b) with no fault
armed is `ReceiptMismatch`: `cancel` (`PublicationRevoked`), `recover_source` per item (rule 1; the recovered
twin's `bytes().as_ptr()` must equal the entry's lease pointer), `retire(None)`, `acknowledge`, the fresh
destinations taken back and dropped, and the CALLER drops the host entry (`remove_at`, `digest_mismatches +=
1`) exactly as `VERIFY FAILED` does: the host bytes no longer match what was written. Under
`MEMRA_KV_HOST_FAULT=flip-demote` (the gate-box diagnostic that corrupts the image after the D2H receipt by
design, day 15) the door prints the named injected difference and requires against the completion's own
checksums instead, so the verify arm catches it at promote and the failure gate's digest cell keeps its OFF
shape (`FAULT`, `demote:`, `VERIFY FAILED`), the day-15 precedent at bind.

### Refusals, typed

`HostPromoteFailure::{Failed, Refused, ReceiptMismatch, Latched}` from `device_entry_from_host`: `Failed` is the
OFF meaning (`promote failed (..)`, `rejected_allocs`; a device alloc refusal under ON books the same);
`Refused` is a contract refusal with the entry intact and the ledger clean (`promote refused (contracts door):
..; serving without the host entry`); `ReceiptMismatch` drops the host entry; `Latched` (an unobservable
completion, a ticket that did not retire or acknowledge, a destination that did not come back: the ledger holds
state the server cannot reach, and in-flight is exactly one batch) is one `TIER DISABLED` line. Pre-submit
unwinds drop the ops' original device handles and the host twins before `take_plane` (review finding 1);
the abort drains before every release or retire and discards no result (review finding 2). A mixed entry
(some planes `Contract`, some `Pinned`) cannot be built by one demote and is refused by name rather than
half-routed; an all-`Pinned` entry keeps the OFF `plane_up` (stated: none exists under the door today).

### Landed (day 16, `25891baa9`; receipts in `DAY16.md`, `pro-single-day16/`, replay `verify-day16.py`)

As censused, with these points fixed while writing it: the source twins are minted by the one engine
addition `CudaTransfers::retain_host` (`tier_transfer.rs`, after `retain_device`; owner thread and context,
`AlreadyReleased` on a released backing); the route is `host_kv_planes_from_contract` (`worker.rs`), its
unwind `host_promote_contract_abort` (cancel, recover every source with its pointer checked against the
entry's lease, drain, retire against the consumer fence only if published, acknowledge, release the
producer after a drain, take every fresh plane back through its retained twin; nothing discarded);
`device_entry_from_host(engine, src, tier: Option<(&HostTierContext, HostTierEntryClass)>)` selects the
route under `Some((tier, class)) if src.glm.is_none() && host_entry_has_contract_plane(src)` and keeps
`plane_up` (both `htod_u8_into` calls) for everything else; `host_promote_prefix_hit` books the four
outcomes (`Failed`: `rejected_allocs` and the OFF line; `Refused`: `promote refused (contracts door): ..;
serving without the host entry`; `ReceiptMismatch`: `digest_mismatches`, `remove_at`, `..; host entry
dropped, cold path serves`; `Latched`: `host.disable`). The fault cell has two sides
(`HostTierContext::take_fault(demote)`), the demote route takes only `PreSubmit`/`PostPublish` and the
promote route only `PromotePreSubmit`/`PromotePostPublish`. The boot line ends `... (Option B); KV plane
H2D through the same engine on promote (Option C)` and prints the device ledger at three budgets. The
receipt line's digest uses the D2H line's domain so the two lines of one entry carry one digest. The
GPU cells ran on the card first (`gputests`: `6 passed; 0 failed`, the B pair and the four C cells).

Receipts (`DAY16.md`, `pro-single-day16/`, `verify-day16.py` `DAY16 REPLAY: PASS`, 203 checks): the fault
gate `ALL GREEN` on four cells (the promote cells' aborted ticket sequence consumed, `seq=2` skipped, no
`Capacity`), identity default and plain `ALL GREEN` OFF and ON with one H2D receipt per ON promote whose
digest equals its D2H's, failure `1 FAILURE(S)` both arms with the flip named at the H2D and `VERIFY
FAILED` after it, lane A's tenant fix arm `PASS` with 8 D2H and 2 H2D receipts, serve-smoke and lane B's
gates line-identical. The WC pair's first cell: `WC-DESTINATIONS.md` "Results" (demote median 37.8 OFF
against 169.2 ms ON, N=10; the promote's own share 4.5 against 33.2; a first-touch step in both arms).

### Review fixes (PR #605, findings 1 and 2; `DAY16.md` review section)

Two classification defects in the promote abort, both real: a partially accepted batch had every rejected
index pushed as a leak (a rejected slot is `None` in the engine and `recover_source` answers `Rejected`),
so a clean unwind latched the tier (fixed: the abort recovers exactly the ACCEPTED items it is handed);
and the `ready_view` loop inferred `published` from its index, which can disagree with the engine (whose
`ready_view` publishes after `owner.ready_view`, `with_destination` before its fallible steps) and then
takes the wrong arm, leaving the ticket un-retired and every destination refused (fixed: the abort asks
the engine through `cancel`, `PublicationRevoked` recovers, `AlreadyPublished` takes the consumer-fence
arm). Both paths are injectable one-shot faults, `contract-promote-reject` and `contract-promote-readyview`,
exercised by two GPU unit cells and two new cells of `tools/kv-host-contract-fault-gate.sh`.

### One-shot faults and the receipt line

`MEMRA_KV_HOST_FAULT=contract-promote-presubmit` (the producer fence refused before any op is submitted:
every fresh destination released, every twin dropped, entry intact and writable, tier on, the next promote
completes) and `contract-promote-postpublish` (a refusal after `ready_view` published every item: the ticket
retires against its consumer fence and is acknowledged, the twins drop, the destinations release, tier on,
the next promote completes). The demote route takes only its own two faults and the promote route only
its own, so one boot arms exactly one side. Receipt line per ON promote, before `promote:`: `[prefix-host]
contracts door H2D receipt: ticket issuer=<n> seq=<n> epochs=0/1/1 items=<n> (<k> KV planes[, draft])
complete=<n> require=ok checksums_sha256=<hex> published retired acknowledged`, where the digest is over the
same ordered item checksums under the same domain as the D2H line, so for one entry the H2D line's digest
EQUALS the D2H line's digest of its demote (the replay checks it).

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

`DAY13.md`, `pro-single-day13/` (plain surface); `DAY14.md`, `pro-single-day14/` (draft-bearing surface
under the gates' default spec environment, replay `verify-day14.py`); `DAY15.md`, `pro-single-day15/`
(Option B: the D2H contract receipt before every ON demote across the identity, failure, tenant-reclaim
cells, serve-smoke and lane B's gates unchanged, replay `verify-day15.py`). OFF and ON on the same binary, same prompts, N=1,
`executed-not-qualified`, one RTX PRO 6000 Blackwell at its 600 W limit, through the
canonical collector (`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`).

## Removal rule

A negative, flat or superseded receipt deletes the door in the same PR: the env read, the
boot wiring, `host_tier_context` and its helpers, the FLAGS.md row, this doc's status, and
the tests; the verdict moves to the "Removed doors" ledger. The sidecar route it exercises
(lane B's) is the lead's to keep or drop. If Option B lands and the door has served two weeks
with no rollback, the seam is deleted and the constructor becomes the naked default.

## Review table for the decide-by (2026-10-05), written day 18 (`DAY18.md`)

Every row is `executed-not-qualified` development evidence; the target card is one RTX PRO 6000 Blackwell
at 600 W unless the row says otherwise. "Banked" means the receipt and its replay are in this repository (or
on lane A's branch where named). The door's decision is the review's; this table is its input list.

| Owed cell | Receipt | Verdict line (verbatim) | State |
|---|---|---|---|
| Identity gate, default (spec) environment, OFF against ON | `pro-single-day22/cells/identity-default-{off,on}` (day 22, Move 1 whole: the ON log carries `promote submitted off the tick ... request parked`, the H2D receipt `items=34 ... published retired acknowledged`, `promote published off the tick: ticket complete after 1 poll(s)`); `pro-single-day21/cells/identity-default-{off,on}` (A's demote slice); `pro-single-day14/`, `pro-single-day15/`, `pro-single-day16/hostgate-identity-{off,on}-default` (`verify-day16.py`: `DAY16 REPLAY: PASS`, 203 checks) | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` both arms, the same 13 verdict lines, equal demote bytes, ON D2H then H2D receipts `items=34 (16 KV planes, draft) complete=34 require=ok` | banked (day 13 finding 2, `5 FAILURE(S)` under ON on the spec surface, fixed day 14 by the draft-plane surface) |
| Identity gate, plain environment, OFF against ON | `pro-single-day22/cells/identity-plain-{off,on}` (day 22, Move 1 whole, `items=32`); `pro-single-day21/cells/identity-plain-{off,on}`; `pro-single-day13/`, `-day15/`, `-day16/hostgate-identity-{off,on}-plain` | `ALL GREEN`, 13 lines, demotes `89 / 160.5MB`, `86 / 160.4MB`, 1 promote, 1 `verify ok`; ON receipts `items=32 (16 KV planes)` | banked |
| Failure gate, default and plain, OFF against ON, plus the whole-budget arm | `pro-single-day22/cells/failure-{default,plain}-{off,on}` and `failure-default-pct100-{off,on}` (day 22, tree `98170f182` with Move 1 whole); `pro-single-day21/cells/failure-{default,plain}-{off,on}` (day 21, tree `9be3f7373` with lane A's Move 1 slice; earlier `pro-single-day13/` to `-day16/`) | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` in all four share-cap arms on the target card (days 21 and 22) and on the RTX 5090 (`rtx5090-day21/`), 15 `ok` each, arm line `pool-full refusal arm: tenant share cap 50%`; the whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`) RUN on the target card day 22, door OFF and ON, `ALL GREEN`, 14 `ok` each (the tenant-reject count belongs to the share-cap arm), arm line `pool-full refusal arm: whole host budget (... insert-path skip demote after the copy)`, server line `[prefix-host] skip demote: entry 159.9MB > host budget 1MB`; until day 22 that arm was asserted by pattern against lane D's banked day-8 log only; under the door it runs the whole Move 1 contract (submit, D2H receipt, publish) before the insert-path refusal (`DAY22.md`, recorded for the lead); digest cell ON unchanged: receipt `seq=1`, `FAULT: flipped one demoted K byte`, `VERIFY FAILED` at promote. Days 13 to 17 read `1 FAILURE(S)` on `FAIL: pool-full refusal is LOUD and named` in both arms: the gate matched the insert-path `skip demote: entry` line, reachable only at `MEMRA_KV_HOST_TENANT_PCT=100`, while the default 50 (`49d1d6f65`) refuses before the copy with the typed share-cap line (suffix `405466cf7`); the gate's match moved to that cited text (`7efab005d`, `DAY21.md`), the server unchanged | banked; resolved day 21, never the door's |
| Contract fault gate, six cells (presubmit, postpublish, promote-presubmit, promote-postpublish, promote-reject, promote-readyview), with the floor | `pro-single-day22/cells/fault-{default,plain}` (day 22, tree `98170f182`, Move 1 whole, the lead's floor from #626); `pro-single-day21/cells/fault-{default,plain}` (day 21; earlier `pro-single-day16/faultgate-fix/`) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 65 `ok:` on day 22 in the default and the plain environment on the target card (`items=34`, `items=32`), the floor line `ok: promote-reject: the entry carries at least two planes, so the reject is partial (items=N >= 2)` visible in both; day 21 read 64 `ok:` before the floor on the target card and on the RTX 5090 (`items=18`, `items=16`). The floor (revuto on #626, added by the lead): the reject cell's self-consistency check `1 of M` equals `items=N` had no floor, so a regression registering one plane would have passed a cell whose purpose is a PARTIAL reject; `items=N >= 2` closes it. Until day 21 the promote-reject cell hardcoded the 27B's `1 of 34 items` (red on the 9B, lane A day 17, and it would have been red in the plain environment on the 27B); it now reads the total from the server's own r2 D2H receipt (`7efab005d`) | banked; resolved day 21 |
| Review-round cells (PR #599 findings 1 and 2; PR #605 findings 1 and 2) | `pro-single-day15-review/` (`verify-day15-review.py`), `pro-single-day16-review/` (`verify-day16-review.py`) | `DAY15 REVIEW REPLAY: PASS` (46 checks); `DAY16 REVIEW REPLAY: PASS` (42 checks) | banked |
| Lane A's tenant reclaim fix arm, OFF against ON | `pro-single-day15/`, `pro-single-day16/tenant-{off,on}` | `PASS` both arms, eight receipts | banked |
| Serve smoke and lane B's two gates, OFF against ON | `pro-single-day16/smoke-{off,on}`, `bevict-{off,on}`, `bnewest-{off,on}` | line-identical OFF against ON | banked |
| The door's cost with WRITE-COMBINED destinations (the pre-`for_device` engine), N=5 per arm per order, both orders, one lock hold | `pro-single-day16/wc-pair2-retry3/` (`wc-pair.py`: `WC PAIR REPLAY: PASS`) | demote ON 169.2 ms pooled against OFF 38.6 (steady state 136 to 140 against 6 to 8); promote's own share 33.2 against 4.5 | banked as the record; SUPERSEDED on this card class by `PinnedKind::for_device` (`docs/decisions/PINNED-DESTINATIONS.md`, ruling 23) |
| The door's cost with CACHED destinations (a `for_device` binary), same cell | lane A day 15, `research/spill-a-20260919/pro-single-day15/wc-pair/` on `origin/lane/spill-a-20260919` (`DAY15.md` there; replay with this lane's `wc-pair.py`: `WC PAIR REPLAY: PASS (12 checks)`) | demote pooled OFF **37.8** against ON **113.6** ms (steady state 6.1 to 6.9 against 81.8 to 83.0); promote 11.4 against 88.7; promote minus inline demote **4.4** against **5.8**; 36 to 51 C, 491 W peak under 600 W | banked on A's branch (the lead integrates) |
| The hash micro-cell (the split of the delta into the hash and the ticket lifecycle), both cards | `pro-single-day18/hashmicro/`, `rtx5090-day18/hashmicro/` (`day18-replay.py`: `DAY18 REPLAY hashmicro: PASS (8 checks)` on both) | target card: `cached_ms=77.922 wc_ms=1698.063 heap_ms=77.990 ... cached_gbps=2.153 wc_gbps=0.099 ... wc_over_cached=21.792`; RTX 5090 Laptop host: `cached_ms=37.339 wc_ms=1431.613 heap_ms=37.480 ... cached_gbps=4.493 wc_gbps=0.117 ... wc_over_cached=38.340` (full lines in `DAY18.md`) | banked today |
| The arena pair (door OFF both arms; whether the arena is the cheaper host shape at all) | `pro-single-day17/arena-pair/` (`arena-pair.py`: `ARENA PAIR REPLAY: PASS (18 checks)`) | `... -> arena_first_touch_absent; arena_not_slower` | banked; an input to the arena item, not the door's cost |

**Arithmetic for the review (across sittings on the same box, not a same-window measurement).** On the
target card one engine hash pass over 160 MiB of cached pinned memory is 77.9 ms (N=10, range 77.8 to
78.0); lane A's steady-state demote delta with cached destinations is about 76 ms (82 against 6.1). The
demote delta is therefore one hash pass, within 2 ms, and the ticket lifecycle's share is below the
resolution of the two cells. Of the census's three hashes per plane (`progress` at completion, the
server's at take, `bind_tier_image`'s), only the completion checksum is new under ON on the demote side
(bind's checksum is code unchanged and runs in OFF as well), which is consistent with one pass. The
promote side does not fit as simply: the census puts one completion checksum over the host source at
promote (Option C), a 77.9 ms pass, but the measured promote-share delta is 1.4 ms (5.8 against 4.4);
either that hash is not inside the promote's own window as paired, or it is not where the census puts it.
Write-combined: one pass over 160 MiB of write-combined memory is 1698 ms here (lane A's day-13 harness
read 1711 ms with another implementation), yet the day-16 pair with write-combined destinations read a
demote delta of about 130 ms; the ON demote of day 16 cannot have hashed 160 MB of write-combined memory
at this rate, so which bytes the completion checksum read on that binary is a census question, not
settled here. Both are named as open for the review; nothing is inferred beyond the numbers.

**Still missing at the decide-by, stated.**
1. The arena under the door: `MEMRA_GLM5_TP_KV_HOST=1` is refused with the door at boot (its fixed backing
   is not governor-charged and its slices are not leases); the lease handoff is engine work, not started;
   day 17's pair says the arena removes the first-touch step (about 35 ms on the first three demotes per
   boot) and is otherwise equal at steady state, which is what a pricing would weigh. Scoped by ruling 28
   (lead, integ27): the handoff stays scoped until this review, where the budget question (one pinned budget
   or two) is decided with the door. Re-read day 22: unchanged.
2. The DFlash tail slice: no drafter artifact identity is derivable from a GGUF digest and no gate boots a
   DFlash drafter on the card; no cell exists. Re-read day 22 against day 20 (`DAY20.md`, memra#365): day 20
   bounded the STANDALONE whole-prompt tap sink (the prefill tap buffer in `generate_spec_dspark` and
   `generate_spec_dflash`), a prime-time allocation shape; the tail slice is the host tier's image of the draft
   KV TAIL (`dspark_draft`) under the door, whose two blockers (identity, gate) day 20 did not touch. The
   question survives; it is not answered. Day 20 did record the export directory's byte manifest (`config.json`
   and `model.safetensors` sha256, equal to #370's `qualification.json`), which is the identity input the slice
   would bind; nothing binds it. Pre-registered, no cell exists.
3. Verify digest v3 (the draft plane inside `MEMRA_KV_HOST_VERIFY`): not landed; the day-14 finding 4 item.
4. RESOLVED day 21 (`DAY21.md`, `7efab005d`): the pool-full failure-gate line was the gate's, stale against the
   default tenant share cap; `ALL GREEN` in all four arms on both cards on the tree carrying A's slice; day 22
   added the whole-budget arm as a run receipt (both door arms, target card) and the fault gate's floor receipt.
5. The RTX 5090 class: every door cell above ran on the target card; `PinnedKind::for_device` leaves that
   class write-combined, where today's micro-cell reads 1431.6 ms per 160 MiB hash pass (0.117 GB/s); the
   door's demote cost on that class is unmeasured (no pair cell there) and, by the arithmetic above, would
   be the write-combined pass unless the census question in the paragraph above resolves otherwise. The
   per-hardware rule wants that pair before any default on that class.
6. The promote-side census question and the day-16 write-combined contradiction named above.
