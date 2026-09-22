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
The review's input list in its final form is the last section of this file ("Review table for the
decide-by (2026-10-05), final form (day 23)"): owed cells, cost table, correctness table, what is still
missing and why, and the decision question stated without an answer.

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

## Review table for the decide-by (2026-10-05), final form (day 23, `DAY23.md`; first written day 18)

Every row is `executed-not-qualified` development evidence. The target card is one RTX PRO 6000 Blackwell at
600 W through the canonical collector (`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`);
the RTX 5090 rows are the local RTX 5090 Laptop GPU with the Qwen3.5-9B NVFP4 MTP artifact under `flock` on
`/tmp/memra-5090.lock`; the target card runs the Qwen3.8-27B NVFP4-Q5K MTP artifact. "Banked" means the
receipt and its replay are in this repository (or on lane A's branch where named). Timing rows carry their N
and regime in the cited receipt; a same-window pair is named as one, everything else is a same-box
cross-sitting reading. The door's decision is the review's; these tables are its input list, and the
paragraph at the end states the question without answering it.

### A. Owed cells: what the door owes, and where each receipt is

| Owed cell | Tree | Receipt | Verdict line (verbatim) | State |
|---|---|---|---|---|
| Identity gate, default (spec) environment, OFF against ON | `98170f182` (day 22, Move 1 whole: A day 18 plus #622, #626); `9be3f7373` (day 21, A's demote slice); `3df0cb2b3` (A day 18 run 2); `1b354be59`, `70038ed01`, `30e433c4c` (days 16, 15, 14); `5ecfd262c` (day 13, the red finding) | `pro-single-day22/cells/identity-default-{off,on}` (the ON log carries `promote submitted off the tick ... request parked`, the H2D receipt `items=34 ... published retired acknowledged`, `promote published off the tick: ticket complete after 1 poll(s)`); `pro-single-day21/cells/identity-default-{off,on}`; `pro-single-day14/`, `-day15/`, `-day16/hostgate-identity-{off,on}-default` (`verify-day16.py`: `DAY16 REPLAY: PASS`, 203 checks) | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` both arms, 12 `ok`, the same 13 verdict lines, equal demote bytes, ON D2H then H2D receipts `items=34 (16 KV planes, draft) complete=34 require=ok` | banked (day 13 finding 2, `5 FAILURE(S)` under ON on the spec surface, fixed day 14 by the draft-plane surface) |
| Identity gate, plain environment (`MEMRA_SERVE_SPEC=0`), OFF against ON | as above | `pro-single-day22/cells/identity-plain-{off,on}` (`items=32`); `pro-single-day21/cells/identity-plain-{off,on}`; `pro-single-day13/`, `-day15/`, `-day16/hostgate-identity-{off,on}-plain` | `ALL GREEN (teeth=0)`, 13 lines, demotes `89 / 160.5MB`, `86 / 160.4MB`, 1 promote, 1 `verify ok`; ON receipts `items=32 (16 KV planes)` | banked |
| Failure gate, default and plain, OFF against ON, plus the whole-budget arm | `98170f182` (day 22); `9be3f7373` (day 21); `3df0cb2b3` (A day 18); days 13 to 16 | `pro-single-day22/cells/failure-{default,plain}-{off,on}` and `failure-default-pct100-{off,on}`; `pro-single-day21/cells/failure-{default,plain}-{off,on}`; `rtx5090-day21/failure-*` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` in all four share-cap arms (15 `ok`, arm line `pool-full refusal arm: tenant share cap 50%`); the whole-budget arm `MEMRA_KV_HOST_TENANT_PCT=100` run day 22 door OFF and ON, `ALL GREEN`, 14 `ok` each, arm line `pool-full refusal arm: whole host budget (... insert-path skip demote after the copy)`, server line `[prefix-host] skip demote: entry 159.9MB > host budget 1MB`; under the door that arm runs the whole Move 1 contract (submit, D2H receipt, publish, 160 MB) before the insert-path refusal (`DAY22.md`, recorded for the lead); digest cell ON: receipt `seq=1`, `FAULT: flipped one demoted K byte`, `VERIFY FAILED` at promote. Days 13 to 17 read `1 FAILURE(S)` on `FAIL: pool-full refusal is LOUD and named`: the gate matched the insert-path line reachable only at `TENANT_PCT=100` while the server default 50 refuses before the copy with the typed share-cap line; the gate's match moved (`7efab005d`, `DAY21.md`), the server unchanged | banked; resolved day 21, never the door's |
| Contract fault gate, six cells (presubmit, postpublish, promote-presubmit, promote-postpublish, promote-reject, promote-readyview), with the floor and the ticket accounting clause | `913095404` (day 25, the lead's `receipt_seq_accounts` clause from integ36, 67 ok, both cards, both environments); `98170f182` (day 22, the lead's floor from #626); `9be3f7373` (day 21); `3df0cb2b3` (A day 18, 64 ok); `1b354be59` (day 16, `faultgate-fix/`) | `pro-single-day25/cells/fault-{default,plain}`; `rtx5090-day25/fault-{default,plain}`; `pro-single-day22/cells/fault-{default,plain}`; `pro-single-day21/cells/fault-{default,plain}`; `rtx5090-day22/fault-{default,plain}`; `rtx5090-day21/fault-*` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 67 `ok:` (day 25, both environments, both cards) with the accounting clause's own line, verbatim: default `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1` and `receipt seq=2 expected 2 + 0 capture ticket(s) submitted before it = 2`; plain `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3` and `receipt seq=4 expected 2 + 2 capture ticket(s) submitted before it = 4` (the same four lines on the 9B and the 27B). The accounting clause (integ36, from C day 24's finding): the receipt clause used to pin the demote's receipt to a literal `seq=1`/`seq=2`; since Move 2 slice 1 the plain arm's seeds submit capture tickets on the same issuer and each consumes a sequence, so the plain arm read `2 FAILURE(S)` on the 5090 (day 24, twice) while its demote completed `require=ok` at `seq=3`/`seq=4`. The clause now matches any `seq` and asserts it equals the submitted demote's and equals the expected D2H count plus the capture tickets submitted before the demote's own submission line. 65 `ok:` before the clause (day 22; `items=34` / `items=32` on the 27B, `items=18` / `items=16` on the 9B), the floor line `ok: promote-reject: the entry carries at least two planes, so the reject is partial (items=N >= 2)`; 64 `ok:` before the floor (day 21). The floor (revuto on #626): the `1 of M` equals `items=N` self-consistency had no floor, so a regression registering one plane would have passed a cell whose purpose is a PARTIAL reject. Until day 21 the reject cell hardcoded the 27B's `1 of 34 items` (red on the 9B); it reads the total from the server's own r2 D2H receipt now (`7efab005d`) | banked; resolved day 21; the seq pin resolved day 25 |
| Review-round cells (PR #599 findings 1 and 2; PR #605 findings 1 and 2) | `70038ed01` review tree, `1b354be59` review tree | `pro-single-day15-review/` (`verify-day15-review.py`), `pro-single-day16-review/` (`verify-day16-review.py`) | `DAY15 REVIEW REPLAY: PASS` (46 checks); `DAY16 REVIEW REPLAY: PASS` (42 checks) | banked |
| Hit gate, door OFF against ON, the ON arm with the host tier ARMED (the gate's identity clause over the door's whole-entry restore route on its plain hits) | `b1e9c75b6` (day 27, both cards) | `pro-single-day27/cells/hit-{off,on}`, `rtx5090-day27/hit-{off,on}` | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms on both cards, 61 and 68 `ok`, the ON arm's engagement lines quoted in the correctness table | banked; every earlier hit-gate ON arm (A days 17 to 21 on the target card and 5090 days 17 and 18; C 5090 days 24 and 26) ran with the tier unarmed and covered the tick program only, kept as receipts with that note. DAY 28: the gate takes `--external-lock FD` (the identity gate's shape; under the collector's hold it takes no lock of its own, `LOCK.json` owner `collector`, teeth `tools/test_spec_on_cache_hit_gate_lock.sh` `ALL GREEN (7 ok)`, `DAY28.md`), so a later door battery can run it inside the collector's one hold; every hit-gate receipt to date ran the default arm under the gate's own `flock`, and no GPU run under the collector existed until day 29. DAY 29: run under the collector's hold on the target card (`pro-single-day29/hitgate/`, `7349ef932`, one hold, OFF then ON): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms, 61 and 68 `ok:`, each arm's `LOCK.json` `{"owner": "collector", "lock": "/tmp/memra-gpu.lock", "mechanism": "inherited-flock-same-open-description", ...}`, the gate's first line `lock: collector's inherited FD 3 on /tmp/memra-gpu.lock (no flock of this gate's own)`; ON door lines 18, restore route lines 8, OFF 0 (the day-27 figures) |
| Hit gate, door OFF against ON, the ON arm ARMED, over the door's restore route on its DRAFT-BEARING hits (lane A day 23, Move 2 owed item 2) | `2b850b2b0` (A day 23, target card) | `research/spill-a-20260919/pro-single-day23/box/gates/hitgate-{off,on}` on `origin/lane/spill-a-20260919` (`DAY23.md` there) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms, 61 and 68 `ok`, `ok: door arm: 19 route submission(s) across the two boots`; the spec-on boot's 12 draft-bearing hits `restore submitted off the tick: ... 34 planes ...; draft plane N rows (X KB) in the batch`, 13 `D2D restore receipt ... require=ok`, 12 `draft plane ready`, 12 `spec restore: ... + draft plane from cache`, zero refused, latched, declined or disagreement lines; every `spec==plain byte identity` ok with BOTH sides restored through the route | banked on A's branch (the lead integrates); the RTX 5090 pair of this cell not run |
| Hit gate, door OFF against ON, the ON arm ARMED, over the door's CAPTURE route on its spec-boundary publishes (lane A day 24, Move 2 owed item 2, the capture half) | `185c57b4f` (A day 24, target card) | `research/spill-a-20260919/pro-single-day24/box/gates/hitgate-{off,on}` on `origin/lane/spill-a-20260919` (`DAY24.md` there) | see `DAY24.md` Task 3: the spec-on boot's `insert (spec-boundary)` publishes routed as `capture submitted off the tick (spec-boundary): ... 34 planes ...; draft plane N rows (X KB) in the batch` with `items=34 ... require=ok` receipts and `capture published off the tick (spec-boundary)`, the day-23 restores still engaging, and every `spec==plain byte identity` ok | banked on A's branch (the lead integrates); the RTX 5090 pair of this cell not run |
| The door's full acceptance set on the tree that hashes the bundle off the tick AND parks a hit on the `Hashing` entry (A days 28 and 29, rulings 39 and 40): identity x4, failure x2 (the `digest` cell in), the ten-cell contract fault gate (the two hash cells and the demote cells' publication wait in), twin x2, hit OFF/ON with the day-24 census, unit 8 + 5 + 10, both cards, one sitting per card | `867655368` code, binary built at `29a1cc366` (gate script fixes `89318289b`, `259c75f62`, `06e290374`; no `crates/` change after `29a1cc366`) | `research/spill-a-20260919/pro-single-day29/box/gates/`, `.../rtx5090-day29/` on `origin/lane/spill-a-20260919` (`DAY29.md` there) | target card: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each), failure x2 `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL), twin x2 `-> PASS`, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 / 68 ok with `capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13` and `30 route submission(s)`; RTX 5090: identity default ON `ALL GREEN (teeth=0)`, fault default and plain `ALL GREEN` (123 ok), hit OFF/ON `ALL GREEN (qwen)` 61 / 68 ok, the same census; the day-28 tree read `4 FAILURE(S)` / `1 FAILURE(S)` / `23 FAILURE(S)` in the default arm on both cards from one cause (a hit inside the `Hashing` window was a miss) | banked on A's branch (the lead integrates); the typed count `hit parked on a Hashing entry` reads 1 per identity default-ON boot on both cards (the request 0.3 to 0.4 ms after the hand-off; 35 re-parks across the 73.2 ms hash on the target card, 5 across 12.9 ms on the 5090), 2 per fault-gate promote cell, 1 in the `digest` cell, 0 in every plain-arm, demote, d2d and hash cell; A's reading: the day-28 red was the window, not the hash |
| The D2H half of Move 2 owed item 1 (A day 30): the demote's 96 recurrent f32 planes ride the KV ticket as typed D2H spans into a cached pinned staging set, one landing over items and spans, taken back before the retire; the door's full acceptance set on that tree (identity x4, failure x2, the ten-cell contract fault gate, twin x2, hit OFF/ON with the day-24 census, unit 10 + 6 + 11 + 3, the day-28 double-park cell), both cards, one sitting per card | `a8d6b1df5` (tier conformance and engine seam `97a9e091f`, worker `fc637d26a`; receipts `39378d6e4`) | `research/spill-a-20260919/pro-single-day30/box/`, `.../rtx5090-day30/` on `origin/lane/spill-a-20260919` (`DAY30.md` there) | target card: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each), failure x2 `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL), twin x2 `-> PASS`, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 / 68 ok with the day-24 census, `DAY30 A2 pre-submit steady N=80 median=0.62 min=0.58 max=1.10 ... -> PASS` (day 29 by the same reader: `median=6.05 ... -> FAIL`), `DAY30 A3 logs=92 copy_complete_lines=132 receipts_paired=132 ... -> PASS`, `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; RTX 5090: fault default and plain `ALL GREEN` (123 ok), hit OFF/ON `ALL GREEN (qwen)` 61 / 68 ok with the same census, identity default ON `ALL GREEN (teeth=0)` on the rerun (the first pass `NOT RUN: the card never freed in 15 waits`), `DAY30 A3 ... receipts_paired=34 ... -> PASS` | on integ47 (`160929a92`), not on `origin/main` `9717e8d57` at the time of writing (day 39 read); integrable as a complete D2H half; the H2D and D2D halves, the governor charge of the staging (157.9 MB per context on the 27B), the strong-form receipt, a span-refusal fault cell and returning the staging to the pool after a refused receipt stay owed (`DAY30.md` sections 8 and 9) |
| The receipt's price (Move 2 slice 3, the D2D receipt term): ONE digest kernel against the copy it attests, on one 158 MiB span, event-timed on the copy stream, N=5 per order, both orders (A's cell (v) and its 5090 twin; per card, never compared across cards) | A `bb1a2b212` (day 22, target card); `8803f4b6c` (C day 28, RTX 5090) | A `research/spill-a-20260919/pro-single-day22/box/unit-cell/command.log` (the collector's `command.gpu.csv` beside it); C `rtx5090-day28/price/test.log` (`card.{before,after}.csv`, no compute app) | target card: `D2D-RECEIPT PRICE order=copy-first ... copy_median=0.158 digest_median=0.168 pair_median=0.335 pair_over_copy=2.12` and `order=digest-first ... copy_median=0.156 digest_median=0.167 pair_median=0.335 pair_over_copy=2.15`; RTX 5090 Laptop GPU: `order=copy-first ... copy_median=0.394 digest_median=0.221 pair_median=0.442 pair_over_copy=1.12` and `order=digest-first ... copy_median=0.395 digest_median=0.221 pair_median=0.441 pair_over_copy=1.12` (full lines in `DAY22.md` and `DAY28.md`) | banked on both cards; by the day-19 rule's clause the PAIR exceeds the copy on both cards (A reported it verbatim and relaxed nothing; the lead's reading: the rule did not weigh which stream carries it); the single digest exceeds the copy on the target card (1.06x) and not on the 5090 (0.56x), per card |
| The isolating stall cell (Move 2 owed item 3): an intruder with no on-tick compute of its own for the restore class (a whole-entry zero-suffix hit of a 5152-token on-grid entry, served from boundary logits), the capture class by the same intruder against a prime-only boot in the same hold; door OFF against ON, N=5 per arm per order, both orders, two passes in opposite order, one collector hold, 8192 MB cache so no demote or promote lands inside an arm | `61dc1a52a` (C day 28; `crates/` equal to the integ38 tip `643ecbb28` plus nothing) | `pro-single-day28/box/stall/ev/pass{1,2}/{prime,capture-off,capture-on,exact-off,exact-on}/run/receipt.json` (`replays.log` ten `STALL REPLAY: PASS`; `reading.log`; the collector's `command.gpu.csv`) | restore class `stall-exact-off ... stall_median=8.9` / `9.0`, `stall-exact-on ... stall_median=9.1` (IQR 0.1 each), `on_minus_off=+0.2 unc=0.2` / `0.1`; capture class `stall-capture-off ... stall_median=283.7`, `stall-capture-on ... stall_median=284.4`, `on_minus_off=+0.7 unc=0.2` both passes; the prime-only boot `stall-prime ... stall_median=301.5` (IQR 1.5), 17 ms ABOVE the same prime inside the cache-on boots, so the pre-registered subtraction reads `share=-17.8` / `-17.1` with the wrong sign and the capture arm's own share stays unread; full lines in `DAY28.md` | banked; the restore class isolated at about 9 ms in BOTH arms (the allocation and the recurrent f32 copies that stay on the owner stream, owed item 1), the rows' move unresolvable by the tenant (+0.2); the capture class's door delta +0.7 ms; a cache-off boot's prime is not a cache-on boot's prime (finding, not explained). DAY 30 (`pro-single-day30/`, `55b8da077`, `DAY30.md`): the capture share READ against a cache-ON boot whose 308 MB seed insert is refused by the typed oversize line before any copy (`MEMRA_PREFIX_CACHE_MB=128`; the admit-time seed boundary still stops the prime at 5088): base `283.1` / `283.2` (so day 28's 17 ms was the prime's tail chunk past the seed boundary on a second tick, not a capture cost), `share off +0.6 (unc 0.3) isolated` / `+0.5 (unc 0.3) isolated`, `share on +1.2 (unc 0.2) isolated` / `+1.2 (unc 0.3) isolated`, `capture on-off +0.7` both passes, the refused-on control `-0.1 (unc 0.3) under_resolution`; eight boots, one hold, two passes, N=5 per arm per order, 33 to 61 C, 32 to 501 W, `admissible=True` A day 25 priced the retire seam's part of the capture share on the target card: the `Block` settle a session retire forces held the owner thread 0.37 to 0.44 ms (median 0.41, N=11), entered 106 to 204 ms after submission, the same as the tick-top poll's 0.37 to 0.38; the seam waits on no copy (`spill-a-20260919/DAY25.md` Task 2, `pro-single-day25/box/gates/hitgate-on/`). |
| Lane A's tenant reclaim fix arm, OFF against ON | `70038ed01`, `1b354be59` | `pro-single-day15/`, `pro-single-day16/tenant-{off,on}` | `PASS` both arms, eight receipts | banked |
| Serve smoke and lane B's two gates, OFF against ON | `1b354be59` | `pro-single-day16/smoke-{off,on}`, `bevict-{off,on}`, `bnewest-{off,on}` | line-identical OFF against ON | banked |
| The door's cost with WRITE-COMBINED destinations (the pre-`for_device` engine), N=5 per arm per order, both orders, one lock hold | `1b354be59` | `pro-single-day16/wc-pair2-retry3/` (`wc-pair.py`: `WC PAIR REPLAY: PASS`) | demote ON 169.2 ms pooled against OFF 38.6 (steady state 136 to 140 against 6 to 8); promote's own share 33.2 against 4.5 | banked as the record; SUPERSEDED on this card class by `PinnedKind::for_device` (`docs/decisions/PINNED-DESTINATIONS.md`, ruling 23) |
| The door's cost with CACHED destinations (a `for_device` binary), same cell | lane A day 15 (engine source equal to `main` `a51e29abb`) | `research/spill-a-20260919/pro-single-day15/wc-pair/` on `origin/lane/spill-a-20260919` (`DAY15.md` there; replay with this lane's `wc-pair.py`: `WC PAIR REPLAY: PASS (12 checks)`) | demote pooled OFF **37.8** against ON **113.6** ms (steady state 6.1 to 6.9 against 81.8 to 83.0); promote 11.4 against 88.7; promote minus inline demote **4.4** against **5.8**; 36 to 51 C, 491 W peak under 600 W | banked on A's branch (the lead integrates) |
| The hash micro-cell (the split of the delta into the hash and the ticket lifecycle), both cards | `e16bc69e8` | `pro-single-day18/hashmicro/`, `rtx5090-day18/hashmicro/` (`day18-replay.py`: `DAY18 REPLAY hashmicro: PASS (8 checks)` on both) | target card: `cached_ms=77.922 wc_ms=1698.063 heap_ms=77.990 ... cached_gbps=2.153 wc_gbps=0.099 ... wc_over_cached=21.792`; RTX 5090 Laptop host: `cached_ms=37.339 wc_ms=1431.613 heap_ms=37.480 ... cached_gbps=4.493 wc_gbps=0.117 ... wc_over_cached=38.340` (full lines in `DAY18.md`) | banked |
| The arena pair (door OFF both arms; whether the arena is the cheaper host shape at all) | `0b55f7b39` | `pro-single-day17/arena-pair/` (`arena-pair.py`: `ARENA PAIR REPLAY: PASS (18 checks)`) | `ARENA-PAIR rule first_touch_page_o1=35.4 first_touch_page_o2=35.8 first_touch_arena_o1=0.0 first_touch_arena_o2=0.0 steady_demote_page=6.1 steady_demote_arena=6.2 demote_page=38.0 demote_arena=6.2 promote_page=11.3 promote_arena=6.6 promote_excl_page=4.5 promote_excl_arena=0.4 N=5/arm/order pooled=10 orders=2 ... -> arena_first_touch_absent; arena_not_slower` | banked; an input to the arena item, not the door's cost |
| The stall cell, five arms (prime; demote OFF, ON; promote OFF, ON), the on-tick door (before Move 1) | A day 16, `1646d421b` (engine source equal to `main` `653c997f4`) | `research/spill-a-20260919/pro-single-day16/box/stall-{prime,off,on}/ev/` (`stall_cell.py --replay`: `STALL REPLAY: PASS` x5) | `stall-demote-off ... stall_median=117.5`, `stall-promote-off ... stall_median=85.0`, `stall-demote-on ... stall_median=193.5 ... server_demote_ms=[112.8, 118.1, ...]`, `stall-promote-on ... stall_median=162.8 ... server_promote_ms=[122.7, 123.5, 88.8, ...]`; idle p50 13.4, p99 14.8 in every cell | banked on A's branch and on `main` (#626 carried the day-16 records) |
| The stall cell, demote ON after Move 1's demote half (the D2H off the tick) | A day 17, `fc46e230d` | `research/spill-a-20260919/pro-single-day17/box/stall-on/ev/` | `stall-demote-on ... arm_max=163.4 stall_median=149.6 ... server_demote_ms=[127.9, 134.0, 133.4, ...]` (submission to publication); the promote arm of the same boot, recorded not claimed: `stall_median=86.4` | banked on A's branch |
| The stall cell, promote ON after Move 1's promote half (the H2D off the tick, the request parked) | A day 18 run 2, `3df0cb2b3` (run 1, on the tree before A's hook fix: `stall_median=157.8`, `flat`, ten `settled synchronously by a promote` lines) | `research/spill-a-20260919/pro-single-day18/box/stall-on/ev/` (`box/replays.log`) | `stall-promote-on ... arm_p99=92.6 arm_max=131.1 stall_median=81.9 stall_min=81.5 stall_max=117.7 server_demote_ms=[207.0, 208.1, 172.7, ...] server_promote_ms=[60.8, 61.9, 26.4, 26.1, ...]` -> `at_off` (3.1 ms under day 16's OFF), `promote_half_flat`; `stall-demote-on ... stall_median=149.7` | banked on A's branch; taken WITH the owner-thread spin that #627 round 2 removed |
| The stall cell on MAIN's tree after #627 (the parked-only wait), promote OFF and ON, demote OFF and ON, one lock hold (this day) | `91b0d4e08` (`crates/` equal to `main` `0713c1a79`) | `pro-single-day23/stall/ev/{off,on}/{demote,promote}/receipt.json` (`replays.log`: `STALL REPLAY: PASS` x4; `day23-stall-reading.py`) | `stall-promote-on ... arm_p99=92.5 arm_max=131.0 stall_median=81.9 stall_min=81.4 stall_max=117.6 server_demote_ms=[207.7, 207.8, 172.6, ...] server_promote_ms=[61.3, 61.6, 26.4, 25.8, 25.9, 26.0, 25.9, 26.1, 25.9, 26.0]`; `stall-promote-off ... arm_max=134.2 stall_median=85.2 ... server_promote_ms=[45.0, 46.5, 11.2, 10.6, ...]`; `stall-demote-on ... arm_max=163.1 stall_median=149.5 ... server_demote_ms=[127.6, 132.6, 132.5, ...]`; `stall-demote-off ... arm_max=132.1 stall_median=117.5 ... server_demote_ms=[38.1, 42.9, 42.9, ...]`; reading `admissible=True P1=at_or_under_day18 P2=within_wait P3=on_at_off P4=off_stable P5=demote_half_unchanged P6=off_stable`; publish `1 poll(s), 19.5` to `19.7ms` x10; zero `settled synchronously by a promote`; 32 to 51 C, 33 to 363 W under 600 W | banked (`DAY23.md`); the parked-only wait moved neither the tenant's stall nor the promote's window; the one same-window OFF against ON pair of the promote arm on one tree |
| Move 1 decision cell (i), same window, both classes: door ON on the copy stream (arm X, today's tree) against door ON on the owner stream (arm Y, the day-16 tree `1646d421b` built on the box, engine source equal to `main` `653c997f4`), the day-16 script's `on` boot in both, twenty boots interleaved `XYXYXYXYXY` then `YXYXYXYXYX` in ONE collector hold (five boots per arm per order), a dry boot of the day-16 tree first | X `7349ef932` (`crates/` = the integ38 tip `cd7161bc7`; binary built at `f5398f8fb`, `crates/` equal), Y `1646d421b` (this build `a8d737c0…`; A's day-16 build of the same source read `a447fec4…`) | `pro-single-day29/stall/ev/{o1,o2}/bNN-{x,y}/{demote,promote}/receipt.json` (`replays.log` 41 `STALL REPLAY: PASS`, zero FAIL; `reading.log`; `binary-{x,y}.sha256`; `day16-crates-equality.txt` `lines=0`; the collector's `command.gpu.csv`, 8981 samples) | demote `y_minus_x=+43.4 unc=0.6 -> isolated` (order 1: Y 193.4 against X 150.0) and `+43.0 unc=0.1 -> isolated` (order 2: Y 193.1 against X 150.0), N=5 per arm per order; promote `y_minus_x=+13.3 unc=0.4 -> isolated` (Y 162.9 against X 149.5) and `+12.8 unc=0.2 -> isolated` (Y 162.5 against X 149.7); `DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=150.0 idle_p99_sitting=14.9 -> clause_not_met`, `class=promote stall_median(second stream)=149.6 idle_p99_sitting=14.9 -> clause_not_met`, `DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False admissible=True); executed-not-qualified`; 33 to 51 C, 32 to 361 W under 600 W; tenant text sha `264b120d487de2c9` in all 41 receipts of both trees | banked (`DAY29.md`); the copy stream takes 43 ms off the demote's tenant stall and 13 ms off the promote's in the same window, and neither class reaches idle p99. FINDING, not tuned: arm X's promote arm reads 149.6, not the 81.9 of A day 18 and C day 23 (the tree before Move 2): on today's tree the promote intruder's hit parks TWICE (`promote published off the tick ... 19.6ms`, then `[prefix-cache] restore submitted off the tick ... request parked`, `90.3ms to re-admission`, 100 of 100 runs) and its inline demote publishes `after 1 poll(s), 22.4ms` (day 23: `97.0ms`), so the demote's two on-tick hashes land on one tick and the promote arm's tenant stall equals the demote arm's (150.0) |

### B. The cost table: door OFF against ON on the target card (64-token entry, 159.9 MB, ms)

| Quantity | OFF | ON | Shape | Receipt |
|---|---|---|---|---|
| Demote stall median (the tenant's stretched tick), the on-tick door | 117.5 | 193.5 | A day 16, `1646d421b`, N=5 per arm per order, both orders, 43 to 50 C, 88 to 329 W | A `pro-single-day16/box/stall-{off,on}` |
| Demote stall median after Move 1's demote half (the D2H off the tick) | 117.5 (day 16) | 149.6 (A day 17), 149.7 (A day 18 run 2), 149.5 (this day, same window as 117.5 OFF); day 29 same-window ON against ON: copy stream 150.0 / 150.0 against owner stream 193.4 / 193.1 (the day-16 tree rebuilt, N=5 boots per arm per order, `y_minus_x` +43.4 / +43.0 `isolated`) | cross-sitting against day 16 except this day's pair and the day-29 pair | A `pro-single-day17/box/stall-on`, `-day18/box/stall-on`; `pro-single-day23/stall/ev` |
| Promote stall median, the on-tick door | 85.0 | 162.8 | A day 16 | as above |
| Promote stall median after Move 1 whole (the H2D off the tick, request parked) | 85.0 (day 16); 85.2 (this day) | 81.9 (A day 18 run 2, with the spin); 81.9 (this day, the parked-only wait; same window as the OFF beside it); day 29 same-window ON against ON on the integ38 tree: copy stream 149.5 / 149.7 against owner stream 162.9 / 162.5 (`y_minus_x` +13.3 / +12.8 `isolated`); the 149.6 against the 81.9 of days 18 and 23 is a cross-sitting reading across trees: on today's tree the hit parks twice (the promote, then Move 2's restore off the tick) and the inline demote's two hashes land on one tick (`DAY29.md` finding) | this day's OFF against ON is the one same-window pair of the promote arm on one tree; day 29's ON against ON is the one same-window pair of the two Move 1 programs | `pro-single-day23/stall/ev` |
| Server demote line, steady state, cached destinations | 6.1 to 6.9 (inline in a promote), 36.9 to 43.0 (a fresh entry: first touch) | 81.8 to 83.0 synchronous (A day 15 pair; day 16 stall `82.0` to `82.9`); 132 to 134 (A day 17) and 172 (A day 18) submission to publication off the tick; 131.9 to 132.6 in the demote arm and 171.8 to 172.6 inline in the promote arm this day | the ON figure after Move 1 spans submission to publication and is not the tenant's cost | A `pro-single-day15/wc-pair`, `-day16/box`, `-day17/box`, `-day18/box`; `pro-single-day23` |
| Server promote line, steady state | 10.4 to 11.3 | 87.9 to 88.8 synchronous (day 16); 25.9 to 26.4 submission to publication (A day 18 run 2); 25.8 to 26.4 this day | as above | as above |
| The cached-destination pair (the door's copy-and-hash cost, no stall): demote pooled | 37.8 (steady 6.1 to 6.9) | 113.6 (steady 81.8 to 83.0) | A day 15, `for_device` cached, N=5 per arm per order, both orders, one hold, 36 to 51 C, 491 W peak | A `pro-single-day15/wc-pair/` |
| The cached-destination pair: promote, and promote minus its inline demote | 11.4; 4.4 | 88.7; 5.8 | as above | as above |
| The write-combined pair (superseded on this class by ruling 23) | demote 38.6 (steady 6 to 8); promote share 4.5 | demote 169.2 (steady 136 to 140); promote share 33.2 | day 16, `1b354be59`, N=5 per arm per order, both orders, one hold | `pro-single-day16/wc-pair2-retry3/` |
| The hash micro-cell: one SHA-256 pass over 160 MiB on the target card's host | n/a (OFF hashes nothing) | cached 77.9 (range 77.8 to 78.0, 2.153 GB/s); write-combined 1698.1 (0.099 GB/s); heap 78.0 | `hash-micro --bytes 167772160 --n 5`, N=5 per kind per order, two orders, 36 to 38 C | `pro-single-day18/hashmicro/` |
| The hash micro-cell on the RTX 5090 class host | n/a | cached 37.3 (4.493 GB/s); write-combined 1431.6 (0.117 GB/s) | as above, 58 to 59 C | `rtx5090-day18/hashmicro/` |
| The ticket lifecycle's share (arithmetic across sittings on one box, not a measurement) | 0 | the cached steady demote delta is about 76 ms (82 against 6.1); one hash pass is 77.9 ms; so the lifecycle's synchronous share is below the two cells' resolution (about 2 ms). After Move 1 the lifecycle is the off-tick 172 ms submission to publication (A day 18) of which the tenant's tick pays the difference between the ON and OFF demote stall medians (149.7 against 117.5 cross-sitting; this day's same-window pair 149.5 against 117.5, 32.0 ms) | see the arithmetic paragraph below | `DAY18.md`, `HOSTPREFIX-DOOR.md` |
| The arena pair (door OFF both arms) | page-pinned: first touch 35.4 / 35.8, steady demote 6.1, promote 11.3, promote excl. demote 4.5 | arena: first touch 0.0 / 0.0, steady demote 6.2, promote 6.6, promote excl. demote 0.4 | day 17, N=5 per arm per order, two orders, 33 to 51 C, 493 W peak | `pro-single-day17/arena-pair/` |
| The restore class alone, isolated (a whole-entry zero-suffix hit, 5152 tokens, 309.9 MB: no prime of its own): the tenant's stall | 8.9 (pass 1), 9.0 (pass 2), IQR 0.1 | 9.1, 9.1, IQR 0.1; `on_minus_off` +0.2 / +0.2 (unc 0.2 / 0.1) | C day 28, `61dc1a52a`, N=5 per arm per order, both orders, two passes in opposite order, one hold, 33 to 60 C, 33 to 501 W; `arm_p99` 15.6 in both arms (no split), ON `server_restore_ms` 14.5 x10 (the tick period; the copy lands within it) | `pro-single-day28/box/stall/ev`; `DAY28.md` |
| The capture class under a 5088-token seed with nothing evicting (8192 MB cache), the same intruder prime in both arms | 283.7 / 283.7 (IQR 0.2) | 284.4 / 284.4 (IQR 0.1); `on_minus_off` +0.7 / +0.7 (unc 0.2) | as above; the prime-only boot (cache off) read 301.5 in both passes, so the arm's own share is not read by the subtraction (`DAY28.md`); DAY 30: against a cache-ON boot with the insert refused before any copy (base 283.1 / 283.2) the arm's own share reads `+0.6` / `+0.5` OFF and `+1.2` / `+1.2` ON (`isolated`, unc 0.2 to 0.3), `on_minus_off +0.7` both passes (`pro-single-day30/reading.log`, `DAY30.md`) | as above; `pro-single-day30/stall/ev` |
| The D2D receipt's price on the target card (Move 2 slice 3, A's cell (v)): one 158 MiB span on the copy stream, the copy against ONE digest kernel and against the PAIR the receipt needs (source and destination digests) | copy 0.158 (copy-first), 0.156 (digest-first): the OFF program copies and attests nothing | digest 0.168 / 0.167; pair 0.335 / 0.335; `pair_over_copy` 2.12 / 2.15 | A day 22, `bb1a2b212`, event-timed, N=5 per order, both orders, one warm pass, one collector hold, the card's regime in the collector's `command.gpu.csv` (600 W envelope); off the tick (0.34 ms against a 14 ms tick, the owner thread reads 2 KiB of pinned lanes at the settle) | A `pro-single-day22/box/unit-cell/command.log`; `DAY22.md` cell (v) |
| The D2D receipt's price on the RTX 5090 Laptop GPU (cell (v)'s twin; this card's own figure, never compared to the row above) | copy 0.394 (copy-first), 0.395 (digest-first) | digest 0.221 / 0.221; pair 0.442 / 0.441; `pair_over_copy` 1.12 / 1.12 | C day 28, `8803f4b6c`, the same test protocol under `flock /tmp/memra-5090.lock`, one sitting, 58 C before and after, the card off P8 into the cell (the first digest-first copy sample 0.369 is the ramp), no compute app, `power.limit [N/A]` (laptop envelope) | `rtx5090-day28/price/test.log`; `DAY28.md` Task 3 |
| The double park (C day 29's finding), the promote arm of the day-16 script on one tree, same window: the tenant's stall; the request's end-to-end latency | 85.4 / 85.3 (stall, per order); 115.5 / 115.4 (e2e median, 50 runs per order) | 149.4 / 149.4 (stall; `on_minus_off=+64.0 / +64.2 unc=0.1 -> isolated`); 221.4 / 221.4 (e2e; `+105.8 / +105.9 unc=1.2 / 1.1 -> isolated`); in 100 of 100 ON runs `request parked` twice and `90.1ms to re-admission` (89.7 to 91.5) = the tick's decode 13.46 + the inline demote's two on-tick hashes 74.9 + residual 1.8, the poll-to-re-admit slack 0.10, the copy itself about 0.5 (the cell (v) row); the tenant's +64 is the demote's hashes on one tick (57.9 of 64.1), not the second park | A day 25, `71ee2a64d`, twenty interleaved boots (ON OFF .. / OFF ON ..), N=5 boots per arm per order, both orders, one hold, 20 of 20 replays PASS, 33 to 52 C, 32 to 362 W | A `pro-single-day25/box/double-park/ev/` on `origin/lane/spill-a-20260919`, reading `day25-double-park-reading.py` (`DAY25.md` Task 1) |
| The retire-settle share of the spec-boundary capture (Move 2 owed item 3): what the owner thread held at the `Block` settle a session retire forces, per capture | n/a (OFF captures on the tick) | held 0.37 to 0.44 ms (median 0.41, N=11, `settled synchronously by a session retire`), the tick-top `Poll` 0.37 to 0.38 (N=3); entered 106 to 204 ms after submission; share of the submission-to-completion span 0.2 to 0.4 percent: the copy had landed, the 0.4 ms is the settle's fixed cost, not a wait on the copy | A day 25, the same binary, the hit gate ON on the target card (`ALL GREEN (qwen)`, 68 ok), the publish line's new clause `the settle held the owner thread H ms, entered A ms after submission` (no new `MEMRA_*` read) | A `pro-single-day25/box/gates/hitgate-on/qwen-{on,off}-server.log`, reading `day25-retire-reading.py` (`DAY25.md` Task 2) |
| The double park AFTER the promoted-pin refusal (A day 26, ruling 36; the day-25 pair's shape byte-for-byte on the tree that refuses the route when the promote published at this tick top names the hit entry), target card, same window: the tenant's stall; the request's end-to-end latency | 85.3 / 85.3 (stall, per order); 115.3 / 115.4 (e2e median, 50 runs per order) | 81.8 / 81.8 (stall, IQR 0.0; `on_minus_off=-3.4 unc=0.1 -> isolated` both orders: the door's ON stall now BELOW OFF, 149.4 before the refusal); 206.8 / 206.7 (e2e; `+91.4 unc=1.2 / 1.1 -> isolated`; 221.4 before): 100 of 100 ON runs `request parked` once, `restore submitted` 0, one typed refusal; the tenant's two stretched ticks 92.4 (decode, publish, copy, the hit's prime) and 95.3 (decode, the demote's two hashes 74.8, the finished request's token and retire), sum 187.8 against 185.4 before; the request still pays the hashes because its token is emitted a tick after its prime (`advance_sample_emit`), so its saving is 14.6, not 90.1; the demote's `in` 172.3 (97.2 before) is the poll waiting a longer tick, `in - completion` 74.8 unchanged. RTX 5090: no pair (the stall harness's promote arm never ran on the 5090 before or after; stated missing) | A day 26, `e008bf502`, one collector hold, twenty interleaved boots, N=5 per arm per order, both orders, 20 of 20 replays PASS, 32 to 52 C, 32 to 361 W | `spill-a-20260919/pro-single-day26/box/double-park/ev`; `DAY26.md` |
| The same cell on the day-27 tree (`ad4f229e0`, the day-26 code): the baseline option (a) is to be measured against (A day 27 section 3) | 85.2 / 85.3 (stall, per order); 115.4 / 115.5 (e2e median, 50 runs per order); `demote_in median=6.2 promote_in median=10.7` | 81.9 / 81.8 (stall; `on_minus_off=-3.4 / -3.5 unc=0.1 -> isolated`); 206.6 / 206.7 (e2e; `+91.3 unc=0.9 / 1.0 -> isolated`); `demote_in-completion median=74.8`, `demote_in median=172.2`, tenant top gaps 95.3 and 92.4, sum 187.8; equal to day 26 within 0.1 ms | A day 27, the same shape, one collector hold, twenty interleaved boots, N=5 boots per arm per order, both orders, 20 of 20 replays PASS, 33 to 51 C, 33 to 360 W | `spill-a-20260919/pro-single-day27/box/double-park/ev`, `box/reading-day25.log`, `box/reading-day26.log` (regenerated on C day 34 from `ev/` with A's two reading scripts, equal); `DAY27.md` section 3 (integ43) |
| The attribution of the demote's on-tick share (A day 27 section 1, read from the code on `cb9fa5ef3`, no cell): what the 74.8 is | 0 (OFF never computes the bundle checksum: `hpx.tier` is `None`, `bind_tier_image` returns at `worker.rs:9298-9300`) | ONE SHA-256 pass of `bind_tier_image`'s `StateBundle` checksum over the whole host image (`worker.rs:9344`, from `host_demote_publish` `12258` / `12265`, on the worker thread inside the tick): about 157 MB recurrent f32 in pageable heap (`HostF32::Heap`, no receipt partner) plus 1.9 MB KV, the logits and the hidden row; the two Move 1 receipt hashes (`tier_transfer.rs:1710-1712` at the poll, `9440-9451` at bind) cover the KV planes only, about 0.9 ms each on this host; on the RTX 5090 the 21 to 23 ms is the same pass at that host's heap rate (12.2 ms per 54.8 MB by A's arithmetic) plus a write-combined read of the small KV share | arithmetic against the day-18 micro-cell (2.153 GB/s: 74.2 ms per 159.8 MB) and the day-25 to day-27 decompositions (`demote_in-completion median=74.9 / 74.8 / 74.8`, N_runs=100 each); the one unmeasured term is the 9B entry's KV byte split (no line prints it) | `spill-a-20260919/DAY27.md` section 1; `OWNER-THREAD-OFFLOAD.md` day 27 (integ43) |
| The digest micro-cell on the target card's host: SHA-256 (`memra_tier::contracts::checksum`) against the slice-3 four-lane digest (`memra_tier::conformance::receipt_digest`) over 160 MiB of heap | n/a (OFF hashes nothing) | `sha_ms=77.589 lanes_ms=70.737 lanes_over_sha=0.912` (`sha_range=77.538..77.877 lanes_range=70.662..71.044`, 2.162 against 2.372 GB/s, both stable) | A day 27, `day27-digest-micro/` (both engine programs by path dependency), N=5 per order, two orders interleaved call by call, pooled N=10, one collector hold, the card idle at 33 C, 33.50 W; A's pre-registered rule (`lanes_over_sha < 1` with disjoint ranges) read TRUE: the four-lane program 9 percent cheaper here, neither at memory speed, so (b') is worth about 7 ms of the 74.8 (A's reading) | `spill-a-20260919/pro-single-day27/box/digest-micro/ev/digest-micro.log` (integ43) |
| The digest micro-cell on the RTX 5090 class host (this host's own figure, never compared to the row above) | n/a | `sha_ms=37.111 lanes_ms=55.078 lanes_over_sha=1.484` (`sha_range=36.837..40.676 lanes_range=54.490..56.115`, 4.521 against 3.046 GB/s, both stable) | A day 27, the same cell, one collector hold on `/tmp/memra-5090.lock`, the card idle at 56 C, 16.54 W; A's rule read FALSE: the four-lane program 48 percent slower than SHA-NI here; digests byte-identical to the target host's for both programs | `spill-a-20260919/rtx5090-day27/digest-micro/ev/digest-micro.log` (integ43) |
| The same cell on the day-28 tree (`45f824a75`, option (a): the bundle hash on the helper, the entry `Hashing` until the digests land; A day 28, ruling 39) | 85.2 / 85.2 (stall, per order); 115.4 / 115.4 (e2e median, 50 runs per order); `demote_in median=6.2 promote_in median=10.6` | 81.8 / 81.9 (stall; `on_minus_off=-3.4 unc=0.1 -> isolated` both orders); 132.3 / 132.2 (e2e; `+16.9 / +16.9 -> isolated`, from +91.3: the +15.8 the token-emission reading predicted); the demote's owner-thread `owner in-completion median=7.40 min=7.17 max=45.01` (N=100; from 74.8; the max is the first two demotes' pinned first touch), the helper `hashed_in_ms median=73.2` per 157.9 MB off the tick, wall `in - completion` 86.5 (it now holds the hash and a tick top), tenant top gaps 95.3 and 19.0 (from 95.3 and 92.4) | A day 28, the day-26 shape byte-for-byte, one collector hold, twenty interleaved boots, N=5 boots per arm per order, both orders, 20 of 20 replays PASS, 33 to 52 C, 31.7 to 359.6 W; `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; the same sitting's default-arm gates RED (a hit inside the `Hashing` window was a miss) | `spill-a-20260919/pro-single-day28/box/double-park/ev`, `box/reading-day28.log`; `DAY28.md` |
| The same cell on the day-29 tree (`867655368`, option 2a: a hit on the `Hashing` entry parks one tick; A day 29, ruling 40), the acceptance's clauses 1a to 1c re-read | 85.4 / 85.1 (stall, per order); 115.5 / 115.3 (e2e median, 50 runs per order); `demote_in median=6.2 promote_in median=10.7` | 81.7 / 81.8 (stall; `on_minus_off=-3.6 / -3.3 unc=0.1 -> isolated`); 132.3 / 132.2 (e2e; `+16.8 / +16.9 -> isolated`); `owner in-completion median=7.39 min=7.20 max=45.18` (N=100), helper `73.1` (72.9 to 73.5), wall `in - completion` 86.5, `settle modes={'tick-top poll': 100}`, tenant top gaps 95.3 and 18.9: equal to day 28 within 0.1 ms in every figure (no hit falls in the window in this cell: `parked_per_run=[1]` is the promote park) | A day 29, the same shape, one collector hold, twenty interleaved boots, N=5 boots per arm per order, both orders, 20 of 20 replays PASS, 33 to 50 C, 32.7 to 358.1 W; `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; the same sitting's gates ALL GREEN in every arm (section A) | `spill-a-20260919/pro-single-day29/box/double-park/ev`, `box/reading-day28.log`, `box/reading-day25.log`, `box/reading-day26.log`; `DAY29.md` sections 4 and 5 |
| The same cell on the day-30 tree (`a8d6b1df5`, Move 2 owed item 1's D2H half: the recurrent planes ride the ticket as D2H spans; A day 30) | 85.2 / 85.5 (stall, per order); 115.5 / 115.6 (e2e median, 50 runs per order); `demote_in median=6.2 promote_in median=10.8` | 76.9 / 76.9 (stall; `on_minus_off=-8.3 unc=0.2` / `-8.6 unc=0.1 -> isolated`); 127.5 / 127.2 (e2e; `+12.0 unc=1.1` / `+11.7 unc=1.0 -> isolated`); `owner in-completion N=100 median=1.96 min=1.83 max=3.03` (day 29: 7.39, max 45.18), pre-submit steady `N=80 median=0.62` (day 29: 6.05), demotes 2 and 3 of a boot 1.26 / 1.27 (day 29: 42.89 / 42.27: DAY28's first touch was the heap `Vec` pages), helper `79.8` (79.5 to 106.6; day 29: 73.1, the staging copy moved onto it), wall `in - completion` 86.7, `settle modes={'tick-top poll': 100}`, tenant top gaps 90.3 and 19.0 | A day 30, the same shape (day 26's script byte for byte), one collector hold, twenty interleaved boots, N=5 boots per arm per order, both orders, 20 of 20 replays PASS, 33 to 50 C, 33.0 to 361.9 W; the same sitting's gates ALL GREEN in every arm (section A) | `spill-a-20260919/pro-single-day30/box/double-park/ev`, `box/reading-day30-doublepark.log`, `box/reading-day28.log`, `box/reading-day25.log`, `box/counts.log`; `DAY30.md` sections 4 and 5 |

### B-5090. The cost table on the RTX 5090 class: door OFF against ON on the local RTX 5090 Laptop GPU, the 9B (per card; never divided into B)

| Quantity | OFF | ON | Shape | Receipt |
|---|---|---|---|---|
| Demote stall median (the tenant's worst tick minus its p50), the plain 64-token entry `54.6MB, 16 items` | `63.7` (IQR 2.0, pass 1); `63.0` (IQR 2.3, pass 2) | `67.0` (IQR 2.6, pass 1); `63.2` (IQR 1.5, pass 2); `on_minus_off=+3.3 unc=3.3 -> under_resolution`, `+0.3 unc=2.7 -> under_resolution` | C day 35, `091a931c0`, A day 16's shape on the 9B (cache 64 MB, host 8192 MB, `MEMRA_SERVE_SPEC=0`, the tenant 400 tokens at a 7.3 ms tick, A's intruder shapes byte-for-byte), six boots in ONE hold on `/tmp/memra-5090.lock`, two passes in opposite order, N=5 per arm per order per boot, N=10 pooled per pass; 58 to 89 C, 30.64 to 175.33 W, `power.limit [N/A]`; 10 of 10 replays PASS; pass 1 beside an unidentified co-tenant (card-wide memory 20.3 to 23.2 GB against 7.6 to 9.8 GB in pass 2), pass 2 clean | `rtx5090-day35/stall/ev/pass{1,2}/{off,on}/demote/`, `rtx5090-day35/reading.log` |
| Promote stall median (the promote-then-hit shape, A day 26's refusal firing once per run) | `47.7` (IQR 0.9); `47.5` (IQR 2.4) | `49.3` (IQR 12.5); `50.1` (IQR 1.3); `on_minus_off=+1.6 unc=12.5 -> under_resolution`, `+2.6 unc=2.7 -> under_resolution` | as above; per ON run one `promote submitted ... request parked`, one `H2D receipt require=ok`, one `promote published off the tick` (19.7 to 20.3 ms), one `restore not routed (contracts door)`, zero `restore submitted` | `rtx5090-day35/stall/ev/pass{1,2}/{off,on}/promote/` |
| The server's own lines: demote `in` | 17.1 to 26.9 (N=9 per pass) | 54.6 to 70.2 (N=9 per pass); `from submission to completion` median 44.0 / 41.0; `in - completion` median 23.8 (21.7 to 26.2) / 21.6 (20.6 to 23.4) | as above; the `in - completion` segment about 1.8 to 2.0 of day 33's 11.9 ms heap pass at the entry size on this host | `rtx5090-day35/reading.log` (`DAY35 ATTRIBUTION`) |
| The server's own lines: promote `in`, and the inline demote inside the promote window | promote 8.2 to 10.3 steady (27.2 to 31.6 first touch); inline demote 4.6 to 6.6 steady | promote 24.7 to 26.2 steady (41.6 to 45.6 first touch), completion 20.1 median; inline demote 78.3 to 83.1 steady (it spans the park), its `in - completion` 21.8 / 21.9 | as above | as above |
| Where the door's share lands in the tenant's ticks (post-hoc description, `day35-gaps-posthoc.py`, written after the run) | demote: two stretched ticks, `top1_median=71.2 / 70.3`, `top2_median=41.3 / 41.9`, sum `112.8 / 112.3`; promote: one stretched tick, `55.2 / 54.8` | demote: `top1_median=74.5 / 70.5`, `top2_median=70.9 / 69.4`, sum `146.9 / 140.0` (+34.1 / +27.7 per demote, the door's share on the SECOND tick, +29.6 / +27.5 over OFF's); promote: two stretched ticks, `56.7 / 57.4` and `40.1 / 40.0`, top-2 sum `97.9 / 97.6` against `65.3 / 64.6` (+32.6 / +33.0 per hit) | the worst tick is the same in both arms in both classes, which is why the rule's `on_minus_off` is under resolution here; a property of the metric on this shape, stated, not a re-reading of the verdict lines | `rtx5090-day35/gaps-posthoc.log` |
| The prime arm (the class's baseline; no door in it) | n/a | n/a | pass 2 `stall_median=278.4` (IQR 6.6), five stretched ticks of 220.9 to 292.6 ms per 5120- to 5123-token prime; pass 1 INADMISSIBLE, 7 of 10 intruders refused by the memory admission beside the co-tenant (`[admit-oom] capacity reject ... available 1794MB`) | `rtx5090-day35/stall/ev/pass{1,2}/prime/prime/` |
| A days 28 and 29 on this card: no cost cell (gates only; the double-park cell has run on the target card alone) | n/a | n/a | A's day-29 identity default-ON boot on the 9B prints the door's own figures, recorded not compared: the helper `hashed in 12.9ms` per 53.7 MB (50 payloads) off the tick, `landed after 6 poll(s)`, `owner in-completion 60.80ms` (the identity gate's verify arm: `pre-submit 51.56`), the parked hit `5 re-park(s)` across the hash; day 28's fault-default `23 FAILURE(S)` on this card is green on the day-29 tree (`123 ok`) | `spill-a-20260919/rtx5090-day29/identity-default-on/ev/host-on-server.log`, `rtx5090-day29/fault-{default,plain}-rerun2/`; `DAY29.md` section 6 |
| A day 30 on this card: no cost cell (gates only) | n/a | n/a | the day-30 tree's own figures, recorded not compared (`counts.sh`, the demote ledger lines per cell dir): the identity default-ON rerun helper `hashed in 30.7ms` per 53.7 MB (50 payloads), `landed after 14 poll(s)`, the parked hit `13 re-park(s)`, `owner in-completion 60.61ms` (the verify arm: `pre-submit 50.28`); fault-default `N=14 pre-submit median=3.77`, helper `median=30.35`, owner in-completion `median=13.21`; fault-plain `N=14 pre-submit median=2.17`, helper `30.00`, owner in-completion `10.57`; day 29 by the same command, fault-default-rerun2 `N=14 pre-submit median=20.38`, helper `12.90`, owner in-completion `29.82`, fault-plain-rerun2 `20.77`, `12.90`, `29.73` (gate traffic across sittings, not a cost cell) | `spill-a-20260919/rtx5090-day30/counts.log`, `rtx5090-day30/baseline-day29-counts.log`, `rtx5090-day30/identity-default-on-rerun/ev/host-on-server.log`; `DAY30.md` section 7 |

### C. The correctness table: identity, failure and fault gates, both cards, both arms

| Gate | Card, artifact | Door OFF | Door ON | Trees with a green receipt (latest first) | Receipts |
|---|---|---|---|---|---|
| Identity, default (spec) | RTX PRO 6000, 27B | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok) | same, 12 ok, D2H and H2D receipts `items=34` | `e008bf502` (A day 26, over the promoted-pin refusal of the restore route; the ON arms' route lines `promote published` 1, `restore submitted` 1, `request parked` 2, one typed refusal), `913095404` (day 25, over the lead's retire-seam settle and refused-submission fence release from integ36), `98170f182` (day 22), `9be3f7373` (day 21), `3df0cb2b3` (A day 18 x2), `1b354be59`, `70038ed01`, `30e433c4c`; red on `5ecfd262c` (day 13, fixed day 14) | `pro-single-day25/cells/identity-default-*`, `pro-single-day22/cells/identity-default-*`, `-day21/`, A `pro-single-day18/box/` |
| Identity, plain | RTX PRO 6000, 27B | `ALL GREEN (teeth=0)` (12 ok) | same, `items=32` | as above (green since day 13; `e008bf502` A day 26 over the promoted-pin refusal; `913095404` day 25 over the retire-seam settle) | `pro-single-day25/cells/identity-plain-*`, `pro-single-day22/cells/identity-plain-*` |
| Identity, default and plain | RTX 5090, 9B | `ALL GREEN (teeth=0)` x2 | `ALL GREEN (teeth=0)` x2 | `9be3f7373` (day 21); `91b0d4e08` (day 23, main's tree after #627, `ALL GREEN (teeth=0)` x4, 12 ok) | `rtx5090-day21/identity-*`; `rtx5090-day23/identity-*` |
| Failure, default and plain, share-cap arm | RTX PRO 6000, 27B | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok) | same | `e008bf502` (A day 26, 15 ok both arms), `98170f182`, `9be3f7373`, `3df0cb2b3`; days 13 to 17 red on the gate's stale line (resolved day 21) | `pro-single-day22/cells/failure-*`, `-day21/` |
| Failure, whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`) | RTX PRO 6000, 27B | `ALL GREEN` (14 ok), `skip demote: entry 159.9MB > host budget 1MB` | `ALL GREEN` (14 ok), the whole contract runs before the same refusal | `98170f182` (day 22, the only run) | `pro-single-day22/cells/failure-default-pct100-*` |
| Failure, default and plain, share-cap arm | RTX 5090, 9B | `ALL GREEN` x2 | `ALL GREEN` x2 | `9be3f7373` (day 21); `91b0d4e08` (day 23, `ALL GREEN` x4, 15 ok) | `rtx5090-day21/failure-*`; `rtx5090-day23/failure-*` |
| Failure, whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`), default and plain | RTX 5090, 9B | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x2 (14 ok, 0 FAIL), `[prefix-host] skip demote: entry 54.8MB > host budget 1MB` (default) / `entry 54.6MB` (plain), twice per boot | `ALL GREEN` x2 (14 ok), the same refusals, each after the whole Move 1 contract ran (`demote submitted off the tick ... 18 items` / `16 items`, `D2H receipt ... require=ok`, `demote published off the tick ... 36.2ms` / `24.9ms from submission to completion`) | `934a6da3a` (day 31; binary built at `5cef58f09`, `crates/` equal) | `rtx5090-day31/gates/failure-{default,plain}-pct100-{off,on}/` (`gate.log`, `verdict.txt`, `poolfull-demote-lines.txt`), `DAY31.md` |
| Contract fault (ON by construction), default and plain | RTX PRO 6000, 27B | n/a | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 67 ok with the floor and the ticket accounting clause (day 25: default `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1`, plain `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3`, and the postpublish twins `2 + 0 = 2`, `2 + 2 = 4`); 65 ok before the clause (`items=34` / `items=32`); the same lines on the local 5090 (9B, `rtx5090-day25/fault-*`) | `e008bf502` (93 ok, A day 26, over the promoted-pin refusal: the four promote cells read `restore submitted` 0 and one typed refusal each, the `d2d-restore` cell still exactly one restore submitted), `913095404` (67 ok, day 25), `98170f182` (65 ok), `9be3f7373` (64 ok), `3df0cb2b3` (64 ok), `1b354be59` | `pro-single-day25/cells/fault-*`, `pro-single-day22/cells/fault-*`, `-day21/`, `-day16/faultgate-fix/` |
| Contract fault, default and plain | RTX 5090, 9B | n/a | `ALL GREEN`, 65 ok with the floor (`items=18` / `items=16`) | `98170f182` (day 22, 65 ok), `9be3f7373` (day 21, 64 ok); `91b0d4e08` (day 23, 65 ok x2, floor line, `items=18` / `items=16`) | `rtx5090-day22/fault-*`, `rtx5090-day21/fault-*`; `rtx5090-day23/fault-*` |
| The sixteen door cells on the Move 2 slice-2 tree (A day 21 through integ37: identity x4, failure x4, fault x2, hit x2, twin x2, the two unit cells) | RTX 5090, 9B (the 27B for the twin) | `ALL GREEN` / `PASS` in every OFF cell (day 26): identity `(teeth=0)` x2 (12 ok), failure x2 (15 ok), hit (61 ok), twin27 `-> PASS` | `ALL GREEN` / `PASS` in every ON cell (day 26): identity `(teeth=0)` x2 (12 ok), failure x2 (15 ok), fault x2 (67 ok, accounting `1 + 0` and `1 + 2`), hit (61 ok), twin27 `-> PASS`, unit-server 8 passed, unit-engine 2 passed (`d2d_capture_*`, `d2d_restore_*`); the restore route engaged in `identity-plain-on` (two restores, `64 tokens, 16 planes (53.6MB)`, seq=6 and 7, under `teeth=0`) and `fault-plain` (four) | `b74269af5` (day 26, the only run on this class; the target card's is A day 21 `1350f118b`) | `rtx5090-day26/` (`DAY26.md`; attempt 1 in `attempt1-oom/`, twelve boot failures behind a foreign 22 GB holder, cause quoted); DAY 27 NOTE: the two hit cells here ran with the host tier UNARMED (the gate exported no `MEMRA_KV_HOST_MB`; the server's own line `[kv-host-contracts] MEMRA_KV_HOST_CONTRACTS=1 with no host tier on this boot (MEMRA_KV_HOST_MB=0): nothing to route, no program identity built`, zero `[prefix-host]` lines), so their `ALL GREEN` OFF and ON covered the tick program in both arms; the armed runs are the day-27 rows below, these stay as receipts |
| Hit gate (`spec-on-cache-hit-gate.sh qwen`), door OFF against ON with the host tier ARMED in the ON arm (`MEMRA_KV_HOST_MB=8192`, the identity gate's budget; the gate's day-27 door arm asserts the arming and door lines, no latch, and at least one route submission) | RTX PRO 6000, 27B | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok; `door lines: armed=0 door_on=0`) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok, 0 FAIL): both boots `[prefix-host] on: budget 8590MB pinned cacheable host RAM (MEMRA_KV_HOST_MB, startup budget policy) ...` and `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): 1 model program identities ...`; `ok: door arm: 7 route submission(s) across the two boots`: spec-off twin `capture submitted off the tick (seed): 64 tokens, 32 planes (158.8MB)` x2 and `restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=2` / `seq=3` / `seq=5` (its r2, r3, g2 hits; landed `after 1 poll(s), 2.1ms`), spec-on boot one capture (the `samp-noplane` seed) and one restore (the np hit, `seq=2`); zero refused, dropped or latched lines; identity r1, r2, r3, g1, g2 `spec==plain byte identity` with the plain side restored through the door's route and the spec side's rows on draft-bearing entries under the tick program (census: spec-on 11 `insert (spec-boundary)` + 1 `insert (seed)`, twin 2 `insert (seed)`) | `e008bf502` (A day 26, both arms, the ON census equal to day 24's exactly per ruling 36: 12/12/13/13, 2/2/3/3, 30 route submissions, 11 draft-plane captures, zero typed refusals); `b1e9c75b6` (day 27, first armed run of this gate's ON arm on any card); `c1454a5a4` (the stop() tree, the same two lines, 61 and 68 ok, `pro-single-day27-stopfix/`) | `pro-single-day27/cells/hit-{off,on}` (`DAY27.md`; the gate under its own `flock` on `/tmp/memra-gpu.lock`, card idle before and after) |
| Hit gate, door OFF against ON with the host tier ARMED in the ON arm | RTX 5090, 9B | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok, 0 FAIL), the identical shape at `16 planes (53.6MB)`: both boots armed and door ON, 7 route submissions (3 captures, restores `seq=2` / `seq=3` / `seq=5` on the twin's r2, r3, g2 and `seq=2` on the spec-on np hit, landed `2.1` to `2.2ms`), zero refused, dropped or latched lines, identity held | `cd57afeec` (A day 26, release source `e008bf502`, over the promoted-pin refusal: OFF 61 ok, ON 68 ok, the ON census 12/12/13/13 and 2/2/3/3 with 30 route submissions and 11 draft-plane captures, zero typed refusals, equal to the target card's day-24 and day-26 counts; `spill-a-20260919/rtx5090-day26/`); `b1e9c75b6` (day 27); the stop() hardening tree `c1454a5a4` (day 27, `rtx5090-day27-stopfix/`) | `rtx5090-day27/hit-{off,on}` (`DAY27.md`) |
| Identity, failure, fault on MAIN's tree after #627 | RTX PRO 6000, 27B | `ALL GREEN` in every cell (day 23): identity `(teeth=0)` x4 (12 ok), failure x4 share-cap (15 ok) and x2 whole-budget (14 ok, `skip demote: entry X MB > host budget B MB`), fault x2 (65 ok, floor, `items=34` / `items=32`) | `ALL GREEN` in every cell (day 23): identity `(teeth=0)` x4 (12 ok), failure x4 share-cap (15 ok) and x2 whole-budget (14 ok, `skip demote: entry X MB > host budget B MB`), fault x2 (65 ok, floor, `items=34` / `items=32`) | `91b0d4e08` | `pro-single-day23-gates/cells/<cell>/` (twelve cells, collector `pro-single-day23-gates/collector/`, zero lock retries, no compute app in any of the 24 snapshots) |
| Identity x4, failure x2, fault (ten cells), twin x2, hit OFF/ON on the tree that hashes the bundle off the tick and parks a hit on the `Hashing` entry (A day 29, rulings 39 and 40) | RTX PRO 6000, 27B | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x2 (12 ok), `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), twin `-> PASS`, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | `ALL GREEN (teeth=0)` x2 (12 ok; the default arm's r3 `hit parked on a Hashing entry ... 157.9MB on the hash helper for 0.3ms`, then `promote submitted`, then `cached=64 lcp=64`), `ALL GREEN` (15 ok, the `digest` cell's `VERIFY FAILED` caught after the park), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL: the four promote cells with 2 parked hits each, the two demote cells with `the boot's last publication landed before stop (bounded 15 s wait)`, the two hash cells `hand-off ticket(s) ['3'], refusal ticket(s) ['3']`), twin `-> PASS`, hit `ALL GREEN (qwen)` (68 ok, the day-24 census) | `867655368` (binary at `29a1cc366`; the day-28 tree `45f824a75` read `4 FAILURE(S)`, `1 FAILURE(S)`, `23 FAILURE(S)` in the default arm) | `spill-a-20260919/pro-single-day29/box/gates/`; `DAY29.md` section 5 |
| Identity default ON, fault default and plain, hit OFF/ON on the same tree | RTX 5090, 9B | hit `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok; 1 parked hit, `53.7MB on the hash helper for 0.4ms`, `5 re-park(s)`), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` x2 (123 ok each, on the fixed gate `06e290374`; the gate's first two shapes on this card: an abort with no verdict, then `2 FAILURE(S)` on `the next demote publishes`, both in `DAY29.md` section 2), hit `ALL GREEN (qwen)` (68 ok, the day-24 census) | `867655368` (the day-28 tree read `fault-default` `23 FAILURE(S)` here) | `spill-a-20260919/rtx5090-day29/`; `DAY29.md` section 6 |
| Identity x4, failure x2, fault (ten cells), twin x2, hit OFF/ON on the tree whose demote carries the recurrent planes as D2H spans under the ticket (A day 30, Move 2 owed item 1's D2H half) | RTX PRO 6000, 27B | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x2 (12 ok), `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), twin `-> PASS`, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | `ALL GREEN (teeth=0)` x2 (12 ok; the default arm's `hit parked on a Hashing entry ... 157.9MB on the hash helper for 0.5ms`, `50 re-park(s)` across the 105.3 ms hash), `ALL GREEN` (15 ok, the `digest` cell's `VERIFY FAILED` caught after the park), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL: the four promote cells with 2 parked hits each, the two hash cells `hand-off ticket(s) ['3'], refusal ticket(s) ['3']`, the `postpublish` cell's `demote failed (tier D2H receipt refused: injected failure (MEMRA_KV_HOST_FAULT=contract-postpublish)); nothing demoted`), twin `-> PASS`, hit `ALL GREEN (qwen)` (68 ok, the day-24 census); every copy-complete line `items=128 (32 KV, 96 f32 spans)` or `items=130 (34 KV, 96 f32 spans)` (A3, 132 of 132) | `a8d6b1df5` | `spill-a-20260919/pro-single-day30/box/gates/`; `DAY30.md` section 6 |
| Identity default ON, fault default and plain, hit OFF/ON on the same tree | RTX 5090, 9B | hit `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok; 1 parked hit, `53.7MB on the hash helper for 0.4ms`, `13 re-park(s)`; run by the battery's `identity-rerun` arm after the first pass's `NOT RUN: the card never freed in 15 waits`), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` x2 (123 ok each), hit `ALL GREEN (qwen)` (68 ok, the day-24 census); copy-complete lines `items=64 (16 KV, 48 f32 spans)` / `items=66 (18 KV, 48 f32 spans)` (A3, 34 of 34) | `a8d6b1df5` (the gate scripts read at `39378d6e4`, no `crates/` or `tools/` change) | `spill-a-20260919/rtx5090-day30/`; `DAY30.md` section 7 |
| The same five cells on the lane merged with main after #652 (`HostHashWorker::close` with the latch bound), under one collector hold (A day 30 section 11) | RTX 5090, 9B | hit `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` x2 (123 ok each), hit `ALL GREEN (qwen)` (68 ok, the day-24 census); A3 34 of 34; `hash helper joined (the tier latched off)` 8 lines, `joined (shutdown)` 15, `detached` 0 | `2a83224ab` (`4eeb76dfd` moves no `crates/` or `tools/` file against it) | `spill-a-20260919/rtx5090-day30-merge/`; `DAY30.md` section 11 |

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
settled here. After Move 1 the copy and the hash leave the tick: the demote's tenant cost fell from 193.5
to 149.6 (day 16 to day 17) and the promote's from 162.8 to 81.9 (day 16 to day 18), both cross-sitting;
what remains in the tenant's tick on the demote side (about 32 ms over OFF, cross-sitting) is not
attributed by any cell. Both are named as open for the review; nothing is inferred beyond the numbers.

### D. Still missing at the decide-by, stated, with the reason

1. **The arena under the door.** `MEMRA_GLM5_TP_KV_HOST=1` is refused with the door at boot (its fixed backing
   is not governor-charged and its slices are not leases); the lease handoff is engine work, not started.
   Day 17's pair says the arena removes the first-touch step (about 35 ms on the first three demotes per boot)
   and is otherwise equal at steady state, which is what a pricing would weigh. Scoped by ruling 28 (lead,
   integ27): the handoff stays scoped until this review, where the budget question (one pinned budget or two)
   is decided with the door. Unchanged days 22 and 23.
2. **The DFlash tail slice.** No drafter artifact identity is derivable from a GGUF digest and no gate boots a
   DFlash drafter on the card; no cell exists. Day 20 (`DAY20.md`, memra#365) bounded the STANDALONE
   whole-prompt tap sink (the prefill tap buffer in `generate_spec_dspark` and `generate_spec_dflash`), a
   prime-time allocation shape; the tail slice is the host tier's image of the draft KV TAIL (`dspark_draft`)
   under the door, whose two blockers day 20 did not touch. Day 20 did record the export directory's byte
   manifest (`config.json` and `model.safetensors` sha256, equal to #370's `qualification.json`), which is
   the identity input the slice would bind; nothing binds it. Pre-registered; the cell is not pre-registered
   in a runnable shape because its gate (a DFlash drafter booted on the card) does not exist.
3. **Verify digest v3** (the draft plane inside `MEMRA_KV_HOST_VERIFY`): not landed; the day-14 finding 4 item.
4. RESOLVED day 21 (`7efab005d`): the pool-full failure-gate line was the gate's; day 22 added the whole-budget
   arm as a run receipt and the fault gate's floor receipt.
5. **The RTX 5090 class pair.** RESOLVED day 31 (`DAY31.md`, `rtx5090-day31/pair/`): the day-16 `wc-cell` shape
   adapted to the 9B (cache 64 MB, host tier 8192 MB, the default spec boot, 64-token draft-bearing entries of
   54.8 MB demoted as `18 items`), four boots `o1-off, o1-on, o2-on, o2-off` in ONE collector hold on the `rtx5090`
   rig (35 s), N=5 per arm per order, both orders, pooled N=10; the collector's 250 ms CSV 54 to 74 C, 9.5 to
   169.9 W, the cell's own 1 s CSV 55 to 74 C, 27.7 to 169.3 W, `power.limit [N/A]`, no compute app before or
   after; `WC PAIR REPLAY: PASS (12 checks)`; the write-combined premise printed inside the hold by
   `tier-transfer-gate roundtrip`: `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined
   flags=4` (`driver_flags=6` at six sizes). The server's boot lines do not print the arm (the door line names the
   engine, the `[prefix-host] on:` line names the OFF tier's cached pool), so that print is the receipt. This card's
   own rows (ms; the server's `demote: ... in Y` and `promote: ... in Y` lines; never compared to section B):

   | Quantity | OFF | ON | Receipt |
   |---|---|---|---|
   | Demote pooled (r2 to r6 of each boot) | `median 20.0 (N=10, min 4.8, max 24.1, IQR 18.0)`; per boot `[23.1, 23.8, 21.8, 5.1, 4.8, 5.3]`, `[18.2, 24.1, 22.6, 5.4, 6.5, 17.6]` | `median 57.3 (N=10, min 38.5, max 60.4, IQR 19.3)`; per boot `[57.8, 60.4, 58.9, 39.7, 39.5, 39.7]`, `[57.7, 58.8, 56.9, 39.4, 38.5, 38.7]`; the door's `demote published ... from submission to completion` `median 26.5 (N=12, min 17.3, max 37.4, IQR 18.1)` | `rtx5090-day31/pair/reading.log`, `pair/wc-pair/ev/*-server.log` |
   | Promote pooled (r3 to r7) | `median 15.6 (N=10, min 8.6, max 27.7, IQR 17.6)` | `median 21.1 (N=10, min 20.1, max 42.4, IQR 18.5)`; the door's `promote published ... from submission to completion` `median 14.9 (N=10, min 14.6, max 15.9, IQR 0.7)` | as above |
   | Promote minus its inline demote | `median 3.5 (N=10, min 3.4, max 4.1, IQR 0.4)` | `median -19.3 (N=10, min -37.4, max -15.4, IQR 7.4)`: negative because on this tree the inline demote publishes at a later tick top than the promote (log order per hit: `demote submitted`, `promote published`, `D2H receipt`, `demote published`), so day 16's subtraction does not isolate the ON promote | as above |
   | The verdict line, verbatim | | `DAY31 PAIR VERDICT: demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated; promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated; parked per boot [0, 10, 10, 0]; pinned=write-combined; admissible=True` | `rtx5090-day31/pair/reading.log` |

   Read from the receipt, not tuned: the raw lists step in both arms (the first three demotes of a boot 18 to 24
   OFF and 57 to 60 ON, the later ones 5 to 7 OFF and 38 to 40 ON; A day 17's first-touch step on this card too),
   so the pooled medians straddle the step and the IQRs are 18 to 20. Every ON boot reads `parked=10,
   restore_submitted=5` for its 5 promotes: the day-29 double park on this card, on draft-bearing entries (A day
   26's approved proposal 1 is the named change). The micro-cell's 1431.6 ms per 160 MiB write-combined pass would
   put about 490 ms per pass on 54.8 MB; the ON demote's `in` figure reads 38.5 to 60.4, so the two hashes are not
   reading write-combined memory at that rate on this tree; which memory they read is item 6's question, not
   answered here. The per-hardware rule's pair on this class now exists as one cell with these N and this regime;
   the tenant-stall cell on this class exists since day 35 (the RTX 5090 cost table below section B; `DAY35.md`): both
   classes `under_resolution` in both passes by the worst-tick rule, the door's shares on a second stretched tick.
6. **The promote-side census question and the day-16 write-combined contradiction** named in the arithmetic.
   RESOLVED day 27 and day 33: A's code census (`spill-a-20260919/DAY27.md` section 1, integ43; the paragraph below,
   A's) names the memory each hash reads and what the 74.8 is, and answers the promote side (its item 5: hash 3 over
   the 1.9 MB of KV source leases inside `promote_completion`, the recurrent state receipted by nothing at promote,
   which fits the pair's 1.4 ms promote-share delta); C's RTX 5090 measurement (`DAY33.md`, `rtx5090-day33/hashwc/`;
   the second paragraph below) refuted the write-combined-stream reading (`H1 refuted`) and left H2 against H3 to the
   census, which A's paragraph settles as H2's form (heap). The one term still unmeasured is the 9B entry's KV byte
   split (no line prints it; adding one is engine code), carried in the packet's item 7. ANSWERED day 27 (A, `spill-a-20260919/DAY27.md` section 1, read from
   the code on `cb9fa5ef3`, no cell): the two Move 1 receipt hashes (`progress` at the poll, `bind_tier_image`'s
   per-plane check) cover the KV planes only, about 1.9 MB on the 27B's 64-token entry (a few MB on the 9B; not
   split by any line), in the card's pinned kind (cached on the target, write-combined on the 5090); the demote's
   `in - completion` (74.8 ms target, 21 to 23 ms 5090) is ONE SHA-256 pass of `bind_tier_image`'s `StateBundle`
   checksum over the WHOLE image, about 157 MB of it the recurrent f32 state in pageable heap (`HostF32::Heap`,
   `reserve_image` with no arena) that never crossed the contract and has no receipt partner, plus the pre-submit
   f32 D2H and the insert. The write-combined pass of the micro-cell (1431.6 ms per 160 MiB) applies to the one
   to two percent of the entry that lives in the write-combined leases, which is why the 5090 reads 21 to 23 and
   not 490 per pass: one heap-rate pass (37.5 ms per 160 MiB here) over the entry plus a write-combined read of
   the small KV share. The OFF arm never computes the bundle checksum (`hpx.tier` is `None` under OFF), so the
   door's demote cost is that pass, not Move 1's copy. Day 16's write-combined delta on this card is the same
   reading. The promote-side census question: the promote's `completion` figure contains the engine's hash 3 over
   the KV source leases (1.9 MB), and the recurrent state is not receipted at promote (the restore copies it on
   the owner stream; `MEMRA_KV_HOST_VERIFY=1` alone digests it).
   The RTX 5090 measurement, day 33 (`DAY33.md`, `rtx5090-day33/hashwc/`; 9B pair-cell entry size 54,800,000 B and 160 MiB,
   one collector hold, N=5 per kind per order, both orders, N=10 pooled, 56 to 58 C, premise `PINNED-DEFAULT ...
   kind=write-combined flags=4` inside the hold; this card's figures, compared to nothing from the target card): the engine
   hash over write-combined memory runs at 0.116 GB/s at BOTH sizes (`wc_ms=473.846` at 54.8 MB, `1450.827` at 160 MiB;
   `wc_over_cached` 39.9 and 39.7), cached and heap at 4.6 GB/s (`cached_ms=11.862`, `heap_ms=11.939` at 54.8 MB); a
   `memcpy` of the write-combined buffer into cached pinned memory plus the hash of the copy is 224.2 ms at 54.8 MB
   (memcpy 212.3, hash 11.8; 0.474 of the single pass at both sizes). Against the day-31 ON demotes (`in` median 48.3, max
   60.4, N=12; `in` minus `from submission to completion` median 21.6, min 21.2, max 23.0, N=12): verbatim `DAY33 HASH-WC
   VERDICT: ... H1-single fits=False H1-twostep fits=False -> H1 refuted (no WC read route fits the ON demote's wall time; H2
   or H3 stands, separated by the code census, not by this cell); pinned=write-combined; admissible=True`. Read: the two
   demote-side hashes on this tree do not read the write-combined destination by any route measured (the fastest single
   pass 468.8 ms is 7.8x the slowest ON demote, the fastest two-step 222.0 ms is 3.7x); the post-completion segment is
   consistent with up to two hashes over cached or heap memory (`two_hashes_cached_ms=23.723` against 21.6, max 23.0).
   Whether they read a cached copy, a staging buffer or a device digest (H2), or whether the door's leases are not the
   printed arm (H3), is the code census: A day 27's paragraph above settles it as H2's form (pageable heap for about
   98 percent of the hashed bytes, the write-combined leases for the KV planes only). The promote-side half of this
   item is A's item 5 above, not day 33's measurement.
7. **The door gates on MAIN's tree after #627 on the target card.** RESOLVED day 23: twelve cells on `91b0d4e08` through the collector, all `ALL GREEN` (`pro-single-day23-gates/`, `DAY23.md`).
8. **The 5090 door gates on the tree after #627.** RESOLVED day 23: ten cells on `91b0d4e08` (fault x2 65 ok, failure x4, identity x4) `ALL GREEN` (`rtx5090-day23/`, `DAY23.md`); the whole-budget arm excepted (item 9).
9. **The whole-budget arm on the RTX 5090.** RESOLVED day 31: four cells (`failure-{default,plain}-pct100-{off,on}`,
   `MEMRA_KV_HOST_TENANT_PCT=100`, cache 64 MB) on `934a6da3a`, all `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (14 ok,
   0 FAIL, no lock retry, no compute app before or after, 54 to 61 C); the refusal `[prefix-host] skip demote: entry
   54.8MB > host budget 1MB` (default) and `entry 54.6MB > host budget 1MB` (plain) twice per boot in every arm; in
   the ON arms each refusal follows the whole Move 1 contract (`demote submitted off the tick: 64 tokens, 54.8MB,
   ticket seq=3, 18 items ...`, `D2H receipt ... items=18 (8 KV planes, draft) ... require=ok`, `demote published off
   the tick: ticket seq=3 complete after 1 poll(s), 36.2ms from submission to completion`; plain `54.6MB ... 16
   items`, `24.9ms`), as the 27B receipt showed on the target card (`rtx5090-day31/gates/`, section C row).
10. **Slice 3 and the spec-boundary route: the publishes still on the tick after Move 2 slice 1 (day 24 census,
    `DAY24.md`; no code).** Slice 1 (`prefix_capture_off_tick`, A day 20) routes only `prefix_insert_from_session`,
    the `seed` and `lcp-split` publishes of a plain-primed session. Still on the tick program are: (a) every
    `prefix_insert_from_spec_boundary` publish: the MTP drain sweep's `spec-boundary` capture (worker.rs, the tick
    boundary after the burst, one capture per prime stop, `sp.draft_plane_ref()` as the draft source), the
    `dspark-boundary` publish (with the exported drafter tail) and the `glm5-boundary` publish (latent tails); (b)
    the fanout leader's snapshot and the pause sweep's boundary snapshot (`prefix_snapshot` direct, A's day-20
    statement); (c) every `CaptureRoute::OnTick` refusal of slice 1 itself (TP shards, latent planes, an SWA ring, a
    cache not at the boundary, the latch). On the 27B at the served context (`MEMRA_CTX=8192`, the spec default
    boot) the spec-boundary publish moves per entry: the trunk KV planes `pos x 29.7 KB` (the receipts' slope,
    308.0 MB at 5088 tokens against 158.8 MB at 64: 243 MB at 8192, 1.9 MB at the gates' 64-token entries), the MTP
    draft plane K and V `pos x (k_tok_bytes + v_tok_bytes)` sliced from the spec session's persistent draft scratch
    (a 64-token spec-boundary entry is 158.9 MB against the plain seed's 158.8 MB, so about 0.1 MB at 64 tokens,
    about 1.9 KB per token, one layer's K and V, about 15 MB at 8192), the recurrent state and boundary logits from
    the `SpecBoundaryCapture` the spec engine already owns (taken at the burst boundary; no copy at publish), the
    f32 `last_h`, and on DSPARK the exported draft tail (a fixed about 85 MB per entry, `export_tail`). Each plane is
    `alloc_u8` (`cuMemAllocAsync` plus a memset, stream-ordered) then `copy_u8_into` (`memcpy_dtod` on the owner
    stream), then `insert_demoting` (whose eviction demote is Move 1's route). What an off-tick route would need: two
    borrowed source spans instead of slice 1's one, the live trunk planes `[0..pos)` (slice 1's class, append-only
    below the prime boundary) and the draft scratch `[0..pos)`, whose owner is `memra_engine::spec::SpecSession`
    (`scratch: MtpScratch`, `draft_plane_ref`; append-only below the prompt end for the session's lifetime, the
    true-hidden refresh rewrites generated positions only, `None` when ring-backed); a producer event recorded on
    the owner stream at the drain sweep (by then the burst has committed past `pos`, so every kernel that wrote a
    row below `pos` precedes the event); the borrow's lifetime, which is the open question: the sweep runs at the
    tick boundary and the session may retire or park before the copy lands, so a retiring or parking spec session
    with a `Capturing` entry must settle it first (slice 1's settle-before-drop, extended to the spec session's
    scratch) or the publish refuses to the tick program; one `Capturing` entry per worker as today; and slice 3's
    receipt term covering both plane classes (the draft plane binds as `Role::Draft`, `items=34` on the 27B). The
    DSPARK tail is the drafter's own export (`dspark.draft_kv().export_tail`), owned and fenced by the drafter, and
    stays outside that route unless the export becomes a capture item. DAY 24 (lane A, `research/spill-a-20260919/DAY24.md`, tree `185c57b4f`): the spec-boundary capture LANDED as designed here. Slice 1's submit half is now one core both publishers call (`host_capture_submit` over `CaptureSubmit`; the seed route's admission, recurrent clones and lines unchanged) and `prefix_spec_capture_off_tick` is the spec-boundary route, asked after the publisher's early returns and before the latent arm: the two borrowed source spans (rows `[0..pos)` of every live trunk plane and of the spec session's draft scratch through `draft_plane_ref`) ride ONE batch of slice 1's class behind one producer event recorded at the drain sweep after the burst committed past `pos`, `items=34` on the 27B against 32 plain; the recurrent state, boundary logits and `last_h` come owned from the `SpecBoundaryCapture` (no copy at publish); the `Capturing` entry owns the fresh planes of both classes (`CapturePlaneClass`, `Done { kv, draft }`, `shell.draft` at the landing) and publishes both or neither, a plane of either class that does not come back latching with nothing published; slice 3's receipt term covers the draft items as two more lanes; publication is the slice-1 tick-top publication through `insert_demoting` with `why = spec-boundary` and the OFF trace role `spec-snapshot` riding the pending state. Item 10's open question, the borrow's lifetime, is answered by naming the one path that frees a source inside the tick: the MTP demotion (`s.spec.take()` then `into_demoted`, which drops the `MtpScratch`), which now settles a pending capture `Block` before it consumes the session (`a spec demotion`; the dspark twin the same, stated); the retire, park and shutdown seams already settled it since slice 1. Refused BY NAME to the OFF program: latent planes or capture tails (`glm5-boundary`), a TP cache, a trunk layer with `0 < len < pos`, a capture whose snapshot is not at `pos`, a cache with no KV plane; a draft source shorter than the boundary mirrors the OFF program's own line and routes the trunk alone. Stated difference from OFF: an alloc or registration failure of a draft destination refuses the WHOLE capture where OFF publishes trunk-only (no token moves; a missing entry is a cold prime). Still outside, by name: the DFlash `export_tail` (`dspark-boundary`, a different publisher), TP shards and latent planes, the fanout leader's and pause sweep's `prefix_snapshot` publishes, and the recurrent f32 state (Move 2 owed item 1). The tier rule is `memra_tier::conformance::d2d_capture_draft_publish` with the red arm `d2d_capture_draft_published_with_the_draft_unlanded_fails` (contracts 87 passed).

11. **The draft-bearing restore: what the restore route refuses by name, what a draft plane's restore would need,
    and how much of the hit gate it is (day 26 census, `DAY26.md`; no code).** Slice 2's route
    (`host_restore_park_probe`, worker.rs) reaches its class check only past six silent gates: a `Restoring`
    request already pending, the route's latch, the host tier not armed (`hpx.armed()`, a `MEMRA_KV_HOST_MB`
    budget above zero and no latch), no copy stream, an empty request id, a vision or capture request, a
    continuation-reuse hit, or a `lookup` miss (an entry of at least 64 tokens exactly prefixing the prompt).
    Since A day 26 (lead ruling 36 on `DAY25.md` proposal 1; `spill-a-20260919/DAY26.md`) one TYPED refusal sits
    between the lookup and the class check: when `hpx.promoted_pin`, the one-tick insertion pin of the promote
    published at this tick top, names the entry the lookup found (`host_restore_promoted_this_admission`), the
    route prints `[prefix-cache] restore not routed (contracts door): the entry was promoted for this admission
    (insertion pin id=P, N tokens, model M); the tick program copies it` and the request takes the tick program's
    device-hit copy in the same admission (C day 29's double park removed by shape; the line sits outside the hit
    gate's `refused (contracts door)` and `restore refused` counters, and the promote's ticket seq is not on the
    pin, so the line does not carry it). At
    the class check it refuses BY NAME, silently (`return false`, no line; the typed `restore refused (contracts
    door)` line fires only after it, on the cache allocation, the OFF validation, a vanished pin or the submit):
    `e.tp.is_some()` (TP shards), any `e.latent` plane (GLM latent tails, the `glm5-boundary` publish),
    `e.draft.is_some()` (the MTP draft plane, every `insert (spec-boundary)` entry), `e.dspark_draft.is_some()`
    (the DFlash tail, `dspark-boundary`), `e.pos != e.toks.len()`, empty boundary logits, and an entry with no
    KV plane. What the draft plane's restore would need, read from the OFF program it would have to equal
    (`spec_session_from_restored_deferred`, spec.rs 9196): today the trunk cache is restored first
    (`prefix_restore_at`, the carrier), then the spec session is built and its `MtpScratch` is ALLOCATED inside
    that constructor, the entry's `draft.k` and `draft.v` are copied into `scratch.kv.k` and `scratch.kv.v` with
    `copy_u8_into` on the owner stream (`pos x k_tok_bytes` and `pos x v_tok_bytes` bytes, about 1.9 KB per
    token on the 27B), then `scratch.set_len(pos)`, all after the geometry checks (`draft_len == pos`, the
    entry's `k_tok_bytes`/`v_tok_bytes` equal to the model's scratch layout, `pos <= scratch.cap`, a ring-backed
    scratch refused) and after `spec_restore_refusal` decided at admission that this REQUEST takes the draft at
    all (a sampled request under the load guard or the penalty window, or an entry without `last_h`, serves
    plain on the same entry). So the destination side needs: the scratch allocated at the probe, before the
    session exists, owned by the pending `Restoring` state beside the trunk cache and never by a session while
    the copy is in flight; a borrowed `CudaViewMut<u8>` destination of exactly `pos x k_tok_bytes` and `pos x
    v_tok_bytes` into it; the source borrowed from the pinned entry's `draft.k`/`draft.v` under the SAME pin the
    trunk restore takes (no second guarantee); a producer fence recorded on the owner stream after the
    recurrent-state copies as today; the reader fence, rule 3's owner-stream wait installed at the settle
    before the deferred prime's first draft-head read of rows `[0..pos)` (the walk reads the scratch, so the
    wait must precede session construction or the constructor must take a ready, pre-filled scratch);
    `spec_restore_refusal` evaluated at the probe, before the submit, so a request that would serve plain on a
    draft-bearing entry submits the trunk restore alone (or takes the tick program as today) and never a draft
    plane it will not read; the geometry checks moved to the probe's validation; and slice 3's receipt term over
    both plane classes (the draft plane binds as `Role::Draft`: `items=34` on the 27B and `items=18` on the 9B
    against 32 and 16 plain). The DFlash tail (a fixed about 85 MB `export_tail`, owned and fenced by the
    drafter) stays outside unless the export becomes a capture item. How much of the hit gate this is: the
    qwen arm's spec-on boot publishes 12 entries, 11 `insert (spec-boundary)` (draft-bearing) and 1 `insert
    (seed)` (the `samp-noplane` namespace, plain), and serves 13 hits, 12 on draft-bearing entries and 1 on the
    plain one; its spec-off twin boot publishes 2 `insert (seed)` entries and serves 3 hits, all plain; per
    gate run 11 of 14 entries and 12 of 16 hits are draft-bearing, and the counts are IDENTICAL on both rigs
    (target card, A day 21 `pro-single-day21/box/gates/hitgate-{off,on}`; RTX 5090, 9B, `rtx5090-day24/hit-on`
    and `rtx5090-day26/hit-{off,on}`). One step before the class check, though: the hit gate boots with no
    `MEMRA_KV_HOST_MB` (zero `[prefix-host]` lines in every hit-gate log on both rigs), so `hpx.armed()` is
    false and the route is off for its 4 plain hits too; the hit gate's identity clause will cover the route,
    for the plain hits now and the draft-bearing ones when their restore lands, only on a boot that arms the
    host tier. Nothing here is built. DAY 27: the gate's ON arm arms the tier (`MEMRA_KV_HOST_MB=8192`) and asserts the door engaged (`5c8c72447`); run OFF and ON on both cards (`b1e9c75b6`): `ALL GREEN` in every arm, the plain side's r2, r3 and g2 hits restored through the route (`restore submitted off the tick ... seq=2` / `3` / `5`, landed after one poll) and the spec side's rows on draft-bearing entries under the tick program, byte-identical across the two programs; the identity clause now covers the route for the plain hits, and the draft-bearing restore stays owed (the correctness table rows, `DAY27.md`). DAY 23 (lane A, `research/spill-a-20260919/DAY23.md`, tree `2b850b2b0`): the draft-bearing restore LANDED as designed here (the scratch allocated at the probe and owned by the `Restoring` state, borrowed destinations of exactly `pos x tok_bytes`, the source under the trunk's pin, the draft K and V rows as two more items of the same batch and receipt, rule 3's wait before session construction through `spec_session_from_restored_ready`, `spec_restore_refusal` and the geometry checks at the probe); on the target card the hit gate's armed ON arm read `ALL GREEN (qwen)` (68 ok) with `19 route submission(s)`: the spec-on boot's 12 draft-bearing hits `restore submitted off the tick: 64 tokens, 34 planes (158.9MB), ...; draft plane 64 rows (118.8KB) in the batch` (and the 96- and 128-token shapes), receipts `items=34 ... require=ok`, `draft plane ready`, `spec restore: ... + draft plane from cache`, zero refused, latched, declined or disagreement lines, every `spec==plain byte identity` ok; identity default ON routed its two draft-bearing hits the same way (`ALL GREEN (teeth=0)`); the identity clause now covers the route for the draft-bearing hits too (the first such receipt on any card). Still outside: the DFlash tail, TP and latent entries (refused by name), and the spec-boundary CAPTURE route (item 10).

### E. The decision question, stated and not answered

Promotion would make the contract program the naked default of the host tier on the target card class:
every pageable D2H and every contract-routed H2D would carry a ticket, a per-item completion event, a
completion checksum over the plane, a typed receipt with epochs and a require verdict, and the fail-closed
settle arms, with the OFF program and its `MEMRA_KV_HOST_CONTRACTS` read deleted; Move 1's off-tick copies
would be the only demote and promote path, and the arena path would have to take the lease handoff or stay
refused. Deletion would remove the door, `host_tier_context` and its helpers, the FLAGS.md row, the six fault
cells and the contract receipts, and would keep the synchronous OFF program (one host-blocking synchronize
per plane at demote, an owner-stream `htod_u8_into` at promote, no receipt, no byte attestation, no typed
unwind), so the door's offload (Move 1) and its attestation would be lost together with its cost and its
open items. The receipts say: identity `teeth=0` in every arm on both cards on every tree since day 14, the
failure and fault gates green with the floor on both cards, the contract's copy-and-hash cost on cached
destinations about one hash pass per 160 MB entry (76 to 78 ms) at demote and 1.4 ms at promote, the
tenant's stall for a demote 149.6 against 117.5 and for a promote 81.9 against 85.0 after Move 1 (cross-
sitting; this day's same-window pair on main's tree after #627: promote 81.9 against 85.2, demote 149.5 against 117.5), the write-combined class 20 to 40 times slower per hash
pass and unmeasured as a pair, the D2D receipt's pair of digests 2.12x to 2.15x the copy it attests on the target
card and 1.12x on the RTX 5090 (0.34 and 0.44 ms per 158 MiB, on the copy stream, off the tick; per card), the restore class isolated on day 28 at 8.9 to 9.0 OFF against 9.1 ON and the
capture class's door delta +0.7 (both under a millisecond, both ON above OFF: Move 2 moved the rows and the tenant's
stall did not follow; the 9 ms is the allocation and the recurrent copies still on the owner stream), the arena handoff and the DFlash tail slice not built, the promote-side
hash not where the census puts it, and Move 1's decision cell (i) read in the same window on day 29 (the copy stream against the
owner stream, door ON in both): the demote's tenant stall 150.0 against 193.4 / 193.1 and the promote's 149.6 against 162.9 /
162.5, both `isolated`, cell (i)'s clause `stall_median(second stream) <= idle p99` NOT MET for either class (idle p99 14.9),
and the promote arm on today's tree at 149.6 where the tree before Move 2 read 81.9 (the hit parks twice, the inline
demote's hashes on one tick; a finding for the lead, not tuned). Whether that trade is a default, a longer door with a new date, or a
deletion is the owner's call at the review; nothing in this lane's records answers it.

DAY 30: the owner's input for the decide-by is assembled as ONE page plus an appendix in `DOOR-DECISION-PACKET.md`
(this directory): the question above, what the door is today, the correctness table (every gate, both arms, both
cards, tree and receipt path, verbatim verdicts), the cost table (every stall cell with its N, order and regime,
per card), the open findings, and the three outcomes the door hygiene rule allows with what each would require,
stated without a recommendation; every number there was re-read from its receipt file on day 30 (`DAY30.md`). It
is a draft for the owner and becomes a `docs/decisions/` record only after the owner decides.

DAY 32: the neighbouring door `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-09-23) has its own packet in the same shape,
`ADMIT-BY-MEMORY-DECISION-PACKET.md` (this directory). Its part (b) reaches this tier through
`host_demote_prefix_ref(engine, host, entry, ContractD2h::OnTick)` (`evict_all_demoting`, worker.rs), so under both doors
ON the admission flush's demote is the day-16 synchronous tick program of Move 1 owed item 3, not the copy-stream route;
with the host tier unarmed it is `PrefixCache::evict_all`. No cell has booted the two doors together (`DAY32.md`).

DAY 34: what changed since the packet's first draft (day 30), in one paragraph. The RTX 5090 class has its pair and
whole-budget receipts (day 31, items 5 and 9). The promoted-pin refusal landed (A day 26, ruling 37, on `main` since #647):
on the promote-then-hit shape the tenant's stall reads 81.8 against OFF's 85.3 (`on_minus_off=-3.4 unc=0.1 -> isolated`),
the request's e2e +91.4 over OFF, the demote's on-tick share `demote_in-completion median=74.8` unchanged, and the same
cell on the day-27 tree read the same within 0.1 ms. That share is now attributed (A day 27 section 1, C day 33, item 6):
it is `bind_tier_image`'s one SHA-256 pass over the whole host image, about 157 MB of it recurrent f32 in pageable heap
that never crossed the contract and has no receipt partner, a pass OFF never computes; Move 1's two receipt hashes are
about 1.8 ms over the 1.9 MB of KV planes; on the RTX 5090 no write-combined read route fits the ON demote (the fastest
WC pass 7.8x the slowest demote), and its 21 to 23 ms is the same pass at that host's heap rate plus the small
write-combined KV share. The digest micro-cell read the four-lane program 9 percent cheaper than SHA-256 on the target
host (`lanes_over_sha=0.912`) and 48 percent dearer on the 5090 host (`1.484`), neither at memory speed, so no digest swap
reaches the tick's cost. Ruling 39 (lead, integ43) approved option (a), the bundle hash on a helper thread with the entry
`Hashing` until the digests land and the receipt term unchanged, for A day 28 with its pre-registered acceptance gate
(`in - completion <= 12.0`, stall ON at or under OFF + 2.0, e2e `on_minus_off <= +20.0`, every gate green in both arms on
both cards, two fault cells, a bitwise digest unit cell, no flag); its receipt does not exist at the time of writing. The
question of this section is unchanged and still not answered here; what the review weighs has moved from an unattributed
32 ms in the demote arm's stall to a named pass with an approved arm that has not yet been measured.

DAY 35: the RTX 5090 class has its tenant-stall cell (the table B-5090 above; `DAY35.md`, `rtx5090-day35/`). By the day-16
rule (the tenant's worst tick minus its p50) the door's demote and promote arms are under resolution on this card in both passes
(`+3.3 unc 3.3`, `+0.3 unc 2.7`; `+1.6 unc 12.5`, `+2.6 unc 2.7`), with every door line present and every receipt `require=ok`;
the door's share is there, on a second stretched tick the rule does not read (about 70 ms against 41 for a demote, a second
40 ms tick for a promote), and the ON demote's `in - completion` on this host reads 21.6 to 23.8 ms, about two of day 33's heap
passes at the entry size. Pass 1 ran beside an unidentified co-tenant of about 12.6 GB and its prime arm is inadmissible for
that reason; pass 2 is clean. The question of this section is unchanged and still not answered here.

DAY 36: what changed since day 34, counted from the receipt files (`DAY36.md`; the commands in the packet's appendix A,
outputs under `day36-cpu/`). Option (a) landed on A day 28: the bundle hash runs on one helper thread per
`HostTierContext` (`hashed in` `median=73.20` ms per `98 payloads (157.9MB)` on the target card, off the tick), and the
demote's owner-thread `in - completion` reads `median=7.40` (A's N=100; `DAY28 CLAUSE 1c ... rule <=12.0 -> PASS`) from
day 27's 74.8; clauses 1a, 1b, 3, 4 and 5 passed, and clause 2 read "**Clause 2 FAILS on the target card**" (identity
default ON `4 FAILURE(S) (teeth=0)`, failure ON `1 FAILURE(S)`, contract fault `23 FAILURE(S)`; the RTX 5090's default
fault arm the same 23) for one cause, a hit inside the `Hashing` window missing. Option 2a (ruling 40) landed on A day
29: that hit parks one tick at a time until the digests land (1 line in the host-on boot of each card's identity default-ON arm), and A
day 29 read "**Clause 2 PASSES on both cards.**" with clause 1 unchanged (`median=7.39`, stall 81.7 / 81.8 against OFF
85.4 / 85.1, e2e `+16.8` / `+16.9`). Ruling 41 (integ45) makes the day-28 and day-29 code the door's serving path, closes
Move 1 owed items 2 and 2a, and names the pre-submit segment (Move 2 owed item 1; its D2H half landed since on A day 30, on integ47 (`origin/lane/spill-integ47-20260923` at `160929a92`), not on `origin/main` `9717e8d57` at the time of writing, `DAY30 A2 pre-submit steady N=80 median=0.62 min=0.58 max=1.10 boots_on=10 demotes_per_boot=[11] rule N>=80 median<=1.5 max<=3.0 -> PASS`, `DAY28 VERDICT clauses_failed=0 -> ALL PASS`) as the demote's
remaining owner-thread cost: `median=6.07` on the 4th to 11th demote of a boot and `42.19` on the first three (N=80 and
N=30, day 28). integ45's RTX 5090 receipts on the battery tree `1c540e050`, counted by this lane's command: hit gate OFF
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok:`) and ON `ALL GREEN (qwen)` (68 `ok:`, `ok: door arm: 30 route
submission(s) across the two boots`); contract fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 `ok:`
each, 0 `FAIL:`, the ten cells with `hash-helper-gone` and `hash-never-lands` 14 `ok:` each); the serve smoke
`serve-smoke: 0 failed` (the spec, gemma4 and Q35 arms SKIP for absent files) and the D2D cells `test result: ok. 5
passed; 0 failed`. Not on `main` at the time of writing (`origin/main` `ca5a90e2a`). Scope: every day-28 and day-29
stall figure is the promote-then-hit shape; no demote-class tenant-stall cell has run on this code on either card. The
question of this section is unchanged and still not answered here.

DAY 37: the RTX 5090 class has a demote-class tenant-stall cell on the option (a) tree (`DAY37.md`, `rtx5090-day37/`).
Day 35's cell ran unchanged with two binaries in one hold, interleaved in both orders: `091a931c0` (day 35's tree, no
option (a)) and `8b889dcdf` (the day-28 and day-29 code, on `main` since #652), six boots each, 40 receipts, every one
`REPLAY: PASS`, under the rule pre-registered in `DAY37.md` section 1 before the first boot. By the day-16 rule the
tenant's worst tick minus its p50 does not separate the binaries in any arm or in either class's difference in
differences (`did-demote o1=-2.3/5.8 o2=-6.4/7.3 under_resolution`, `did-promote o1=-1.8/8.8 o2=+0.7/8.1
under_resolution`), and `ON - OFF` stays under resolution within each binary (demote `+4.4 unc 4.9` on `091a931c0`,
`+0.1 unc 5.7` on `8b889dcdf`). By the pre-registered secondary quantity, the tenant's two largest gaps summed, the
demote-on and promote-on arms `moved` (`-21.7/5.2`, `-24.5/8.5`; `-23.1/8.7`, `-21.4/9.2`), both OFF controls and the
prime control did not, and both DiDs `moved` (`did-demote o1=-24.2/8.2 o2=-29.0/12.7 moved`, `did-promote
o1=-24.0/14.2 o2=-19.1/9.8 moved`): the worst tick alone does not separate the binaries and the two largest gaps
summed do; the reader does not split the sum per tick. The demote's owner-thread ledger inside the cell: `owner_held=40.29` over the first three demotes of a boot
and `41.85` over the 4th on (N=12, N=24), `pre_submit=23.49` and `25.06`, `hashed_in=12.9` off the tick, 0 parked
hits, 0 reparks, 0 detaches. The target card's demote class on the option (a) tree is measured by no line. The question
of this section is unchanged and still not answered here.

DAY 38: no card ran; day 37's cell is split per tick and the 9B entry's bytes are split from banked logs (`DAY38.md`,
`day38-cpu/`). A post-hoc reader over day 37's 40 receipts, pre-registered at `1bb7f4dd7` before it ran, names the
tenant's stretched ticks by position after the fire (tick 1 the first gap over 3 x p50 at or after the fire's gap, tick
2 the next) and applies day 37's rule to each tick unchanged, with day 37's admissibility. On tick 2, the tick whose top
polls the demote's ticket, the demote-on arm `moved` (`-20.4/2.2`, `-21.0/2.5`) and so did the demote DiD
(`did-demote o1=-21.7/3.2 o2=-22.0/3.8 moved`); on tick 1 every arm and both DiDs read `under_resolution`
(`did-demote o1=-2.3/5.9 o2=-6.5/7.7`); the demote OFF control reads `under_resolution` on both ticks; `DAY38
HYPOTHESIS ... -> consistent with H`. Promote tick 2 is `not_defined`: option (a)'s promote-on stretches one tick in 40 of
40 runs. Within each binary, demote ON minus OFF on tick 2 reads `+29.7 unc 3.8` on `091a931c0` and `+7.0 unc 3.5` on
`8b889dcdf`, both isolated; the second is the size of the ledger's `copy_settle` (8.48 / 8.30 ms), a match, not a timed
attribution. The 9B plain entry's 54.6 MB host image is 950,272 B of KV planes (exact, the D2D receipt, equal to 14848
B/token x 64), 256 B of token ids and 53.7 MB of heap payloads: logits in (950,272, 1,099,744) B and conv + ssm + hidden
in [52,599,728, 52,699,728) B by the one-decimal rounding of the printed figures. No banked line splits conv, ssm and
hidden; a per-`HostHashSlot` tally on `demote copy complete off the tick` would, and it is engine code. The target
card's demote class on the option (a) tree is still measured by no line. The question of this section is unchanged and
still not answered here.
