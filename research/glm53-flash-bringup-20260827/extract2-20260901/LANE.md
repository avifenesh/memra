# lane/glm5-extract2 — EXTRACTION PHASE 2

Owner law: everything shareable lands as GENERAL functionality, with families as CONSUMERS.
Phase 1 = `research/glm53-flash-bringup-20260827/extract-general-20260831/LANE.md` (read its
classification table first; this doc continues its numbering and its discipline).

Base: `origin/lane/glm53-flash-bringup` @ `89d164ae3` (fetched at lane start; the CURRENT head,
absorbing the bankfix consolidation and the 1M depth re-price). Worktree
`~/projects/wt-glm5-extract2`, branch `lane/glm5-extract2`. Pushed, not self-merged.

---

## 0. PREMISE CORRECTION — three of the assigned items are not on the base

The task assigned items 4, 6 and 8 as "NEW GENERICS LANDED SINCE PHASE 1", stating the
in-flight lanes had all landed. **Measured, not assumed:** they have not. Receipts:

| lane | head | on origin? | merged into bringup? | carries |
|---|---|---|---|---|
| `lane/glm5-tp-transport` | `fed175303` | **NO — local only** | no (`git merge-base --is-ancestor` = NO) | `MEMRA_GLM5_TP_TRANSPORT`; new `crates/memra-engine/src/glm5_tp_transport.rs`; deltas in `glm5_tp.rs`, `hybrid_forward.rs`, `glm5_tp_gate.rs`, `docs/FLAGS.md`; `research/.../tp-transport-20260901/HEALTH.sh` + `peer-read-probe.cu` |
| `lane/glm5-host-audit` | `4c010f1e1` | yes | no | `MEMRA_WORKER_AFFINITY`; new `crates/memra-server/src/affinity.rs`; deltas in server `lib.rs`/`worker.rs`, `docs/FLAGS.md`; `apply_penalties_dense` in `crates/memra-sampling/src/lib.rs` |
| `lane/glm5-accrace` | `09da7209a` | NO — local only | no | (nothing in this lane's scope) |

Two independent corroborations that nothing from those lanes reached the base:

- `git grep MEMRA_GLM5_TP_TRANSPORT` / `MEMRA_WORKER_AFFINITY` / `apply_penalties_dense` over
  `crates/ docs/ tools/` at `89d164ae3`: **zero hits**.
- `git diff 2bbc18d82...HEAD -- crates/` (phase-1 merge point → base) adds **zero** new
  `MEMRA_GLM5_*` flag names and **zero** glm5-named `fn`/`struct`/`enum`/`static` in any
  non-glm5-owned file. The only general-but-glm5-named flags in the tree are the ones phase 1
  already named.

**RESOLVED MID-LANE for two of the three.** `lane/glm5-tp-transport` was pushed and merged into
bringup at `469e03a81` while this lane's first battery was running. Per the owner's
absorb-the-newest order it was merged in (commit `ba8274271`, two conflicts: the gate preamble
— resolved as the UNION of their transport pin and this lane's general-name clears — and the
`perf-ci.jsonl` append, resolved as the ts-ordered union of both sides' rows). Items **4** and
**8a** are therefore EXECUTED (§2.7, §2.8). Item **6** and item **8b** followed the same course a few hours later:
`lane/glm5-host-audit` landed on main at `4c2b884ca` while this lane's fourth battery was
finishing, and both were executed on top (§2.9, §3.4).

**NET: every assigned item is closed.** Nothing in this lane is deferred for want of content. The
only two things still deferred are deferred on their MERITS, not on availability — the per-session
DraftSource trait (§4.6) and the batched-verify generalization (§4.5), both waiting on a second
hybrid spec family that does not exist yet.

Consequence for what remains, and it is a deliberate call rather than a scope dodge:
**renaming a flag that does not exist in the base is not an extraction, it is an invention** — and merging an
unpushed lane's whole delta into the extraction lane to reach its flag is exactly the
"manufactured merge hell" phase 1 refused for door H. So items 4, 6 and 8 are PLANNED here
with recipes precise enough to be a mechanical apply the hour their lanes land (§4), and every
item whose content IS on the base is EXECUTED (§2, §3).

The base did, however, already carry the two chains phase 1 listed as in-flight (the moe-loc
door D/H chain and the ep-diet chain — phase 1's own merge at `2bbc18d82` absorbed them). So
phase-1 PLANNED items #7 (door H) and the ep-diet doors were executable, and are executed.

Also unfound: `LAW:general-functionality-first` is cited in the task as living verbatim in
darklanes `agent-knowledge/gpu/kernel-craft.md`. It is **not in the corpus** — no file under
`agent-knowledge/` contains that ID or the phrase. The law was executed from the owner order
phase 1 quotes verbatim ("everything that is relevant for general integration should be
extract out") plus phase 1's classification discipline. Flagged for the owner: the law is
being cited as codified while its corpus entry does not exist.

---

## 1. THE PHASE-2 CLASSIFICATION TABLE

Inventory: `git diff 2bbc18d82...HEAD --stat -- crates/` = **41 files, +13,688 / −1,157**.
Classified with phase 1's classes: (a) GENERAL already, (b) general-but-glm5-named
(= extraction target), (c) GLM5-SPECIFIC by construction, (d) POLICY/DOCS.

| class | files | notes |
|---|---|---|
| (a) GENERAL already | 24 | see below |
| (b) general-but-glm5-named | **0 new** | every target in this lane predates the phase-1 merge point (rows 2/5/7 below) or lives in an unlanded lane (rows 4/6/8) |
| (c) GLM5-SPECIFIC | 11 | `hybrid_forward.rs` glm5 arms, `glm_spec.rs`, `glm5_tp.rs`-side content, `glm5_dedup_sched_gpu.rs` (+960, new suite), the glm5 chat/toolcall dialect arms, glm5 worker route |
| (d) POLICY/DOCS | 6 | `execution_manifest.rs`, `config.rs`, `tensor_contract.rs` rows, `model_packs/mod.rs`, `build.rs`, FLAGS/KERNELS rows |

(a) GENERAL already, representative units (no move needed, they land on main as engine
capabilities): `memra-gguf/src/placement.rs` **+484, NEW** — pure contiguous multi-GPU stage
placement, byte-costs only, zero glm5 mentions; `tp.rs` +1538 and `pp.rs` +675 (shared spec
grammar, boundary/stage machinery); `parallel.rs` +970; `decode_batch.rs` +931 and
`decode.rs` +57; `cu/qmatvec.cu` +599 and `cu/moe_router.cu` +145 (generic kernel classes);
`memra-gguf/src/source.rs` +781 (sharded-source/bulk transport); new bins
`nvfp4_bank_oracle.rs` +391 and `verify_shape_bisect.rs` +220; `memra-cli/src/lib.rs`;
`memra-kv`; `vision_pre.rs`/`vision_gemma.rs`; `model_packs/hy3/mod.rs` +276 (a second
family's pack — (c) for *hy3*, and the first evidence that the EP/transport doors need
general names).

### The eight assigned items, verdicts

| # | item | general seam it becomes | class | verdict |
|---|---|---|---|---|
| 1 | DraftSource seam (`Glm5DflashDrafter`, the drafter load contract, tap resolution, source selection) | `dflash::DflashDrafter` + `dflash::load_drafter` + `dflash::resolve_tap_layers` + `spec::DraftSourceKind` / `resolve_draft_source_kind` | STRUCTURAL, split | **EXECUTED for the model-level + selection halves; the per-session-state trait PLANNED** (§4.6 — reason and sketch) |
| 2 | `MEMRA_GLM5_HTOD_DIET` | `MEMRA_HTOD_DIET` (+ counter `HTOD_DIET_AVOIDED`) | MECHANICAL | **EXECUTED** |
| 3 | door D device-tables + vrows MoE pair naming | already `MEMRA_MOE_*` and generic | AUDIT | **VERIFIED generic; residue extracted** (§3.1) |
| 4 | `MEMRA_GLM5_TP_TRANSPORT` (peer-pull transport) | `MEMRA_TP_TRANSPORT` + `tp_transport.rs` (module renamed, tags caller-owned) | MECHANICAL | **EXECUTED — its lane landed mid-lane and was absorbed** (§2.7) |
| 5 | `MEMRA_GLM5_EP_DIET`, `MEMRA_GLM5_EP_GROUPED_PRIME` | `MEMRA_EP_DIET`, `MEMRA_EP_GROUPED_PRIME` | MECHANICAL | **EXECUTED** |
| 6 | `MEMRA_WORKER_AFFINITY` | `MEMRA_WORKER_CPUSET` (coordinator ruling) | MECHANICAL | **EXECUTED — its lane landed on main mid-lane** (§2.9) |
| 7 | `MEMRA_MOE_VROWS_DEDUP_ORDER` / `_DOWN_TMAJ` | already general | AUDIT | **VERIFIED generic; no residue** (§3.1) |
| 8 | `HEALTH.sh` + `peer-read-probe.cu` → `tools/`; `apply_penalties_dense` | `tools/box-health.sh` + `tools/peer-read-probe.cu` with contracts and a docs section; family-agnostic sampler | mixed | **8a EXECUTED** (§2.8); **8b VERIFIED family-agnostic, no move needed** (§3.4) |
| 9 | phase-1 aliases drift check | — | AUDIT | **NO DRIFT — all five hold** (§3.2) |
| C | batched verify walk generalization | hybrid-family verify contract | STRUCTURAL | **PLANNED — second consumer still not real** (§4.5) |

---

## 2. EXECUTED — what moved

Commit `847d1155d`. **No behavior change on any path a banked receipt touches**: every old
flag name honored, receipt tags byte-identical, refusal bytes preserved, counters keep
counting, no hot-path arithmetic touched (renames are read-site only).

### 2.1 The flag-alias law for BOOLEAN doors — the piece phase 1 was missing

Phase 1 built one composition per door SHAPE it moved, and each was hand-rolled:
`MEMRA_VERIFY_WS` OFF-wins (default ON), `MEMRA_SPEC_TRACE` general-wins-loudly (levelled),
`MEMRA_EP_MAP` refuse-at-load (valued). The shape this lane needed — a **default-OFF boolean
door read PER CALL** — had no law. Now it does, in one place
(`lib.rs::alias_door_from` / `alias_door`):

- either name `=1` arms the door; `=0` on either name is a deliberate pin, never an arming;
  nothing but `"1"` arms (no truthiness guessing);
- an AGREEING pair resolves to the general name;
- a DISAGREEING pair **falls the door CLOSED to the shipped program** and prints one
  `[flag-alias]` stderr line naming BOTH flags, once per process;
- the resolver returns the **armed name**, so every downstream refusal cites the flag the
  operator actually typed — the `resolve_ep_map_env` contract, extended to booleans.

Why it falls closed rather than refusing hard: this is read per call inside the round, and an
abort in the GPU worker thread exits the process and kills every live session. A per-call door
refuses by NOT ARMING; the loud line is the operator's receipt that neither value won. The
pure resolver is unit-tested three ways with no env mutation (default/arming, agreeing pair,
disagreeing pair naming both + saying which way it falls).

### 2.2 `MEMRA_GLM5_HTOD_DIET` → `MEMRA_HTOD_DIET` (door H)

Door H was engine-generic all along and only the flag was family-named. Both of its classes
are "the host uploaded bytes the device already had", and neither is family knowledge:

1. the ungated shared-expert add's `vec![1.0f32; t]` re-upload — applies to **every** MoE
   family whose plan carries no `ffn_gate_inp_shexp`;
2. the latent-plane `len_d` i32 mirror's synchronizing pageable copy — applies to **every**
   latent-KV consumer; `Engine::i32_mirror_store` is an `Engine` method, not a family method.

Renamed with it: `GLM5_HTOD_DIET_AVOIDED` → `HTOD_DIET_AVOIDED`,
`glm5_htod_diet_avoided()` → `htod_diet_avoided()`, `glm5_htod_diet_on()` → `htod_diet_on()`.

### 2.3 `MEMRA_GLM5_EP_DIET` → `MEMRA_EP_DIET`, `MEMRA_GLM5_EP_GROUPED_PRIME` → `MEMRA_EP_GROUPED_PRIME`

Both doors name a class, not a family. `EP_DIET` names a MOVEMENT class (one bulk peer
activation fan-out per layer-call instead of per-token uploads; compact peer staging with one
bulk return instead of a per-slot round-trip dribble; one scatter launch instead of the
`t*n_used` sequential axpy chain). `EP_GROUPED_PRIME` names a PROGRAM SHAPE (run the family's
own chunked grouped MoE prefill per rank over each rank's resident expert slab, then add the
peer's bulk-returned partial). The hy3 pack landing on this same base is the first concrete
second consumer in sight.

**The glm5 TP-2 walk stays the CONSUMER** and keeps everything family-shaped: its per-slot
expert kernels, its slot-ordered fmaf combine, its CSR, and its `GLM5_EP_*` engagement
counters in `glm5_tp.rs` (phase 1 classifies the TP-2 shard maps and walk as (c) — that is
unchanged). Only the door reader moved.

The load-time co-refusal in `hybrid.rs` ("`{flag}=1` is set but `MEMRA_GLM5_TP` is off") no
longer iterates two literal names: it resolves through the doors and reports the armed name.
So its bytes are **unchanged** for the alias (what every banked script sets) and correct for
the general name.

### 2.4 The draft-source seam (phase-1 item b#6) — the half that is real today

`DraftSourcePlan` was always general (it is the PLAN's statement). What was glm5-named was
everything on the ENGINE side of it. Three of the four pieces are family-agnostic by content
and moved:

- **`dflash::DflashDrafter`** — the model-level holder `{ drafter weights, byte-identity pin }`.
  Literally zero glm5 content. `glm_spec` re-exports it as `Glm5DflashDrafter`, so glm5's call
  sites and gates keep the name they were written against (the `glm5_conf_keep` pattern).
- **`dflash::load_drafter`** — every drafter↔target validation plus the sha256 identity pin:
  the checkpoint is a `DFlash2DraftModel`; `cfg.hidden == n_embd`; `target_layer_ids` name
  valid trunk layers; `mask_token_id` is inside the target vocab. All four are properties of
  the PAIR, not of glm5. `hybrid.rs` keeps only what is genuinely glm5: the family flag name
  and the `is_glm5_next()` route. Error bytes unchanged — the general fn prefixes
  `{flag}={dir}` exactly as the inline chain did.
- **`dflash::resolve_tap_layers`** — pure, with the red-arm SHIFT as a *parameter*. This is the
  one place where the family boundary is load-bearing: `MEMRA_GLM5_DFLASH_GATE_RED` is a gate
  instrument, and phase 1 classified `_GATE_RED` / `_GATE_SAME_DEV` as instruments that are
  never generalized. So the family reads its own red-arm env, prints its own `[glm5-spec]`
  tag, and hands the shift in. Error bytes unchanged (`"glm5 DFlash2"` is the label).
- **`spec::DraftSourceKind` + `resolve_draft_source_kind`** — the uniform selection law, pure:
  (1) a LOADED drafter IS the source (the operator asked for it by name; a set drafter flag
  that cannot load is already a loud boot failure); (2) else the embedded head, and only under
  a plan that declares `Embedded` — a loaded head under a plan that does not claim it is a
  load-path bug, refused by name rather than drafted from; (3) else refuse before drafting.
  glm5 selects through it, with a `debug_assert` that the general law and the family's drafter
  presence agree.

One intentional behavior delta, named because it is the only one in the lane: the glm5
no-draft-source refusal message now carries the general law's reason in a `[...]` suffix. No
test, gate, or box script greps that string (checked). And the case the resolver newly refuses
(a loaded embedded head under a plan that does not declare `Embedded`) is unreachable on glm5
— the pack declares `Embedded` and the head only loads under it — so this is strictly a
correctness tightening for the next family, not a glm5 change.

### 2.5 Gate arms for the renames — the rename is exercised, not asserted

A rename whose general name is never driven end-to-end is a rename that works on paper. Both
renamed door families gained arms in the same commit:

- `glm5_moe_loc_doors_gpu` door-H test, three arms in ONE process (the door is read per call):
  alias ON (the banked path, unchanged), **GENERAL name ON** with the counter required to
  move, and a **disagreeing pair** required to leave the counter FLAT *while the value still
  lands through the shipped synchronizing form* — fail-closed, not fail-broken.
- `glm5-tp-gate` gained **B2G** (diet armed via `MEMRA_EP_DIET`, alias unset, engagement
  counter must be > 0) and **B2X** (`MEMRA_EP_DIET=1` + `MEMRA_GLM5_EP_DIET=0`, dispatch
  counter must be 0). Its preamble now also `rm_env`s the two general names, because a leaked
  general name would disagree with the alias pins and silently defeat the ON twins.

Note for future runners, learned here: **a process-wide env pin cannot cover a per-call door's
alias.** Pinning `MEMRA_HTOD_DIET=0` for a whole suite would DISAGREE with that suite's own
alias-ON arm and fall the door closed — the pin would break the arm it was meant to cover.
Alias coverage for per-call doors belongs INSIDE the arms.

### 2.6 FLAGS.md, same PR (the flag-rename law)

Three new general rows (`MEMRA_EP_DIET`, `MEMRA_EP_GROUPED_PRIME`, `MEMRA_HTOD_DIET`) carry
the full documentation — original lane text preserved verbatim beneath a generality lead, plus
the alias-law paragraph (both arms, rollback seam, receipts pointer). The three glm5 rows
shrink to alias pointers. `tools/check-flags.sh`: green, **760 runtime literal reads, no
uncovered names, no grandfather list** — every alias covered.
### 2.7 Item 4 — the TP transport seam goes general

`crates/memra-engine/src/glm5_tp_transport.rs` → `tp_transport.rs`. The generality claim is
about CONTENT, and it is checkable: not one function in that module knows a layer type, a
mixer, an expert or a model family. It is `Rank`, a publication event, and named
point-to-point hop shapes over dense blocks (`fanout_f32`/`_i32`, `gather_halves`,
`concat_halves_on_root`, `host_stage_block`, `host_row_to`, `return_block_to_root`,
`return_row_to_root`) plus an arm-time byte-integrity pull ladder. `pp.rs`'s `BoundarySlot` is
where the shape came from; hy3 and step TP are the consumers in sight.

- `Glm5TpTransport` → `TpTransport`.
- the movement census `GLM5_TP_*` → `TP_*` with its snapshot readers, because the census counts
  the TRANSPORT's hops. (Contrast with the `GLM5_EP_*` counters in `glm5_tp.rs`, which count
  the glm5 EP WALK and therefore keep their names — the line between the two is exactly the
  line between the seam and its consumer.)
- **the flag takes the VALUED law, not the boolean one.** `MEMRA_TP_TRANSPORT` primary,
  `MEMRA_GLM5_TP_TRANSPORT` alias, and a disagreeing pair **REFUSES THE LOAD** naming both
  (`tp_transport::resolve_transport`). Why this differs from §2.2/§2.3's fall-closed, written
  where the code is: falling closed is only correct for a door read PER CALL inside the round,
  where an abort would kill every live session. A transport is resolved ONCE at arm time,
  before any session exists — so a refused load is cheap and a load that silently picked a
  transport the operator did not choose is the expensive outcome. `transport_env()` returns the
  ARMED NAME, so the parse refusal and BOTH peer-access grant labels cite the flag the operator
  typed.
- **receipt tags are the caller's** (phase 1's `[glm5-phase]` rule): `arm_transport` and
  `transport_census_line` take a `tag`; glm5 passes `GLM5_TP_TRANSPORT_TAG` = `glm5-tp-transport`.
  All four announce/census/ladder lines the tp-transport lane banked hours earlier keep their
  exact bytes, and a second family gets its own marker instead of inheriting glm5's.
- glm5 keeps: the flag NAME it documents, the tag, the rank geometry, the owner-first law, the
  shard maps, and `same_device_gate`.
- gate arms: **XG** (peer-pull armed through the GENERAL name with the alias UNSET, `peer_pulls`
  must be > 0) and **XD** (disagreeing pair must refuse the load naming both). The existing
  X0/X1/X2/X3/XT/XF keep driving the alias, which is the banked-receipt path.

Deliberately NOT renamed: `MEMRA_GLM5_TP` itself. Phase 1 classifies shard maps as (c) by owner
call, and the flag's value IS a shard map (`all@0,1`).

### 2.8 Item 8a — two lane artifacts become first-class fleet tools

`research/glm53-flash-bringup-20260827/tp-transport-20260901/HEALTH.sh` → `tools/box-health.sh`
and `.../peer-read-probe.cu` → `tools/peer-read-probe.cu`. Neither contains anything glm5, TP,
or even memra: every check is a host/driver/fabric fact that silently degrades ANY workload
(a persistent 400-of-600 W cap costing 25.3% of dense prefill; the false-600W ~600 MHz
degradation; a PCIe link at Gen2 x16 that ran 3.5 h of production undetected; a 256 MB BAR1; an
out-of-range CPU affinity mask; IOMMU translated mode; ACS ReqRedir; P-state normalization).

Promotion is more than `git mv`, and this is the part that makes it a tool rather than a moved
file:

- the NAME says what it checks (`HEALTH.sh` in a lane dir tells you nothing at a fleet prompt);
- a real CONTRACT in the header: usage, OUTDIR-is-the-receipt, exit 0 = fit to measure /
  exit 1 = do not open the window, and WARN semantics stated (facts to RECORD beside every
  number from the box, not things to fix — IOMMU mode and ACS on a VM);
- the DEPENDENCY list with its teeth: a missing `nvcc` is a HARD-FAIL, never a skip, because
  section 8 is the only check that catches the driver staging SM-issued peer access through
  system memory — the check that matters most is the one easiest to silently lose;
- an explicit note that it reads only files, `/proc` and `nvidia-smi`, so a SCHEDULED runner can
  call it (the scheduled-jobs-must-not-depend-on-interactive-state law);
- `docs/TESTING.md` gains **"Box health before measurement"**, placed immediately before the
  multi-GPU section: what the tool answers, the degradation cases, the `nvcc` build line for the
  probe, the probe's four exit codes, why `-p2p a` = `NS` is EXPECTED on SM120, and why `ncu` is
  deliberately absent (profiling every rank deadlocks). A tool nobody can find is a tool nobody
  runs.

Engine comments that pointed at the lane paths now point at `tools/`.


---

### 2.9 Item 6 — `MEMRA_WORKER_AFFINITY` → `MEMRA_WORKER_CPUSET`

`lane/glm5-host-audit` landed on main while this lane's fourth battery was finishing, so the last
blocked rename executed too. Coordinator ruling: `*_CPUSET` is the convention, and the reason is
not cosmetic — the VALUE is a cpuset spec (`8-15,136-143`, `ccx:0`), so "affinity" named the
mechanism rather than the thing being set.

Nothing about the door was ever family-specific: it pins the ONE `memra-gpu-worker` thread that
every family's decode runs on. `MEMRA_WORKER_CPUSET` primary, `MEMRA_WORKER_AFFINITY` honored.

- **Valued, resolved once on the STARTUP thread → refuse-at-boot**, the third instance of that
  shape in this lane (`MEMRA_EP_MAP`, `MEMRA_TP_TRANSPORT`, now this) and deliberately not the
  per-call doors' fall-closed: no request exists yet, and a server silently pinned to a cpuset
  the operator did not choose is worse than a boot that says why it stopped.
- `parse_affinity` → `parse_affinity_named(flag, ..)`: every one of its five refusal messages
  now interpolates the ARMED flag name, so a banked script that sets the alias reads its own
  name back instead of being told about a flag it never set.
- **Receipt lines byte-unchanged.** `affinity-identity-gate.sh` and the host-audit window runner
  grep `[worker-affinity]` and `[worker-affinity] off`; both are untouched, as are the `engaged
  … effective=` readback and `REFUSED` lines. The ONE change is the OFF line's parenthetical,
  which now reads `(MEMRA_WORKER_CPUSET / MEMRA_WORKER_AFFINITY unset or =0)` — checked against
  every consumer first, and nothing greps past `off`. A line that named only one of two live
  flags would be a small lie in a receipt.
- two new unit tests (either-name-honored with the armed name returned; disagreeing pair refuses
  naming both), and the 5 pre-existing parser tests re-pointed at the general name.

---

## 3. AUDITS

### 3.1 Items 3 + 7 — door D device-tables, the vrows MoE pair, and the dedup doors

VERDICT: **generic in the plumbing, no glm5 coupling.** Evidence, not assertion:

- No symbol anywhere in `crates/` matches a glm5-and-vrows/moe/dedup/tables/order/tmaj name
  pattern (`fn *glm5*vrows*` etc.): zero hits.
- Kernel names are all family-free: `moe_vrows_tables_from_sel`, `moe_vrows_order_from_sel`,
  `moe_gate_up_preclamp8_q8_rows{,_ord}`, `moe_down8_fma_q8_rows{,_tmaj}`.
- Flags, statics and readers are all `MEMRA_MOE_VROWS_*` / `MOE_VROWS_*`.
- The device-table build hangs off `Engine`, not off a glm5 type; door H's
  `Engine::i32_mirror_store` and `Engine::shexp_ones` likewise.

RESIDUE EXTRACTED (three doc lines that named glm5 for what is an arithmetic or plan
property — this is exactly what "extract residue" means for a naming audit):

1. `moe_gate_up_preclamp8_q8` was documented as "glm5_next's PRE-clamped … twin". It is the
   kernel class for **any** MoE family whose activation clamps the gate before the silu; the
   door names the arithmetic, not the family. Reworded, glm5_next named as the first such
   family.
2. `Engine::shexp_ones` cited `MEMRA_GLM5_HTOD_DIET` and read as a GLM-5.3-Flash fact; now
   cites `MEMRA_HTOD_DIET` and states the plan property (`no ffn_gate_inp_shexp`) with the
   glm5 count kept as the measured instance.
3. `Engine::i32_mirror_store` cited `MEMRA_GLM5_HTOD_DIET`; now `MEMRA_HTOD_DIET`.

The remaining `lane/glm5-*` mentions in those blocks are **provenance citations** (which lane
measured it) and stay — phase 1's rule: cite the source, and provenance is not coupling.

### 3.2 Item 9 — phase-1 alias drift check on the current head

Several merges landed on bringup since phase 1 (the moe-loc chain, ep-diet, dedup, door-r, the
bankfix consolidation, the 1M battery). **No drift: all five phase-1 extractions hold.**

| phase-1 extraction | still wired at `89d164ae3`? |
|---|---|
| `MEMRA_VERIFY_WS` + `MEMRA_GLM5_VERIFY_WS` alias, OFF-wins | yes — `verify_ws_on_from` intact, both names read, FLAGS rows present, door-W gates still pin the alias |
| `MEMRA_SPEC_TRACE` + `MEMRA_GLM5_SPEC_TRACE` alias, general-wins-loudly | yes — `spec_phase.rs` resolver reads both, the loud override line intact, glm5 passes its own `[glm5-phase]` tags |
| `MEMRA_EP_MAP` + `MEMRA_GLM5_EP_MAP` alias, refuse-at-load | yes — `ep_map::resolve_ep_map_env` intact; `glm5_tp.rs` + `hybrid.rs` consume it; `glm5-tp-gate` arms H1–H5/R4/M drive the alias |
| `tp::refuse_door_composition` | yes — glm5 delegates through `refuse_glm5_tp_door_composition` with its own table; `tp.rs` carries the format unit tests |
| `spec::spec_conf_keep` | yes — re-exported as `glm5_conf_keep`, call site and the 10 unit assertions intact |

### 3.4 Item 8b — `apply_penalties_dense`

VERDICT: **already general, no move needed** — and the strongest possible form of that verdict,
because it is structural rather than an inspection result. `memra-sampling`'s `[dependencies]`
section is **EMPTY**: the crate cannot name a family type, an engine type, or a CUDA type, so
family coupling is impossible by construction rather than absent by luck. `apply_penalties_dense`
is a private method over `&mut [(u32, f32)]` candidate slices. The three family strings in the
whole file are provenance citations (which lane measured the host cost, and a Qwen vocab size as
the worked example of ~152k SipHash probes per token). Nothing to extract.

### 3.3 Item 8 (superseded by §3.4) — `apply_penalties_dense`

Not on the base at all: it exists only in `lane/glm5-host-audit`'s
`crates/memra-sampling/src/lib.rs`. The classification the task asked for can be stated from
that lane's content — it is a dense-logit penalty apply in `memra-sampling`, a crate with no
family types at all, so it is family-agnostic **by crate boundary**, not by inspection. But
"confirm no glm5 coupling" against code that is not in the tree would be a claim about a diff,
not a gate on a tree. It is deferred to §4.4 with the rest of that lane's items.

---

## 4. PLANNED — recipes precise enough to be a mechanical apply

### 4.1 Item 4 — EXECUTED, see §2.7

Its lane landed mid-lane and was absorbed. The recipe that was written here (valued-flag alias
shape, seam hoist, gate twins, FLAGS row in the same commit) was followed as written; §2.7 is
what actually shipped.

### 4.2 Item 6 — EXECUTED, see §2.9

Its lane landed on main mid-lane. The recipe written here (valued-flag alias shape refusing at
boot, FLAGS row in the same commit) was followed as written.

### 4.3 Item 8a — EXECUTED, see §2.8

### 4.4 Item 8b — VERIFIED, see §3.4

### 4.5 Item C — the batched verify walk

Phase 1 deferred this to "the second hybrid spec family". That trigger is **still not met**:
the hy3 pack landed on this base, but a pack is a plan + tensor contract, not a spec walk —
there is no second family with a verify walk to generalize against. Cutting the contract now
would fix the shape to glm5's mixer dispatch, which is the opposite of the goal. The
substantive prep that IS worth doing when the trigger fires is recorded in phase 1's row #8
and needs nothing from this lane.

### 4.6 Item 1 remainder — the per-session DraftSource trait

The model-level and selection halves are executed (§2.4). The per-session half is deferred,
and here is the reason in code terms rather than as a scheduling excuse:

`Glm5DraftState`'s two arms are not family-agnostic. `NativeMtp` maintenance calls
`glm5_mtp_plane_reset` + `glm5_seed_row` against the MLA latent plane; `Dflash2` maintenance
drains `sess.cache.hc_taps` (the hc-contract tap sink) through `glm5_tap_drain`; the retained
draft-q type carries the family's rank space and its filtered per-slot stats; and the round's
rollback interacts with the KDA scan stash. A trait over that has exactly ONE implementor
whose every associated type is a glm5 type — a decorative cut, made blind, on the hottest
file in the lane program (`glm_spec.rs` / `hybrid_forward.rs` / `worker.rs` are in three
in-flight lanes' diffs).

The sketch the second consumer should start from, so it is not re-derived:

```rust
// engine-level, once a second family's session state exists to design against
pub trait DraftRounds {
    /// The family's per-session draft state (glm5: Glm5DraftState).
    type State;
    /// The family's retained draft-q for one round (glm5: Glm5DraftQ).
    type DraftQ;
    /// The family's session/cache handle.
    type Session;

    /// Which source this session is pinned to — resolved ONCE through the already-general
    /// `spec::resolve_draft_source_kind`, so this is a getter, not a policy.
    fn kind(&self, state: &Self::State) -> spec::DraftSourceKind;

    /// ONE round's drafts. `k` is the SHARED K policy (`spec::choose_spec_k` /
    /// `MEMRA_SPEC_K`) and the confidence break is the SHARED `spec::spec_conf_keep` —
    /// both already general, which is why they are parameters here and not trait items.
    fn round_drafts(
        &self, sess: &mut Self::Session, state: &mut Self::State,
        k: usize, sp: Option<&spec::SpecSampling>, p_min: f32, pmin0: bool,
    ) -> Res<(Vec<u32>, Self::DraftQ)>;

    /// Source-keyed state maintenance after the trunk rolled back to `keep`.
    fn maintain(
        &self, sess: &mut Self::Session, state: &mut Self::State,
        keep: usize, round_tokens: &[u32],
    ) -> Res<()>;
}
```

What is ALREADY shared and must NOT be re-invented inside that trait: the accept/rollback
arbitration, `spec::spec_conf_keep`, the `MEMRA_SPEC_K` / `MEMRA_SPEC_PMIN` / `PMIN0` surface,
`spec::sample_boundary_token`, the `spec_phase.rs` timers, `dflash::DflashDrafter` /
`load_drafter` / `resolve_tap_layers`, and `spec::resolve_draft_source_kind`. After this lane,
a second family's spec bring-up writes only `round_drafts` and `maintain`.

---

## 5. THE ALIAS LIST (cumulative, after phase 2)

| general name | family alias (honored) | composition on a disagreeing pair | shape |
|---|---|---|---|
| `MEMRA_VERIFY_WS` | `MEMRA_GLM5_VERIFY_WS` | OFF wins (either `=0` disables) | bool, default ON (phase 1) |
| `MEMRA_SPEC_TRACE` | `MEMRA_GLM5_SPEC_TRACE` | general wins, one loud `[spec-trace]` line | levelled (phase 1) |
| `MEMRA_EP_MAP` | `MEMRA_GLM5_EP_MAP` | **refuses at load**, naming both | valued path (phase 1) |
| `MEMRA_HTOD_DIET` | `MEMRA_GLM5_HTOD_DIET` | **falls closed**, one loud `[flag-alias]` line naming both | bool, default OFF, per call (phase 2) |
| `MEMRA_EP_DIET` | `MEMRA_GLM5_EP_DIET` | **falls closed**, one loud `[flag-alias]` line naming both | bool, default OFF, per call (phase 2) |
| `MEMRA_EP_GROUPED_PRIME` | `MEMRA_GLM5_EP_GROUPED_PRIME` | **falls closed**, one loud `[flag-alias]` line naming both | bool, default OFF, per call (phase 2) |
| `MEMRA_TP_TRANSPORT` | `MEMRA_GLM5_TP_TRANSPORT` | **refuses the load**, naming both | valued, resolved once at arm time (phase 2) |
| `MEMRA_WORKER_CPUSET` | `MEMRA_WORKER_AFFINITY` | **refuses the boot**, naming both | valued, resolved once on the startup thread (phase 2) |

**The shape of the law follows the READ, not the flag's importance** — one line, because getting
this backwards is how a future rename goes wrong: a door read PER CALL inside the round falls
closed (an abort in the GPU worker thread exits the process and kills every live session); a
valued seam resolved ONCE at arm time refuses the load (no session exists yet, and a silently
substituted transport/map is worse than a refused boot).

Renamed non-flag symbols (phase 2): `Glm5DflashDrafter` → `dflash::DflashDrafter` (old name
re-exported); `glm5_htod_diet_on` → `htod_diet_on`; `glm5_htod_diet_avoided` →
`htod_diet_avoided`; `GLM5_HTOD_DIET_AVOIDED` → `HTOD_DIET_AVOIDED`; `glm5_ep_diet_on` →
`ep_diet_on`; `glm5_ep_grouped_prime_on` → `ep_grouped_prime_on`; `hybrid::sha256_file_hex8`
now `pub(crate)`; module `glm5_tp_transport` → `tp_transport`; `Glm5TpTransport` →
`TpTransport`; the movement census `GLM5_TP_*` → `TP_*` with its readers; files
`research/.../HEALTH.sh` → `tools/box-health.sh` and `research/.../peer-read-probe.cu` →
`tools/peer-read-probe.cu`.

Deliberately KEEPING their glm5 names, with the reason, so nobody "finishes the job" wrongly:
the `GLM5_EP_*` engagement counters in `glm5_tp.rs` (they count the glm5 EP WALK, the consumer,
not the seam — that is exactly the seam/consumer line); the `[glm5-tp-transport]` and
`[glm5-phase]` receipt TAGS (caller-owned by law, so banked receipts keep their bytes);
`MEMRA_GLM5_TP` (its value IS a shard map, and phase 1 classifies shard maps as (c) by owner
call); `MEMRA_GLM5_SPEC` / `_MTP` / `_VISION*` (family route + family weights); every
`_GATE_RED` / `_GATE_SAME_DEV` (gate instruments, never serving flags).

Every OFF arm of every new door stays pinned `=0` (never unset) in the gate bins, per the
vacuous-arm law.

---

## 6. GATE TABLE

Runner: `run-battery.sh` in this directory (phase 1's, plus the two suites that landed since
and the alias-coverage note). Runner law upheld: `--include-ignored` everywhere, and a green
suite reporting `0 passed` counts as **FAILED**. Rig discipline: `flock /tmp/memra-5090.lock`,
`NVIDIA_TF32_OVERRIDE=0`, exactness only (no timing claims from this card). The lock was
verified clear before launch (`/proc/locks` carried no entry for the lockfile's inode, no
`sccache` fd-holder) and `sccache --start-server` was pre-warmed OUTSIDE the lock, per the
phase-1 sccache-flock finding.

**TEN FULL RUNS**, each from scratch rather than spot-checked, because every trigger touched
a hot file:

| run | tree | why re-run | receipts |
|---|---|---|---|
| 1 | `e05dab650` | the §2.1-§2.6 flag + draft-source work | `receipts-run1/`, `battery-run1.console` |
| 2 | `fd444a76f` | tp-transport landed on bringup and was absorbed (`ba8274271`); items 4 + 8a executed on top, and the transport rename MOVED A MODULE and touched `hybrid_forward.rs` + `glm5_tp.rs` | `receipts-run2/`, `battery-run2.console` |
| 3 | `55647f30b` | **`origin/main` absorbed** (step37 bank-v3 + hy3 c1/native-tune), with a real content×shape conflict resolved in `hybrid_forward.rs`'s device-routed EP arm | `receipts-run3/`, `battery-run3.console` |
| 4 | `fec2d762c` | **`origin/main` absorbed again** — the coordinator's glm53 bringup→main merge landed, so main now carries this lane's base and the merge-to-main became a fast-forward | `receipts-run4/`, `battery-run4.console` |
| 5 | `909e0c221` | **host-audit landed on main and was absorbed** — items 6 + 8b executed on top (the last two blocked items) | `receipts-run5/`, `battery-run5.console` |
| 6 | `018eeb74f` | **`origin/main` absorbed a sixth time** (nvfp4 quad-symbol) — it touched `qmatvec.cu`, which carries the very vrows kernels doors D/E/M gate, so the door suites had to re-run rather than be reasoned about | `receipts-run6/`, `battery-run6.console` |
| 7 | `146ed3aa5` | **the PEER-REVIEW fixes** (§8) — two real defects plus the hygiene set, including a change to the glm5 session's draft-source selection, so nothing was taken on reasoning | `receipts-run7/`, `battery-run7.console` |
| 8 | `272b2cce9` | **the VERIFICATION-PASS fixes** (§8.1) — the door-H clear moved into the three matrix gate binaries, so the gates themselves changed | `receipts-run8/`, `battery-run8.console` |
| 9 | `9a522b642` | **`origin/main` absorbed a sixth time** — and this one RANK-WIDENED the extracted module itself (§9), so the transport was replaced wholesale and re-layered | `receipts-run9/`, `battery-run9.console` |
| 10 | `4ef756b73` | **`origin/main` absorbed a seventh time** (v0.123.0 + lane/glm5-accrace) — all engine code auto-merged, but it landed +113 in `glm_spec.rs`, 46 in `hybrid_forward.rs` and 177 in `glm5_spec_ppn_gate.rs`, i.e. the files carrying this lane's draft-source seam, door readers and door-H clear | `receipts/`, `battery.console` |

All ten verdict lines read `extract2 battery: ALL GATES PASS`. The table below is RUN 10 — the
exact tree PR #77 carries. Runs 3 through 6 are the load-bearing ones for the MERGE claim: they are the runs that gate this lane's
extractions against main's newly landed engine work, which is what makes that claim a measurement
rather than an expectation. main moved SEVEN times during this lane (bringup; host-audit; step37
bank-v3; hy3 c1/native-tune; nvfp4 quad-symbol; glm5-spec-tp + glm5-composition + down8-default-ON
+ qwen4exp; v0.123.0 + glm5-accrace) and **every absorb was re-gated rather than assumed** — which
is the whole reason there are ten runs and not one. Two of those absorbs were not routine: the
sixth rank-widened the extracted module itself (§9) and the seventh landed in the three files
carrying this lane's seam, readers and clear. **Runs 7 and 8 are not main moves** — they gate the two review rounds (§8, §8.1). Run 7 covered a
change to the glm5 session's draft-source selection; run 8 covered a change to three gate
binaries. Run 8 gates the SHIPPED code.

**The one non-green line in four runs, and why it is not a finding.** Run 4's perf stage returned
`qwen9b-plain-short: 135.98 tok/s [WARN] — -1.60% vs median 138.19` (`perf stage: 0 fail, 1 warn`,
exit 0, correctness stage GREEN). The perf stage is a tripwire, not evidence, and a single rig
number is never a verdict — but "probably noise" is not an answer either, so it was settled by
DIFF rather than by re-measuring: `git diff 55647f30b..fec2d762c -- crates/` is **one line, and it
is a comment** (an `origin/main's hy3` → `main's hy3` reword). Zero non-comment, non-whitespace
lines changed between the 138.73 run and the 135.98 run, so no executable byte differs and the
delta cannot have been caused by this lane. The four rows across the lane — 137.83 / 138.19 /
138.73 / 135.98 — span 2.0%, which is the rig's own spread on a throttling laptop 5090
(LAW:rig-exactness-only: this card yields exactness, never timing claims).

| gate | invocation | verdict |
|---|---|---|
| 30 GPU suites, default (ship) arm | `cargo test -p memra-engine --test <s> -- --include-ignored --test-threads=1` under flock + TF32-off; phase-1's 27 + the two that landed since (`glm5_ep_diet_doors_gpu` 3/3, `glm5_dedup_sched_gpu` 6/6) + `glm5_vision_gpu` with the real shard 2/2 | **ALL GREEN, all non-zero** (2..18 tests each; receipts `receipts/<suite>.log`). The renamed-door suites: `glm5_moe_loc_doors_gpu` 4/4 (door D + door H through `htod_diet_*`), `glm5_matvec_doors_gpu` 5/5, `glm5_spec_session_gpu` 10/10 (door W through the phase-1 pool), `glm5_dflash_session_gpu` 10/10 (the whole DFlash2 draft source through the hoisted `load_drafter` + `resolve_tap_layers`, red arm included) |
| door-H ALIAS arms (explicit prints banked) | same suite re-run `--nocapture` (`receipts/door-h-alias-nocapture.log`) | `door H len_d PASS: both arms land the value, counter anchored ON / flat OFF` · `door H alias PASS: general name engages, disagreeing pair falls closed with the value still landing` · exactly ONE `[flag-alias] MEMRA_HTOD_DIET="1" and MEMRA_GLM5_HTOD_DIET="0" disagree …` line |
| glm5-tp-gate P=16 N=12 | `cargo run -q --release -p memra-engine --bin glm5-tp-gate -- 16 12`, flock + TF32-off (`receipts/tp-gate-p16-n12.log`, exit=0) | **ALL ARMS PASS** — now over BOTH fixtures main added (`TP-2` kda2/mla2/4-expert and the `TP-4` quad kda4/mla4/8-expert, same-device multi-context emulation). The EP-door arms are each other's red: **B2G** `MEMRA_EP_DIET=1` with the alias UNSET → diet layer-calls **207**, decode BYTE-IDENTICAL to plain (28 t=1 steps × 32 logits + tape), prime band 3.637e-5 (band 2e-4); **B2X** `MEMRA_EP_DIET=1` + `MEMRA_GLM5_EP_DIET=0` → diet layer-calls **0** with one `[flag-alias]` line. Same assertion shape, 207 vs 0 — the arms are non-vacuous by construction, not by inspection. B2 (alias) 207/152/225 counters unchanged, B3 grouped-prime still falls closed on the Q8_0 bank (dispatches 0), M2/R2D/R3D unchanged, H1–H5/R4/M still drive `MEMRA_GLM5_EP_MAP` with refusal bytes unchanged |
| glm5-tp-gate TRANSPORT arms (item 4) | same invocation | **X0** pin held: `peer_pulls=0` across the whole `=0` battery with `host_legs=49967 host_syncs=26732` (a pin that is not holding reads exactly like a passing gate otherwise) · **X1** engagement `peer_pulls=1674 host_legs=0 host_syncs=0 pub_events=6696` · **XT** transport-vs-transport BYTE-IDENTICAL at decode AND prime, `max_rel=0.000e0` exactly — the transport moves bytes, it does not compute · **XF** unknown spelling refuses by name · **XG** (new) `MEMRA_TP_TRANSPORT=peer-pull` with the alias UNSET → `peer_pulls=1674`, decode byte-identical to plain, prime band 3.637e-5 · **XD** (new) disagreeing pair REFUSES THE LOAD with both names in the message |
| ppn matrix, 9 arms | `glm5-spec-ppn-gate` 2/3-stage set, incl. the `MEMRA_SPEC_TRACE=2` twin | **24/24 PASS lines per arm, exit=0 every arm.** Phase-1 drift re-proved: the spec-trace twin still engages the moved timers — **20** `[glm5-phase]`/`[glm5-phase-v]` lines vs **0** on the control arm, tag bytes unchanged |
| hppn matrix, 10 arms | `glm5-hyper-ppn-gate` banked set | 6/6 PASS lines per arm, exit=0 every arm |
| hbatch matrix, 10 arms | `glm5-hyper-batch-gate` banked ladder | 3/3 PASS lines per arm, exit=0 every arm |
| memra-server suite | `cargo test -q --release -p memra-server` (`receipts/memra-server-suite.log`) | **517/517** (phase 1 banked 492; main's own work plus this lane's 3 cpuset-alias tests account for the rest) — plus the same two empty 0-test targets, the ep-place shape |
| engine lib units | `cargo test -p memra-engine --lib` under flock+TF32-off (`receipts/engine-lib-units.log`) | **325 passed / 2 ignored**. This lane added **17** unit tests: 3 `alias_door_from`, 3 `resolve_tap_layers`, 5 transport alias, 3 cpuset alias (incl. the two-spellings-of-off arm), 3 `resolve_draft_source_kind` — the last three added at review, because the function had shipped untested (phase 1: 275) — includes this lane's 11 new: three `alias_door_from` cases (default/arming, agreeing pair, disagreeing pair naming both), three `resolve_tap_layers` cases (resolution, caller-owned shift, empty + out-of-trunk + SHIFTED-tap refusals), and five transport cases (the parse refusal asserted under BOTH flag names, plus alias resolution across unset/either/agreeing/disagreeing — including the deliberately literal case that `"0"` vs `"host-canonical"` is a DISAGREEMENT rather than a normalize-then-compare, so an operator who typed two things gets told) |
| check-flags | `tools/check-flags.sh` | green — **772 runtime literal reads, no uncovered names, no grandfather list**; all EIGHT general/alias pairs covered |
| clippy | `cargo clippy --workspace --all-targets` | **zero warnings** — the main merge surfaced 12 in main's newly landed code under this toolchain (6 manual-`is_multiple_of` in `lib.rs`, 6 redundant `u64`->`u64` casts in `tp.rs` where `device_ptr` already returns `u64`); fixed in-lane rather than disclaimed as someone else's |
| fmt | `cargo fmt --all --check` | clean |
| local-ci --perf | `tools/local-ci.sh --perf` (`receipts/local-ci-perf.log`, exit=0) | correctness stage **GREEN** (kernel-check 107 cells, decode-batch-gate 4 configs, graph-warmup-stress + canary, serve-stress c=64, spec-on-cache-hit ×2, drafter-attach wiring gate); perf stage **0 fail 0 warn**, qwen9b-plain-short **138.22 tok/s [OK]** vs the rolling median. Across ten runs: 137.83 / 138.19 / 138.73 / 135.98[WARN, settled by diff] / 138.45 / 138.18 / 137.41 / 137.56 / 138.22 / 138.22 — plus one **[FAIL] 132.85 that is NOT in that list and is named here rather than dropped**: run 9's first perf attempt hit a CONTENDED window (another lane's `local-ci` serve-stress holding 21.4 GB of the card, load average 32). The runner detected it, waited 600 s, retried, recorded `window_clean=false` and failed the cell — its own hygiene working. Unlike the earlier WARN this one could NOT be settled by diff, because real code had changed (main's rank-widening), so it was settled the only honest way: re-measured on a clean window once the co-resident left, giving the 138.22 [OK] above. A 2.0% spread on a throttling laptop card is why no timing claim is ever made from it (LAW:rig-exactness-only) |
| byte identity where a rename touches a hot path | — | the flag renames are READ-SITE ONLY (env name → same value → same branch), so identity is structural; the transport rename additionally MOVED a module, and that is the one place a rename could have moved bytes — so it carries the direct proof: **XT compares the two transports to each other bit for bit at decode AND prime and reads `max_rel=0.000e0`**. Plus tp-gate B2/B2G/XG decode arms byte-identical to plain, door D/H arms bitwise, and the 30-suite battery's every standing exactness bar |

Rig discipline held throughout: exactness only (no timing row is claimed from this card), and the
battery correctly QUEUED behind another lane's `tools/local-ci.sh --perf` on the rig flock rather
than running lock-less — the lock was in legitimate use, not wedged (holder alive, no `sccache`
fd-holder), so nothing was cleared.

---

## 7. INTEGRATION MAP UPDATE (for the bringup → main merge)

Phase 1's §3 integration map gains, in the **General engine features** set:

- the flag-alias law for boolean doors (`alias_door_from`) — the fourth and last door shape,
  so a future rename has a law to follow instead of a hand-roll;
- `MEMRA_HTOD_DIET` — engine-generic HtoD hygiene (resident ones buffer + async i32 mirror);
- `MEMRA_EP_DIET` / `MEMRA_EP_GROUPED_PRIME` — general EP dispatch/prime doors;
- the draft-source seam: `dflash::DflashDrafter`, `dflash::load_drafter`,
  `dflash::resolve_tap_layers`, `spec::DraftSourceKind` + `resolve_draft_source_kind`;
- `memra-gguf::placement` — pure multi-GPU stage placement (already general on this base, but
  it is new since phase 1 wrote its map, so it belongs in the reviewable general set);
- **the TP transport seam** (`tp_transport.rs`, `MEMRA_TP_TRANSPORT`): `Rank`, the publication
  link, the named hop shapes, the arm-time byte-integrity pull ladder, and the movement census —
  reviewable as an engine capability with no glm5 acceptance needed, and the XT arm proves the
  arms are byte-identical to each other;
- **two fleet tools**: `tools/box-health.sh` and `tools/peer-read-probe.cu`, with the
  `docs/TESTING.md` "Box health before measurement" section. These are the most immediately
  reusable things in the whole merge: any box window, any family, any lane;
- `MEMRA_WORKER_CPUSET` — the worker-thread cpuset pin, which pins the one thread every family's
  decode runs on and was never family-specific.

And in **glm5 model-support set**: unchanged in substance — the glm5 TP-2 EP walk, its
counters, the shard maps, the session state, and every `MEMRA_GLM5_*` family flag phase 1
named as not-a-target. Support state after merge is still NativeReference-class until the
real-artifact qualification lane banks its receipts; **this merge does not move the darklanes
roster.**

**Main should get FIRST, independent of this merge** (added to phase 1's list):

- the flag-alias law itself — main has renames coming from every lane, and three hand-rolled
  compositions in one file is how the fourth one gets it wrong;
- `memra-gguf::placement` (pure, no CUDA, no family) is cherry-pickable today.

**Nothing is left blocked.** Both lanes this doc opened by reporting as unlanded (tp-transport,
host-audit) landed while this lane ran, and both were absorbed and executed on top rather than
left as recipes — §2.7/§2.8 and §2.9/§3.4. The general set above is complete as written.

---

## 8. PEER REVIEW (PR #77) AND WHAT IT CHANGED

An independent reviewer went at the zero-behavior-change claim with the brief to break it.
Verdict: **no BLOCKER**; the claim held on every mechanically checkable path, and the riskiest
item — the `hybrid_forward.rs` merge resolution — was confirmed clean by a whole-file diff
against `origin/main` (nothing but module-path renames), including the two-constants trap
(`NVFP4_EP_DEVICE_ROUTER_BATCH_CAP` at the guard, `NVFP4_EP_DEVICE_BATCH_CAP` at the capacity)
and the untouched non-device-routed announce site.

It found **two real defects**, both fixed:

1. **`tp_transport.rs` — the byte-integrity ladder refusal was the ONE refusal site never
   parameterized.** It hardcoded `MEMRA_GLM5_TP_TRANSPORT` in both the message and the printed
   rollback, on a path reachable with only the general name set (the XG arm proves that call
   shape). Following its own advice — set the alias to `0` while the general name is
   `peer-pull` — produces a disagreeing pair, which `resolve_transport` then refuses: a fabric
   failure turned into a second, unrelated refusal. It also contradicted §2.7's own claim of
   "both peer-access grant labels", when there are three refusal sites. Now `{armed_flag}`.
2. **`tools/box-health.sh` — `set -e` was switched ON mid-script**, after §2.8 gave the tool a
   formal exit-code contract. The script runs `set -u -o pipefail` and never `set -e`, so
   section 8's `set +e … set -e` *enabled* errexit for sections 9-10: a hard-failing box could
   exit non-zero with no HARD-FAIL recorded, no summary line, and no "DO NOT OPEN THE WINDOW" —
   breaking the contract in the same commit that wrote it. (Also `set -e 2>/dev/null || true` is
   a no-op guard; `set -e` cannot fail.) Removed, with the reason in the code.

And a set of honesty/hygiene findings, all fixed rather than argued with:

- **Over-claimed generality.** Three FLAGS.md rows said hy3/step "arm the same door" in the
  present tense; there is exactly one consumer of each today. Put in the honest tense, and the
  silent case named: the armed-but-inert co-refusal is scoped to glm5-class plans, so setting an
  EP door on a non-glm5 deployment does nothing and says nothing — and after the verification
  pass caught that the note had landed in only ONE of the three rows, on all three (the transport
  row for the same reason by a different mechanism: `transport_env()` has exactly one caller).
- **Stale pointers in the row I wrote**: `glm5_tp_transport::parse_transport` (renamed in the
  same PR), and receipts pointing at `HEALTH.sh` / `peer-read-probe.cu` at paths this PR
  emptied.
- **`tools/README.md` says "when you add a tool, add its line"; this PR added two and indexed
  neither** — while §2.8 argued at length that a tool nobody can find is a tool nobody runs.
  Both indexed.
- **The rename did not reach three doc comments** (`memra-server/src/lib.rs`,
  `worker.rs`, `glm5_tp.rs`) that still presented the alias as the flag's name.
- **The draft-source seam was decoration in release.** `source_kind` was consumed only by a
  `debug_assert!`, which is compiled out of every gate and prod binary, so the walk still
  re-derived the answer locally — and the assertion was structurally unfalsifiable. The session
  state now keys on the RESOLVED KIND, so the seam is load-bearing where it ships, with the
  impossible mix refusing loudly instead of silently drafting from the MTP plane. Its refusal
  message was also self-contradictory ("requires a draft source … [a head IS loaded]"); recomposed
  to be true on both branches.
- **`resolve_draft_source_kind` shipped untested**, including a branch unreachable on glm5. Three
  tests added — an unreachable refusal with no arm is an untested refusal, and the next family is
  what makes it reachable.
- **An env leak on the panic path** in the door-H disagreement arm (two keys left armed for every
  later test in a `--test-threads=1` file). Replaced with a `Drop`-scoped multi-key guard.
- **Door H's general name was cleared nowhere**, so a leaked `MEMRA_HTOD_DIET=0` would have made
  the shell-driven door-H ON arms vacuous — they assert bit-identity, which passes either way.
  My first fix put the clear in `glm5_tp_gate`, which is the one gate that never arms door H at
  all; the verification pass caught that and named the four surfaces that ARE exposed. The clear
  now sits in the three gates whose matrix runners drive `MEMRA_GLM5_HTOD_DIET=1` from the shell
  (`glm5_spec_ppn_gate`, `glm5_hyper_ppn_gate`, `glm5_hyper_batch_gate`), and it clears only the
  GENERAL name so the caller's alias cannot be outvoted. The box script
  (`struct-battery-20260831/box/c1_dh.sh`) drives a serving binary and is out of this lane's
  reach; it is named here so the next box window sets both names or neither.
- **A cost claim slightly stronger than the code**: "read-site only" is true of the arithmetic,
  but honoring two names doubles the `env::var` count on a per-call door. Now stated, with the
  `OnceLock` trade named in advance for the day a door moves to a per-token site.
- **Affinity inherited the transport module's deliberate literalism** (`0` vs `off` is a
  disagreement) with neither a note nor a test, on a flag whose OFF vocabulary is far wider.
  Documented and pinned with an arm, because "surprising but intended" is what a later reader
  helpfully "fixes".
- **Merge archaeology in engine code** whose own diff disproved it, and a stale run count in this
  doc's gate table. Both corrected.

### 8.1 THE VERIFICATION PASS (second review of the fixes)

The reviewer was asked to verify the fixes rather than re-review, and it did not rubber-stamp
them. Both real defects verified fixed; four things were still wrong, and they are the more
interesting half of the review:

- **The ladder rollback advice was still capable of creating a disagreeing pair** — narrower than
  the original defect, not gone. With BOTH names set to the same value the armed name is the
  general one, so "roll back with `MEMRA_TP_TRANSPORT=0`" leaves the alias still asking for
  `peer-pull`, and that disagreement refuses the load. Advice that creates a second failure is
  worse than no advice. Fixed with `set_transport_names` / `rollback_advice`, which name every
  name the operator actually set — and the doc says why the armed name is not enough here when it
  is enough everywhere else.
- **My door-H clear landed in the wrong binary.** See the bullet above. The lesson is the one this
  lane keeps re-learning: I fixed the door I had open in front of me instead of the four the
  finding named, and my own added comment ("it is not armed anywhere in THIS gate") said so
  without my noticing.
- **The silent-case note landed in one of three rows** while §8 claimed it generally.
- **An unsourced cert line.** The gate table claimed "519 server" for
  `cargo test -p memra-server --lib` — a number in no receipt, credited to an invocation the
  battery never runs (phase 4 is the release package suite). The banked figure is **517**, and the
  neighbouring row still said 505 from an earlier run. This is exactly LAW:cert-lines-carry-invocations
  and it was in my own table: both rows now carry the invocation, the receipt path, and the banked
  number. Also corrected: "main moved SIX times" over a five-item list, and the claim that run 7
  gates against main's engine work (it does not — it gates the review fixes, which is a better
  reason, just a different one).

Verified-correct on the substantive fix: the reviewer traced `tap_layers` rather than assuming, and
confirmed `source_kind == Dflash2` ⟺ `dflash_src.is_some()` ⟺ `tap_layers.is_some()`, so every
reachable cell of the new three-way match takes the arm it took before — identical program — and
the new refusal arm is uninhabitable rather than reachable-and-wrong. It also noted a benefit I had
not claimed: dropping the `_` catch-all means a future third `DraftSourceKind` variant now fails to
compile instead of silently falling into the MTP arm.

Two changes no gate can cover, named rather than implied: the ladder message (the rig cannot make
the byte-integrity ladder fail — it is a compile-checked format string) and the new
`(Dflash2, _, _)` refusal arm (uninhabitable, so never executed).

---

## 9. THE SIXTH AND LAST MAIN ABSORB — and a clippy finding that is NOT this lane's to fix

`origin/main` moved again while the PR sat (`a71ee11c0`: lane/glm5-spec-tp, lane/glm5-composition,
down8 default-ON, qwen4exp). One of those, **lane/glm5-composition, rank-widened the very module
this lane renamed** — `Rank::Root`/`Rank::Peer` became plain indices, every hop shape takes
per-rank parts, the publication link carries one pub + one release event PER RANK, and the ladder
runs over every ordered rank pair. 39 conflict hunks across `tp_transport.rs` (21),
`hybrid_forward.rs` (12) and `glm5_tp.rs` (6).

**Resolution: take THEIRS wholesale, then re-apply the rename layer.** Resolving 39 hunks by hand
in a freshly-landed perf lane is how a merge silently loses a hop; the rename layer, by contrast,
is a closed and mechanical set — module path, `Glm5TpTransport` → `TpTransport`, the six census
counters and their readers, the flag alias resolver, the caller-owned tag, `set_transport_names`
/`rollback_advice`, and the door readers. So main's rank-widened semantics are in verbatim and this
lane's names sit on top. `glm5-tp-gate` also needed its four added arms re-shaped: main refactored
`tp_arm(..)` into `run_tp_arm(&cx2, ..)`, so B2G/B2X/XG/XD now take the shared harness context.

### The clippy finding, stated rather than disowned or absorbed

The merged tree carries **~200 clippy warning sites**, and the honest accounting matters:

- **Every one of them is in code byte-identical to `origin/main`.** ~180 are in files this branch
  does not touch at all (`qwen4exp_gpu.rs`, the qwen4exp pack and gates, `memra-reference`,
  `model_plan.rs`, `config.rs` — all arrived with main's qwen4exp lane). The other 16 are in the
  four files this lane does touch, but at lines that are main's verbatim: 13
  `needless_range_loop` in the new `for r in 1..ranks` rank walks (counts identical in both
  trees: 4 in `glm5_tp.rs`, 5 in `tp_transport.rs`), plus two `very_complex_type` signatures and
  one `into_owned` in `glm5_tp_gate.rs`, all three greppable in main's blob.
- **This lane introduces zero new warnings.** Its own 12-warning find earlier in the lane (main's
  `is_multiple_of`/redundant-cast set) WAS fixed in-lane, because it was 12 lines and mechanical.
- **~200 is a different thing, and rewriting index loops inside a just-gated perf lane to satisfy
  a style lint is a correctness risk for no benefit to this PR** — it would also bury an
  extraction review under an unrelated cleanup.

So per the no-pre-existing-disclaimer law's own alternative — *fix in-lane OR schedule a named
lane* — this is scheduled, not disowned: **`lane/clippy-zero-restore`**, scoped to main's qwen4exp
lane plus the 13 rank-walk loops, to be run against `origin/main` directly rather than through a
feature branch. Flagged to the owner because it means **`main` is currently clippy-dirty and CI is
not gating it**, which is the finding that outlives this PR.
