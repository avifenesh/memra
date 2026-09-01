# lane/glm5-extract-general — extract the generally-applicable machinery out of glm5 naming/seams

Owner order (verbatim): "everything that is relevant for general integration should be
extract out." Base: `origin/lane/glm53-flash-bringup` @ `a5d608b07` (the ep-place merge
head; the moe-loc merge chain had NOT landed on origin at branch time — its door D/H work
is classified below as in-flight and only PLANNED here). Scope of the audit:
`git diff origin/main...origin/lane/glm53-flash-bringup --stat -- crates/` =
163 files, +59,910 / -3,726.

Goal: the eventual bringup->main integration ships GENERAL engine capabilities plus a
glm5 model-support set — not a glm5 fork. glm5 stays a CONSUMER of every extracted seam.

## STEP 1 — the inventory

Classification classes:

- **(a) GENERAL already** — generic name, no glm5 coupling; lands on main as-is.
- **(b) GENERAL BUT GLM5-NAMED/GATED** — the extraction targets (table below).
- **(c) GLM5-SPECIFIC by construction** — the model-support set; lands as glm5 support.
- **(d) POLICY/DOCS** — capability tables, plan policy fields, doc rows.

Counts over the 163 changed crates/ files (a mixed file is counted once, by its dominant
class, with the minority class named in the notes):

| class | files | insertions (approx) |
|---|---|---|
| (a) GENERAL already | 74 | ~12,000 |
| (b) general-but-glm5-named (extraction targets live in these) | 4 (lib.rs, glm_spec.rs, ep_map.rs, glm5_tp.rs — each mixed with (a)/(c) content) | ~1,400 extractable |
| (c) GLM5-SPECIFIC | 78 | ~45,000 |
| (d) POLICY/DOCS | 7 | ~1,500 |

### (a) GENERAL already (representative units — no move needed)

- `cu/qmatvec.cu` +547: `matvec_bf16_f32acc_x4_tcols` / `_tcols16` / `_x1` weight-once
  t-column classes; `moe_gate_up_preclamp8_q8_rows` / `moe_down8_fma_q8_rows` (+ `_w4`
  packed twins). Generic kernel classes, generic flags (`MEMRA_BF16_TCOLS_WIDE`,
  `MEMRA_BF16_TCOLS_X1`, `MEMRA_MOE_VROWS_PACK`).
- `cu/kernels.cu` +93: `topk_rows_shard_f32` + merge (`MEMRA_TOPK_SHARDS`, generic).
- `cu/flash_attn.cu`, `cu/f16_prefill.cu`, `cu/hybrid.cu`: generic / hybrid-class kernels.
- `mmq_ffi.rs`, `f16_ffi.rs`: FFI for the generic classes.
- `tp.rs` +200: `parse_layer_specs_for_trunk` shared spec grammar (glm5_tp already
  consumes it: `Glm5TpLayerSpec = crate::tp::StepEpLayerSpec`), bulk-transport fixes.
- `pp.rs` +96: plan-driven trailing MTP cache plane in `new_cache*` — plan-generic.
- `spec.rs` +981: shared spec machinery (`SpecSampling`, `host_u01` exposure,
  `MEMRA_SPEC_PMIN`/`PMIN0` capture) — the seam glm5 consumes with NO new flags.
- `dflash.rs` +297: `DflashDraft` reuse surface (`ctx_features`/`ingest_ctx`/
  `forward_round`), `DsparkDraftSample::Selector` — the q38 machinery verbatim,
  now multi-family.
- `memra-kv/lib.rs` +1,011: latent plane snapshot/restore, `truncate_index_pool_keys`,
  `HcTapSink`, ppn cache planes — latent/hybrid-class generic.
- `memra-tokenizer/unicode.rs` +721: generic unicode machinery (1 glm5 mention).
- `memra-gguf/source.rs` +489: bulk-transport / sharded-source fixes (6 glm5 mentions).
- `moesd.rs`, `moe_cache.rs`, `graph_update.rs`, `parallel.rs`, `prime_graph.rs`,
  `round_stream.rs`, `spill.rs`, `parity_geometry.rs`: small generic fixes.
- dsv4 units (`cu/dsv4_gpu.cu`, `dsv4_ffi.rs`, `dsv4_gpu.rs`, gguf dsv4*, dsv4 bins):
  deepseek-v4 family work riding the lane — no glm5 coupling.
- server small units (`health.rs`, `auth.rs`, `constrained.rs`, `toolcall.rs` core,
  `embed_api.rs`, `anthropic.rs`, `darklane.rs`): generic route/intake fixes (toolcall
  carries a glm5 dialect arm — that arm is (c)).
- `memra-reference/hidden_trace.rs` +80: generic hidden-state tap tracing.
- gate-bin maintenance (~35 bins with small diffs): generic gate upkeep.

### (b) THE EXTRACTION TARGETS — verdicts

| # | glm5-named unit (where) | general seam it becomes | mechanical or structural | verdict |
|---|---|---|---|---|
| 1 | `MEMRA_GLM5_VERIFY_WS` + `Glm5VerifyWs` + `GLM5_VERIFY_WS_HITS` (engine `lib.rs`; comment refs in `hybrid_forward.rs`, gates in `glm5_matvec_doors_gpu.rs` / `glm5_spec_session_gpu.rs`) | `VerifyWs` size-keyed free-list pool + `MEMRA_VERIFY_WS` (+ `verify_ws_hits()` counter). Family-agnostic by content: `CudaSlice` free-lists keyed on exact length, nothing glm5 in the type. | MECHANICAL (rename + re-gate) | **EXECUTED**. Old flag `MEMRA_GLM5_VERIFY_WS` stays honored: OFF-wins composition (either name `=0` disables; default ON) — zero churn for banked box scripts and the struct-battery arms. Engagement tag `[glm5-verify-ws] engaged` kept BYTE-IDENTICAL (mv-battery `serve.sh`/`build_attrib.sh` grep it); tag generalization deferred to a receipt boundary. |
| 2 | `MEMRA_GLM5_SPEC_TRACE` phase timers: `Glm5PhaseNs` + trace-level readers + verify sub-split atomics (`glm_spec.rs`) | `spec_phase.rs`: `SpecPhaseNs` with tag-parameterized `emit()`, `MEMRA_SPEC_TRACE` levels 1/2 — the draft/verify/accept/roll/maint split applies to every spec family; vkda/vmla/vffn sub-buckets are mixer-class (KDA/MLA are multi-family), not glm5. | MECHANICAL (module move + re-gate) | **EXECUTED**. `MEMRA_GLM5_SPEC_TRACE` stays honored as the family alias; the general name wins when both are set (one loud stderr line names the override — never silently dead). Emit tags `[glm5-phase]` / `[glm5-phase-v]` unchanged (passed by the glm5 caller) so flip-battery cell-2 receipts stay comparable. |
| 3 | `MEMRA_GLM5_EP_MAP` consumption (`glm5_tp.rs` reader, `hybrid.rs` co-refusal) + `ep_map.rs` module doc | `MEMRA_EP_MAP` as the fleet flag, resolved in `ep_map.rs` (`ep_map_env()`): the `memra-ep-map-v1` reader is fleet-shared by design — hy3/qwen adopt the same flag + reader; the glm5 TP-2 loader keeps its own geometry laws (ranks=2, entry_rank=0, layer cover). | MECHANICAL (flag generalization + doc sweep; reader was already family-agnostic code) | **EXECUTED**. `MEMRA_GLM5_EP_MAP` stays honored as the family alias; both set to DIFFERENT values refuses loudly at load (fail-closed, names both). Errors name the flag that actually armed the load, so `glm5-tp-gate` H1–H5/R4 stay green unchanged. |
| 4 | co-refusal / composition-matrix pattern (`GLM5_TP_REFUSED_DOOR_FLAGS` + `refuse_glm5_tp_door_composition`, `glm5_tp.rs`) | `tp::refuse_door_composition(primary, table, armed)` — the pure composition law is seam-generic (any parallel door x optimization-door matrix); each family keeps its own TABLE. | MECHANICAL (helper hoist; glm5 delegates) | **EXECUTED**. Error format byte-identical (`"{primary} + {flag}: unproven composition, refused ({why})"`). glm5's table content stays glm5 — the PATTERN is what main gets. |
| 5 | K-policy surface (`glm5_k` admit, `glm5_pmin`/`glm5_pmin0`/`glm5_conf_keep`) | Already general at the worker: glm5 admits through the SHARED `choose_spec_k` / `MEMRA_SPEC_K` pin and consumes `MEMRA_SPEC_PMIN`/`PMIN0` with no new flag (module-doc law). Residue: the pure chain-break rule `glm5_conf_keep` promotes to `spec::spec_conf_keep`. | MECHANICAL (one pure fn move) | **EXECUTED**. glm5's K clamp (K+1 <= 15 decode-exact knee; DFlash2 block bound) is family geometry and stays. |
| 6 | DraftSource seam (`Glm5DraftState` / `Glm5DflashDrafter`, `glm_spec.rs`; `DraftSourcePlan` in model_plan is ALREADY general) | An engine-level family-agnostic draft-source contract (source-blind verify: accept/rollback/commit/receipts shared) so the next spec family selects NativeMtp/Dflash2/external uniformly. | STRUCTURAL (trait/module design across `glm_spec.rs`/`hybrid.rs`/`worker.rs`) | **PLANNED, not executed** — those files are hot in the in-flight lanes (ep-diet/door-r both carry `glm_spec.rs`+`hybrid_forward.rs`+`lib.rs` deltas) and the seam needs a design pass (per-session state carries glm5 session internals). Plan below. |
| 7 | Door D device dispatch-table build + door H HtoD diet (moe-loc chain: `MEMRA_MOE_VROWS_DEV_TABLES`, `MEMRA_GLM5_HTOD_DIET`, `Engine::i32_mirror_store`) | Door D is ALREADY generically named (`MEMRA_MOE_*`) and MoE-generic by design; door H's `i32_mirror_store` is an Engine-generic method behind a glm5-named flag. | in-flight — NOT IN THIS BASE | **PLANNED**: when the moe-loc chain lands on bringup, rename `MEMRA_GLM5_HTOD_DIET` -> `MEMRA_HTOD_DIET` with the same alias pattern as #1 (one flag reader, OFF-wins). Do it in a follow-up ON TOP of the landed chain — executing it here against un-landed work is manufactured merge hell. |
| 8 | `MEMRA_GLM5_VERIFY_BATCH` batched verify walk | The batched-verify CONCEPT (one t=K+1 mixer call per layer, sequential recurrence in-kernel) generalizes to any hybrid spec family; the implementation is the glm5 walk's own mixer dispatch. | STRUCTURAL | **PLANNED** — natural extraction trigger is the second hybrid spec family; premature trait-cutting here churns the exact files the in-flight lanes hold. |

Not extraction targets (named so nobody re-litigates): `MEMRA_GLM5_SPEC` (family route
master, the `MEMRA_DSPARK` precedent), `MEMRA_GLM5_MTP` (the family's embedded NextN head),
`MEMRA_GLM5_TP` (family shard map — owner classifies shard maps (c)), `MEMRA_GLM5_VISION*`
(tower), `MEMRA_GLM5_*_GATE_RED` / `_GATE_SAME_DEV` (gate instruments, never serving flags).

### (c) GLM5-SPECIFIC by construction (stays the model-support set)

mHC walks (`hyper.rs`, the hc arms in `hybrid_forward.rs`); KDA kernels + module
(`cu/kda.cu`, `kda.rs`); MLA kernels (`cu/mla_attn.cu`, `mla.rs`, `mla_ffi.rs`); the glm5
TP-2 shard maps and walk (`glm5_tp.rs` minus the two hoisted seams); the spec session +
KDA rollback (`glm_spec.rs` minus the timers); vision tower (`vision_glm5.rs`,
`vision.rs` glue, reference tower); the `glm5_next` model pack, config/hf_mapping/
tensor_contract/model_plan rows; the glm5 chat template arms (`chat.rs`,
`template_is_glm5`); the worker glm5 route (session field, `step_glm5_spec`, demotion,
vision spans); every `glm5_*` test suite and gate bin; `glm5_checkpoint_runner`;
micro-gguf fixtures.

### (d) POLICY/DOCS

`execution_manifest.rs` (the `GLM5_SPEC` capability manifest — deliberately its own
table, `mtp_spec_capable` unextended); model_plan policy fields (`DraftSourcePlan`,
sampling defaults); docs/FLAGS.md, docs/KERNELS.md, docs/MODELS.md rows (outside the
crates/ diff scope but ride the same merge).

## STEP 2 — what moved (this lane's commits)

Executed extractions (all NO-BEHAVIOR-CHANGE: bit-gates stay bit, engagement counters
keep counting, receipt tags byte-identical, every old flag name still works):

1. **ep-map flag generalization** — `MEMRA_EP_MAP` primary, `MEMRA_GLM5_EP_MAP` alias;
   conflicting dual set refuses loudly at load. `ep_map.rs` owns the env seam + de-glm5'd
   module doc; `glm5_tp.rs` + `hybrid.rs` consume it; errors name the armed flag.
2. **verify-workspace generalization** — `Glm5VerifyWs` -> `VerifyWs`,
   `Engine::glm5_verify_ws` -> `verify_ws`, `GLM5_VERIFY_WS_HITS` -> `VERIFY_WS_HITS`,
   `glm5_verify_ws_hits()` -> `verify_ws_hits()`; flag `MEMRA_VERIFY_WS` with OFF-wins
   alias composition. Engage announce line byte-identical.
3. **spec phase timers** — new `spec_phase.rs` (`SpecPhaseNs`, `spec_trace_on/level`,
   verify sub-split accumulators); `MEMRA_SPEC_TRACE` with glm5 alias; glm_spec.rs is a
   consumer passing its own `[glm5-phase]`/`[glm5-phase-v]` tags.
4. **composition-refusal helper** — `tp::refuse_door_composition`; glm5 delegates with
   its own table; error bytes unchanged.
5. **conf-keep rule** — `spec::spec_conf_keep`; glm_spec re-exports for its call site.

FLAGS.md: three new general rows (`MEMRA_EP_MAP`, `MEMRA_VERIFY_WS`, `MEMRA_SPEC_TRACE`)
carry the full documentation; the three glm5 rows shrink to alias pointers (same PR, per
the flag-rename law). Alias semantics chosen per churn: the glm5 names are what every
banked gate script, box battery, and the in-flight lanes set TODAY — refusing them would
break `glm5-tp-gate` arms and the struct-battery mid-bank; honoring them costs one env
read. The general name wins/composes as documented per flag above.

### Files touched by this lane (for in-flight-lane warnings)

Exactly these ten (the extraction commit's stat):
`crates/memra-engine/src/ep_map.rs` (+77), `glm5_tp.rs` (~60), `glm_spec.rs` (-167/+~30,
the timer block moved out), `hybrid.rs` (15, one co-refusal site), `lib.rs` (121, the
verify-ws block + one `pub mod` line), `spec.rs` (+21), NEW `spec_phase.rs` (+178),
`tp.rs` (+42), `tests/glm5_spec_session_gpu.rs` (6, the renamed counter fn), and
`docs/FLAGS.md` (row moves). `hybrid_forward.rs`, `moesd.rs`, `worker.rs`,
`glm5_matvec_doors_gpu.rs` deliberately UNTOUCHED (in-flight / receipt surfaces).

Highest collision risk vs lane/glm5-ep-diet + lane/glm5-door-r (both hold deltas in
`lib.rs` @ the Engine struct/counters region, `hybrid_forward.rs`, `glm_spec.rs` @~1218,
`moesd.rs`): the `lib.rs` Engine-struct field rename `glm5_verify_ws` -> `verify_ws`
(their door D fields insert adjacent at ~1054) and the `lib.rs` counters region (their
+148 lines insert at ~1596, just above the renamed reader). The `glm_spec.rs` timer-block
removal sits ~800 lines above their hunk; low risk. On conflict, the resolution rule is
one-directional: THEIR content, THIS lane's names.

## STEP 3 — the integration map (bringup -> main after extraction)

What the merge ships, in two named sets:

**General engine features** (reviewable as engine capabilities, no glm5 acceptance
needed): the matvec bf16 tcols/x1 kernel classes + doors T/X/K (generic flags, banked
mv-battery receipts); the topk shard split; the MoE verify-rows pair + door M; the
verify-workspace pool (`MEMRA_VERIFY_WS`) and the `SCRATCH_ALLOC_CALLS` census; the spec
phase-timer instrument (`MEMRA_SPEC_TRACE`); the `memra-ep-map-v1` reader + `MEMRA_EP_MAP`
fleet seam (+ `tools/build_expert_placement_map.py`); the composition-refusal law
(`tp::refuse_door_composition`); the shared layer-spec grammar; the shared spec K/pmin
surface (`spec_conf_keep`); the bulk-transport and latent-KV machinery; `DflashDraft`
multi-family reuse + `HcTapSink`; unicode/tokenizer generalizations; dsv4 units.

**glm5 model-support set**: everything in (c) + the family flags + the `GLM5_SPEC`
manifest + suites/gates. Support state after merge remains NativeReference-class until
the real-artifact qualification lane banks its receipts — the merge does NOT move the
darklanes roster.

**Main should get FIRST (independent of this merge):**

- The vacuous-gate sweep pattern (92626d44e / 62e00491b lesson, now law in this repo's
  gates): every A/B gate on a default-ON flag pins the OFF arm `=0` — unset is not an
  arm. MEASURED on main in this lane: the nine `remove_var`-using gate bins
  (concat_prime_probe, decode_batch_gate, fa_hd128_check, kernel_check, mmvq_bisect,
  pp2_gate, ppn_bench, ppn_gate, w4a4_gate) unset stage/route flags whose DOCUMENTED
  default is off/unset — clearing to the reference arm, not the vacuous class. The full
  audit (each unset name vs its FLAGS.md default column, re-run at every future
  default-ON flip) is the cheap named follow-up; the flip commit is the trigger point.
- The zero-test `--ignored` hole: main is CLEAN today — no runner uses bare
  `--ignored`; the one doc-comment invocation (`mla_fixture_load_gpu.rs`) targets a
  suite whose only test IS `#[ignore]`, so it runs 1 test, not 0. The RUNNER-side law
  (`--include-ignored` + assert non-zero counts; the moe-loc `kda_fixture_gpu`
  "ok. 0 passed; 3 filtered out" lesson) should land in main's TESTING.md with the
  battery-runner template when the moe-loc chain merges forward — this lane's
  `run-battery.sh` implements it (a green suite reporting `0 passed` counts as FAILED).
- The `MEMRA_EP_MAP` general seam (this lane) unblocks hy3/qwen placement-map adoption
  without waiting for the full glm5 merge, if cherry-picked with `ep_map.rs` + the tool.

**Planned structural follow-ups (NOT this lane; coordinate with in-flight owners):**

- DraftSource trait-level seam (b#6) — after ep-diet/door-r/struct-battery land.
- `MEMRA_GLM5_HTOD_DIET` -> `MEMRA_HTOD_DIET` (b#7) — on top of the landed moe-loc chain.
- Batched-verify generalization (b#8) — at the second hybrid spec family.
- Receipt-tag generalization (`[glm5-verify-ws]`, `[glm5-phase]`) — at a receipt
  boundary, with the box scripts updated in the same commit.

## Incidental finding: the sccache-flock rig deadlock (2026-08-31, fixed in-lane)

Found live while queueing this lane's battery: `/tmp/memra-5090.lock` was held by a DEAD
pid (`/proc/locks` names it; `fuser` shows `sccache`) for ~80 minutes with three lanes'
suites queued behind it and the 5090 idle. Mechanism: a cargo build running UNDER the
rig flock spawns the sccache daemon, which inherits the lock fd; the flock'd command
exits, but the open file description — and the exclusive flock — live on in the daemon.
Release = kill the fd-holding sccache daemon (it restarts on demand). In-lane fixes:
`run-battery.sh` pre-warms the server OUTSIDE the lock, and `lock-watchdog.sh` clears
exactly the poisoned state (dead holder + fd-holding daemon). Fleet promotion candidate:
agent-knowledge/gpu — every rig battery should pre-warm sccache before its first flock.

## Gates (this lane, the MERGED extracted tree)

Tree = the extraction commits + `origin/lane/glm53-flash-bringup` @ 6c548e3c2 absorbed
(the moe-loc merge chain landed mid-lane and was merged in per the owner's
absorb-the-newest order; two lib.rs conflicts resolved their-content/this-lane's-names;
one auto-merge artifact — a stale orphaned doc line — caught by clippy and dropped).
Receipts in `receipts/`. Runner law upheld: `--include-ignored` everywhere, a green
suite reporting `0 passed` counts as FAILED.

| gate | invocation | verdict |
|---|---|---|
| 28 GPU suites, default (ship) arm | `cargo test -p memra-engine --test <s> -- --include-ignored --test-threads=1` under flock+TF32-off; standing 18 (incl. the newly landed `glm5_moe_loc_doors_gpu`) + 9 extra glm5 suites + `glm5_vision_gpu` w/ real shard | ALL GREEN, non-zero counts (4..18 tests each; receipts `receipts/<suite>.log`) — incl. `glm5_spec_session_gpu` 10/10 (door-W ws gate through the renamed pool + counter) and `glm5_matvec_doors_gpu` 4/4 |
| glm5-tp-gate P=16 N=12 | `cargo run -q --release --bin glm5-tp-gate -- 16 12`, flock+TF32-off | **ALL ARMS PASS** — H1–H5/R4/M drive `MEMRA_GLM5_EP_MAP` (the alias) end-to-end through the new `ep_map_env()` resolver; refusal bytes unchanged (receipt `receipts/tp-gate-p16-n12.log`; the battery.console FAIL line is this runner's own bin-name typo `glm5_tp_gate`, fixed in the script and re-run) |
| ppn matrix 9 arms (incl. `MEMRA_SPEC_TRACE=2` twin) | `glm5-spec-ppn-gate` 2/3-stage arm set | 24/24 PASS lines per arm; the spec-trace twin proves the GENERAL flag engages the moved timers: 20 `[glm5-phase]`/`[glm5-phase-v]` lines, tag bytes unchanged, control arm 0 lines |
| hppn matrix 10 arms | `glm5-hyper-ppn-gate` banked arm set | 6/6 PASS lines per arm |
| hbatch matrix 10 arms | `glm5-hyper-batch-gate` banked ladder | 3/3 PASS lines per arm |
| memra-server suite | `cargo test -q --release -p memra-server` | 492/492 (plus two empty 0-test targets, the ep-place shape) |
| engine lib units | `cargo test -p memra-engine --lib` | 275 passed / 2 ignored (standing) — incl. the new ep_map env-resolution, spec_phase level-resolution, verify-ws OFF-wins, tp door-composition byte-format tests |
| check-flags | `tools/check-flags.sh` | green — 746 runtime reads covered, both old and new names |
| clippy | `cargo clippy --workspace --all-targets` | zero warnings |
| fmt | `cargo fmt --check` | clean |
| local-ci --perf | `tools/local-ci.sh --perf` | exit 0 — correctness stage green, perf stage 0 fail 0 warn; qwen9b-plain-short 136.58 tok/s [OK] vs the rolling median, row banked at git 4721342e0 (`receipts/local-ci-perf.log`) |

---

## PHASE 2 LANDED — this map is superseded in three rows (2026-09-01)

`lane/glm5-extract2` (`research/glm53-flash-bringup-20260827/extract2-20260901/LANE.md`)
executed the rest. Read that doc's §7 for the merged integration map; the deltas to §3 above:

- **"Planned structural follow-ups" row 2 is DONE**: `MEMRA_GLM5_HTOD_DIET` -> `MEMRA_HTOD_DIET`
  (door H is engine-generic HtoD hygiene), with `HTOD_DIET_AVOIDED` / `htod_diet_avoided()`.
- **"Planned structural follow-ups" row 1 is DONE IN HALF**: the DraftSource seam's model-level
  and selection halves are extracted (`dflash::DflashDrafter`, `dflash::load_drafter`,
  `dflash::resolve_tap_layers`, `spec::DraftSourceKind` + `resolve_draft_source_kind`). The
  per-session-state trait stays deferred, with the reason in code terms and a trait sketch in
  extract2 §4.6 — its trigger is unchanged (the second hybrid spec family's session state).
- **"Planned structural follow-ups" row 3 (batched verify) is UNCHANGED**: the hy3 pack landed
  on the same base, but a model pack is not a verify walk, so the second-consumer trigger is
  still not met.
- **NEW general features to add to the "General engine features" set**: the flag-alias law for
  BOOLEAN doors (`alias_door_from` — the fourth and last door shape, so future renames follow a
  law instead of a hand-roll); `MEMRA_EP_DIET` and `MEMRA_EP_GROUPED_PRIME` (general EP
  dispatch/prime doors); the draft-source seam above; `memra-gguf::placement` (pure multi-GPU
  stage placement, landed on bringup after this doc was written).
- **NEW for "Main should get FIRST"**: the flag-alias law itself, and `memra-gguf::placement`
  (pure, no CUDA, no family — cherry-pickable today).

Not extracted, and NOT because they were declined: the transport seam + its fleet tools
(`lane/glm5-tp-transport`, LOCAL-ONLY, never pushed to origin) and the
`MEMRA_WORKER_AFFINITY` -> `MEMRA_WORKER_CPUSET` rename plus the `apply_penalties_dense` audit
(`lane/glm5-host-audit`, on origin, not merged into bringup). Their content is not on the
bringup head at all; extract2 §4.1/§4.2/§4.3/§4.4 carry apply-ready recipes.
