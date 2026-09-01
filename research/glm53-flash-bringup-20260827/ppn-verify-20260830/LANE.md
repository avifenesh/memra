# glm5_next ppN verify twin (lane/glm5-ppn-verify, 2026-08-30)

The LAST engine blocker on the glm5 spec critical path: the worker spec route
(lane/glm5-spec-routing @ 19d49a0b1) admitted single-device placements only — the
capability refused sharded (`MEMRA_PP_STAGES>1`) placements by name, and the serving shape
is 3-card ppN (SPLITS=15,30). This lane lands the `[t, streams, n_embd]` stage-split twin
of the T-parallel verify walk, the per-stage rollback, the last-stage MTP chain, and the
capability lift — gated, red-proven, fail-closed for everything outside the gated set.

Base: origin/lane/glm5-spec-routing @ 19d49a0b1 (spec-routing was NOT yet merged to
origin/lane/glm53-flash-bringup at lane open — bringup head cc718b988 has no spec-routing
ancestor — so the lane bases on the spec-routing head directly, as pre-agreed).

## What was built

### 1. The walk under the split (`glm_spec.rs`)

`glm5_verify_rows` now owns its ppN door exactly as the batched decode walk does
(`decode_step_batch_hyper_ppn`, decode_batch.rs — the named mirror):

- The layer loop is extracted into `glm5_verify_range(lo, hi)` (the `hyper_range_decode` /
  `decode_batch_layers` precedent): the unsplit walk and every stage run the SAME code over
  their own range, so split-vs-unsplit bit-identity is structural. At `lo=0, hi=n_layers`
  the launch sequence is identical to the pre-extraction walk — re-proven by re-running the
  whole pre-existing tparallel battery (below).
- `glm5_verify_rows_ppn`: `MEMRA_PP_STREAMS=0` same-stream seam (boundary clone pairs) +
  the per-stage arm (`rt.enter(s)` scopes, per-stage engines, per-stage pos_rows uploads,
  `rt.tx`/`rt.rx` of the `[t, streams, n_embd]` payload per cut, `fence_stages_behind` at
  entry). ROW CHAINING NEEDS NO NEW TRANSFER PATH: row r+1 at layer il depends only on row
  r at layer il (KDA state / MLA latent), never on a later layer — so the straight
  layer-range split preserves the chaining and the boundary carries exactly what plain ppN
  decode carries, t rows of it.
- DRAIN CONTRACT (new, named): unlike its decode twins the verify walk returns DEVICE
  buffers with no terminal dtoh, so the per-stage arm synchronizes the LAST stage's stream
  before returning — the TX-wait chain transitively settles every earlier stage (pp.rs
  multi-stream law), making the logits, collapsed rows and ckpt columns safe for the
  caller's streams.
- The trunk exit is extracted (`glm5_verify_head`) and runs on the last stage's engine.

### 2. Rollback under the split

- KDA ckpt columns are cloned inside `glm5_verify_range` through the range's engine — under
  a split each column lives on (and is stream-ordered with) its layer's OWNING stage.
- `glm5_verify_rollback` restores stage by stage: `rt.enter(s)` + the stage engine for
  layers `[fence[s], fence[s+1])` — MLA `len`/`len_d` truncate + `truncate_index_pool_keys`
  clamp + KDA column restore all land on the stage streams the walk writes on, so ordering
  is by stream, no extra fence. Door shut: the old single-engine loop, unchanged
  (`glm5_rollback_layer` is the shared per-layer body).
- `glm5_mtp_plane_reset` writes the plane's `len_d` through the HEAD engine — the MTP plane
  is on the last stage (`pp::new_cache*` maps trailing MTP/NextN planes there).

### 3. Draft chain on the last stage — no DtoH bounce, no cross-device transfer

`glm5_head_engine`: the LAST stage's engine when the door is open with per-stage streams,
else the caller's engine by identity. Everything head-side rides it: the MTP chain
(`mtp_head_forward_mla_cached`), the session warm, draft argmax dtoh, the sampled p/q
kernels, seed-row copies. The carrier residency closes BY PLACEMENT: `prime_cache`'s ppN
twin publishes `hiddens`/h_seed on the last stage's engine, the verify walk's `collapsed`
rows are produced there, the MTP block's weights and plane live there, and the draft
projects through the trunk lm head which the loader puts there — so NO transfer is
required anywhere in the draft chain (the answer to the residency question: none needed,
none added).

Loader fix (hybrid.rs): the embedded MTP block's tensors — and the FR-Spec trim's gathered
draft-head rows — now upload through `pp::layer_engine(e, n_trunk, il>=n_trunk)` = the LAST
stage's engine. They uploaded through the primary before, which was invisible on the rig
(devices unset → primary) and on the serving config (`worker_device` puts the primary ON
the head stage) but wrong for any placement whose primary is not the last stage — exactly
the box gate's `MEMRA_PP_DEVICES=0,1` arm.

### 4. Capability lift (fail-closed for everything ungated)

- Engine seams: `glm5_verify_rows` + `glm5_spec_session_new` no longer refuse `pp_cuts`;
  both refuse an UNQUALIFIED pipeline rewrite by name (`RewriteSurface::Pipeline`), same as
  every other hc ppN walk. `glm5_spec_session_new` allocates through
  `pp::new_cache_planned` (stage-owned planes; door shut = byte-identical plain path).
- Worker capability (`glm5_spec_capable`): sharded placements admit ONLY through
  `glm5_sharded_placement_admits` — **stages 2 and 3** (the gated set: SPLITS=24 and
  SPLITS=15,30 classes) AND no `MEMRA_STEP_TP`/`MEMRA_STEP_EP` composition AND
  `rewrite_allowed(Pipeline)`. Stages >= 4: no gate receipt, refuses. TP/EP: never
  co-gated, refuses (the dspark precedent). The GLM5_SPEC manifest itself is UNCHANGED —
  placements are not operation classes; the gate lives at the capability seam where the
  refusal lived.
- FLAGS.md `MEMRA_GLM5_SPEC` row updated in the same lane (capability + remaining box arm).
  `tools/check-flags.sh` green. No new flag: the ppN door (`MEMRA_PP_STAGES`) and the spec
  master (`MEMRA_GLM5_SPEC`) already exist; their composition is what this lane gates.

## Gate table (rig 5090, TF32 off, flock-serialized, 2026-08-30)

Pre-existing single-device batteries RE-RUN over the extracted walk (the structural
bit-identity claim's receipt):

| gate | result |
|---|---|
| `glm5_tparallel_verify_gpu` (7: walk identity, accept-j, stale-KDA red, pool-key red, e2e K=1..7 + forced arms, rollback-disabled red, FR-Spec battery) | PASS 7/7, 2.59s (`rerun-tparallel-20260830.log`) |
| `glm5_spec_session_gpu` (7: served bursts, j-sweep, sampled twin, EOS, red arms, receipt log) + `glm5_mtp_head_gpu` (5) | PASS (`rerun-session-mtp-20260830.log`) |
| worker `glm5_sharded_placement_matrix_is_exact` (2/3 admit; 4+ refuse; TP/EP refuse; off/empty TP values are not TP) | PASS |
| worker `glm5_route_wiring_is_live_in_comment_stripped_source` extended: capability anchors now include `glm5_sharded_placement_admits(` + `RewriteSurface::Pipeline` as invocations | PASS |
| memra-gguf `glm5_spec_class_matrix` (4) unchanged-green | PASS |
| CPU suites: memra-server 473 / memra-gguf --lib 183 / memra-engine --lib 252 | PASS, 0 failed |
| `cargo fmt` + workspace `cargo clippy --all-targets` | zero warnings |
| `tools/check-flags.sh` | green, no uncovered names |
| `tools/local-ci.sh --perf` (qwen9b fixture present) | ALL GREEN, exit=0: correctness stages green, perf stage 0 fail 0 warn, qwen9b-plain-short 139.79 tok/s [OK] (window_clean=false — persistent co-resident lanes on the rig, same posture as the spec-routing lane's settled run; the number sits above the 138.97 median regardless) |

NEW `glm5-spec-ppn-gate` (binary, one placement per invocation; 23 arms per invocation:
W0 plain-ppN-decode re-pin, W1 verify-walk rows bit-identity, A accept-j j=0..7,
E e2e tapes K=1..7 natural + forced-accept K=3/7 + forced-rejection sweep, R1 stale-KDA
red, R2 pool-key tripwire red, R3 rollback-disabled red; non-vacuity asserted: hc
topology, door open, state classes split across stages). **ALL 8 placements: 23/23 PASS,
0 fail, all 3 reds bite** — red signatures match the single-device battery's (stale-KDA
379 differing logits; pool-key "claims 8 finished pools but the cache holds only 6";
rollback-disabled "latent cache overflow — 59 + 1 rows exceeds capacity 59"):

| placement (all P=24 N=20 K=7) | fence | result |
|---|---|---|
| stages=2 even | [0,2,4] | 23/23 PASS (`10-n2-even.log`) |
| stages=2 SPLITS=1 / SPLITS=3 | [0,1,4] / [0,3,4] | 23/23 PASS each (`11-…`,`12-…`) |
| stages=2 MEMRA_PP_STREAMS=0 (same-stream seam) | [0,2,4] | 23/23 PASS (`13-…`) |
| stages=2 MEMRA_PP_OVERLAP=0 | [0,2,4] | 23/23 PASS (`14-…`) |
| stages=3 even (the SPLITS=15,30 class) | [0,1,2,4] | 23/23 PASS (`16-n3-even.log`) |
| stages=3 SPLITS=1,3 asym | [0,1,3,4] | 23/23 PASS (`17-…`) |
| stages=3 MEMRA_PP_STREAMS=0 | [0,1,2,4] | 23/23 PASS (`18-…`) |

Repro:
```
cargo build --release -p memra-engine --bin glm5-spec-ppn-gate
bash research/glm53-flash-bringup-20260827/ppn-verify-20260830/run-spec-ppn-gate.sh
```

## The box cross-device twin — RUN, ALL GREEN (2026-08-30, same day)

Coordinator opened a box window (cards 0/1/2; card 3 co-tenant untouched). The full
battery re-fought with real cross-device placements — peer transport and weight sharding
live — on 4x RTX PRO 6000 Blackwell Workstation, CUDA 12.8, TF32 off, correctness-only.
Lane head 50aee8879 rebuilt ON the box (4m43s, strings-probed — the rebuild-attribution
law); receipts + per-arm logs in `box-twin/` (`RECEIPTS.md`):

| placement | result |
|---|---|
| stages=2 `MEMRA_PP_DEVICES=0,1` | 23/23 PASS, reds 3/3 |
| stages=2 `MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0` (bring-up placement) | 23/23 PASS, reds 3/3 |
| stages=3 `MEMRA_PP_DEVICES=0,1,2` (the serving shape's class) | 23/23 PASS, reds 3/3 |
| stages=3 `MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_SPLITS=1,3` (asym) | 23/23 PASS, reds 3/3 |

Named: the literal SPLITS=15,30 cuts exist only at 45 trunk layers — the fixture harness
cannot take them; the serving shape's CLASS is the two stages=3 three-device arms, and the
literal-cut twin belongs to the real-artifact battery, which runs the deployed placement
itself.

With the twin green the capability lift is fully receipted for its gated set, and the box
A/B prerequisite chain (tparallel LANE §Box A/B) advances to: **real-artifact
accept/rollback battery** on the deployed NVFP4 placement (SPLITS=15,30) -> acceptance
measurement + trim A/B (the glm5 ranks mint is DONE — darklanes
`research/glm53-ranks-mint-20260830/`, three classes, gated) -> interleaved x5 A/B with
the vendor-default sampled twin + spec-engagement receipts + the 8-turn cache-on twin.
Note also: spec-routing (19d49a0b1) merged to bringup while this lane ran — the base note
above records lane-open state.

## Deliberately NOT in this lane

- Cross-device receipts of any kind (box arm above).
- Stages >= 4, TP/EP compositions, the deferred-readback (pipelined) arm — each refused by
  name (`decode_step_h_ppn_deferred` still `refuse_hyper`s; nothing here changed that).
- Real-artifact receipts, admission pricing of the K-column transient, session
  parking/resume — unchanged follow-ups from spec-routing-20260830.
