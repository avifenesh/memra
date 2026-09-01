# glm5 DFlash2 as an ALTERNATE DRAFT SOURCE (lane/glm5-dflash-draft-src, 2026-08-30)

Wire the pinned incoai/GLM-5.3-Flash-DFlash2 drafter into `Glm5SpecSession` beside the
native MTP head, so the box can run the end-to-end served three-way comparison:
**plain vs native-MTP-spec vs DFlash2-spec**.

**License state: owner holds written approval from the DFlash2 owners (2026-08-30) for use
of `incoai/GLM-5.3-Flash-DFlash2` beyond probe/eval.** Record of that approval governs this
lane; artifact identity stays pinned regardless: revision
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, `model.safetensors` sha256
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` (boot receipt pins the
first 8 hex, `b33c0347`).

## 1. The seam design

`Glm5SpecSession` gets a pinned-per-session draft source (`glm_spec.rs::Glm5DraftState`):

| source | flag | drafts K candidates by | per-session state |
|---|---|---|---|
| `NativeMtp` | `MEMRA_GLM5_MTP=1` | `mtp_head_forward_mla_cached` chain (unchanged, byte-identical code path) | MTP latent plane + pending `(token, h_seed)` pairs |
| `Dflash2` | `MEMRA_GLM5_DFLASH=<dir-or-hf-spec>` | the q38 DFlash2 machinery VERBATIM: `ctx_features` -> `ingest_ctx` -> `forward_round` -> `dflash2_propose_greedy/_sampled` (mask-fill harvest, selector walk) | drafter `DflashKv` (ctx-feature cache) + pending host feature rows |

The seam sits exactly where the session asks for K candidates given (anchor, position):
`glm5_spec_round` step 1+2 became one source-keyed match. **Everything from the verify walk
on is SHARED and untouched** — `glm5_verify_rows` (t=K+1 batched-decode-class walk), the
greedy longest-prefix accept, the sampled rejection walk, `glm5_verify_rollback` (KDA state
columns + MLA latent truncate + kpool clamp), commit bookkeeping, receipts, K policy. The
exactness invariant lives in verify: a draft source can only move acceptance, never output.

Sampled arm: the DFlash2 source records its true proposal (`DsparkDraftSample::Selector`,
candidate-set softmax q) and the accept rides the SAME `dspark_accept_sampled` rejection
walk the q38 serve route ships (`u_j*q_j < p_j`, filtered p from the verify rows, sparse-q
residual at the reject slot, fresh-stats bonus from row K) with THIS session's Philox
counters (`uctr` selector/accept draws, `sctr` bonus/residual) — session-continuity law,
burst-split invariance gated.

FR-Spec trim: consumed through the MTP struct exactly as the dspark arm does
(`shared_head_head` + `d2t`, remap after top-k, verify full-vocab). With the head NOT
loaded (the normal DFlash2 boot) trim is unavailable — stated in the boot receipt; glm5
ranks are unminted anyway (owner lane), so nothing is lost today.

### The q38 pattern: the native MTP head is NOT loaded (owner requirement)

`MEMRA_GLM5_DFLASH` alone enables the route. `glm5_spec_session_new`,
`spec.rs::generate_spec` routing and worker `glm5_spec_capable` all accept
`mtp.is_some() || glm5_dflash.is_some()`; the MTP-specific warm/plane-reset/re-seed steps
are scoped to the native arm. On the real artifact that keeps the whole `layers.45` NextN
block (a full MoE trunk layer: 288 routed experts + MLA + indexer) OFF the card.

**Named cost, not forked:** the plan's trailing MTP CACHE PLANE still allocates
(`Cache::new_planned` walks `plan.mtp_blocks` structurally; skipping it would fork cache
layout against the latent-plane hole postmortem). Cost on the real artifact: latent width
512 f32 = 2 KiB/token (PLACEMENT-RECEIPT.md) -> **16 MiB at ctx 8192** plus the lazily
allocated index tail ring — noise against the multi-GB head saving. VRAM-at-ready per arm
is a banked receipt of the box three-way window; the rig mini-fixture delta is printed by
`gpu_draft_source_selection_matrix` (mechanism proof at mini scale).

Drafter weights load ONCE per model on the HEAD engine
(`pp::layer_engine(e, n_trunk, n_trunk)` — the stage that owns the trunk lm_head the
drafter projects through and where every round's chain runs). Set-but-unloadable is a loud
boot failure, never a silent plain fallback.

K bound: the drafter blocks 8 = anchor + 7 drafts (its trained mask pattern). The worker
clamps operator K pins to 7 with a logged line; the engine burst refuses K > 7 by name.

## 2. The drafter's measured input contract (probe ground truth, not guessed)

From `../dflash2-probe-20260829/` (RECEIPTS.md + `capture_worker.sh` + `score_dflash2.py`):

* **Features: the STREAM-MEAN (`hc_contract`) of the COMPLETED trunk layer output** at the
  drafter config's `target_layer_ids` = plan layers **[5, 14, 24, 33, 42]**, f32, one row
  per position. `score_dflash2.py` asserts the drafter's own `target_layer_ids == LAYERS`
  1:1 against the capture's `MEMRA_TRACE_LAYER_ROWS_LAYERS=5,14,24,33,42`, and the capture
  seam (commit 81d6601c0) emits at the END of layer il's iteration in the hc walk — i.e.
  memra plan-layer indices, completed output, mean over the `hc_mult` streams (the SGLang
  glm5_next integration's pinned aux-hidden definition). This banked 0.731 acc@1 / 3.06
  tokens-per-verify-cycle (4.66 tool-wire) teacher-forced on real agent traffic — the
  contract is measured, and the tap-shift red arm exists precisely because a wrong layer
  set is fluent and silent.
* Feature layout: rows `[pos, n_taps*4096]`, taps concatenated in config order — exactly
  the `fc` input (`ctx_features` = `hidden_norm(fc(taps))`).
* Context = features of every COMMITTED position `[0, start)`; the anchor enters as noise
  row 0 (its own feature joins the context only after its verify row commits) — the
  probe's `F_feat[new_lo:start]` advance, mirrored by the round's keep-rows ingest.
* Noise block: target `embed_tokens` rows for `[anchor, MASK(154856) x 7]`, raw (no
  scale); drafter output rows 1..8 project through the target `lm_head` (drafter has no
  head of its own; both are unquantized in the NVFP4 artifact's ignore list).
* Drafter class facts (load-validated): hidden 4096 == n_embd, 5-layer qwen3-class,
  vocab 154880 == target text vocab, block 8, sliding window 2048 (non-causal symmetric,
  `is_causal=false`, all layers sliding), conv kernel 2 group 16, selector rank 256
  top_k 16.

Implementation: features flow through the new HOST sink `Cache::hc_taps`
(`memra-kv::HcTapSink`) filled at three walk sites — `hyper_range_prime` (ppN prime, all
stage arms), `prime_chunk_hyper` (single-engine chunked prime; chunk base = `cache.pos` at
entry, the `dflash_taps.base` precedent), and `glm5_verify_range` (the verify walk, so
accepted rows feed the next round). Contraction on-device (`hyper::contract_mean`, the
`memra_dsv4_hc_mean` kernel), then one dtoh per tapped layer per walk. Host staging is the
deliberate first-light choice: it makes the seam ppN-placement-invariant (tap layers span
stage devices on the 2/3-stage box shapes) and matches the probe's own capture arithmetic;
a device-resident tap diet is a named follow-up if the box A/B shows it in the round wall.

## 3. Gate table (rig: 5090, NVIDIA_TF32_OVERRIDE=0, flock /tmp/memra-5090.lock, exactness only)

`crates/memra-engine/tests/glm5_dflash_session_gpu.rs` — mini glm5 hc fixture (the
session-gate fixture) + a mini DFlash2 checkpoint dir written through the REAL loader
(config census, safetensors, precision seam); trunk loads WITHOUT the MTP head throughout.

| # | arm | asserts | result |
|---|---|---|---|
| 1 | GREEN served greedy byte identity | dflash-source burst tape == plain decode tape, K=1..7, worker-sized bursts | see RESULTS |
| 2 | GREEN forced-rejection j-sweep | every partial-accept j, continuation across burst boundaries, byte-identical | see RESULTS |
| 3 | RED tap-shift (`MEMRA_GLM5_DFLASH_GATE_RED=tap-shift`) | wrong features (taps +1 layer) CHANGE the draft stream (feature seam live), never improve acceptance, tape stays byte-identical | see RESULTS |
| 4 | RED rollback disabled + forced rejections | diverges or fails loudly through the dflash session | see RESULTS |
| 5 | GREEN sampled twin | pinned-seed deterministic, burst-split invariant (session Philox), seed-sensitive | see RESULTS |
| 6 | GREEN EOS | mid-burst EOS finishes the session; post-EOS burst empty | see RESULTS |
| 7 | RED K bound | K=8 refuses naming the drafter block | see RESULTS |
| 8 | selection matrix (subprocess logs) | dflash2 armed (head NOT loaded — the VRAM note), both-armed (dflash2 wins, stated), native-mtp line, fail-closed warn, flag-off = zero `[glm5-spec]` lines; VRAM-at-ready printed per arm | see RESULTS |

Native-source regression: `glm5_spec_session_gpu` + `glm5_tparallel_verify_gpu` re-run on
this head (the seam refactor must leave the native arm byte-identical).

Acceptance NUMBERS on the rig are mechanism-only: the mini drafter is random, so it
accepts near chance on both gate-3 arms. Teacher-forced acceptance on a real-artifact
sample is NOT cheaply available on the rig (the target is a 190 GB 4-card artifact);
**acceptance lands on the box** — the probe's 0.73 acc@1 band is the sanity bar there.

## 4. Flags

* `MEMRA_GLM5_DFLASH` — new, **default OFF by design**: the flip needs the box three-way
  comparison; no serving-shape numbers exist yet (probe numbers are teacher-forced).
  FLAGS.md row lands in this PR (default, both arms, rollback seam = unset, receipts
  pointer here).
* `MEMRA_GLM5_DFLASH_GATE_RED` — gate-harness knob (never a serving flag), FLAGS.md row.
* `MEMRA_GLM5_SPEC` remains the one master flag; `MEMRA_GLM5_MTP` unchanged.

## 5. Remaining box cells (the three-way window — box B, after this lane lands)

All arms: same box, same binary, same artifact, same env EXCEPT the source flags (the
comparability requirement); interleaved x5 per the A/B protocol law; arm identity proven
per boot (nonce + the boot selection receipt, never health-200); engagement receipts
grepped from the server log (`[glm5-spec] draft source = ...`, `[glm5-acc] ...` per burst
— never-serve-greedy law).

1. **Boot receipts + VRAM-at-ready per arm**: plain / `MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1`
   / `MEMRA_GLM5_DFLASH=<pinned dir> MEMRA_GLM5_SPEC=1` — bank `nvidia-smi` at ready and
   the boot lines; drafter sha8 must read `b33c0347`.
2. **Real-artifact acceptance sanity (greedy)**: served bursts on the banked gpf-ab pool
   shapes, `[glm5-acc]` cumulative acceptance vs the probe band (3.06 tokens/cycle class;
   tool-wire stretches higher). A collapse to ~0 here = feature-contract wiring defect
   (the tap-shift red arm's failure class), not a perf result.
3. **Tap-shift red arm on the real artifact** (one short cell): acceptance collapse
   magnitude vs arm 2, tape byte-identical — the 0.73 acc@1 sanity band made a box
   receipt.
4. **Three-way perf window**: plain vs native-MTP vs DFlash2 — TTFT + tok/s + acceptance,
   greedy AND vendor-default sampled (request with NO sampling params, spec-engagement
   receipt from the log), interleaved x5, per the serving-decision cell laws (multi-turn
   8-turn larger-prompt cache-on twin included per the 2026-08-21 owner law).
5. **K sweep on the winner** (K in {2,3,5,7} for DFlash2; the shared policy table's depths
   for MTP) before any default-flip proposal.
6. **ppN placement arm**: the serving split (stages 2/3) with the dflash source — the tap
   host-staging works per-stage by construction, but the box banks the cross-device twin
   receipt (the same named final gate the ppn-verify lane carries for the native arm).

Flip condition for `MEMRA_GLM5_DFLASH` default-ON (unchanged from the FLAGS row): the
three-way window shows DFlash2 winning at equal exactness, with sampled rows and cache-on
twins, and the rollout follows the model-rollout protocol (pinned commit, gate bundle,
`glm5-spec.v1` receipt for sealed bundles).

## 6. RESULTS (rig, 5090, TF32 off, flock; head = this lane rebased onto 9053d538d)

Full battery `glm5_dflash_session_gpu`: **9/9 PASS** (`rig-gates.log`, banked from the
rebased head — the base moved under the lane mid-flight when the MLA-TC default flip
landed; rebased, rebuilt, re-run; post-rebase binary suffixes verified against the rebase
timestamp per the rebuild-after-checkout law).

| # | arm | receipt |
|---|---|---|
| 1 | greedy byte identity K=1..7 | 7/7 byte-identical over worker-sized bursts (`0/19..0/133 accepted` — the mini drafter is random; identity is the claim, acceptance is the box's) |
| 2 | forced-rejection j-sweep | byte-identical AND **schedule-exact 15/42 accepted** — the non-vacuity teeth; they immediately caught their own class (a 20-token reference tape starved the final round's deep accepts: 14/15) |
| 3 | RED tap-shift | tape byte-identical both arms, DRAFT STREAM DIVERGED (feature seam live), acceptance not improved; collapse magnitude = box cell 3 |
| 4 | RED rollback disabled | fails LOUDLY: `layer 1: latent cache overflow — 59 + 1 rows exceeds capacity 59` (un-rolled-back latent appends overflow the plane) |
| 5 | sampled twin | pinned-seed deterministic, burst-split invariant, seed-sensitive |
| 6 | EOS | finishes the session; post-EOS burst empty |
| 7 | K bound | K=8 refuses naming the drafter block |
| 8 | selection matrix | dflash2 line + head-NOT-loaded note + burst ran; both-armed states dflash2 wins; native-mtp line; fail-closed warn; flag-off = zero `[glm5-spec]` lines. Mini-fixture VRAM-at-ready print: no measurable delta at this scale (the mini NextN head is one dense 128-wide block; device-global mem_get_info also sees co-tenant rig noise) — the mechanism receipt is the boot line + the harness assert `model.mtp.is_none()`; per-arm VRAM numbers are box cell 1 |

Native-source regression on the same head: `glm5_spec_session_gpu` **7/7**,
`glm5_tparallel_verify_gpu` **7/7**, `glm5-spec-ppn-gate` stages 2 and 3 **ALL ARMS
PASS** — the seam refactor left the native arm intact, ppN included.
