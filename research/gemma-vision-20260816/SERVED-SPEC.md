# Gemma-4-31B SERVED SPEC — the assistant drafter through memra-server (lane/gemma-batched, 2026-08-17)

The funded 3-stage arc (sizing: SERVED-SPEC-SIZING.md). All three stages landed; the
spec route is DEFAULT-ON with a drafter attached, per owner law, with the receipts below.

## Shipping config

`MEMRA_DRAFT=<gemma-4-31B-it-official-Q8_0-MTP.gguf>` +
`MEMRA_GEMMA_DRAFT_RANKS=<gemma31b-ranks-32768.gguf.txt>` (447k own-gen FR trim) +
`MEMRA_GEMMA_TRIM_ADAPT=512`, K=5 (the seam default). Trunk: NVFP4mix (official
weights); lineage law holds (official head ↔ official-weights trunk).
Seams: `MEMRA_GEMMA4_SPEC` unset = armed at K=5 when MEMRA_DRAFT present; `0` = plain
kill switch; explicit K >= 1 = that depth; garbage refuses loud. `MEMRA_SPEC_DFLASH`
together with the spec route refuses loud at boot (3f4597f02 law).
**Workaround pin, see the regression below: cells/gates on the Japan box ran with
`MEMRA_Q8RP=0`.**

## Stage 1 — burst-scoped GemmaSpecSession (d08a9b593)

Engine session (`gemma_spec_session_new`/`_burst`, gemma_spec.rs): carries the eager
round loop's exact cross-round state (cache, post-norm h, pending token, adaptive-depth
kc/prev_full); rounds never exit mid-way; overshoot committed and counted; visible
emission stops at first EOS while commit accounting keeps every kept row.
`into_demoted()` hands (cache, pending, committed) to the plain path.

**Banked boundary battery** (`gemma-spec-session-gate`, written BEFORE the refactor):
widths {1, K-1, K, K+1, 32} each byte-identical to one-shot `generate_spec_gemma` AND
plain greedy (128 toks); cache-rows == committed + emitted-prefix invariants; demote
mid-burst → plain continuation == plain greedy. ALL GREEN on 5090 (Q4_0 QAT + QAT-Q8
head), ranks-trim AND full-head configs, and re-run green on the merged state.
**Prompt ids (deterministic re-bank):** `2 105 2364 107 155122 1217 14820 3927 4146
236764 607 614 2591 236761 106 107 105 4368 107` (the accept5 receipt prompt).

## Stage 2 — worker wiring (a7261f3a4)

`step_gemma_spec` mirrors the qwen spec-burst arm (greedy-only): turn-1 prime inside
the session, one burst per tick (MEMRA_SPEC_BURST default 32), emission via
`emit_spec_token_events` (one event per public id, `spec_visible_len` budget clamp).
Admission mirrors the Q38 policy shape, each piece verified against gemma4's arms:
greedy only (penalties excluded — pure-argmax verify), unconstrained, text-only, SOLO
admission (spec on single-stream; batch arrivals take plain-batched), cold sessions
only (no prefix/reuse pools in v1 — nothing to downgrade on cache-hit yet, so the Q38
downgrade-on-cache-hit table has no gemma analogue to mirror). gspec sessions excluded
from prefill/eager/batched-chunk scheduling by predicate; s.cache stays None until
demote.

**LOAD-KEY FINDING (disclosed, not a defect):** `MEMRA_DRAFT` at load forces the
documented stream-k GEMM form for n_embd >= 3500 (hybrid.rs "governs the PRIME too"),
so a drafter-armed boot's plain completions can differ from a no-drafter boot's at
near-tie logits — a load-config class difference. The serving law is therefore
WITHIN-BOOT: `serve-gemma4-spec-gate` proves spec-admitted == plain-fallback
byte-identical on the same trunk (phase-2 decoy holds a slot so the probe takes the
plain path; ambiguity checks: phase 1 must add [gspec-acc] lines, phase 2 must add
zero). Gate prompts: the 6 fixed prompts in `tools/serve-gemma4-spec-gate.sh` (in-repo,
commit-pinned).

## Merged shipping state (8fea1c8b6 + aee786f8d)

lane/gemma-pnfold + lane/gemma-fused2 merged per coordination; the batched arm rides
pnfold's `gemma4_layer_tail_add_nq_pn` tail front, so batched == eager holds by
construction at either MEMRA_G4_PNFOLD value. Merged-state re-gates: decode-batch-gate
config ALL GREEN, session battery ALL GREEN, serve batch gate ALL GREEN vs the folded
eager stream (5090, Q4_0). E4B flag restated from PN-FOLD-RESULTS.md: E4B serving
boots `MEMRA_G4_PNFOLD=0` until the E4B lane re-gates; nothing in this lane serves E4B
through any new path.

## REGRESSION FILED — capacity-keyed Q8RP default garbles NVFP4 on the PRO 6000 class

At pnfold HEAD (6cc3e016c) and at this lane's merged HEAD, PLAIN gemma-gate decode on
the NVFP4mix artifact emits token-0 garbage on the Japan box, and the spec path dies
with CUDA_ERROR_ILLEGAL_ADDRESS (which then poisons the context — every subsequent
request 500s). Bisect receipts (box GPU1, all with the accept5 prompt ids):
- 194782617 (pre-pnfold): plain GREEN (canonical `[100, 45518, 107, ...]`).
- 6cc3e016c (pnfold HEAD): plain GARBAGE; kill-switch sweep — `MEMRA_G4_PNFOLD=0`,
  `MEMRA_PDL_NVFP4=0`, `MEMRA_PDL=0`, `MEMRA_NVFP4_FUSED2=0` all still garbage;
  **`MEMRA_Q8RP=0` restores the canonical continuation.**
- aee786f8d + `MEMRA_Q8RP=0`: plain 61.13 tok/s (matches the pnfold ledger ~60.05),
  spec 156.47 tok/s (2.56×), 128/128 agreement.
The capacity-keyed Q8RP default (eb915bfc2, fused2 range) never engages on the 24GB
5090 (why every local gate was green) and corrupts on this 96GB card class. The
5090-vs-PRO-6000 asymmetry means the fused2/gap lane's own NVFP4 receipts need
re-examination — their serving numbers were measured somewhere this default engaged.
**Owner/gap-lane action needed; every Japan cell below runs with the `MEMRA_Q8RP=0`
pin, disclosed.** This lane does not flip another lane's default.

**Coordinator confirmation (2026-08-17):** the recipe-probe lane independently proved
the same root (03c727043 — Q8RP default NaNs gemma4 prefill on Q8_0-carrying
artifacts on fresh 96GB boots; pad-spam serving). Standing directives, encoded here:
- Every server/cell boot in this arc carries `MEMRA_Q8RP=0` explicitly and every
  banked receipt notes the env (the cells jsonl carries an appended annotation record;
  the stage-3 gates/cells all booted through run-stage3.sh which exports the pin).
- Receipts that PREDATE the Q8RP arm need no pin: the batched-arm cells at 1af8bfb59
  and the flip-confirm cells at 194782617 ran before eb915bfc2 entered this lane
  (pre-pnfold-merge commits have no Q8RP code at all), and all 5090 receipts are
  additionally immune (24GB — the capacity key never engages).
- Baseline hygiene: the pnfold c1 ledger (60.05 plain) and this lane's single-probe
  61.13 CLI read are SUSPECT-PENDING-RECERTIFICATION; the trustworthy NVFP4 c1 plain
  baseline is the recipe-probe's `MEMRA_Q8RP=0` cells — **58.10 control** (58.76 with
  the Q6_K-embd variant if the owner blesses it).
- **OPEN OBLIGATION:** when the gap lane's Q8RP fix lands and re-certifies, the
  stage-3 final cells (spec c1 prose/code, batch c8/c16, mixed) RE-RUN on the fixed
  state with the mirror ENGAGED — the shipping config must be what's measured. Until
  then the numbers below are the pinned-config record, not the final shipping bank.

## Stage 3 — served cells (Japan GPU1, NVFP4mix, 450W, MEMRA_Q8RP=0, 5 reps)

Server booted once on the shipping config (drafter default-on). Gates first, same boot
class: spec serve gate ALL GREEN (6 prompts, within-boot spec == plain), batch serve
gate ALL GREEN (8 prompts, eager == batch-c1 == batch-c4). Cells
(`tools/gemma4-spec-cells.py`, prompts recorded verbatim in the script; SSE-timed
decode rates exclude prime, the bench convention; receipts `served-spec-cells.jsonl` +
box `/data/memra/evidence/gemma-batched-20260817/`):

| cell | median (min–max, n=5) | note |
|---|---|---|
| spec-c1 prose decode | **135.5 tok/s** (132.6–135.6) | ttft p50 0.31s |
| spec-c1 code decode | **211.3 tok/s** (207.7–211.4) | ttft p50 0.21s |
| batch-c8 aggregate | **258.1** (255.1–258.2) | above the 245.6 flip receipt (+pnfold) |
| batch-c16 aggregate | **269.1** (268.8–269.4) | above the 257.3 flip receipt |
| mixed: spec side | **112.1 tok/s** (110.4–112.1) | live stream while c8 batch runs |
| mixed: batch side | **169.8 agg** (168.3–169.9) | c8 under a live spec stream |

- **The mixed coexistence cell is GREEN** — the acceptance criterion the whole-request
  shortcut failed by construction: a spec stream holds 112 tok/s while c8 batch traffic
  simultaneously clears 170 agg on the same GPU; neither side starves. 160 [gspec-acc]
  lines over the run; the ambiguity discipline held per cell.
- **Served-vs-bench delta, honestly:** CLI one-shot at the same HEAD/artifact on the
  receipt prompt reads 156.5 tok/s; the served prose cell reads 135.5 on a DIFFERENT
  prose prompt (in-repo). The gap mixes prompt-class acceptance and per-burst serve
  overhead (per-burst scratch + host emission); the code-class cell (211.3 served)
  shows the wiring does not cap the route. A same-prompt served-vs-CLI split is a
  follow-up nicety, not a blocker.
- Spec on c1 vs plain batched c1 (~60 tok/s): **2.3–3.5× single-stream win served.**

## Verdict (pre-recert)

All three stages landed, gated, and pushed per stage. The route is DEFAULT-ON with a
drafter attached (mixed cell green → owner law), kill switches everywhere, refuse-loud
on every ambiguous env. Remaining niceties (non-blocking): round-cadence on_commit
streaming, prefix/reuse pool for gemma spec sessions, auto-demote policy on high
concurrency (handoff is gated and live; policy wiring deliberately minimal in v1),
same-prompt served-vs-CLI delta cell.

# FINAL BANK — integrated shipping config (lane/gemma-ship, 2026-08-17)

The OPEN OBLIGATION fired and closed: the gap lane root-caused and fixed Q8RP
(abf155e8 — a pre-existing `build_q4_rp_swap` hijack of non-Q4_0 tensors; qtype guard
now in the swap fn; mirror-on == mirror-off byte-identical including prefill). This
section REPLACES the pinned-config record above as the shipping bank.

**Integrated config (commit shas):** lane/gemma-ship = lane/gemma-batched
(516bc0d9f) + lane/gemma-pnfold @ e85125e9d (carries abf155e81 Q8RP fix, 51a06edce
recert audit, pn-fold 4066b240e) + lane/gemma-fused2 (3bc22344e PDL + 48db56287
fused2 + eb915bfc2 Q8RP default — ancestors via pnfold). Ship-lane commits:
0708e658e (merge), 13afd1fff (output-sample gates + pin retirement), 2c52c41a1
(battery wrapper + E4B receipt). NO MEMRA_Q8RP pin anywhere — the mirror is ENGAGED;
the fix is the guard, the per-boot output-sample gate is the standing sentinel.

**Artifact swap executed:** owner-blessed Q6_K-embd NVFP4mix
(`gemma-4-31B-it-NVFP4mix-embdQ6K.gguf`) is the shipping trunk; LANE-STATUS.md on
the box updated. Head/ranks unchanged (official-Q8 MTP + 447k trim + 512 adapt, K=5
seam default).

**E4B merge flag GATED, not assumed:** the dc site is unreachable for E4B by
construction (decode-dc-gate refuses: "e4b has no device-counter decode step");
E4B's served arm (eager) is byte-identical fold-on vs fold-off over 24 tokens,
coherent stream (local card receipt).

**Identity gates on the shipping artifact (Japan GPU0, default env, mirror engaged):**
spec serve gate ALL GREEN (within-boot spec == plain, 6 prompts, boot-sample
non-degenerate, ambiguity both phases), batch serve gate ALL GREEN (8 prompts,
eager == batch-c1 == batch-c4, both boot-samples non-degenerate).

**Final shipping battery** (Japan GPU0 @450W, interleaved ×5, alternating
default/kill-switch boot order, per-boot output-sample gates 10/10 green, zero
anomalies; receipts `ship-cells.jsonl` + box `/data/memra/evidence/gemma-ship-20260817/`):

| cell | median (min–max, n=5) |
|---|---|
| plain c1 (kill-switch reference) | **61.5 tok/s** (61.4–61.5) — coheres with the gap lane's 60.46 recert + Q6K's ~+1% |
| spec c1 prose decode | **121.3 tok/s** (121.3–121.5) — 1.97× plain |
| spec c1 code decode | **219.7 tok/s** (219.5–220.0) — 3.57× plain |
| batch c8 aggregate | **271.7** (271.5–272.0) |
| batch c16 aggregate | **277.8** (277.7–278.1) |
| mixed: spec side | **107.8** (107.8–108.6) |
| mixed: batch side | **169.9** (169.9–170.3) |

spec ttft p50 0.25s. Config-delta note vs the pinned-config record (base embd +
mirror off): prose spec reads lower (121.3 vs 135.5) and code higher (219.7 vs
211.3) — the Q6K embd + engaged mirror shift draft acceptance by class; reps are
dead flat and every identity gate is green, so this is the config's true shape, not
noise or wrongness. Every serving-quality claim rides the shipping artifact now.

**This lane is the release candidate.** No main merge, no version bump, no
catalog/pricing — the release train is the owner's call.

## Prose-delta decomposition (embd A/B, 2026-08-17) — ONE-VARIABLE VERDICT

The ship bank's config-delta (prose spec 135.5 → 121.3, code 211.3 → 219.7) had two
candidates: the Q6K embd or the pnfold×assistant-drafter interaction (pnfold's
acceptance re-bank was dflash-only). Interleaved ×5 A/B at ship HEAD, mirror engaged,
default env, ONLY the trunk embd varying (`tools/gemma4-embd-ab.sh`; receipts
`embd-ab.jsonl` + box `.../embd-ab/`; per-boot output-samples green, ambiguity
guards green, all reps dead flat):

| cell | Q8_0 embd | Q6K embd |
|---|---|---|
| spec c1 prose | **137.5** (137.5–137.5) | 121.5 (121.5–121.5) |
| spec c1 code | 215.7 (215.7–215.8) | **220.1** (220.1–220.1) |

**The embd is the lever; pnfold×assistant is EXONERATED** (prose fully recovers on
Q8_0 embd at a HEAD that includes pnfold — in fact 137.5 slightly beats the 135.5
pinned record, consistent with pnfold's spec win). Mechanism per the acceptance
receipts: the drafter's proposals ride the embd distribution — prose acceptance
~0.40 on Q8_0-embd vs ~0.36 on Q6K; code the reverse (~0.68 vs ~0.72).

**Per-class economics for the owner's serve-best call** (all on the fixed mirror):
- **Q6K embd:** plain c1 +1.1% (61.5), code spec +2.0% (220.1), prose spec −11.6%.
- **Q8_0 embd:** prose spec +13.2% (137.5), code spec −2.0%, plain −1.1%.

Recommendation: pick the shipping embd on traffic mix — code/agentic-dominant
traffic (the OR-board positioning) keeps **Q6K**; prose/chat-dominant traffic swaps
to **Q8_0 embd**. Both artifacts stay minted on the box; the swap is a one-path
`MEMRA_MODELS` change with no engine or seam implications, and the identity gates
are green on both trunks.

## ffn_down Q6_K promotion re-bank (2026-08-17) — **FINAL SHIPPING BANK (owner-ruled)**

**Owner ruling ("accepted your call"): `gemma-4-31B-it-shipQ6K-downQ6K.gguf` IS the
shipping trunk — final, not provisional.** The table below is the FINAL shipping
bank; the embdQ6K bank above stays recorded as the minted-fallback reference, with
the per-class trade table as the decision record. The shipping trunk's identity
gates are exactly the banked ones from this ladder: spec serve gate ALL GREEN
(within-boot spec == plain, 6 prompts), batch serve gate ALL GREEN (8 prompts,
eager == batch-c1 == batch-c4), per-boot output-samples 12/12 non-degenerate
(2 gate boots + 10 battery boots) — all measured ON the downQ6K trunk at ship HEAD,
default env, mirror engaged. No further gate work is owed for the trunk decision.

**Follow-on (gap lane, not this lane):** Q6_K batched-dequant tuning targets the c8
−5.9% mechanism; its wins would re-bank c8/c16 later. ~~Until then THIS bank is the
reference.~~ **Superseded by the FINAL BANK v2 below (zoofusion merge).**

# FINAL BANK v2 — release candidate (zoofusion integrated, 2026-08-17)

lane/gemma-zoofusion merged (a707ec3c6 eager qkv fold, ad2c75704 capacity-keyed f16
prefill mirrors, 1c0c93230 certification, 6007bcce2 receipts) — merge commit
3a5c1f6b3, ladder at f62c5fb25. Shipping trunk unchanged (owner-ruled
shipQ6K-downQ6K). Mirror engagement PROVEN per boot: `[q4f16] prefill fp16 mirrors
built: 110 dense trunk tensors` + `[q8rp] split-plane decode mirrors built: 50`.

**Identity gates on the merged state (Japan GPU0, default env, mirrors engaged):**
spec serve gate ALL GREEN (within-boot spec == plain), batch serve gate ALL GREEN,
per-boot output-samples all green, local decode-batch + spec-session batteries ALL
GREEN (5090, Q4_0).

**Final battery** (Japan @450W, interleaved ×5, alternating default/kill-switch
boots, TTFT banked per cell; receipts `ship-cells-v2.jsonl` + box
`.../zoofusion/`):

| cell | v2 median | vs v1 | ttft p50 |
|---|---|---|---|
| plain c1 | 64.5 | +0.1% | **0.055s** |
| spec c1 prose | 131.3 | +0.2% | 0.271s |
| spec c1 code | **229.6** | **+5.5%** | 0.192s |
| batch c8 | **268.2** | **+4.8%** | **0.029s** |
| batch c16 | 273.0 | −0.4% | 0.056s |
| mixed spec / batch | 111.2 / **175.0** | +4.4% / **+9.0%** | 0.27s / 0.57s |

**PROTOCOL RECONCILIATION (read before comparing c8 numbers):** this battery's cells
are PREFIX-CACHED fixed-prompt loads (decode-dominated; TTFT tiny by construction).
The fusion lane's certification cells are COLD-PROMPT/UNCACHED (~150-token fresh
prompts): its c8 read 173.5 → 236.0 (+36%) and ttft 1.609s → 0.431s (−73%). Both are
real; neither is a regression of the other — cold-prompt c8 is prefill-dominated
(where the f16 mirrors live), cached c8 is decode-dominated. The serving
conversation should quote: cached c8 268.2 / cold c8 236.0 / cold ttft 0.43s p50.

**Item-5 check (c8 decode vs the embdQ6K reference):** v2 cached c8 = 268.2 vs
271.7 embdQ6K — the downQ6K c8 dent shrank from −5.9% to **−1.3%** after the
prefill fix. Not material; the sized follow-up (Q6_K m=8 decode kernel, 840GB/s vs
1.26TB/s, varlen playbook) would close and exceed it — NOT built in this round per
scope.

**Argmax-margin gate finding (filed, non-blocking):** the run-gen prefill-vs-decode
gate reads flips=2/bad=1 on gemma-31B × board-2048 INVARIANTLY — same result at
pre-merge HEAD (5090, Q4_0) and on the box (downQ6K) with mirrors ON and OFF
(receipts `.../zoofusion/` + `.../zoofusion/f16off/`). Both flips are
margin-explained (config_delta > margin); the FAIL is the default --max-flips 1
budget, i.e. the gate's own documented two-legitimate-arithmetics near-tie class
with a stale default calibration for this model — not a zoofusion or mirror defect
(the mirror-off invariance is the proof). Two actions for the owning lanes: (a) the
gate needs a gemma-31B calibration row; (b) the fusion lane's "argmax gate PASSES"
cert line is not reproducible from its receipts (no invocation recorded) — future
certs should bank the invocation. Serving correctness is unaffected: the byte-level
serve-stream identity gates (spec==plain, eager==batched) are green through
mirror-engaged boots, and dflash acceptance is exactly unmoved on the fusion tape.

**This is the release-candidate FINAL bank.** ~~No main merge, no version bump, no
catalog/pricing — the owner's serving conversation opens from here.~~
**Superseded by FINAL BANK v3 below (q6kb merge).**

# FINAL BANK v3 — release candidate CLOSING bank (q6kb integrated, 2026-08-17)

lane/gemma-q6kb merged (caab5a822 capacity-keyed KQRP K-quant split-plane mirrors —
Q6_K batched tier off the misaligned GGUF walk, kernel 88→66µs; a0fe2f2de results) —
merge 39d354334. Diff scope verified: one hybrid.rs admission block (the Q8RP
capacity pattern; 24GB rigs refuse by construction), no serving/scheduler code.
Mirror engagement PROVEN per boot: `[q8rp] split-plane decode mirrors built: 111
tensors` (50 → 111 — the K-quant set joined) + `[q4f16] prefill fp16 mirrors: 110`.

**Gates on the merged state:** calibrated argmax gate PASS on the shipping trunk;
spec + batch serve gates ALL GREEN; per-boot output-samples all green; local
decode-batch + spec-session batteries ALL GREEN (5090, Q4_0).

**FINAL BANK v3** (Japan @450W, interleaved ×5, default env, BOTH protocols, TTFT
banked; receipts `ship-cells-v3.jsonl` + box `.../q6kb-v3/`):

| cell | v3 median | vs v2 | ttft p50 |
|---|---|---|---|
| plain c1 | **65.7** | +1.8% | 0.055s |
| spec c1 prose | **138.5** | **+5.5%** | 0.260s |
| spec c1 code | **243.0** | **+5.8%** | 0.184s |
| batch c8 (cached) | **282.6** | **+5.4%** | 0.027s |
| batch c8 (cold) | **235.8** | new cell | 0.181s |
| batch c16 | **287.5** | **+5.3%** | 0.054s |
| mixed spec / batch | **117.0 / 187.0** | +5.2% / +6.9% | 0.26s / 0.45s |

- Every cell moved the predicted direction (q6kb's ~+5% batch / +2% plain confirmed
  served); zero adverse findings; all reps dead flat.
- The cold-c8 cell independently reproduces the fusion cert's cold protocol (235.8
  vs their 236.0) — the two protocols now reconcile inside ONE battery.
- **The downQ6K c8 dent vs the embdQ6K reference is CLOSED AND INVERTED**: cached
  c8 282.6 vs 271.7 (+4.0%) — zoofusion + q6kb together retired the trade the trunk
  ruling accepted.
- Spec prose (138.5) now exceeds even the Q8_0-embd arm's 137.5 — the embd trade's
  prose cost is fully recovered on the shipping trunk.

**Release-candidate CLOSING bank** unless the owner rules otherwise. No main merge,
no version bump, no catalog/pricing — the serving conversation opens from here.

### Original re-bank record (pre-ruling analysis, kept verbatim)

The recipe-vdown lane's ffn_down Q6_K passed the blessed quality bar (+4.6% plain on
its cells, agree 8/24 == the embd-Q6K bar, exactness green — lane/gemma-recipe-vdown
e71006a81). Promotion ladder ran same-discipline on `shipQ6K-downQ6K.gguf` at ship
HEAD (identity gates ALL GREEN — within-boot spec==plain, batch identity, per-boot
output-samples 10/10 + both gate boots; battery interleaved ×5, default env, mirror
engaged; receipts `ship-cells-downq6k.jsonl` + box `.../downq6k/`):

| cell | downQ6K | vs embdQ6K bank |
|---|---|---|
| plain c1 | **64.4** | **+4.8%** (probe's +4.6% confirmed served) |
| spec c1 prose | **131.1** | **+8.1%** (acceptance 0.421 vs 0.388 — NOT codec-insensitive on serving cells) |
| spec c1 code | 217.7 | −0.9% |
| batch c8 | 255.8 | **−5.9%** |
| batch c16 | 274.1 | −1.3% |
| mixed spec / batch | 106.5 / 160.5 | −1.2% / −5.5% |

**Two findings the probe's plain-c1 cells could not see:**
1. Prose spec acceptance IMPROVES on downQ6K (0.42 vs 0.39) — "codec-insensitive"
   held on the probe's cells, not on the serving classes; the prose gain compounds
   verify-cost and acceptance.
2. **Batch aggregate REGRESSES ~6% at c8** (255.8 vs 271.7; mixed batch side −5.5%).
   Mechanism consistent with Q6_K dequant costing more per weight than Q8_0 in the
   batched m=8 tier while the byte savings pay off in the BW-bound m=1 tier.

**Per-class economics (downQ6K vs embdQ6K, both quality-free):** interactive/c1
traffic wins everywhere that matters (+4.8 plain, +8.1 prose spec, code ~flat);
batch/harvest-heavy traffic pays ~6% at c8. Per the serve-best law this is an OWNER
traffic-mix call, so the swap is **PROVISIONAL**: banked here, staged on the box,
finalize on the owner's word. Both trunks minted; both banks exist at ship HEAD, so
the zoo-fusion arc can baseline against whichever the owner picks.
