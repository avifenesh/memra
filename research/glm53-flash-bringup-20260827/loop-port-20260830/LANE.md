# glm5 spec LOOP-PORT: the mature-loop mechanisms, in the banked port order (2026-08-30)

Closing the gap between the first-generation glm5 spec loop and the mature qwen/step loops.
The prize, restated from the 3way decision window (lane/glm5-3way-decision, banked in
`../3way-decision-20260830/LANE.md`):

- round wall = **31.6 + 20.1·K ms** vs a **28.24 ms** plain step on the 3-card serving shape;
- **a 0.67 ms fixed-cost saving flips K=1 positive today** (needs 51.93 ms vs measured 52.6);
- DFlash2 is the drafter of record (+31.3% acceptance at +2.8% round cost);
- target: spec-ON decode > plain on the 3-card serving shape.

Map: darklanes `research/glm5-spec-maturity-20260830/LESSONS.md` (the mechanism table with
file:line anchors, the owner's-word mapping, the port order, and the anti-lesson —
MEMRA_SPEC_ADAPT accepted-run K is REFUTED for fixed-cost regimes; not ported).

- Base: `origin/lane/glm53-flash-bringup` @ `a5334254132672ac1845dbcdf789d5ffccb9d5bd`
  (consol-3w merge; includes lane/glm5-dflash-draft-src f8f35bd91 — the 3way binary's code),
  then MERGED FORWARD to `a76b5b398` (the glm5-docs-sweep advance) after the ports landed —
  every gate below re-run on the merged head; one conflict, `research/tune-data/perf-ci.jsonl`,
  resolved append-only (BOTH sides' rows kept in timestamp order, JSON re-validated).
- Branch: `lane/glm5-loop-port`, worktree `~/projects/wt-glm5-loop-port`.
- Rig law: 5090 exactness-only (flock /tmp/memra-5090.lock, NVIDIA_TF32_OVERRIDE=0);
  every timing number in this doc is RIG-RELATIVE (mini fixtures) and clearly labeled so —
  the box-B flip battery (3way cells 4+6 unchanged) re-prices on the serving shape as a
  separate window. Predicted round-wall arithmetic uses the 3way lane's measured terms.

## Port order (each step lands with its exactness gate GREEN before the next)

0. **Phase timers** (`MEMRA_GLM5_SPEC_TRACE=1`) — the 3way lane's named gap ("the engine has
   NO per-phase timing at this head"). Every subsequent port is attributable.
1. **One merged host sync per round** — device accept argmaxes + deferred tap readbacks +
   device-argmax native chain + async len_d mirrors. THE K=1 FLIP LIVES HERE.
2. **Confidence gating** — MEMRA_SPEC_PMIN/PMIN0 honored by the glm5 loop (native chain
   p-of-pick; DFlash2 selector-q tau-slot truncation). No new flags.
3. **Verify-row marginal diet** — per-row copy/alloc hoists; KDA clone diet; MLA rows.
   Fold-ins: batched native MTP-plane warm fill (TTFT), one-way demote arm.

GATES per step: served spec-vs-plain greedy byte identity K=1..7 (glm5_spec_session_gpu +
glm5_dflash_session_gpu), accept-j rollback identity (glm5_tparallel_verify_gpu), sampled
Philox determinism + split invariance, red arms re-run (they must still bite), spec-ppn-gate
stages 2+3, full server suite, plus the NEW timer receipts.

## Sync census at the base head (the thing port 1 kills), greedy round, DFlash2 arm

Counted by reading `glm_spec.rs` @ a53342541 (line refs at this head):

| # | sync | where | size |
|---|---|---|---|
| 1 | `dtoh(&vals_d)` + `dtoh_u32(&idx_d)` + `dtoh(&hproj_d)` | dflash2_propose_greedy (drafter-internal, 3 small dtohs) | ~nd·(2·16+256) f32 |
| 2 | **5× `dtoh(&contracted)`** | glm5_hc_tap INSIDE the verify walk, one per tapped layer | (k+1)·4096 f32 each |
| 3 | **`dtoh(&vlogits)`** + host argmax ×(k+1) | greedy accept (glm_spec.rs:1351) | (k+1)·n_vocab f32 (~2.4 MB at K=3) |
| 4 | 11× `memcpy_htod` len_d | glm5_verify_rollback MLA arm (:766) | 4 B each, each a host sync |
| 5 | k+1 × `htod_i32` pos_rows + 1 htod embedded | walk entry | small |

Native-MTP arm replaces row 1 with **K× full-d_vocab `dtoh(&d_logits)` + host argmax**
(glm_spec.rs:1293) — the per-draft full-vocab readback the map names.

## Port log

### PORT 0 — phase timers (`MEMRA_GLM5_SPEC_TRACE=1`): LANDED @ a545c1a56

Per-burst `[glm5-phase]` line (rounds + draft/verify/accept/roll/maint totals + per-round
means), the dspark `MEMRA_SPEC_STATS` clock pattern: phase boundaries synchronize the
stream, so the numbers are attribution SHARES, never round walls and never perf rows.
Default OFF, OFF arm byte-identical. The round now returns its true drafted count
(== K until the confidence port truncates it). Gates: tparallel 7/7, spec_session 7/7,
dflash_session 9/9, mtp_head 5/5, spec-ppn stages 2+3 ALL ARMS; trace fires
(receipts/glm5-phase-*.txt). FLAGS.md row in the same commit.

### PORT 1 — one merged host sync per round (the K=1 flip): LANDED

Three seams, all unconditional (no new flag BY DESIGN: each replacement carries a proven
bit-identical contract — `argmax_token_device_col` is "bit-identical to host argmax,
argmax_gate-validated" by its own doc — and the byte-identity batteries are the proof; a
kill-switch fork would double the gate matrix for arms provably equal. The rollback seam
for the whole loop remains `MEMRA_GLM5_SPEC=0`, still the prod default):

1. **Greedy accept** (glm_spec.rs, the map's #2): the `(K+1) x n_vocab` logits DtoH + host
   argmax per row -> per-row `argmax_token_device_col` into one `[K+1]` u32 buffer + ONE
   tiny readback. At the real head (n_vocab ~154k) the old path moved ~2.4 MB over PCIe and
   scanned ~600k floats on host per K=3 round; the new path launches 2(K+1) tiny kernels
   and reads (K+1) u32.
2. **Native draft chain** (the map's per-draft full-vocab DtoH, old :1293): per-draft
   `argmax_token_device` + one 4-byte `dtoh_u32_one`, with the #87 sentinel guard mirrored
   from the spec.rs graph chain. Native stays one small sync per draft (the zero-sync
   device chain is spec.rs graph machinery; native is the fallback arm, not the drafter of
   record — deliberately not ported).
3. **DFlash2 tap sink** (the map's #17, "5 host syncs inside every verify walk"): the
   verify-round `HcTapSink` is now DEVICE-STAGED — `glm5_hc_tap` D2Ds the contracted rows
   into per-slot device buffers on the walking (stage) engine, and `glm5_tap_drain` reads
   all slots back at the round's ONE post-walk sync point. Prime sinks stay host-staged BY
   DESIGN (a [prompt, hidden] x 5 device transient at 16k depth is ~1.3 GiB; the prime's
   per-chunk DtoH amortizes over >= 256 rows and DFlash2 TTFT is near-constant already).

Also in this port: sampled accept call sites now key on `drafts.len()`/`rows.len()`
(identical today; makes the confidence port purely additive).

**Rig-relative fixture traces** (5090, mini fixture: 4 layers, hidden 128, VOCAB 32, K=7 —
NOT a pricing instrument; the removed DtoH is bytes-proportional and the fixture vocab is
4800x smaller than the real head's):

| arm | phase | before ms/round | after ms/round | reading |
|---|---|---|---|---|
| DFlash2 | verify | 4.842-5.171 | 4.643-4.828 | the 5 in-walk tap DtoHs left the walk (~-0.2) |
| DFlash2 | maint  | 0.001 | 0.030-0.034 | the tap drain moved here (post-walk, off the dispatch path) |
| DFlash2 | accept | 0.009 | 0.042-0.044 | at VOCAB=32 the old DtoH was ~128 B — launch overhead dominates the fixture; at 154k vocab the balance inverts by ~3 orders of magnitude |
| native | draft | 1.510-1.537 | 1.592-1.633 | same fixture artifact (per-draft 128 B DtoH -> device argmax launches) |

The fixture-scale accept/draft regressions are the EXPECTED artifact of a bytes-
proportional win measured at 1/4800th of the real row size; the port's real-scale claim
rides the 3way arithmetic (below) and re-prices on box B (cells 4+6 unchanged).

**Predicted round-wall arithmetic** (3way lane's measured terms, greedy K=1, DFlash2):
fixed cost 31.6 ms includes, per round: one ~1.2 MB accept DtoH + host argmax scan
(~0.3-0.8 ms on the serving box), 5 in-walk tap syncs (5x sync latency + walk-dispatch
serialization), and the accept-side sync latency. The port removes all three seams'
blocking behavior; the K=1 flip needs 0.67 ms of them. q38's banked receipt for the same
slice: "2 blocking DtoH/round -> 1 ... blocking ~1.7 ms" — 2.5x the whole K=1 gap.

Gates after port 1 (5090, flock, TF32 off): tparallel 7/7 GREEN, spec_session 7/7 GREEN,
dflash_session 9/9 GREEN (incl. tap-shift red arm still biting — the device-staged
features are consumed live), mtp_head 5/5 GREEN, spec-ppn-gate stages=2 AND 3 ALL ARMS
(the R2 pool-key + R3 rollback-disabled reds still bite by name). clippy zero, fmt clean.

### PORT 2 — confidence gating (MEMRA_SPEC_PMIN / MEMRA_SPEC_PMIN0, no new flags): LANDED

The step37 shipping family (`PMIN=0.5 PMIN0=1` is its serving env), honored by the glm5
loop as the map's item 2 prescribes — the spec.rs knobs are consumed, not re-invented:

- **Native chain (p-of-pick)**: after each pick, p = the head's softmax confidence in it
  (`prob_of_token_device`, the spec.rs `g_p` statistic; one 4-byte read per armed draft).
  The spec.rs break verbatim: `p < p_min && (ki > 0 || pmin0)` — the chain stops BEFORE
  the next full-MoE-layer forward is paid; a discarded sampled draw's Philox advance
  stands (eager parity). Unarmed rounds pay nothing.
- **DFlash2 (tau-slot over selector q)**: the sampled walk already recorded `q_chosen`;
  the greedy walk now records its T=1 twin (`dflash2_walk_greedy_q` — same walk, same
  argmax, q is bookkeeping; every existing greedy caller delegates through it unchanged).
  `glm5_conf_keep` truncates the proposal at the first sub-threshold slot PRE-verify (the
  owner's "take only high confidence offers" dspark form) — a truncated round rides DOWN
  the `31.6 + 20.1*K` line; rejection sampling stays exact for any proposal prefix.
- **Zero-draft rounds (PMIN0)**: verify batch = the anchor row alone, m=1 = a plain
  decode step (the llama.cpp mechanism spec.rs vendored; "acceptance 76% at mean len 2.5"
  is the banked receipt). Greedy rides the general accept (j=0, row-0 argmax); sampled
  takes `glm5_sampled_bonus` — the full-accept filtered-Gumbel draw, extracted from
  `glm5_sampled_accept` byte-for-byte and shared.
- The per-round `[glm5-acc]`/usage.spec counters now carry TRUE drafted counts; the
  `[glm5-spec] draft confidence gate armed: PMIN=..` boot line is the engagement receipt.
- **Deliberate default: BOTH UNSET (gate off).** The tau/PMIN value is a per-model
  measurement (q27 lost 1.9% at PMIN=0.3 on one pack; K=3 chains measured flat-to-negative
  on the 5090 rig class) — the box-B tau ladder prices glm5's before any serving env
  carries it. FLAGS.md rows updated with the glm5 scope in this commit.

NEW GATES (decisive, cannot pass by accident — `pmin_override` on `Glm5SpecKnobs` drives
the arms because the env pair latches OnceLock-once per process):
`gpu_confidence_gate_truncates_drafts_never_the_tape` (native battery) and
`gpu_dflash_confidence_gate_truncates_drafts_never_the_tape` (DFlash2 battery): p_min=1.1
is never cleared, so PMIN0 must force drafted==0 on EVERY round with the tape still
byte-identical to plain (each round IS a plain step), and !PMIN0 must force exactly the
slot-0 survivor; plus the sampled zero-draft pinned-seed twin (deterministic, no stall).
CPU pin: `conf_keep_matches_the_spec_rs_break_semantics` (incl. the non-latching slot-0
miss and the strict-< boundary).

Gates after port 2 (5090, flock, TF32 off): tparallel 7/7, spec_session 8/8 (new gate
in), dflash_session 10/10 (new gate in), mtp_head 5/5, spec-ppn-gate stages 2+3 ALL ARMS,
lib conf_keep test green. clippy zero (lib + tests), fmt clean.

### PORT 3 — verify-row marginal diet, KDA arm + copy hoists: LANDED

**KDA snapshot + scan-input replay (the map's #6, the module doc's own named
GdnStash/ReplaySSM diet).** The per-row (conv, ssm) column clones — 4 MiB x 34 layers
per column, ~0.95 GiB transient at K=7, ~408 MiB of D2D per K=3 round — are retired:

- the walk clones the resident ssm state ONCE per layer per round (before row 0) and
  STEALS each row's scan-input buffers (`kda::KdaScanInputs`: q_l2/k_l2/v_conv/g_log/beta,
  ~160 KB/row/layer, ZERO copies — the step allocated them either way; `kda_core` moves
  them out instead of dropping them);
- partial-accept rollback REBUILDS the state by re-issuing rows 0..keep's original t=1
  `memra_kda_scan_s128` launches from the snapshot (`kda::kda_scan_replay`, ping-pong on
  the resident pair, ends under the `ssm_state` name per `kda_cached`'s swap discipline).
  **Byte-identical BY CONSTRUCTION**: each replay is the very launch that produced the
  retired clone — same kernel, same inputs, same t=1 shape. No T-invariance argument is
  even needed on this path (the batched t=keep form would need one; deliberately not
  taken);
- the conv ring stays per-row cloned (288 KiB = 1.4% of the ssm plane; a replay arm for
  it would stash pre-conv inputs for no material win);
- D2D clone traffic per K=3 round: ~408 MiB -> ~136 MiB + ~16 MiB stolen stash (no new
  traffic); walk launch count per KDA layer: 2/row -> 1/row + 1/round. Rollback pays
  keep tiny scan launches per layer on PARTIAL accepts only (full accepts: nothing,
  unchanged).

**Copy/alloc hoist**: `h_row` is one buffer per layer-range walk instead of one
allocation per (row, layer) — t x n_layers allocations retired per round (the dsv4
stream-ordered-pool churn lesson).

**MLA fa-rows batching (the map's #5) is deliberately NOT in this port**: glm5's MLA is
latent/DSA-gathered and per-row `mla_attn_cached` is what the byte-identity contract is
built on; a seqs-form batched verify append is a new kernel class needing its own
per-row-bit-exactness gate (the map prices it Medium for exactly this reason). It is the
named next lever on the 20.1 ms/K marginal after box B re-prices what ports 0-3 bought.

Gates after port 3 (5090, flock, TF32 off): tparallel 7/7 (accept-j-then-continue byte
identity for EVERY j — the replay rollback's decisive gate — plus the stale-KDA red
still biting), spec_session 8/8, dflash_session 10/10, mtp_head 5/5, kda_fixture 3/3,
spec-ppn-gate stages 2+3 ALL ARMS. clippy zero, fmt clean. Phase trace (rig fixture,
receipts/glm5-phase-port3-dflash.txt): verify 4.60-4.62 ms/round (port-1: 4.64-4.83),
roll 0.020-0.021 (port-1: 0.016 — the partial-accept replay trades a few tiny launches
for the retired 4 MiB clones; at real geometry the traffic term dominates the other way).

### FOLD-IN A — batched MTP-plane warm fill (map #4): LANDED

`glm5_spec_session_new` no longer warms the native plane token-by-token (~400 tok/s =
the measured +2.5 s TTFT per 1k prompt tokens, spec-battery flip condition 1 BY NAME);
`glm5_mtp_plane_fill` runs chunked t-parallel passes: batched embed/enorm/hnorm, the
eh_proj concat via `place_rows_strided` (2 launches, not 2t copies), one prime-class
`mla_attn_cached` append per 512-row chunk (output discarded — the plane rows are the
product; the MoE FFN + head of the per-token chain never run for prompt positions).
MTP rows are independent given the trunk hiddens, so the fill is structurally exact;
the t>1 attention class can only move DRAFTS, never output. Native stays the fallback
arm (DFlash2 of record has near-constant TTFT already); the fix unblocks it as one.
Expected effect (map row): native pool TTFT 2.220 -> ~OFF-class, the 8-turn 11.8-19.8
s/turn class collapses — box B prices it if the native arm is ever re-measured.

### FOLD-IN B — one-way demotion handoff (map #8, ship-safety): LANDED

`Glm5SpecSession` gains the demote seam and the worker's spec-gate HIGH sweep now
covers `glm5_on` rows (previously "the sweep covers spec and dspark sessions only" —
the map's named gap: a session admitted on an idle box kept bursting serially when
load arrived). glm5's live anchor is the CARRIED-PENDING shape (emitted, not yet
committed), so `HybridModel::glm5_spec_into_demoted` flushes it through ONE plain
decode step (the `spec_flush_pending` analogue) and hands `(cache, next_pred)` with
next_pred = the flush argmax, unemitted — exactly the `device_next` contract. One-way
by design (draft state + Philox counters drop, VRAM freed); greedy-only, sampled
refuses BY NAME (the MTP sweep's exclusion verbatim — philox law); the worker arm
mirrors the dspark arm including the `MEMRA_SPEC_DEMOTE_AT` test door and the loud
SKIPPED line on any non-handoff shape. NEW GATE
`gpu_demote_handoff_continues_byte_identical`: a mid-stream demote splices
byte-identically into the never-demoted plain tape (cache rows == prompt + emitted;
next_pred == the tape's next token; plain continuation identical), sampled refusal
named.

## Predicted round wall (the deliverable arithmetic; box B re-prices)

Baseline (3way cell 6, measured): `round_wall = 31.6 + 20.1*K ms`, plain step 28.24 ms,
K=1 wall 52.6 ms vs the 51.93 ms tie bar — short 0.67 ms.

What the ports removed from the FIXED term (all per round, all previously blocking the
host inside or immediately after the walk):

| seam | before | after |
|---|---|---|
| greedy accept | (K+1) x n_vocab DtoH (~1.2 MB @K=1) + host argmax scan over ~310k floats | 2(K+1) argmax kernels + ONE (K+1)-u32 readback |
| DFlash2 taps | 5 blocking in-walk DtoHs (each a stream sync serializing walk dispatch) | 5 async D2Ds + one post-walk drain at the round's single sync point |
| native chain | K x full-d_vocab DtoH + host argmax | K x device argmax + 4-byte readback |
| KDA ckpt | 2 clone launches/row (4.3 MiB), ~408 MiB D2D @K=3 | 1 conv clone/row + 1 ssm snapshot/round (~136+16 MiB @K=3) |
| alloc churn | t x n_layers h_row allocs/round | 1/walk |

The q38 receipt for the accept-readback slice alone: "2 blocking DtoH/round -> 1 ...
blocking ~1.7 ms" — 2.5x the whole 0.67 ms K=1 gap. On the serving box's PCIe the
removed accept DtoH + host scan is conservatively 0.3-0.8 ms and the five tap syncs
0.1-0.3 ms; the arithmetic therefore predicts the K=1 flip (needs 0.67 ms) with margin,
and K=2-3 move by the same fixed-term saving plus the KDA-traffic share of the marginal.
NOT A CLAIM — the flip battery is box B's (3way cells 4+6 unchanged, same pools, same
interleave, vendor-default sampled row + usage.spec receipts per the never-serve-greedy
law). Confidence gating (port 2) additionally converts low-confidence rounds into
shorter rounds or plain steps once the box-B tau ladder prices MEMRA_SPEC_PMIN for glm5.

## Flag rows (deliberate defaults, all in FLAGS.md in-lane)

| flag | default | why |
|---|---|---|
| `MEMRA_GLM5_SPEC_TRACE` | OFF | attribution instrument; phase-boundary syncs serialize the round — never a serving mode, numbers are shares not walls |
| `MEMRA_SPEC_PMIN` / `MEMRA_SPEC_PMIN0` (glm5 scope) | unset (off) | per-model measurement law (q27 -1.9% at 0.3; step37 ships 0.5); box-B tau ladder prices glm5 before any serving env carries it |
| ports 1/3 sync-diet + fold-ins | no new flags BY DESIGN | every replacement carries a proven bit-identical contract (argmax_gate tie-break; replay = the original launch re-issued; flush = the accept-identity claim) and the byte-identity batteries are the proof; the loop's rollback seam remains `MEMRA_GLM5_SPEC=0` (still the prod default) |

## Final gate table (5090 rig, flock /tmp/memra-5090.lock, NVIDIA_TF32_OVERRIDE=0)

| gate | result |
|---|---|
| glm5_tparallel_verify_gpu (walk bitwise, accept-j identity, e2e K=1..7, trim, 3 reds) | 7/7 GREEN |
| glm5_spec_session_gpu (served bursts K=1..7, j-sweep, sampled Philox + split invariance, EOS, receipts red/green, confidence gate, demote handoff) | 9/9 GREEN |
| glm5_dflash_session_gpu (served bursts K=1..7, j-sweep, sampled, tap-shift red, block clamp, rollback red, source matrix, confidence gate) | 10/10 GREEN |
| glm5_mtp_head_gpu | 5/5 GREEN |
| kda_fixture_gpu | 3/3 GREEN |
| glm5-spec-ppn-gate stages=2 | ALL ARMS PASS (R2+R3 reds bite) |
| glm5-spec-ppn-gate stages=3 | ALL ARMS PASS |
| memra-server suite | 481/481 (merged head; 480 pre-merge) |
| memra-engine lib | 253 pass (incl. the conf_keep CPU pin) |
| clippy (engine, server, kv, tests) | zero |
| local-ci --perf (merged head `04d40d4e9`) | exit 0 — correctness stage GREEN, perf stage 0 fail / 0 warn (row qwen9b-plain-short 139.36 tok/s; pre-merge run at `fa231d5f7` was 139.17, base head's own rows 139.25-139.32). Every row `window_clean=false`: co-resident rig traffic did not clear in 600 s, the SAME persistent condition the base head's rows record — a house-gate liveness pass, never a perf claim |

Box A/B is NOT this lane's: the flip battery (3way cells 4+6 unchanged) re-runs on box B
as a separate window against this head.
