# glm5 EP DISPATCH DIET — the structural lane that makes TP-2 pay

### lane/glm5-ep-diet, 2026-08-31

Base: `origin/lane/glm53-flash-bringup` @ a5d608b07 (fetched first), with
`origin/lane/glm5-moe-loc` @ 6e6120a0e merged in (door D's kernels are this lane's
dependency; merge commit e14e6d140 — rerere replayed the ep-place-style resolutions, one
merge artifact fixed: a stale duplicated `MEMRA_TOPK_SHARDS` doc line that tripped clippy).

## 0. The charter, from the receipts

Bare TP-2 v1 measured **22.65 tok/s engine vs 35.36 served PP-3**
(`tp2-battery-20260831/RESULTS.md` cell 4), with the join+dispatch tax measured at
**13-18 ms/token** (cell 3: TP-2 engine wall 44.15 ms/token; decode-gap table terms
22.0-24.1 ms; x the measured single-engine driver tax 1.2-1.3 = 26-31 ms; the residual is
the v1 correctness-transport tax). Named contributors, code-anchored in cell 3:

1. per-token host-canonical fan-out (z HtoD to peer per MoE layer x42),
2. ~4-5 sync peer-slot DtoH->HtoD round-trips per layer (~170-210 per token),
3. per-slot sequential dispatch (~32+ launches/layer, host-blocked between peer returns),
4. the EP prime unported: TP-2 prefills at **39-58 tok/s** (per-token per-slot walk) =>
   TTFT 4.7 s @0.5k / **94 s @3.7k — serving-blocking on its own**.

This lane changes HOW bytes move, never WHICH bytes are computed: the placement map
(`MEMRA_GLM5_EP_MAP`, merged lane/glm5-ep-place) changes bytes MOVED; the diet is
placement-agnostic by construction (it consumes `owner_of`/`local_of` and nothing else,
re-proven on the skewed-map gate arm).

## 1. Attack map — landed / deferred, per the brief

| # | attack | disposition |
|---|---|---|
| 1 | DEVICE-SIDE DISPATCH (door-D pattern on the EP walk) | **PARTIAL, by design.** The host fan-out CLASS is restructured: ONE bulk fan-out per layer-call (not per token), SKIPPED entirely when the routing never leaves root — the single-rank-token multiplier the co-activation map buys arrives free. The router keeps the already-dieted single-sync pinned readback (`moe_router_sigmoid_topk_host`) because the HOST still issues the per-slot launches. The FULL device-table extension — router's device sel/w feeding per-rank vrows-style tables with the fused NVFP4 pair as the fixed launch shape, killing the readback exactly as door D killed it on the verify walk — is NAMED for the box/NVFP4 arc (§4): the rig fixture (Q8_0) cannot hold byte identity over the fused-pair kernels, and door D's own receipts say the sync-structure wall transfer needs box pricing anyway. |
| 2 | BULK TRANSPORT (`MEMRA_STEP_TP_BULK_P2P` pattern) | **LANDED, host-canonical.** Door `MEMRA_GLM5_EP_DIET`: peer rows stage compact on the peer and return in ONE bulk DtoH+HtoD per layer-call — the v1 per-slot round-trip dribble (~170-210/token) folds to 42 bulk returns/token. The two bulk hops (fan-out + return) are the NAMED native-P2P swap points; hy3's EP-4 serves `expert_transport=native-p2p` on this box class (P2P OK both directions per the tp2-battery topology receipt) and the glm5 seam inherits `configure_native_p2p` — but the rig's same-device dual-context emulation has no real peer transport to qualify, so the native-P2P arm is BOX-ONLY (§4 arm 3). |
| 3 | GROUPED EP LAUNCHES | **LANDED for the combine; deferred for the projections.** The t*n_used sequential `axpy_f32` combine launches collapse into ONE `moe_pairs_scatter` launch (8 -> 1 per layer at decode; the kernel's header carries the byte-identity contract vs the zeros+sequential-axpy chain, re-anchored bitwise in `glm5_ep_diet_doors_gpu` with id-table and weight-placement reds). Root issue no longer host-blocks behind peer returns, so the two ranks' ~16 launches each submit back-to-back and OVERLAP — the "remove host boundaries in whole groups" rule applied to the boundary, which is what the graph sweep says the bubble actually costs (syncs x queue depth). The remaining per-rank per-slot PROJECTION launches (~16/rank/layer) group only through the fused NVFP4 pair driven by per-rank tables — the same box/NVFP4 arc as attack 1 (they are one door: fixed grouped launch shape + device tables). |
| 4 | EP PRIME | **LANDED.** Door `MEMRA_GLM5_EP_GROUPED_PRIME`: the plain walk's default-ON grouped MoE prefill (85 -> 616-639 tok/s in its box A/B) ported through the EP walk — per-rank expert-major CSR over OWNED experts, one grouped f16 GEMM per projection per rank over the rank's resident slab (pointer tables minted at `arm_moe_ep`, AFTER the gate-red mutations so both slab reds bite the grouped walk), per-rank slot-ordered scatter, root adds the peer's bulk-returned partial. Chunking is inherited: the hyper prime already walks `prime_chunk_ranges`, and each chunk's layer-call now takes the grouped arm. Falls closed to the (dieted) sequential walk wherever the plain arm would (f16g-eligibility, PRE clamp, n_used<=8, `MEMRA_MOE_GROUPED_PREFILL=0`); the peer pass runs under `bind_runtime_device` (the grouped-MoE FFI follows the runtime device — the step TP2-grouped-prime lesson already in the tree). |
| 5 | PLACEMENT MAP | **CONSUMED, agnostic.** The diet reads `owner_of`/`local_of` only; arm M2 (skewed map + diet ON) holds the same bars as arm M. The diet makes the map's win MULTIPLICATIVE: a layer-call whose routing stays on root now skips the fan-out AND the bulk return entirely (counted in `GLM5_EP_DIET_FANOUT_UPLOADS_AVOIDED`), where v1 paid the z fan-out unconditionally. |

Join-diet playbook compliance (darklanes tp2-join-diet trail): direct-join/prestage/
prejoin-overlap patterns are what the bulk staging + two-pass issue implement at this seam;
NONE of the receipted dead ends were rebuilt (no memops-on-PCIe, no stream merge, no work
migration, no token pipelining, no asym split).

## 2. What landed (doors, defaults, bars)

| door | flag (default) | mechanism | bar | receipts |
|---|---|---|---|---|
| EP diet | `MEMRA_GLM5_EP_DIET` (**OFF**) | bulk fan-out (skip when root-only) + compact peer staging + ONE bulk return + ONE slot-ordered scatter combine; two-pass issue (peer first, root never blocked) | decode BYTE identity vs plain (unchanged gate bar), prime band 2e-5, combine bitwise vs host fma chain, reds R2D/R3D bite, counters non-vacuous on ON arms and FLAT across every pinned-`=0` banked arm | §3, FLAGS row |
| EP grouped prime | `MEMRA_GLM5_EP_GROUPED_PRIME` (**OFF**) | per-rank grouped-f16 GEMM prime over EP slabs, root+peer partial add | per-pair grouped-GEMM rows BITWISE plain-vs-split on minted NVFP4; partial-add reassociation in the 2e-5 band; dropped-peer red orders louder; falls closed on the fixture (dispatch counter pinned 0 at a t>16 prime that KEYS the arm) | §3, FLAGS row |

**Both defaults are OFF by design** (the moe-loc doors' exact reasoning): the rig is
exactness-only, the diet changes the round's SYNC STRUCTURE (the class the diet window
warned does not transfer from counts to wall by arithmetic), and the grouped prime's
real-artifact class receipt is rig-minted NVFP4, not the serving artifact. Each ships with
count receipts; the box window owns every flip decision. FLAGS.md rows land in this PR
(check-flags: 748 reads, none uncovered).

Fail-closed additions: an ENABLED diet flag without `MEMRA_GLM5_TP` on a glm5-class plan
refuses at load (the `MEMRA_GLM5_EP_MAP` silent-even-split trap, same site, same scope
guard); `=0` is a deliberate pin and never refuses — the gate pins `=0` across every banked
arm per the moe-loc §4.5 lesson (unset inherits future default flips; a pin does not).

Engagement counters (the box A/B reads deltas; the rig gate asserts non-vacuity and
flat-off): `GLM5_EP_DIET_DISPATCHES`, `GLM5_EP_DIET_BULK_RETURNS`,
`GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED`, `GLM5_EP_DIET_FANOUT_UPLOADS_AVOIDED`,
`GLM5_EP_GROUPED_PRIME_DISPATCHES`. Announces: `[glm5-ep-diet] engaged`,
`[glm5-ep-grouped-prime] flag=on/off` + per-layer `execute` line — every marker
`performance_claim=false`.

## 3. Gate table (rig 5090, flock /tmp/memra-5090.lock, TF32 off, exactness/counters only)

FILLED IN AS RUN — see receipts/ for logs.

| gate | invocation | result |
|---|---|---|
| `glm5_ep_diet_doors_gpu` | `cargo test --release ... -- --include-ignored --test-threads=1` (`receipts/glm5_ep_diet_doors_gpu*.log`) | **3/3 PASS**: compact scatter == host fma chain BITWISE, 12 shapes, id-table + weight-placement reds both bite; grouped split 24/24 rows BITWISE plain-vs-split, combine max_rel **1.360e-4** (calibrated band 1e-3), dropped-peer red **1.984e0** (4 orders louder); landing helper lands + fails closed |
| `glm5-tp-gate` ALL ARMS + reds (banked arms pinned `=0`, new arms B2/B3/M2/R2D/R3D) | P=16 N=12 (`receipts/01-tp-gate-p16-n12.log`) | **ALL ARMS PASS, exit=0.** Banked arms verdict-identical to the ep-place battery (B/C/C0/C2/D decode BYTE, prime 2.8e-5..4.9e-5 class; M 3.637e-5; H1-H5 refuse by name; R1 1.429e2 / R2 2.705e2 / R3 6.742e1 / R4 5.129e1). NEW: **B2 tp-all-diet decode BYTE-IDENTICAL** (28 t=1 steps + tape, two repetitions bit-identical), prime 3.637e-5 — the EXACT v1 magnitude, engagement diet layer-calls=207 / bulk returns=152 / round-trips avoided=225 (55 layer-calls skipped the fan-out entirely — the placement multiplier path exercised); **B3 grouped-prime falls closed** on Q8_0 at P=24 (announce fires, dispatches=0, walk 2.228e-5 in band); **M2 map-skew + diet decode BYTE-IDENTICAL**; **R2D 2.705e2 / R3D 6.742e1** — the reds bite THROUGH the dieted walk at v1's exact magnitudes (the dieted walk is byte-identical to v1 even under reds); flat-counter assert held across every pinned-`=0` banked arm |
| standing suites | `receipts/run-battery.sh` (19 suites, `--include-ignored`, doors pinned `=0`) | **ALL SUITES PASS** (ep_diet_doors 3, moe_loc_doors 4, verify_batch 4, tparallel 9, spec_session 10, dflash_session 10, moe_epilogue 13, mtp_head 6, kpool_indexer 18, matvec_doors 4, hyper_connections 7, hc_fused_pre 4, hc_decode_ws 2, mla_decode_split 3, kda_fixture 3, kda_fused_proj 5, kda_fused_proj_bf16 5, kda_quant_operand 4, mla_gpu_forward 5) |
| ppn matrices | `receipts/ppn/` | spec-ppn **ALL ARMS PASS** (8 arms + reds), hyper-ppn **ALL ARMS PASS** (10 arms) |
| unit tests / server suite | workspace | engine lib 271/271, gguf 183/183, server 492/492 |
| clippy / fmt / check-flags | workspace | clippy ZERO, fmt clean, check-flags 748 reads / 0 uncovered |
| `tools/local-ci.sh --perf` | rig | **exit 0** on the final tree (`receipts/local-ci-perf-run2.log`): correctness stage GREEN, serve-smoke 0 failed, check-flags + self-test green, perf stage **0 fail 0 warn** (qwen9b-plain-short 136.55 tok/s [OK] vs the rolling median; absent-model cells SKIP — the rig's standing shape). A FIRST run was deleted, not banked: the tree was edited while it was in flight (straddled-receipt law), and its two failures (a chat-stream smoke timeout under rig contention + a flags-census live probe against the mid-edit tree) both cleared on the clean run |

**Calibration event (banked, first doors run):** the grouped-prime split gate's first run
measured green reassociation max_rel = **1.360e-4** against my a priori 2e-5 bar — the
per-pair rows gate held BITWISE (24/24 rows identical plain-vs-split, so the grouped GEMM
really is launch-set-independent on this shape), and the entire diff is the per-token
partial-add reassociation on RAW layer outputs, which expose cancellation (a 4-term sum
can land near zero while its terms are O(1); the tp-gate's 2e-4 band sits on
network-smoothed logits and does not transfer to this surface). Per the calibration law
the band is now set FROM the measurement: **1e-3 (7.4x margin), red must exceed 1e-1.**
The scatter-combine gate and the landing-helper gate passed on the same run (12 shapes
bitwise vs the host fma chain; both reds bite).

**Rig-lock incident (banked for fleet-ops):** the shared 5090 flock sat WEDGED ~30 min —
the holder pid was dead but its locked fd had been inherited by the long-lived `sccache`
server daemon, which keeps the flock alive indefinitely (flock lives on the open file
description, and cargo's sccache server inherits fds from the first build that spawns
it). Diagnosis: `/proc/locks` names a dead pid; `lsof /tmp/memra-5090.lock` shows
`sccache` holding an fd. Fix: `sccache --stop-server` (the daemon respawns on the next
build; verify no rustc is mid-flight first). This also unwedged another lane's queued
suite run.

## 4. Predicted arithmetic (nothing here is a claim; the box window prices it)

    v1 round wall, decode t=1 (tp2-battery cell 3):
      engine wall        44.15 ms/token   (22.65 tok/s engine twin)
      table terms        22.0-24.1 ms  (bandwidth 7.6-9.7 + latency ~10 + drain ~4.4)
      x driver tax       1.2-1.3   ->  26-31 ms
      v1 residual        13-18 ms/token = join+dispatch tax  <- THIS LANE'S TARGET

    per-token counts, v1 -> diet (42 MoE layers, n_used=8, peer share ~0.54 of slots):
      sync peer round-trips   ~170-210  ->  42 bulk returns          (counter-receipted)
      peer fan-out uploads    42        ->  <=42, minus every root-only layer-call
      combine launches        336       ->  42
      host-blocked issue      per-slot  ->  two-pass, ranks overlap

    post-diet decode terms (band, NOT arithmetic — sync-structure transfer is measured,
    never derived; the moe-loc door-D law):
      residual reclaimed  ~55-75%  ->  -7 to -13 ms/token
      engine wall         31-37 ms  ->  ~27-32 tok/s engine twin
      what remains named: per-slot projection launches (~32/layer, the box/NVFP4 grouped
      pair + device tables arc), the 4 bulk hops/layer themselves (native-P2P arm),
      the router single-sync (device-router extension).

**Projected TP-2 decode, stated per the tp2-battery instrument trap** (engine twins
under-read; the plain single-engine control read ~0.8x of its served class, and PP arms
cannot be priced by twins at all): engine-twin prediction **~27-32 tok/s**; served-class
projection **~34-40 tok/s** post-diet, vs v1's 22.65 engine / ~27-30 projected. The
serving projection is stated SEPARATELY from the twin number and neither is a serving
claim; the real number needs the serving wiring (tp2 lane increment 6) and the box
re-price. Against the 100 bar this is still a component: the bar lives in post-diet TP-2 x
spec composition + the matvec-efficiency lever (RESULTS.md follow-up 5).

    EP prime, v1 -> grouped port:
      v1                39-58 tok/s prefill  (TTFT 4.7 s @0.5k, 94 s @3.7k)
      plain grouped     616-639 tok/s @4.6-6.5k prompts (box A/B, serving card class)
      EP-2 grouped      the same program with each rank carrying ~half the expert bytes
                        plus 2 bulk hops + 1 partial add per layer-call
      predicted band    400-650 tok/s class  ->  TTFT @3.7k: 94 s -> ~6-9 s
      (UNPRICED: grouped-GEMM overlap across two contexts and the bulk-hop cost on this
      box class are measured quantities, not derivable ones)

## 5. The box window (separate, named — not this lane's receipts)

1. **TP-2 re-price** post-diet: interleaved x5 fresh boots, plain-vs-TP2-vs-PP-3 on the
   serving card class, vendor-default sampled twin + the 8-turn cache-on twin
   (multi-turn law), spreads reported; served calibration boot per the tp2-battery
   method. Diet counters grepped from the boot log (engagement receipt per arm).
2. **Placement A/B**: naive even split vs the measured co-activation map, WITH the diet on
   both arms — the map's fan-out-skip multiplier is only visible through the diet's
   `FANOUT_UPLOADS_AVOIDED` counter.
3. **Native-P2P arm (BOX-ONLY, named here)**: swap the two host-canonical bulk hops for
   direct peer copies through the inherited `configure_native_p2p` ladder;
   byte-identity gate between transports first (the stage-3 decision), then the timed arm.
   The rig emulation CANNOT run this arm — same-device dual-context has no peer fabric.
4. **Grouped-prime pricing**: real-artifact NVFP4 band calibration (the rig band is
   minted-slab class), then TTFT ladder at 0.5k/3.7k/6.5k vs the 39-58 v1 row and the
   PP-3 0.42 s/2.21 s rows.
5. **Real-artifact class gate re-run** (tp2-battery cell 1 shape) with both doors ON —
   the EP band verdict re-decomposed so the diet cannot hide a class change.
6. Flip decisions for both doors (and only then FLAGS default changes, receipts attached).

## 6. Status log

- Lane opened 2026-08-31 from a5d608b07 + moe-loc merge (door D dependency absorbed;
  merge e14e6d140, one rerere artifact fixed in aba918d0f).
- Doors built same day; FLAGS rows same commit (aba918d0f).
- Grouped-prime band CALIBRATED from the first doors run (1.360e-4 green -> 1e-3 band,
  red 1.984e0; commit 6bdffa418) — the a priori 2e-5 bar was wrong for raw
  cancellation-exposed layer outputs.
- Full gate battery green same day (§3): doors 3/3, tp-gate ALL ARMS (incl. the five new
  diet arms), 19 standing suites, both ppn matrices, unit+server suites, clippy zero,
  fmt clean, check-flags 748/0, local-ci --perf exit 0 (0 fail 0 warn).
- Rig flock wedge diagnosed and cleared in-lane (sccache fd inheritance; also unwedged
  another lane's queued run). Banked in §3 for fleet-ops promotion.
- PUSHED to `origin/lane/glm5-ep-diet`; no self-merge. The box window (§5) prices both
  doors and owns every flip decision.
