# TP-2 DIET RE-PRICE — box window (lane/glm5-tpd-battery, 2026-08-31)

The decisive composition of the glm5 TP arc: the **EP dispatch diet** (`MEMRA_GLM5_EP_DIET`)
plus the **co-activation placement map** (`MEMRA_GLM5_EP_MAP`) against the served PP-3 row.

## 0. Setup, pins, and what each prior receipt says

- Box: the glm53 second 4-card (4x RTX PRO 6000 Blackwell Workstation, 97887 MiB, 600 W
  confirmed on all four; 192 cores, 755 GB host RAM). Cards 0,1 for every engine-twin arm;
  0,1,2 for the served PP-3 calibration boot. Port 18400 only for served boots.
- Build: pin **25537ca8e** (`lane/glm5-ep-diet` head) in its OWN clone `/root/memra-tpd`
  (`/root/memra-struct`'s binaries are a DIFFERENT head and were never used), plus **one
  probe-bin-only commit** that prints the five EP-diet engagement counters — the pinned
  probe prints only `ep-peer-slot-dispatches`, and this window's bar needs the counters as
  per-boot receipts INCLUDING on the pinned-`=0` arms, so "flat at zero" is a receipt rather
  than an absent line. No engine path is touched (tp2-battery precedent for probe-bin-only
  commits on a pinned engine head).
- Rebuild attribution (`receipts/build.log`): real build **317 s** (not a
  checkout-attribution 0.04 s "Finished"), `git log -1` banked, binary mtimes == BUILD_END,
  probe sha256 `1154750671914ee4774a90cb30bafd9994af0d485d92c6f8f97345663eabf3f9`, server
  sha256 `c14e222373f237bc7c38f797a4b25455f3ae007127ec39eb4f4eb1c6ba04c76c`. Strings probes
  confirm 7 `glm5-tp-` announce literals, `ep-map armed`, and every diet-door literal
  (`[glm5-ep-diet] engaged`, `[glm5-ep-grouped-prime] flag`, `... execute`, the bulk-staging
  announce text) in BOTH binaries; the probe additionally carries `ep-diet-counters`.
  Banked trap re-confirmed: cargo is not on the non-interactive PATH here, every driver
  sources `~/.cargo/env`.
- Artifact: `/root/models/glm53-nvfp4` (the HF-verified 20-shard NVFP4 mint). Pools: the
  house decode-attribution 10-prompt pool + `l3-WARM` (426 tok) and `l3-A4630` (4626 tok),
  byte-identical to every banked baseline; the tiny-prime pool reproduces the banked
  `prompts-tiny/t0.txt` sha256 `de11681964f01762b7d78110ec332bc0fc74bbf96ba83cce0136f326adceb02b`.
- Arm env: the tp2-battery cell-4 tables VERBATIM (`CUDA_VISIBLE_DEVICES=0,1`
  `MEMRA_GLM5_TP=all@0,1` `MEMRA_RP=0` `MEMRA_MOE_RESIDENT=0` `MEMRA_MOE_SLOTS=16`,
  `NVIDIA_TF32_OVERRIDE=0` `MEMRA_BF16_MMV=1`), so a diet arm differs from the v1 arm by the
  DOOR ALONE. **Named deviation, inherited:** TP arms require `MEMRA_RP=0` (the shard
  builders refuse the rp split-plane mirror by name); RP=0 was proven bit-identical on plain
  in tp2-battery cell 1 row 1.
- Door spelling (owner law): both diet doors default OFF at this head, so every OFF arm
  PINS `=0` explicitly — never unset.
- Instrument trap, load-bearing and stated before any number: engine twins UNDER-READ. The
  eager `prime_cache`+`decode_step` driver reads ~0.8x of served on single-engine walks and
  **0.254x on the PP-3 placement** (0.11x at depth) — PP arms cannot be priced by twins at
  all. Every TP row here is an engine-twin row; the PP-3 comparator is a SERVED row measured
  on this same build (cell 2c).

## 1. Cells and their bars (pre-registered before any timed boot)

0. **Map restore** (untimed) — struct-battery's cell-4 mints died with `/root/out-struct` and
   were never banked; re-derive and check the sha against `56dea5ca...`.
1. **Real-artifact class gate, diet doors ON** (untimed) — decode BYTE for the non-MoE
   classes, the banked EP band for MoE, reds bite; ANY divergence from the banked v1 class
   verdicts STOPS the window.
2. **Diet pricing** (timed, marker, interleaved x3) — v1 vs diet vs the served PP-3 row;
   bulk-transfer counters demanded ON / flat OFF; greedy + one vendor-default sampled row
   per boot; loop-law.
3. **Grouped EP prime** (timed) — prefill tok/s + TTFT@4626 tok vs v1's 94.2 s.
4. **Placement A/B on the DIETED walk** — even vs the banked co-activation map: sign +
   reconciliation (peer-touch cut vs imbalance), the composition verdict.
5. **Spec x TP composition state** — read from FLAGS/source, not measured.

Cells 2, 3 and 4 run as ONE interleaved timed block under a single
`/root/TIMING-IN-FLIGHT` marker (interleaved-A/B protocol law: box drift invalidates
cross-run claims, and the map arm's only honest comparator is the dieted even arm measured
inside the same drift envelope).

## 2. RESULTS

### Cell 0 — the banked co-activation map REPRODUCED BIT-EXACT, and banked this time

One served PLAIN trace boot (`MEMRA_MOE_TRACE` + `MEMRA_MOE_WEIGHT_TRACE` armed together,
agentic tags d00-d05, greedy 256, then struct-battery's class-purity truncation), then the
shared mint tool (`--ranks 2 --entry-rank 0 --expert-count 288 --decode-only`):

| artifact | this window | banked (struct c3/c4) |
|---|---|---|
| `agentic-t1.ids` / `.w` | 63,714 lines each | — |
| t=1 rows in the trace | **63,462** | 63,462 |
| `agentic-t1-coactivation.json` | `56dea5ca5a2502f2...` | `56dea5ca5a2502f2...` **identical** |
| `agentic-t1-even.json` | `e68b419f08ec2217...` | `e68b419f08ec2217...` **identical** |

Three receipts in one: the greedy routing trace reproduces across BUILDS (struct head
`c7d936536` -> ep-diet head `25537ca8e`, both diet doors OFF on the served path), the mint is
deterministic, and the A/B input is the banked artifact rather than a look-alike. Loop-law
0/6. The map JSONs are banked here so the next lane never re-mints them — the gap this cell
had to pay for.

### Cell 1 — CLASS GATE GREEN: the diet reproduces the v1 class verdicts row for row

Teacher-forced on plain single-card reference tapes (the rig-gate shape), full-vocab f32
dumps (prime + first 8 decode steps). Reference determinism first: **54/54 banked
tp2-battery f32 sha256s reproduce** on this build (`analysis/c1-refcheck.txt`), so every diet
row below is measured against a reference byte-equal to the banked one.

| arm (diet ON unless named) | worst norm_rel this window | banked v1 | verdict |
|---|---|---|---|
| `tinykda` TP layers 0-2 (pure KDA shard), tiny prime | **0.0 — BYTE-IDENTICAL** | 0.0 byte | class held |
| `tinymla3` TP layer 3 (MLA+EP), tiny | 2.902e-2 | 2.9e-2 | class held |
| `tinymoe4` TP layer 4 (KDA+EP), tiny | 4.761e-2 | 4.8e-2 | class held |
| `tinytp` TP all@0,1 (full trunk), tiny | 5.249e-2 | 5.2e-2 | class held (identical to 4 digits) |
| `c1tp` TP all@0,1, deep prime | 1.216e-1 | 1.2e-1 | class held |
| RED `swap-wo` THROUGH the dieted walk | **1.065e0** | 0.93-1.05 | bites ~20x above green |
| **diet vs v1**, same build, full trunk | **0.0 — BYTE-IDENTICAL** (12/12 files) | rig arm B2 bar | door is bit-free |
| **map vs diet**, tiny + c1 (cell-4 phase A) | **0.0 — BYTE-IDENTICAL** both | struct 56/56 on v1 | placement-independent |

No silent-wrong signature: the bands are bounded, class-attributed, argmax-preserving where
the banked rows were, and the red is loud. **The diet does not move the real-artifact
numeric class at all** — decode is byte-identical to v1 on the real NVFP4 artifact, which is
stronger than the band the class gate would have accepted.

Engagement counters, per boot (`analysis/c1-engagement.txt`) — the non-vacuous-ON /
flat-OFF bar:

| boot | peer-slot dispatches | diet dispatches | bulk returns | roundtrips avoided | fanout uploads avoided |
|---|---|---|---|---|---|
| `v1` tiny (pinned `=0`) | 9,210 | **0** | **0** | **0** | **0** |
| `diet` tiny | 9,210 | 1,344 | 1,340 | 9,210 | 928 |
| `dietmap` tiny | 4,435 | 1,344 | 997 | 4,435 | 1,271 |
| `diet` c1 (deep) | 501,218 | 42,000 | 41,846 | 501,218 | 82,684 |
| `dietmap` c1 (deep) | 230,314 | 42,000 | 26,287 | 230,314 | 98,243 |
| `plain1` reference | 0 | 0 | 0 | 0 | 0 |

Read the mechanism straight off the table: v1 and diet route IDENTICALLY (9,210 peer slots
both), the diet folds **every** per-slot round-trip into one bulk return per layer-call
(`roundtrips_avoided == peer-slot dispatches`, `bulk_returns` = 1,340 of 1,344 layer-calls),
and the MAP multiplies the diet exactly as the lane predicted: peer-slot dispatches fall to
**0.4815x** (tiny) / **0.4594x** (deep — struct measured 0.4574x on the v1 walk, the same
routing property), bulk returns fall to **0.744x / 0.628x** because whole layer-calls stop
returning anything, and avoided fan-out uploads RISE **1.37x / 1.19x** — the root-only
layer-calls that move zero activation bytes off root. That multiplier is the thing the v1
walk could not express.

Harness finding, receipted: the first `tinykda` boot was failed by THIS HARNESS, not the
engine — a KDA-only sub-shard (`MEMRA_GLM5_TP=0-2@0,1`) reports `moe_ep=0` in its own
preflight, so the diet has no EP layer-call to engage on and its announce legitimately never
fires. The demand is now conditional on the boot's own `moe_ep>0` receipt; the arm was
re-run (`tinykda2`) and is byte-identical both to the plain reference AND to the first boot
(a free boot-determinism receipt). Its tapes were never affected.

### Cell 2 — DIET PRICING: the diet is REAL on decode (+8.77%) and COSTS prefill (-30%)

Timed, `/root/TIMING-IN-FLIGHT` held 14:42:45Z-15:49:10Z for the whole block, three arms
interleaved in a fixed order, fresh boot each, x3. Greedy + one vendor-default sampled row
per boot, 256-token cap, 128-token floor (0 rows excluded). **No escalation triggered:** every
within-arm spread is under 0.5% and every verdict gap is >20x the pooled spread.

| arm | instrument | pool decode tok/s (boot medians) | spread | pool prime | pool TTFT | vendor rows |
|---|---|---|---|---|---|---|
| **PP-3 recipe (15,30)** | **SERVED** (calibration boot, same build) | **34.976** | — | — | **0.428 s** | 34.23 |
| TP-2 v1 (`EP_DIET=0`) | engine twin x3 | **22.654** [22.654/22.658/22.654] | 0.015% | 4.617 s | 4.664 s | 22.57/22.61/22.63 |
| **TP-2 diet** (`EP_DIET=1`) | engine twin x3 | **24.642** [24.647/24.642/24.559] | 0.359% | 5.995 s | 6.037 s | 24.64/24.61/24.56 |
| TP-2 diet + map | engine twin x3 | **22.114** [22.103/22.138/22.114] | 0.156% | 6.020 s | 6.068 s | 22.02/22.05/22.04 |

pooled spread 0.088 tok/s. **diet/v1 = 1.0877** (+1.987 tok/s, 22.6x pooled).
**dietmap/v1 = 0.9761**. Loop-law 0/99. Vendor rows track greedy on every arm (the sampled
traffic shape confirms each verdict, per the never-serve-greedy law).

Cross-window reproduction receipts: v1 lands on the banked **22.65** to the thousandth, and
its peer-slot dispatches are **1,012,568 per pool boot — the tp2-battery banked number
EXACTLY**, identical across all three repetitions (greedy-determinism). The served
calibration reproduces the banked plain-served baseline: 34.976 vs 35.36 (-1.1%), pool TTFT
0.428 s vs 0.42 s, deep `l3-A4630` TTFT **2.209 s vs the banked 2.21 s**. The served boot's
announce gate is green in the negative direction too: no `[glm5-tp-preflight]`, no
`[glm5-ep-diet]`, no `[glm5-ep-grouped-prime]` (the worker refuses TP by design), and cards
0/1/2 hold the stages with card 3 at 1 MiB.

Counters, per boot, identical x3 in every arm:

| arm | peer-slot dispatches | diet dispatches | bulk returns | roundtrips avoided | fanout uploads avoided |
|---|---|---|---|---|---|
| v1 (pinned `=0`) | 1,012,568 | **0** | **0** | **0** | **0** |
| diet | 1,012,568 | 114,786 | 114,382 | **1,012,568** | 137,366 |
| diet + map | **445,157** | 114,786 | **73,944** | 445,157 | **177,804** |

**The arithmetic.** v1 = 44.14 ms/token, diet = 40.58 ms/token: the diet reclaimed
**3.56 ms/token** of the measured 13-18 ms v1 join+dispatch tax — **20-27% of the residual**,
against the lane's predicted 55-75% (7-13 ms). The engine-twin prediction was **27-32 tok/s**;
measured **24.64**. Applying the banked single-root instrument factor (~0.8x), the served-class
projection is **~30.8 tok/s** (v1's was ~28.3), against the lane's predicted 34-40 and against
the **34.976 SERVED PP-3 row measured here**. Stated plainly: the diet is a real, receipted,
reproducible win on the TP-2 decode walk, and it is roughly HALF the predicted win — the
per-slot round-trips were folded completely (`roundtrips_avoided == peer-slot dispatches`) and
the wall did not follow, which is exactly the class the diet window warned about: sync-structure
removal does not transfer from counts to wall by arithmetic. What remains is named and
unchanged: ~32 per-slot projection launches per layer, the 4 bulk hops per layer themselves
(the native-P2P arm), and the router single-sync.

**The prefill regression, and it is the diet's own cost.** Pool prime 4.617 -> 5.995 s
(+29.8%) and pool TTFT 4.664 -> 6.037 s under the diet alone, reproduced x3 in both arms. At
depth it is worse (cell 3): 94.44 -> 135.27 s at 4626 tokens (+43%), prefill 49.0 -> 34.2 tok/s
(0.70x). The bulk fan-out + compact staging + single scatter-combine is a DECODE-shaped
optimisation; at prime `t` the one-launch scatter over a large pair slab and the serialized
bulk return cost more than the per-slot dribble they replace. **Consequence for any flip
decision: the diet door must never ship alone.** It is a pair with the grouped prime.

### Cell 3 — GROUPED EP PRIME: 4x prefill vs v1, 5.7x vs diet-alone, decode-neutral

Same timed block, l3 pool, arms v1 / diet / dietgp interleaved x3, `MEMRA_MOE_GROUPED_PREFILL`
at its ON default in all three (an OFF arm would have pinned it `=0`; none run).

| arm | prompt tok | prime s | prefill tok/s | TTFT s | deep decode tok/s |
|---|---|---|---|---|---|
| v1 | 4626 | 94.444 | 49.0 | 94.497 | 20.416 |
| diet | 4626 | 135.272 | 34.2 | 135.321 | 21.901 |
| **diet + grouped prime** | 4626 | **23.636** | **195.7** | **23.690** | 21.926 |
| v1 | 426 | 7.715 | 55.2 | 7.760 | 22.412 |
| diet | 426 | 10.252 | 41.6 | 10.292 | 24.334 |
| **diet + grouped prime** | 426 | **1.060** | **401.7** | **1.101** | 24.232 |

v1 reproduces the banked 94.2 s TTFT row (94.497 s, +0.3%) and the banked 39-58 tok/s prefill
class (49.0). The grouped prime is **3.99x on TTFT vs v1** (94.50 -> 23.69 s) and **5.71x vs
diet-alone**, i.e. it more than pays back the diet's prefill cost; at the shallow 426-token
row it is **7.05x vs v1** (7.76 -> 1.10 s). Decode is untouched (21.926 vs 21.901), which is
the door's design.

Engagement: `grouped_prime_dispatches = 126` per boot (42 layers x 3 prime chunks) with 42
`[glm5-ep-grouped-prime] execute` lines, identical x3; **0 on both other arms** with 0 execute
lines. The rig's Q8_0 fall-closed arm (B3) does NOT repeat here — the real NVFP4 artifact is
f16g-eligible and the grouped walk fires. Loop-law 0/14.

Honest read against the prediction: the lane predicted a 400-650 tok/s class and TTFT ~6-9 s
at depth. Measured **195.7 tok/s / 23.69 s** at 4626 tokens — the shallow row does reach the
band (401.7 tok/s) but the deep row lands at ~half its floor, so the two unpriced quantities
the lane named (grouped-GEMM overlap across two contexts, and the bulk-hop cost on this box
class) both cost real time at depth. **TP-2 prefill is no longer serving-blocking as a class,
but it is still 10.7x the served PP-3 row (23.69 s vs 2.209 s at the same prompt).**

### Cell 4 — PLACEMENT A/B ON THE DIETED WALK: the map loses HARDER, and the counters say why

Phase A (cell 1): map-vs-diet **BYTE-IDENTICAL** on both the tiny and the c1 pool, so the
timed arm prices a correctness-free lever, same as struct-battery's 56/56 on the v1 walk.

Phase B (the same interleaved x3 block, so both arms share one drift envelope):

| | even split | co-activation map | ratio |
|---|---|---|---|
| pool decode tok/s | **24.642** | **22.114** | **0.8974** |
| pool prime s | 5.995 | 6.020 | 1.004 |
| peer-slot dispatches | 1,012,568 | 445,157 | 0.4396 |
| bulk returns | 114,382 | 73,944 | 0.6465 |
| fanout uploads avoided | 137,366 | 177,804 | 1.294 |

**VERDICT: the map is REFUTED A SECOND TIME, and the diet made it WORSE, not better.**
struct-battery measured 0.9686 (-3.14%) on the v1 walk; on the dieted walk it is **0.8974
(-10.26%)**, a -2.528 tok/s gap at 28.7x the pooled spread. The reconciliation the cell was
built to deliver:

- The map's byte savings are REAL and BIGGER under the diet: peer-slot dispatches 0.4396x
  (matching struct's 0.4396x t4v number exactly — a routing property, reproduced across
  windows and builds), bulk returns 0.6465x because whole layer-calls now return nothing at
  all, and avoided fan-out uploads **1.294x** — the single-rank multiplier the lane predicted
  would make the map's win multiplicative. Every predicted byte-movement effect landed.
- And the wall went the other way, harder. The diet removed the host round-trip serialization
  that used to DOMINATE the round; what is left on the critical path is the slowest rank's
  expert compute, and that is precisely the quantity the map trades away (mint's own named
  trade: expected max-rank touch 5.079 -> 6.894, +36%). Removing the dominant term promoted
  the imbalance term. The lane's "the map prices INSIDE the diet" hypothesis is measured and
  **falsified**: peer-hop count was never this map's cost driver on either walk.
- Therefore: **DO NOT ADOPT `MEMRA_GLM5_EP_MAP` on the glm5 TP-2 walk, dieted or not.** The
  co-activation structure is real (39.2% single-rank t=1 events on agentic traffic) and the
  placement law stands; what is refuted is this SHAPE of it — a 2-rank, entry-rank-pinned,
  balance-sacrificing bundle on a walk whose critical path is per-rank expert compute. A
  placement mint that keeps expected max-rank touch AT the even baseline while still bundling
  (a balance-constrained objective, not a balance-tolerant one) is the next honest attempt,
  and it is a mint-side change, not an engine one.

### Cell 5 — spec x TP composition: still CO-REFUSED, so here is the arithmetic

The gate condition ("TP-2 diet+map approaches or beats PP-3") did NOT fire: diet+map is 22.114
engine twin / ~27.6 served-class projected against the **34.976 served PP-3** row measured in
this window. The best TP-2 arm here is the dieted EVEN split at 24.642 / ~30.8 projected, still
**0.88x of served PP-3**.

State of the composition, read from the tree at this pin, not measured:

- `crates/memra-engine/src/glm_spec.rs:1514` — `generate_spec_glm5` refuses by name: *"glm5
  spec is co-refused while MEMRA_GLM5_TP is armed: the TP-2 walk carries no verify/rollback
  wiring in v1"*.
- `crates/memra-server/src/worker.rs:9725` — the serving worker refuses `MEMRA_GLM5_TP` at
  spawn outright.
- `docs/FLAGS.md` `MEMRA_GLM5_TP`: TP x `HC_FUSED_PRE`, TP x `HC_DECODE_WS`, TP x
  `KDA_FUSED_PROJ`, TP x `MLA_DECODE_SPLIT` all REFUSED; TP x `GLM5_VERIFY_BATCH` needs no
  refusal only because spec sessions are already co-refused.

So the arithmetic instead of a row, using this box's measured multipliers:

    best TP-2 today (diet, even)      24.642 engine twin  ->  ~30.8 served-class projection
    x the vrest DFlash2+PMIN loop     1.77x measured on PP-3
    composed TP-2 x spec ceiling      ~54.5 tok/s
    current best single-stream        71.489 (PP-3 + spec ship config, struct-battery c1)
    the 100 bar from there            1.399x

    TP-2 would have to reach ~56.5 served-class (~45.2 engine twin, 22.1 ms/token) before
    TP-2 x spec merely MATCHES today's 71.5 - another 18.4 ms/token off the current
    40.58 ms/token, i.e. MORE than the entire remaining v1 join tax (13-18 ms) that this
    whole lane set out to remove and got 3.56 ms of.

**The honest verdict for the 100-bar path: it does not run through TP-2.** Post-diet TP-2 is
still below served PP-3 on decode, still 10.7x its TTFT at 4.6k, and its two remaining named
transport levers (native P2P for the 4 bulk hops/layer, device tables for the ~32 per-slot
projection launches) are bounded by the 3.56 ms the far larger round-trip removal actually
bought. The 1.399x to 100 stays on the PP-3 + spec line with the levers that already have
measured mass: the dedup kernel campaign (21.96% repeat fraction = the +5.6-6% ship class) and
the matvec-efficiency lever (bf16-mmv+moe = 65% of GPU time at 57-70% efficiency vs q38's 87%).

## 4. Flip decisions (this window owns them)

- `MEMRA_GLM5_EP_DIET`: **stays default OFF.** It is a +8.77% decode win with a -29.8% pool
  prime / -43% deep prefill cost; alone it makes TP-2 worse for any real prompt shape. It is
  ALSO not exposable: the serving worker refuses `MEMRA_GLM5_TP`, so there is no customer path
  to flip. Recommendation: keep OFF, and treat the door as PAIRED with the grouped prime for
  any future TP work.
- `MEMRA_GLM5_EP_GROUPED_PRIME`: **stays default OFF**, same exposure argument, but it is
  RECOMMENDED as the mandatory companion of the diet: 3.99x TTFT vs v1, 5.71x vs diet-alone,
  126 dispatches/boot engagement, decode-neutral, byte class unchanged.
- `MEMRA_GLM5_EP_MAP`: **DO NOT ADOPT** on this walk. Two independent box windows now measure
  it losing (-3.14% v1, -10.26% dieted) while its byte-movement predictions all land. The
  next attempt is a balance-constrained mint, not this artifact.
- No FLAGS default is changed by this window; the receipts above are the reasons.

## 5. Traps and findings banked from this window

- **A diet-class door can be net-negative on the shape it was not designed for.** The EP diet
  is decode-shaped; at prime it costs 30-43%. Any "removes N syncs" door needs a PREFILL arm
  in its pricing cell, not just a decode arm — this one would have shipped a TTFT regression
  on a decode-only battery.
- **Removing the dominant cost term promotes the next one, and can flip a lever's sign.** The
  placement map got WORSE after the diet because the diet removed the host-round-trip term the
  map was helping and exposed the per-rank-compute term the map hurts. A lever's sign is only
  valid on the walk it was measured on.
- **Counters can be fully green while the wall disagrees.** `roundtrips_avoided` equalled
  peer-slot dispatches exactly (every round-trip folded) and only 20-27% of the attributed
  tax came back. Count receipts prove ENGAGEMENT; they never price it.
- **A window whose input artifact lives in another window's scratch dir starts by re-deriving
  it.** struct-battery's maps died with `/root/out-struct`; cell 0 cost a boot. Mints are cheap
  and deterministic — bank the artifact, not only its sha (done here).
- Harness: a door's announce demand must be conditional on the boot's own applicability
  receipt (`moe_ep>0`), or a legitimate arm fails the gate for structural reasons.
- Perf-CI note (deliberate, written): the pre-push hook refuses a branch touching
  `crates/memra-engine/**` without a fresh `local-ci.sh --perf`. This branch's only engine-dir
  change is the probe BIN's `eprintln!` + env echo (no library, no serving path), and the pin
  25537ca8e already carries a clean `--perf` row (0 fail 0 warn). Pushed with
  `MEMRA_SKIP_PERF_CI=1` knowingly; the skip is logged by the hook and recorded here.

## 3. Window log

- 14:07:46Z WINDOW START line to `/root/BOX-QUEUE.md` (box was FREE: cards 0-3 at 1 MiB, no
  marker, struct-battery DONE line read; its retained `/root/memra-struct` untouched).
- 14:07-14:13Z build (317 s real), attribution + strings probes banked. 14:15Z probe-bin-only
  counter commit + probe-only rebuild (sha changed, `ep-diet-counters` literal present).
- 14:16-14:22Z cell 0: trace boot + mint, map sha REPRODUCED. Banked+pushed.
- 14:23-14:52Z cell 1: 11 tape boots + the fixed `tinykda2` re-run, 9 compare pairs, 54/54
  reference f32 shas reproduced. GREEN — the window proceeds to timed cells. Banked+pushed.
- 14:42:45Z TIMING-IN-FLIGHT raised for the whole timed block; 15:49:10Z marker DOWN (trap-based,
  so a failure could not have left it up). 18 timed boots + 1 served calibration boot, 0
  failures, x3 sufficient on every arm (no escalation rule fired).
- 14:42-15:28Z cell 2 pool (9 boots, v1/diet/dietmap interleaved). 15:28-15:45Z cell 3 deep
  (9 boots, v1/diet/dietgp interleaved). 15:45-15:49Z the served PP-3 calibration boot.
- Totals: 30 engine-twin boots + 2 served boots, 0 failures, loop-law 0 of 127 screened
  (99 pool + 14 deep + 14 served), engagement announces demanded/forbidden per arm on every
  boot, every counter set identical across all three repetitions of each arm.
- Wall: window open 14:07:46Z, cells closed 15:49:10Z box time (~1 h 41 min against a ~3 h
  estimate: build 5.3 min, cell 0 6 min, cell 1 29 min, timed block 66 min, banking the rest).
