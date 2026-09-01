# lane/glm5-1m-battery: the 1M DEPTH RE-PRICE on the current head

Owner bar: **"1M context with high tok/s."** No receipt existed for the CURRENT head at
depth. The only 1M receipt in the bank is the `1m-demo-20260829` lane, and it was measured
on a PRE-MLA-TC, PRE-spec baseline: prefill 161.28 tok/s, TTFD 6,419.8 s (107.0 min),
greedy decode 27.5 @1k decaying to 15.7 @1.035M. This window re-prices that curve on the
head the fleet actually runs, and adds the arm the demo never had: the SHIP CONFIG (spec)
at depth.

Box: rented 4x RTX PRO 6000 Blackwell Workstation Edition 97,887 MiB (box B, vast,
exclusive for this window; IP/port/instance stay out of this repo per the public-boundary
law). 192 cores, 755 GB RAM.

## Pins

| what | value |
|---|---|
| engine head | `lane/glm53-flash-bringup` **92ea07376** ("perf-ci row at the door-r+extraction+struct train") |
| dedup lever | `lane/glm5-dedup` **5848b3d0c** was **NOT** merged into bringup at fetch time — this head does **not** carry the expert-slab dedup lever. Recorded, not assumed. |
| binary | `memra-server` sha256 `4d7dfadc893e8ebba626bdef0cd41cd48bd683c5eaeea29e506f22b7b88a0a4e` |
| build attribution | 283 s real compile (`-j 96` of 192, capped so a concurrent CPU-only mint window kept its core), `BUILD_RC=0`, dirty 0, binary NEWER than every source, strings census 13/13 → `VERIFY_BINARY=GREEN` (`logs/00-verify-binary.txt`) |
| artifact | `/root/models/glm53-nvfp4` (the bit-verified GLM-5.3-Flash NVFP4 mint) |
| drafter | `/root/models/glm53-dflash2`, boot-gated on the server's own `draft source = dflash2 @ b33c0347` line |
| corpus | the 1m-demo Gutenberg file, sha **VERIFIED IDENTICAL**: `a07d4fcd595b57bd3019bb4a16a1a99137c3d04e15b79091183af22141a5d868`. Every rung is the same character slice as the banked demo row, so the demo columns are a true comparator (the 1M rung reports `prompt_tokens=1,035,357` — byte-for-byte the demo's count). |
| port / dirs | 18400, `/root/out-1m`, own clone `/root/memra-1m` |

## Posture, and the one forced deviation

The demo's phase-7 recipe is the ONLY demonstrated 1M config and is carried verbatim except
for one line:

    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=13,26,39
    MEMRA_MOE_SLOTS=12000 MEMRA_MOE_RESIDENT_HEADROOM_GB=36
    MEMRA_CTX=1048576 MEMRA_TIMEOUT_MS_MAX=64800000
    MEMRA_ST_PINNED=1 MEMRA_MOE_FUSED_EPI=1
    MEMRA_MAX_SESSIONS=1 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0
    MEMRA_MOE_VRAM_FRAC=0.35 MEMRA_MOE_HARD_VRAM_FRAC=0.35   <-- THE DEVIATION
    MEMRA_GLM5_VISION=0                                       <-- measurement-scoped

`MEMRA_TIMEOUT_MS_MAX` is the demo's MEASUREMENT-CELL override and must NEVER reach a
fronted deploy: the 90 s ceiling is a platform fact of the fronted route.

The 3-card resident shape is **not usable at 1M**: it OOMs in DSA k-pool selection at layer
31. Reproduced on this very box before this window (`/root/out-1m-b/`: gpu2 peak 97,242 of
97,887 MiB, `[engine-error] class=Overloaded prefill error: layer 31: DSA k-pool selection
failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`).

## FINDING 1 — the 1M request is now REFUSED AT ADMISSION on the demo's own recipe

Two boots on the demo recipe returned **HTTP 429 in under 2 s** — an admission refusal, not
an OOM, and the demonstration would have read as "1M is gone" to anyone probing with a
generic script. The cost model is explicit in the log:

    [admission] request cost: ctx=1035677 path=plain
                = 23936 B/token x ctx + 25575MB prefill-workspace + 18MB fixed = 50382MB
    [admit-oom] VRAM reject: ctx=1035677 has no attainable admission headroom
                (available 39090MB) - HTTP 429

The 1M prime wants **50,382 MB on the limiting device**. What was left was **39,090 MB**.

**The cause is the expert SLRU arena, and the first hypothesis was WRONG.** The arena grows
on demand toward `MEMRA_MOE_VRAM_FRAC` (default **0.85**) of free VRAM per device: measured
here, a card went from **7,126 MiB at boot-ready to ~62,554 MiB after ONE 1,108-token
request**. That ~55 GB of arena is what leaves under 40 GB for a request that needs 50.4 GB.

The first attempt blamed the bf16-resident mirror (`MEMRA_BF16_MMV=1`, carried for
served-env fidelity, whose `[bf16-mmv] RESIDENT` census covers `output.weight`
634,388,480 elements plus `kda_q/k/v/out` 33,554,432 each across 42 layers). **Refuted by
measurement**: removing it moved available headroom the WRONG way, 39,090 MB → 36,740 MB.
Stated because a plausible attribution that survives to a receipt is worse than none
(receipts: `receipts/c1a-bf16-refusal/`, `receipts/c1b-arena-refusal/`).

The fix keeps the demo's demonstrated slot count (12000) and caps the arena by FRACTION at
0.35 soft and hard. That still leaves ~32 GB of arena against a working set of roughly
288 experts x 13 stage layers x ~4.6 MB/slot ~ 17 GB, i.e. about 2x headroom, and the
demo's floor warning is respected: `MEMRA_MOE_SLOTS=256` starves the fused-epi SLRU arm
below `3*n_used`, fails closed to the sequential loop and HALVES prefill to ~40 tok/s, so
the slot count was left alone and prefill tok/s is the starvation detector on every rung.

Why this matters beyond this window: **the demo's 1M equilibrium was calibrated against a
0.85 arena fraction and no longer holds on this head.** A 1M serving posture has to pin the
arena fraction explicitly, or the first request on a fresh boot silently eats the headroom
the second one needs.

## FINDING 2 — grouped MoE prefill, the demo's named "lever 1", does not reach the 1M config

The demo's prefill-gap statement named `MEMRA_MOE_GROUPED_PREFILL` as "the attributed
dominant share" and the top lever. It is **DEFAULT ON at this head** — and it is
**structurally absent on the 1M posture**. The grouped arm runs its per-projection GEMM over
the **local resident slab**; the 1M posture has no resident experts (host-pinned staging plus
a capped on-demand SLRU arena). The boot announces the flag and then never executes:

    [moe-grouped-prefill] flag=on t=1108 il=3 (announce printed in both arms;
        engagement is the per-layer execute line + the dispatch counter)

`execute` lines: **0** on every 1M boot in this window (against `execute layer=44 ...
provenance=resident-slab` on the 3-card resident boot in `/root/out-1m-b/`). So the lever
the demo pointed at for closing the prefill gap pays only on a RESIDENT posture, and the
resident posture is the one that OOMs at 1M. That tension is the finding, and it is why the
prefill numbers below are not the "lever 1 landed" number anyone reading the demo would
predict.

MLA-TC prefill, by contrast, **does** engage on every arm and is the lever that actually
reaches 1M:

    [mla-tc-prefill] engaged: absorb/decompress = strided-batched bf16 TC GEMMs,
        attention = fa_mla_gathered_bf16 (t=1108, t_kv=1108, nh=64, width=1111)

with zero `DECLINED` lines (no cuBLASLt shape declines) on any boot.

## FINDING 3 — the bf16-resident mirror is a capacity/speed trade at 1M, and it is large

Same 1,108-token warm rung, same boot recipe, differing only in the bf16-resident mirror:

| arm | decode tok/s @1k (span / steady p50) |
|---|---|
| `MEMRA_BF16_MMV=1` (today's fleet serving env) | **37.68** / 37.82 |
| mirror off (the demo-exact 1M recipe) | **24.01** / 24.42 |

That is **1.57x of decode at shallow depth**, given up to make the 1M request admissible.
It is not a free choice and it is not a bug: bf16-resident halves decode read traffic for
every preserved non-expert weight, which is exactly why it is in the serving env. The
honest statement is that **today the fleet cannot have both the bf16 decode mirror and a 1M
context on four 96 GB cards** — and note the mirror was NOT what refused 1M (finding 1), so
this trade is about arena+mirror TOGETHER, not the mirror alone. Re-pricing the pair against
the arena fraction is a named follow-up, not a claim this window makes.

## THE 1M PRIME, re-measured (cell 1)

One real 1,035,357-token prompt — the SAME character slice of the SAME sha-verified corpus as
the demo, `cached_tokens=0`, streamed through the serving surface on PP4 — primed and decoded
to a coherent cross-book answer:

| | 1m-demo (pre-MLA-TC) | current head | ratio |
|---|---|---|---|
| prompt_tokens | 1,035,357 | **1,035,357** | identical prompt |
| TTFD (the prime) | 6,419.8 s = 107.0 min | **5,300.55 s = 88.3 min** | **1.211x** |
| prefill | 161.28 tok/s | **195.33 tok/s** | **1.211x** |
| decode, span | 16.04 tok/s | **16.26 tok/s** | 1.014x |
| decode, steady p50 | 15.67 tok/s | **16.61 tok/s** | 1.060x |
| completion tokens | 88 | 182 | |
| error census | 0 | **0** | |
| loop-law screen | — | **0 flagged of 2** | |

Per-card VRAM peak over the whole cell (1,074 samples at 5 s; cards are 97,887 MiB):
**55,418 / 54,746 / 54,746 / 69,850 MiB** (56.6 / 55.9 / 55.9 / 71.4 %), against the demo's
81,945 / 80,121 / 80,089 / 94,905 — the capped arena leaves far more headroom than the demo had.

Output sanity at 1M held, and it is a genuine whole-corpus retrieval, not a template answer:
the greedy answer names *War and Peace* AND *The Count of Monte Cristo*, picks vengeance as the
shared theme, and grounds it in **Prince Andrew's pursuit of Anatole Kurágin** and **Dantès'
methodical revenge** — detail drawn from opposite ends of a 4.2 MB prompt.

**The projection this window was sent to check was wrong by ~8-10x.** MLA-TC's ~8-11 min
estimate implied 1,600-2,200 tok/s at 1M; the measurement is 195.33. Finding 2 is why: MLA-TC
does engage and does fix the attention term, but the term that dominates a 1M prime is
per-token MoE dispatch, and the lever for that (`MEMRA_MOE_GROUPED_PREFILL`) cannot execute on
the only posture that fits 1M.

## FINDING 4 — SPEC AND 1M CONTEXT ARE MUTUALLY EXCLUSIVE TODAY (cells 3, 3diag, 3a)

This is the answer to "today's tok/s at 1M on the ship config", and it is not a number — it is
that **the ship config cannot engage at 1M at all.**

Cell 3 set the full ship env on the PP4 1M posture (DFlash2 @ `b33c0347`, `MEMRA_GLM5_SPEC=1`,
`MEMRA_SPEC_PMIN=0.7`). The boot announced everything you would want to see —

    [glm5-spec] serve route ARMED: draft source = dflash2 @ b33c0347; ... native MTP head NOT loaded
    [spec-gate] policy placement=single-or-non-pp2 LOW=2 HIGH=4 source=placement-default spec-admission=on

— and then **the verify walk never ran**: 0 `[glm5-acc]` lines, 0 `verify walk BATCHED per
layer`, 0 door T/X/K/W announces, 0 `PMIN=0.700`, and decode 23.34 tok/s against the plain
boot's 23.34. Two hypotheses were raised and **both refuted by measurement** (`receipts/c3diag/`):

1. `MEMRA_REUSE_POOL=0`, which this window itself had added — reverting it changed nothing.
2. auto-K choosing K=0 at concurrency 1 — an operator pin `MEMRA_SPEC_K=3` was accepted
   (`[spec-k] operator pin K=3: automatic placement/concurrency/prompt policy ... disabled`)
   and **still** nothing engaged.

The cause is in `worker.rs`:

    fn glm5_sharded_placement_admits(fence_stages, step_tp, step_ep) -> bool {
        (2..=3).contains(&fence_stages) && !tp_set(step_tp) && !tp_set(step_ep)
    }

with the reason in its own doc comment: the verify-walk ppN twin, per-stage rollback and
last-stage MTP chain are red-proven by `glm5-spec-ppn-gate` at **stages 2 and 3 only**, and "a
stage count outside that set has NO gate receipt and refuses … fail-closed is the default,
extended deliberately, never inferred." That is a correct, deliberate design decision. It just
collides head-on with 1M, because **the only demonstrated 1M config is PP4**.

Cell 3a turns the code reading into a one-variable measurement — same binary, same ship spec
env, `MEMRA_CTX=131072` on both arms, same 16k prompt, only the stage count differs:

| arm | spec evidence lines | acceptance | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| **PP4** (splits 13,26,39 — the 1M posture) | **0** | none | 206.19 | **23.10** |
| **PP3** (splits 15,30 — the serving class) | **8** | 0.586 → 0.593 → 0.633 → **0.632** | 194.45 | **27.23** |

Spec is worth **1.179x** at 16k on the placement that admits it (23.10 → 27.23 tok/s), and
**0x at 1M**, because 1M has to be PP4.

**The refusal is SILENT.** No line says spec was declined for this placement. The server prints
`serve route ARMED` at boot and then serves plain forever. Only an engagement gate that DEMANDS
the announces caught it; a generic tok/s probe would have reported "spec is disappointing at
depth" instead of "spec never ran" — precisely the 2026-08-25 DE DFlash2 shape that produced the
never-serve-greedy / verify-with-a-spec-engagement-receipt law.

## The ship config at depth, on the placement that admits it (cell 3b)

Since PP4 refuses spec (finding 4), the ship curve was measured at **PP3** — on the SAME base
recipe as the PP4 plain rows (capped arena, host-pinned staging, no bf16 mirror), with only the
stage count and `MEMRA_CTX` changed, so stage count plus spec env are the only differences and
residency/bf16/grouped-prefill are not silently varying too. Ship spec: DFlash2 `b33c0347`,
`MEMRA_GLM5_SPEC=1`, `MEMRA_SPEC_PMIN=0.7`, **K unset (auto-K)** — and auto-K DOES admit spec
here at concurrency 1, so no operator pin was needed.

Acceptance comes from each response's own `usage.spec` block (rounds / drafted / accepted /
acceptance_rate), which is authoritative per request; its ABSENCE is exactly how the PP4 refusal
was caught.

| rung | arm | prompt_tok | TTFD s | prefill tok/s | decode tok/s | acceptance |
|---|---|---|---|---|---|---|
| 1k | ship greedy | 1,108 | 9.84 | 112.65 | (warm rung) | **0.7917** (12 rounds) |
| 16k | ship greedy | 15,766 | 79.68 | 197.86 | **27.49** | **0.6316** |
| 16k | ship **vendor-default** | 15,766 | 79.57 | 198.14 | **27.78** | **0.6337** |
| 16k | plain (PP4, spec refused) | 15,766 | 76.46 | 206.19 | 23.10 | none |

### THE SHIP vs PLAIN DEPTH CURVE, one placement, one base recipe

| depth | plain decode | SHIP decode | ship/plain | acceptance | ship prefill |
|---|---|---|---|---|---|
| 1k (warm) | 21.05 | (burst-invalid p50) | — | **0.7917** | — |
| **16k** | **21.07** | **27.49** | **1.305x** | **0.6316** | 197.86 tok/s |
| **131k** | **20.37** | **25.64** | **1.259x** | **0.5138** | 200.63 tok/s |

Vendor-default twin at 16k: **27.78 tok/s, acceptance 0.6337** — it tracks greedy, so the uplift
is not a greedy artifact. Cross-boot reproducibility is good: cell 3a's independent PP3 boot
measured 27.23 tok/s / 0.632 acceptance at the same rung, within ~1 %.

### DOES SPEC SURVIVE DEPTH? YES — that is the answer this window was sent for.

The worry was structural: the verify walk pays the same DSA indexer scan x(K+1), so spec could
plausibly be eaten alive by depth. Measured, it is not:

* **acceptance falls hard** — 0.7917 @1k → 0.6316 @16k → 0.5138 @131k (a third of the accepted
  fraction gone over two orders of magnitude);
* **but the uplift barely moves** — 1.305x @16k → **1.259x @131k**, a loss of only 3.5 % of the
  uplift for a 19 % loss of acceptance.

The reason is visible in the plain column: **plain decode is nearly depth-flat too** (21.07 →
20.37, -3.3 %), because the KDA trunk is linear attention and the DSA indexer caps every query at
topk+tail rows. Depth costs both arms about the same fraction, so the RATIO survives even as
acceptance decays. Prefill is likewise depth-flat on this placement (197.86 → 200.63 tok/s).

Practical read: **at the PP3 serving ceiling, spec is still worth ~1.26x at 131k** — the ship
config does not need to be abandoned at depth. It simply cannot be had at 1M at all (finding 4).

## What this window did NOT measure, named

Every omission below is a budget decision, not an oversight. Two ~88-minute 1M primes dominate
any 1M window, and `MEMRA_PREFIX_CACHE_MB=0` is pinned, so **every rung is a fresh cold prime and
nothing is ever reused** — a vendor-default twin at depth D costs a second full prime at depth D.

* **the 1M plain vendor-default twin** — cut after the greedy 1M landed (the probe was
  terminated 102 s in). Its budget went to the ship-config work instead, because the owner's
  question is about the SHIP config. The demo's own plain 1M greedy/sampled pair already agreed
  to 0.01 % on prefill and 0.6 % on decode.
* **the 131k ship vendor-default row** — cut in favour of the plain comparator at 131k, without
  which the deep ship number would have had no baseline on the same placement. The deepest
  sampled ship row is therefore **16k**, and it tracks greedy there (27.78 vs 27.49 tok/s).
* **262k and 525k on both ladders** — the 1M prime came in at 88.3 min instead of the projected
  8-11, which removed roughly two hours of assumed headroom from the plan.
* **cell 4's deep point is 131k, not the briefed 525k** — a `[glm5-phase-v]` split only exists on
  a verify walk, a verify walk only exists on a spec boot, spec only runs on PP2/PP3, and PP3's
  serving ceiling is `MEMRA_CTX=131072`. 525k IS reachable on PP3 with the capped-arena posture
  (roughly 26 GB of planes across three cards) but costs a ~45 min prime that did not fit.
* **the resident fleet serving env** — two boots produced ZERO expert-residency decisions on this
  window's base recipe; recorded as unresolved with the refuted hypothesis attached rather than
  guessed at (`receipts/c3b-fleetenv-residency-denied/UNRESOLVED.txt`). Consequence stated: the
  ship rows here are NOT directly comparable with the banked 70.458 / 71.489 resident+bf16 rows.

## Instrument defects found in this window's own harness (each fixed, each stated)

Three of these were MY gates failing, and two of them aborted a whole cell before its first
request — the same shape as the loud-failures-fail-quietly law, applied to me:

1. **`gates()` asserted the PMIN announce at BOOT.** `PMIN=0.700` prints at the first spec round,
   not at load, so a correctly configured ship boot went GATES RED and cell 3 exited. Moved to
   `engage()`, which is the post-request gate that already existed for exactly this reason.
2. **`gates()` hardcoded `MEMRA_CTX=1048576`.** The PP4-vs-PP3 A/B pins `131072` on both arms so
   stage count is the only variable; the literal RED'd its first arm. Now asserts the LIVE value
   and takes an expected value from `ONEM_EXPECT_CTX`.
3. **`steady_p50` is invalid on a spec arm.** A speculative round emits its whole accepted run in
   one burst, so many interarrival gaps are ~0, the median gap collapses, and the statistic
   explodes — the PP3 spec arm reported **50,393.7 tok/s** beside a sound span of 27.23. The
   span statistic reads only the endpoints and is burst-proof, so span is primary and any p50
   above 3x its own span is reported as `burst` and excluded. Guard applied at REPORT time, not
   in the probe, so the probe never changed mid-cell.
4. **A leaked `vramwatch`** from a killed cell kept sampling into a dead cell's CSV; killed, and
   the kill pattern that missed it is recorded.

## No defaults changed, nothing published

* **No FLAGS default was changed** by this window, and no new flag was introduced. The arena
  fraction (`MEMRA_MOE_VRAM_FRAC` / `MEMRA_MOE_HARD_VRAM_FRAC=0.35`), `MEMRA_GLM5_VISION=0` and
  `MEMRA_TIMEOUT_MS_MAX` are **cell-scoped measurement pins**, reasoned in `box/serve.sh`.
  `MEMRA_TIMEOUT_MS_MAX` must never reach a fronted route.
* **No published product fact was touched**: no price, no roster, no context-window claim, no
  performance claim. Nothing here is a customer-facing number, and per finding 4 the 1M figure
  must NOT become one — a 1M context that takes 88 minutes to first token and cannot run the
  ship config is a capability receipt, not a product.
* **Publicity: deliberately none.** There is a tempting Show-HN angle ("1,035,357 real tokens
  primed and answered, cross-book retrieval, 88 minutes") but posting it would advertise a
  context length we cannot serve at a usable TTFT and cannot pair with speculative decoding.
  The post-worthy version of this material is gated on the prefill work below. Recorded per the
  every-release-ships-with-its-publicity law as an explicit skip, not a silent one.

## Follow-ups this window earned, in priority order

1. **The 1M prefill gap is now attributed, and the top lever does not reach it.** Grouped MoE
   prefill needs a resident slab; the 1M posture cannot be resident. Either teach the grouped
   arm to run against the capped SLRU arena, or find a residency shape that fits 1M planes.
   That is the whole distance between 88 minutes and a servable TTFT.
2. **Extend `glm5-spec-ppn-gate` to stages=4.** This single gate is what blocks spec at 1M. The
   refusal is deliberate and correct given no receipt exists — so produce the receipt. Until
   then, 1M and spec cannot compose, and that should be written down where a serving decision
   would find it.
3. **Make the placement refusal LOUD.** `glm5_sharded_placement_admits` returning false should
   print one line naming the stage count and the admitted range. A silent decline that leaves
   `serve route ARMED` in the log is the exact failure mode the DE DFlash2 incident produced.
4. **Pin the arena fraction in any long-context posture.** The demo's 1M equilibrium was
   calibrated against a 0.85 fraction and no longer holds; the first request on a fresh boot
   otherwise decides the box's capacity.
5. **Re-price the bf16 decode mirror against the arena cap at 1M** (the 37.68-vs-24.01 tok/s
   shallow trade), as a pair, not as two independent knobs.
6. **Resolve the fleet-env residency denial** (`MEMRA_ST_PINNED` suspected, unconfirmed).
## Cell 4 — WHERE THE DEPTH COST SITS (trace=2, UNTIMED, marker down)

`MEMRA_GLM5_SPEC_TRACE=2` on the PP3 ship posture. **These are SHARES, not walls**: level 2
synchronizes at every phase boundary and adds per-layer stream drains, so it serializes what an
untraced round overlaps. They attribute; they never price, and no row here enters a perf table.
(Reassuringly the serialization is small — traced decode 26.83 vs untraced 27.49 tok/s at 16k and
25.02 vs 25.64 at 131k, ~2.4 %, with acceptance reproducing exactly at 0.6316 / 0.5138.)

Per-round verify sub-split, ms/round (two bursts shown where the rung emitted two):

| depth | vkda | (in-kernel scan) | **vmla** | vrest | (vffn) | verify total | round total |
|---|---|---|---|---|---|---|---|
| 1k | 45.515 | 0.365 | **8.443** | 77.677 | 66.093 | 131.636 | 137.472 |
| 16k | 50.297 / 42.657 | 0.378 / 0.359 | **12.802 / 11.876** | 55.614 / 43.999 | 43.297 / 32.860 | 118.714 / 98.532 | 124.763 / 104.560 |
| 131k | 43.103 / 47.248 | 0.360 / 0.373 | **17.617 / 18.387** | 45.806 / 46.960 | 34.548 / 35.038 | 106.525 / 112.595 | 112.649 / 118.706 |

**The prediction is confirmed, and this is the indexer-diet lane's attribution receipt:**

* **`vkda` is FLAT with depth** — 45.5 → ~46.5 → ~45.2 ms/round across two orders of magnitude,
  and its in-kernel `scan` share is flat too (0.36-0.38 ms). KDA is linear attention: its
  per-token work does not grow with the plane. It contributes **nothing** to depth cost.
* **`vmla` GROWS MONOTONICALLY** — 8.443 → ~12.3 → ~18.0 ms/round, **2.13x from 1k to 131k**, and
  its share of verify nearly triples (6.4 % → 16.9 %). This is the MLA + DSA k-pool indexer term.
  **All of the depth cost is here.**
* `vrest` (glue + FFN/MoE + head) is NOT a depth term: it falls 77.7 → ~46 and then flattens,
  tracking `vffn` — i.e. tokens-per-round, which shrinks as acceptance falls, not plane depth.

Note the trace has **no separate indexer bucket**: the indexer sits INSIDE `vmla`, so `vmla`
growth is the indexer+MLA term jointly. That is the honest limit of this instrument, and
separating the two is the first thing an indexer-diet lane should instrument.

**And this explains the flat depth curve.** The round TOTAL actually FALLS with depth (137.5 →
~112-119 ms) because the growing `vmla` (+9.6 ms) is more than offset by shrinking `vffn`
(-31 ms) as acceptance drops and each round verifies fewer rows. So the two depth effects
partially cancel: acceptance decay shrinks the batched FFN work at the same time the indexer
scan grows. That composition — not an absence of indexer cost — is why decode only loses 3.3 %
(plain) and the spec uplift only loses 3.5 % between 16k and 131k.

The lever is therefore real but bounded: at 131k the indexer+MLA term is ~17 % of verify, so a
perfect indexer diet buys at most that share back on this shape — and it grows with depth, which
is exactly why it matters more at 1M than anywhere measured here.
