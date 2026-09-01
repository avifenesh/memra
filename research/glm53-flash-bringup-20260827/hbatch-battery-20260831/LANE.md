# glm5 HYPER-BATCH default-flip battery (lever #6 of the decode-gap attribution)

Lane: `lane/glm5-hbatch-battery` (box B window `hbatch-battery`, 2026-08-30/31).
Owner question: what does the batched decode walk (`decode_step_batch_hyper`, cap 15,
gate `glm5-hyper-batch-gate`) buy at c in {2,4,8,12} on the serving shape, and should
`MEMRA_HYPER_BATCH` flip default ON. Economics context: with the walk OFF, concurrent
sessions decode serially, so the box aggregate equals the single stream (~35.4 tok/s =
~3.06M output tok/day) and the attribution prices the shape unit-negative 3-5x at
saturation (`darklanes research/glm5-decode-gap-20260830/ATTRIBUTION.md` section 7).

## Pins

- Build: memra `34e0c0bf2` (lane/glm53-flash-bringup head = consol merge of
  lane/glm5-flip-battery; contains loop-port `bb8d9e3cc` and the loop-port merge
  `90a7b5210` — ancestry verified). Rebuild-attribution note: cargo Finished instantly
  because `bb8d9e3cc..34e0c0bf2` touches ZERO compiled sources (receipts-only merge,
  verified `git diff --name-only` over `*.rs *.cu *.cpp *.h *.toml Cargo.lock` = empty);
  the binary is the flip-rebattery build (real 1m19s), sha256
  `a87e91fcd57c2879cfefff561d7f84a0fbbfa40f3e2db5a144ebc9f1bf6a8a51`, strings-probed for
  the hyper-batch literals (5x `MEMRA_HYPER_BATCH`, 1x engagement-ON line, 1x
  engagement-OFF line, 3x `decode_step_batch_hyper`). `receipts/build-34e0c0bf2.log`.
- Box: box B, 4x RTX PRO 6000 Blackwell WS 600W, cards 0/1/2 (CVD=0,1,2), port 18400,
  card 3 untouched.
- Serving env: the pinned 3-card recipe (PP_STAGES=3 SPLITS=15,30 DEVICES=0,1,2 +
  BF16_MMV=1 PP_BF16=1 MOE_GROUPED_PREFILL=1 MOE_RESIDENT_GB=98 MOE_SLOTS=16 CTX=131072
  PREFIX_CACHE_MB=0 TF32=0 COMPAT=openai), with **MEMRA_MAX_SESSIONS=16 on BOTH arms**
  (named deviation from the 3way window's 4: covers the c<=12 ladder; receipted per boot).
- Arms: OFF = `MEMRA_HYPER_BATCH` unset (today's default) | ON = `MEMRA_HYPER_BATCH=1`.
  Boot engagement receipts (worker.rs carve-out): ON = `BATCHED DECODE (mHC hyper arm,
  opt-in via MEMRA_HYPER_BATCH=1...)`, OFF = `EAGER-ONLY serving (hyper-connections
  residual — no batched decode arm)`; gates require the right line present AND the other
  absent, plus >=3 RESIDENT, boot nonce in `/proc/pid/environ` (arm-identity law), fluent
  output sample, zero `[glm5-spec]` lines (plain serving both arms).
- Pools: the decode-attribution pool (10 real prompts, 6 code + 4 prose) +
  `l3-ab-prompts.json` (WARM ~0.4k / A4630 ~3.7k / B5550 ~5.5k / C6470 ~6.5k), the same
  files every prior window used. Ladder picks are deterministic index lists; the c=4 pick
  list is byte-for-byte the 3way cell-5 list, so the aggregate is directly comparable to
  the banked 30.4/30.5 receipt.
- Estimators: per-session decode tok/s = streamed `(ct-1)/(t_last-t_first)`; burst
  aggregate = total completion tokens / burst wall (the 3way cell-5 estimator, kept for
  comparability); decode-window aggregate = `sum(ct-1) / (max t_last - min t_first)`
  (banked alongside; excludes the serialized-prefill head of the burst).

## Cell 1 — boot gates + correctness spot: **GREEN**

Boots `c1-off` / `c1-on`: GATES GREEN both arms (engagement line present ON / absent OFF,
3x RESIDENT, nonce verified, samples fluent). Correctness (greedy 256, served path,
exactness only — no timing read from this cell):

| bar | tapes | verdict |
|---|---|---|
| OFF conc c=2 (x2 pairs) vs OFF solo | 4 | identical |
| ON conc c=2 (x2 pairs) vs ON solo | 4 | identical |
| ON conc c=12 (full-width spot) vs ON solo | 12 | identical |
| ON solo vs OFF solo (served-path B=1 class pin) | 4 | identical |
| ON conc vs OFF conc (cross-arm) | 4 | identical |

The gate's red class (cross-session contamination: swapped-row / wrong-cache-slot) does
not reproduce on the served path at any tested width, including the cap-adjacent c=12.
Loop-law: 0/36 flagged. Receipts: `receipts/c1/`.

## Cell 2 — THE LADDER (timed, marker held 20:34:45Z-22:22Z, interleaved x3/arm): **ON WINS EVERY RUNG c>=2**

Six boots (OFF/ON alternating x3 rounds), greedy pool 256, deterministic picks per rung
(c=4 = the 3way cell-5 pick list). X3 SUFFICIENT: no escalation rule fired (within-arm
aggregate rel spreads 0.031-0.188%, all under the 0.5% bar; every ON-OFF gap >> 2x pooled
spread). Loop-law 0/328. All 12 boots GATES GREEN.

| c | OFF agg tok/s | ON agg tok/s | ON/OFF | OFF dw | ON dw | per-sess p50 OFF->ON | TTFT p50 OFF->ON |
|---|---|---|---|---|---|---|---|
| 1 | **35.42** (baseline reproduced) | 35.32 | 0.9970x | 35.42 | 35.32 | 35.4 -> 35.3 | 0.362 -> 0.362 |
| 2 | 33.83 | 37.03 | **1.0946x** | 34.53 | 37.88 | 17.3 -> 19.0 | 0.334 -> 0.332 |
| 4 | 30.24 | 34.73 | **1.1484x** | 30.50 | 35.19 | 8.4 -> 9.8 | 3.439 -> 3.382 |
| 8 | 31.33 | 37.44 | **1.1952x** | 31.47 | 37.75 | 4.3 -> 5.2 | 5.214 -> 5.101 |
| 12 | 31.94 | 38.78 | **1.2141x** | 32.02 | 39.03 | 2.9 -> 3.6 | 6.749 -> 6.607 |

(agg = burst estimator, total completion tokens / burst wall — the 3way cell-5 estimator;
dw = decode-window aggregate, excludes the serialized-prefill head. p95 == p50 to the
tenth on every rung, both arms — the per-session distribution is tight, no straggler tail.
TTFT-max at c=12: OFF 6.89s / ON 6.61s.)

Standing-baseline reproduction: OFF c=1 decode-pool median 35.42 / 35.42 / 35.42 across
the three rounds (flip-battery banked 35.408). The OFF c=4 rung reads 30.24 vs the banked
3way cell-5 30.4/30.5 — same number, same picks, different window.

What the table says:

- **OFF (today's default): concurrency LOSES aggregate.** Eager per-session round-robin
  never recovers the single stream (30.2-33.8 vs 35.4 at every c>=2) — the c=4 flat
  receipt generalizes across the ladder.
- **ON: +9.5% to +21.4% aggregate over OFF, monotone in c**, and c>=8 rises ABOVE the
  single stream (37.4 / 38.8 vs 35.4). Every rung's gap is decisive (spreads 2 orders
  below it).
- **The B=1 cost of the batched body is real but tiny: -0.30%** (35.32 vs 35.42, gap 0.11
  vs 2x pooled spread 0.04) — a single ready session walks `decode_step_batch_hyper` at
  B=1 instead of `decode_step_hyper`.
- **No multiplicative amortization.** Per-session tok/s still collapses ~1/c (ON c=12:
  3.6/session); the batched tick's cost stays near-linear in B on the real artifact —
  the per-session mixer segment (KDA + MLA/kpool, per-row by construction) plus the
  low-overlap routed-expert unions at B<=12 dominate; the amortizable trunk (hc glue +
  shexp + head) is the ~10-21%. The attribution's "~10-20M+ tok/day class" hope for this
  lever alone is REFUTED on this placement (measured, no-throughput-by-analogy).
- Non-monotonicity across c on the same arm (ON c=2 37.0 > c=4 34.7) is pool composition
  (the c=4 pick list carries the 3.7k-deep A4630 prefill in a 30s burst), not a knee; the
  ON/OFF ratio at fixed c is the clean comparison.

Vendor-default sampled twin (round 1, full ladder both arms, NO sampling params on the
wire): tracks greedy within noise on every rung — ON 36.95 / 34.73 / 37.28 / 38.71 vs
greedy 36.97 / 34.71 / 37.44 / 38.79 at c=2/4/8/12; OFF 32.77 / 29.63 / 30.69 / 31.33.
No short-row trap fired at 256 max_tokens (per-c row jsons carry completion_tokens).

8-turn larger-prompt cache twin (round 3, vendor mode, turns 4.6k->7.9k prompt tokens):
per-turn TTFT 2.22->3.40s OFF and 2.23->3.40s ON — identical to the ms, and equal to the
3way plain-arm twin (2.23->3.40). `cached_tokens=0` on every turn of both arms (the
receipted glm5 dead prefix cache, TRAP:glm53:prefix-cache-snapshot-refused;
PREFIX_CACHE_MB=0 pinned in the recipe) — hyper-batch neither helps nor harms multi-turn,
and the cache-conditional rows stay dead on this family.

Receipts: `receipts/c2/` (per-boot gates/identity/vram, per-rung conc jsons + tapes,
vendor twins, twin.json, log-receipts, console).

## Cell 3 — admission interplay at high c (ON arm, count-based): **GREEN**

Boot `c3-on` GATES GREEN. Reference: the gpf-workspace admission receipts
(`MEMRA_ADMIT_PREFILL_WORKSPACE` default ON at this head).

- **Burst A, c=12 ALL-DEEP** (l3 WARM/A4630/B5550/C6470 x3 = ~48k prompt tokens of
  concurrent monolithic prefill, the workspace stress shape): 12/12 HTTP 200, all
  completed (agg 28.63, TTFT ~24.6s serialized prefill wall), **zero engine OOM** — the
  262k-2card failure surface (mid-stream `class=Overloaded`) does not reproduce on the
  3-card shape at this depth.
- **Burst B, c=20 mixed** (over `MEMRA_MAX_SESSIONS=16`): 20/20 HTTP 200. The session bar
  behaves as documented — 16 admitted (TTFT ~12.9s), 4 queued FIFO (TTFT 104-114s, served
  after slots freed, never rejected); `MEMRA_MAX_QUEUE_DEPTH` backstop untouched at +4.
- Log receipts: `admit_oom_lines=0 overloaded_lines=0 shed_queue_lines=0 panic_lines=0`;
  post-stress fresh sample fluent (server healthy).
- Verdict nuance, stated: at these depths NOTHING needed shedding, so this cell proves
  the no-engine-OOM half and the session-bar half; no `[admit-oom]` 429 was provoked
  (the 3-card residency has headroom for 12 concurrent deep prefills). The 429-shape
  itself is already red-armed in the gpf-workspace lane's own receipts.
- Receipt miss, honest: the 1 Hz vramwatch subshell had an ordering bug (checked its
  sentinel file before it was touched) and wrote nothing; per-boot vram-at-ready and the
  no-OOM outcome receipts stand without it.

Receipts: `receipts/c3/`.

## Cell 2b — the c=15 rung (cap width, timed, marker up, interleaved x3/arm)

Single batched chunk at the derived cap (B=15). Picks = all 14 pool items + a duplicate
d00 (named deviation: the pool has 14 items; the duplicate is timing-only). This mix is
DEEPER than the c=12 rung (4 deep prompts incl. B5550/C6470 vs 2), so absolute aggregates
are not cross-c comparable; the ON/OFF ratio at fixed picks is the reading.

| arm | agg median | spread% | dw | per-sess p50 | TTFT p50 / max |
|---|---|---|---|---|---|
| OFF | 30.45 | 0.084 | 30.56 | 2.3 | 12.90 / 13.12 |
| ON | 36.54 | 0.086 | 36.83 | 2.8 | 12.71 / 12.75 |

**ON/OFF = 1.2000x at the cap width** — the multiplier plateaus at ~1.20x from c=8 up.
Loop-law 0/84. Receipts: `receipts/c15/`.

## Cell 4 — economics row (analysis over cells 2/2b, no new hardware time)

Attribution baseline, quoted verbatim (`glm5-decode-gap-20260830/ATTRIBUTION.md` §7):
"output ceiling 3.06M/day x $1.00-2.03/M = **$3.1-6.2/day**; input ceiling at the measured
616-687 tok/s prefill fully saturated = ~55.9M tok/day x $0.20-0.30/M = **$11.2-16.8/day**;
total **$14-23/day ceiling against a 3-4 card PRO 6000 box** (comparable receipted rents:
step37's 2-card box $3.20/hr = $76.8/day; ECON:scaling-gate-pass's 2-card box $45.49/day,
2026-08-13). This shape is unit-negative at saturation by roughly 3-5x."

Measured update:

- Today's default under load is WORSE than the attribution's 3.06M baseline: OFF aggregate
  at c=2..15 reads 30.2-33.8 tok/s = **2.61-2.92M tok/day** — the 3.06M single-stream
  ceiling only exists at exactly c=1.
- Best stable measured rung: **ON at c=12 = 38.78 tok/s burst (39.03 decode-window) =
  3.35M tok/day (3.37M dw)** — **+9.5% vs the 3.06M attribution baseline, +21.4% vs the
  OFF arm at the same load**; the multiplier is a stable ~1.20x from c=8 through the cap.
- Revenue arithmetic at the attribution's anchors: output ceiling moves $3.1-6.2/day ->
  **$3.35-6.80/day**; total ceiling $14-23/day -> **~$14.6-23.6/day** against the same
  receipted rents. **The shape stays unit-negative at saturation by roughly 3-5x** — the
  flip is strictly positive (output-axis $/Mtok at a $76.8/day rent: $25.1/M -> $22.9/M,
  -9.5%) but does NOT change the economics class.
- The attribution's "box ceiling from ~3.1M to ~10-20M+ tok/day class" hope for this lever
  is **REFUTED as a standalone** on this placement (measured; no-throughput-by-analogy):
  the batched tick stays near-linear in B (per-session mixers + low-overlap expert unions),
  so batching buys ~1.2x, not several-x. The several-x aggregate needs the single-stream
  levers (#1-#5) and/or TP (#7) first — hyper-batch then multiplies whatever they buy.

## VERDICT — the default-flip decision packet

**FLIP: `MEMRA_HYPER_BATCH` should be ON for glm5_next serving (deployment now, engine
default in a follow-up lane with the FLAGS.md row updated in the same PR per the
new-flags law).**

For (all measured this window, build 34e0c0bf2, real artifact, deployed 3-card PP
placement): +9.5%/+14.8%/+19.5%/+21.4%/+20.0% aggregate at c=2/4/8/12/15; ON never loses
at any c>=2; TTFT under load <= OFF everywhere; per-session p95==p50 (no straggler tail);
vendor-default sampled twin tracks greedy; correctness bit-identical concurrent-vs-solo
at c=2 and c=12 (the gate's contamination red class does not reproduce on the served
path); admission/session bars clean at c=12-deep and c=20.

Caveats, priced:

1. **B=1 cost -0.30%** single-stream (35.32 vs 35.42; gap 0.11, 2x pooled spread 0.04) —
   a lone session walks `decode_step_batch_hyper` at B=1. Two concurrent sessions repay
   it 30x over.
2. **Residency**: VRAM-at-ready identical across arms (the walk adds no weights);
   c=12 all-deep concurrent prefill does not OOM on this shape.
3. **MAX_SESSIONS interplay**: chunk cap is 15; 16 active sessions form a 15+1 chunk
   split. Pin `MEMRA_MAX_SESSIONS<=15` (or a multiple of 15) on glm5 boxes; the 15+1
   steady state is unmeasured (c=20 burst rode it transiently, cleanly).
4. **K/spec**: both arms plain (`MEMRA_GLM5_SPEC`/`DFLASH` stay OFF per their own NO-FLIP
   receipts). If spec ever flips, the c>2 K-shed sends spec sessions to plain, which then
   ride this walk; blanket K pins at c>2 remain forbidden (3way cell-5, -24%); operator K
   pins clamp to cap-1=14 (worker.rs).
5. Protocol: interleaved x3 per the amended owner protocol (2026-08-30), no escalation
   rule fired (spreads 0.031-0.188%); the FLAGS row's original "x5" phrasing predates the
   amendment. c=1..15 timed + c=20 count-based covers the row's "c=1..32+" intent up to
   the session bar this recipe serves; a >16-session steady-state cell is named follow-up
   if MAX_SESSIONS is ever raised.
6. The 8-turn cache twin ran honestly: glm5 prefix cache is structurally dead
   (cached_tokens=0 both arms, TRAP:glm53:prefix-cache-snapshot-refused) — the FLAGS
   row's "cache-on twin" is unreachable on this family; multi-turn TTFT identical across
   arms to the ms.

Window totals: 16 boots (2 c1 + 12 ladder incl. c15 + 1 c3 + post-stress), 0 boot
failures, GATES GREEN 16/16, byte-identity 36/36 tapes + 8 bars, loop-law 0/448 across
all screened tapes, 1 receipt miss (c3 vramwatch) stated.
