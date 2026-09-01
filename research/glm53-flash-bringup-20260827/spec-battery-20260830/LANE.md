# glm5 spec real-artifact battery + acceptance + trim + flip A/B (box stage, 2026-08-30)

Final box stage of the glm5 spec arc; verdict feeds the owner's spec-flip decision.

- Build: lane/glm53-flash-bringup @ `0a70b35d8102253c0989d953afa42c17e4aa3540` (consolidated
  head: spec routing + ppN verify + gpf-workspace admission + chunk fix + MLA TC).
  Rebuild receipt: `box/build-0a70b35d8.log` (exit 0, incremental over the prior window's
  cache, real 49s; binary strings-probed: 9 `glm5-spec` literals incl both
  `serve route ARMED` variants and `[glm5-acc]`; the serve-time fingerprint field echoed
  the build commit id live, and is redacted in every banked payload per box-scrub policy).
- Placement (deployed recipe, pinned on every boot; deviations named):
  `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 CUDA_VISIBLE_DEVICES=0,1,2
  MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MLA_TC_PREFILL=1
  MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 MEMRA_MAX_SESSIONS=4
  MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 MEMRA_COMPAT=openai` + model pin,
  port-scoped serve harness (`box/serve.sh`: pidfile + /proc exe + boot-nonce identity,
  boot gates: 3x RESIDENT, ARMED/TRIMMED lines on spec arms, zero `[glm5-spec]` on OFF).
  - Named deviations: `MEMRA_TIMEOUT_MS_MAX=600000` on all boots (measurement cell —
    the l3 deep prompts must not 408; FLAGS.md measurement override);
    `MEMRA_PREFIX_CACHE_MB=2000` on the 8-turn cache twin boots only.
  - `MEMRA_MLA_TC_PREFILL=1` pin named: A/B-green twice; default flip pending owner.
- Pools (real prompts only): decode-attribution pool (10 prompts, 6 code + 4 prose,
  `decode-attribution-receipts/prompts.json`), l3-ab deep pool (WARM 1.6k + A4630 14.5k +
  B5550 20k + C6470 23.2k chars, sha256 `de57a7a4...`).
- Ranks (mint lane `darklanes research/glm53-ranks-mint-20260830`, sha-verified on box):
  sxc (agentic, `1804027e...`), prose (`9498ed34...`), mixed (`8461ad2d...`).
- Upstream reference for stage 2: 3.71-5.06 acceptance, 1.36-2.05x decode.
  Card-3 probe reference (lane/glm5-card3-acceptance-probe, probe posture, single card):
  served K3 greedy acc/cycle 1.413 tok/cycle 2.413; trim plateau delta -1.2%.

## Stage log

(appended per stage; each stage banks before the next starts)

### Stage 1 — real-artifact accept/rollback battery (byte identity, untimed): GREEN

Served-path spec-vs-plain greedy byte identity on the deployed 3-card placement,
K in {1,3,5,7}, both pools (14 real prompts), max_tokens 256 (greedy law bound),
tape = reasoning + `\0` + content bytes, non-streaming.

- **56/56 tapes byte-identical to the plain boot** (14 per K arm; `run_pool.py compare`,
  receipts `box/s1/`).
- Mid-stream rejection-heavy coverage: the pool carries natural rejection-heavy rows —
  d02-code runs 125 rounds at acc-rate 0.210 (K5) / 0.150 (K7), l3-B5550 0.189 (K7);
  every one byte-identical, so the rollback path is exercised hundreds of times per arm.
- Loop-law screen: 0 flagged of 70 tapes (tail n-gram + repeated-line, `looplaw_screen.py`
  functions applied to every tape). Aggregates carry no exclusions.
- Boot gates green on all 5 boots: 3x RESIDENT, `[glm5-spec] serve route ARMED ... FULL
  target vocab` on spec arms, zero `[glm5-spec]` lines on the plain boot, boot-nonce
  verified in the serving pid's environ (receipts `box/logs/boot-s1-*.gates|.identity`).
- Acceptance plateau already visible (counts, greedy): rounds per prompt identical at
  K=5 and K=7 for every prompt — acceptance is bounded by actual match length, not K.
- acc@1 (greedy, K=1 arm): per-prompt 0.771-0.939, e.g. d07 0.803, l3-WARM 0.939.

### Stage 2 — acceptance (count-based): DONE

Both pools (14 real prompts: 6 code + 4 prose + 4 l3deep), max_tokens 128 (card3-comparable
shape), greedy = temperature 0, vendor-default = NO sampling params on the wire.
Receipts: `box/acc/` (per-prompt usage.spec + tapes), `[glm5-acc]` per-burst lines +
`[glm5-spec] route=spec K=...` lines in `box/logs/boot-acc-*.log`. Loop-law 0/84.

| arm | mode | acc/cycle | tok/cycle | acc rate | agentic | prose | l3deep |
|---|---|---|---|---|---|---|---|
| notrim K=3 | greedy | 1.443 | 2.443 | 0.481 | 1.482 | 1.502 | 1.332 |
| notrim K=3 | vendor-default | 1.365 | 2.365 | 0.455 | 1.343 | 1.530 | 1.251 |
| notrim K=5 | greedy | 1.473 | 2.473 | 0.295 | 1.523 | 1.540 | 1.342 |
| notrim K=5 | vendor-default | 1.386 | 2.386 | 0.277 | 1.377 | 1.443 | 1.344 |
| notrim nopin | greedy | 1.443 | 2.443 | 0.481 | (byte-equal to K=3 arm) | | |
| notrim nopin | vendor-default | 1.322 | 2.322 | 0.441 | 1.242 | 1.471 | 1.306 |

(class columns = acc/cycle per traffic class)

- **Policy receipt** (nopin boot log): `automatic table: prompt<1024 -> K=3; cold-long ->
  K=3; prompt>=1024 and cached>=1024 -> K=2 (K=5 when the loaded MTP head is rank-trimmed)`.
  nopin greedy counts are byte-identical to the K=3 pin — the flip arm's policy default is
  K=3 on these shapes.
- Sampled acceptance ~= greedy at K=3 (1.365 vs 1.443 acc/cycle; card3 saw the same shape).
- vs card-3 probe (single-card posture): K3 greedy 1.413 there, 1.443 here — consistent.
- vs upstream reference 3.71-5.06 acceptance: we sit at tok/cycle ~2.4; the upstream
  acceptance-length band is NOT reproduced on this artifact/pool (their number likely
  reflects deeper draft chains/tree drafting on their eval set; ours is the served shape).

### Stage 3 — trim A/B (count-based, K PINNED on both sides): DONE

K pinned explicitly on every trim and no-trim arm (the card-3 probe's K-policy finding:
the automatic table's "K=5 when trimmed" row keys on `cached>=1024`, so a trim boot
without a pin does NOT get K=5 — an unpinned trim/no-trim comparison would silently
compare different K). TRIMMED boot line verified on all four trim boots
(`[glm5-spec] serve route ARMED: MTP head loaded; draft head TRIMMED to 32768 rows`).

| arm (ranks) | mode | acc/cycle | tok/cycle | delta vs no-trim |
|---|---|---|---|---|
| sxc K=3 | greedy | 1.435 | 2.435 | -0.55% |
| sxc K=3 | vendor-default | 1.323 | 2.323 | -3.1% |
| sxc K=5 | greedy | 1.468 | 2.468 | -0.34% |
| sxc K=5 | vendor-default | 1.347 | 2.347 | -2.8% |
| prose K=3 (reference) | greedy | 1.441 | 2.441 | -0.14% |
| mixed K=3 (reference) | greedy | 1.438 | 2.438 | -0.35% |

- **Verify stays full-vocab by contract, re-proven on the served path**: trim K=3 greedy
  256-token tapes 14/14 byte-identical to the plain boot (`box/s3/trim-k3-id256`).
  Trim shifts WHICH tokens get drafted (round counts differ by ±1-2 per prompt vs
  no-trim), never how they verify.
- Rank-class choice barely moves acceptance on this pool at matched K (sxc vs prose vs
  mixed within 0.4%); consistent with card3's -1.2% plateau delta at the engine level.
- trim-nopin policy receipt (`box/logs/boot-s3-trim-nopin.log`): short cold prompt
  routes `K=3` — the table's `prompt<1024 -> K=3` row fires; the trimmed-K=5 row is
  unreachable without `cached>=1024`, i.e. trim alone never re-keys K (bug observation
  banked per the probe lane's finding).
- Loader-law observations at this head (recorded, expected): `nextn.eh_proj` now loads
  RESIDENT under bf16_mmv admit (the probe lane's Float-2D eh_proj row is gone);
  remaining `[loader-law] WARNING` rows are `kda_{f,g}_{a,b}` Float-2D (4 per boot,
  every MTP boot).
- Loop-law: 0 flagged of 98 stage-3 tapes.

### Stage 4 — flip A/B (timed, TIMING-IN-FLIGHT held for the whole window): DONE

Interleaved x5 fresh boot pairs, spec OFF vs ON (nopin = K=3 policy default), boot-nonce
arm identity on every boot, streamed greedy decode pool at max_tokens 256 (byte-identical
tapes across arms by stage 1, so tok/s is a pure speed comparison), deep TTFT rows
(l3 WARM 1.6k chars + A4630 14.5k chars), ONE vendor-default sampled row per boot.
Receipts: `box/s4/*/timed.json`, boot logs + `.identity`/`.gates` files.

#### The flip table (median over pool, then median of 5 boots; variance across boots < 0.1%)

| arm | decode tok/s | pool TTFT | A4630 deep TTFT | vendor row |
|---|---|---|---|---|
| spec OFF | **35.4** | **0.362 s** | **2.212 s** | 33.6-34.0 tok/s, no usage.spec, zero `[glm5-spec]` lines |
| spec ON (K=3 policy) | **27.5** | **2.220 s** | **12.258 s** | 19.6-59.3 tok/s (sampled-length noise), usage.spec present every row |
| ON/OFF | **0.777x** | **6.1x worse** | **5.5x worse** | |

- **Decode: spec ON is 22.3% SLOWER** despite tok/cycle 2.44 — the verify walk + K
  sequential MTP-head forwards under the 3-stage ppN split cost more than the accepted
  tokens buy back on this placement.
- **TTFT-unchanged assertion FAILS, attributed**: `glm5_spec_session_new` primes the trunk
  through the normal grouped prefill, then warms the MTP plane with a SEQUENTIAL per-token
  loop over the whole prompt (`for i in 0..plen-1 { mtp_head_forward_mla_cached }`,
  glm_spec.rs — "a t-parallel plane prefill is deliberately out of scope"). Spec TTFT =
  plain prefill + P sequential MoE-layer head forwards; the regression scales with prompt
  length (239-tok prompt: +1.86 s; ~3.7k-tok prompt: +10.0 s).
- Engagement receipts both arms, every boot: ON boots `[glm5-spec] route=spec K=3` +
  `[glm5-acc]` bursts + usage.spec on the vendor-default row (never-serve-greedy law);
  OFF boots zero `[glm5-spec]` lines, no usage.spec.
- Interleave x5 consistency: per-boot medians identical to 0.1 tok/s and 1 ms TTFT across
  all five pairs on both arms (box clock drift nil inside the held window).

#### 8-turn larger-prompt cache twin (owner law; vendor-default; per-turn TTFT + accept)

Twins: {OFF, ON} x {MEMRA_PREFIX_CACHE_MB=2000 (named deviation), 0}, 8 user turns
(A4630 seed + 7 decode-pool prompts, history grows to ~7.9k tokens), max_tokens 128;
plus an EOS-seeking re-run pair at max_tokens 2048. Receipts `box/s4/s4twin*/twin.json`.

- **Prefix cache CANNOT engage for glm5_next at this head** — loud named refusal on every
  snapshot attempt: `[prefix-cache] snapshot failed (latent (MLA/DSA) KV planes are not
  carried by prefix entries); prefix not cached` (boot logs, c2000 twins; budget line
  proves 2097 MB was configured). cached_tokens=0 on every turn of every twin, both arms,
  both budgets, both max_tokens shapes. The engagement receipt is the refusal line itself:
  cache-on vs cache-off is a no-op for glm5 serving today (prefix-latent restore is the
  PRODUCT-SUSPECT lane; entry layout v3 carries latent planes but the snapshot arm stays
  fail-closed here).
  - Corollary: the automatic K table's `cached>=1024 -> K=2` (and trimmed `K=5`) rows can
    never fire for glm5 — the policy default is effectively K=3 everywhere.
- Per-turn TTFT (cache-irrelevant, so this is the depth profile of the prefill
  regression), vendor-default, turns 1->8, prompt 4.6k -> 7.9k tokens:
  - OFF: 2.23 -> 3.40 s (both budgets identical)
  - ON: 11.8 -> 19.8 s (both budgets identical) — **3.5-5.8x worse per turn, growing
    with depth**
- Per-turn acceptance ON: acc/cycle 1.32-1.59 (usage.spec every turn), sampled shape.
- EOS-seeking re-run (max_tokens 2048): turn 1 EOS-terminated at 1241 tokens and the next
  turn STILL read cached_tokens=0 (rules out the parked-on-EOS explanation; the refusal is
  structural). Its ON arm doubles as a depth ladder for the spec prefill regression:
  TTFT 13.0 s @ 4.6k -> 39.1 s @ 16k prompt tokens (~2.5 s per 1k tokens of depth, i.e.
  the sequential MTP warm runs ~400 tok/s); route receipts `route=spec K=3 ... sampled=1
  cold=1` at prompt=16079/18311/20482 (never K=2 — cached is always 0).
- Loop-law: timed tapes greedy = stage-1-identical by construction; vendor rows length-
  bounded; 0 flags.

## Verdict and recommendation (for the owner's flip decision)

**Correctness: GREEN.** 70/70 served-path spec-vs-plain byte-identity (56 stage-1 + 14
trim spot-check), rejection-heavy rows included, loop-law 0 flags across 252 tapes.
The route, verify walk, rollback and trim seam are correct on the deployed placement.

**Acceptance: real but modest.** acc/cycle 1.44 (tok/cycle 2.44) at the K=3 policy
default, sampled ~= greedy; trim costs -0.5% (greedy) to -3% (vendor) at matched K.
The upstream 3.71-5.06 acceptance band does not reproduce on this artifact/pool.

**Performance: NO-FLIP.** On the deployed 3-card ppN placement, spec ON at the policy
default is 22.3% SLOWER on decode (27.5 vs 35.4 tok/s, interleaved x5, variance <0.1%)
and regresses TTFT 5.5-6.1x (scaling with prompt depth: +2.5 s per 1k prompt tokens from
the sequential per-token MTP-plane warm). The accepted tokens do not buy back the
draft+verify cost here, and the prefill regression alone is disqualifying for the
long-prompt agentic traffic this model serves.

**Recommendation: keep MEMRA_GLM5_SPEC default OFF on this placement.** Flip conditions,
in engine-work terms (both are named out-of-scope seams, not tuning knobs):
1. a T-parallel (or batched) MTP-plane prefill — removes the TTFT regression;
2. a cheaper verify cycle under ppN (the walk pays 3-stage pipeline overhead per round) —
   without it, break-even needs tok/cycle > 35.4/27.5 * 2.44 ~= 3.14 at current costs,
   i.e. ~2.1 accepted/cycle vs the measured 1.44.
Re-run stage 4 of this battery (same pools, same interleave) after either lands.
Trim: no acceptance win at matched K on any rank class; not a flip lever. Prefix cache:
glm5 entries refuse to snapshot (latent planes), so cache-conditional K rows are dead
code for this family until the prefix-latent lane clears its PRODUCT-SUSPECT.

## Wall actuals (BOX-QUEUE receipts)

| stage | window | wall |
|---|---|---|
| build + prep (fetch, checkout, rebuild, gates) | 08:30-08:39Z | 9 min |
| stage 1 byte-identity (5 boots, 70 tapes) | 08:39-09:05Z | 25 min |
| stage 2 acceptance (3 boots, 6 cells) | 09:06-09:25Z | 18 min |
| stage 3 trim A/B (5 boots incl nopin receipt, 7 cells + identity) | 09:26-09:51Z | 25 min |
| stage 4 flip A/B (16 boots: 5x2 interleaved + 4 twins + 2 EOS twins) | 09:52-11:05Z | 73 min |
| **total window** | 08:30-11:05Z | **2 h 34 min** (estimate was ~7 h; 32 s warm boots and short cells beat it) |
