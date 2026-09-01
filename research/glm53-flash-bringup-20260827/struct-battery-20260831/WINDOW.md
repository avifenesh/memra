# glm5 STRUCT-BATTERY box window (struct-battery, 2026-08-31)

Owner: struct-battery agent. Box B (glm53 lane, 4x RTX PRO 6000 Blackwell WS 96 GB),
cards 0/1/2 for served cells (CUDA_VISIBLE_DEVICES=0,1,2) + cards 0,1 for the TP-2
engine twins, port 18400, out=/root/out-struct/. The combined box window for THREE
lanes' box plans in one sitting:

- `../moe-loc-20260831/LANE.md` — price doors D (`MEMRA_MOE_VROWS_DEV_TABLES`) + H
  (`MEMRA_GLM5_HTOD_DIET`) on the ship shape (predicted band 71.4-76.6 tok/s composed),
  and run the §4.6 dedup instrument boot (`MEMRA_MOE_VROWS_DEDUP_STAT`) that collapses
  the +0.9%-to-+23% dedup-lever span to one number.
- `../ep-place-20260831/LANE.md` §4 as amended — real-traffic expert traces per class,
  per-class placement mints with the shared tool, and the first-of-fleet naive-vs-
  coactivation placement A/B (engine twin; the served worker refuses MEMRA_GLM5_TP in
  v1 by design).
- `../mv-battery-20260831/WINDOW.md` — the 70.458 tok/s ship winner (DFlash2 + auto-K
  nopin + PMIN0.7 + doors T/X/K/W) is the baseline every priced row compares against.

## Pins

- Build: **c7d936536** = `lane/glm5-struct-battery` head = the moe-loc merge (door D/H +
  dedup instrument, double-fix reconciliation on the doors gate file) on the ep-place
  bringup head a5d608b07. Verified ancestry: contains lane/glm5-moe-loc @ 6e6120a0e AND
  the ep-place merge (ep_map.rs + `moe_vrows_tables_from_sel` + `[moe-vrows-dedup]` all
  in-tree). Own clone /root/memra-struct; build attribution `receipts/build.log`
  (rebuild-attribution law; strings probes: door D announce, dedup line, verify-ws,
  HTOD_DIET, and the probe's ep-map/tp announces).
- Artifact: /root/models/glm53-nvfp4 (the HF-verified 20-shard NVFP4 mint).
- Drafter: /root/models/glm53-dflash2, sha256 re-verified vs pin `b33c03475ba7322c`.
- Serving env: byte-identical to mv-battery (3-card pinned recipe: PP_STAGES=3
  SPLITS=15,30 BF16_MMV=1 PP_BF16=1 MOE_GROUPED_PREFILL=1 MOE_RESIDENT_GB=98
  MOE_SLOTS=16 CTX=131072 MAX_SESSIONS=4 PREFIX_CACHE_MB=0 TF32_OVERRIDE=0
  TIMEOUT_MS_MAX=600000). HYPER_BATCH + VERIFY_BATCH at their ON defaults everywhere.
  **Doors T/X/K/W are DEFAULT ON at this head** — the ship config needs no door flags,
  and any doors-off control would pin all four `=0` explicitly (none run here).
- Engine-twin env (cell 5): the tp2-battery arm table verbatim (CUDA_VISIBLE_DEVICES=0,1
  MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16), plus
  `MEMRA_GLM5_EP_MAP=<minted map>` on the map arm only.
- Baseline: **70.458 tok/s** decode-pool median (mv-battery c4 winner, x3, this exact
  env); the 100 bar needs 1.42x on it.

## Cells

1. **D+H PRICING** (TIMED, marker, interleaved x3, x5 on anomaly): ship config with
   D+H ON vs OFF (both flags pinned `=0` on the OFF arm). Engagement announce
   `[moe-vrows-dev-tables]` demanded ON / forbidden OFF (door H is counter-anchored, no
   announce). Cross-arm tape identity is a STOP bar (both doors carry rig bit gates).
   Deliverable: the current-best number vs 70.458 and vs the 100 bar.
2. **DEDUP STAT** (untimed, count-based): one boot `MEMRA_MOE_VROWS_DEDUP_STAT=1
   MEMRA_MOE_VROWS_DEV_TABLES=0`, real pools (greedy both pools + vendor-shape rows),
   bank the `[moe-vrows-dedup]` lines + per-phase deltas. Decides the dedup lever.
3. **TRACE CAPTURE** (untimed): `MEMRA_MOE_TRACE` + `MEMRA_MOE_WEIGHT_TRACE` boots per
   traffic class — agentic (d00-d05, the SXC-derived decode pool), prose (d06-d09),
   l3 deep (WARM/A4630/B5550/C6470), each on a PLAIN boot (pure t=1 rows, the shape the
   A/B instrument decodes) + one SPEC ship-shape agentic boot (verify-row structure).
   Traces banked with sha256s + t-histograms; boot-health sample rows truncated out.
4. **MAP MINTS** (CPU): shared tool per class x {coactivation, frequency, even},
   `--ranks 2 --entry-rank 0 --expert-count 288 --decode-only`; ship-shape t<=8
   sensitivity mint labeled. Deliverable before any A/B: per-class predicted
   single-rank fraction (1 - peer_touch_fraction) + the per-layer coactivation-vs-even
   scan (the fixture's greedy-loses-on-a-layer finding, checked on real traffic).
5. **PLACEMENT A/B** (TIMED): engine-twin (tp2-battery harness; its banked instrument
   trap — engine twins under-read served — survives because both arms ride the same
   instrument and the deliverable is RELATIVE). Identity spot first (map teacher-forced
   on even tapes; rig arm M bar = decode byte-identical), then even vs coactivation map
   interleaved x3, decode pool, greedy + one vendor-default sampled row per run.
   Deliverables: the relative delta + the stats-vs-measured reconciliation (does
   peer_touch_fraction predict the peer-slot dispatch ratio and the win?).
6. **CENSUS SPOT** on the D+H winner if time allows (duration-bounded nsys, c8-ship
   instrument config; the CUPTI-flush trap fix).

Bank per cell: rig worktree `/tmp/wt-ml` on `lane/glm5-struct-battery` (branch base =
the built head), receipts under this dir, identity scrubbed (system_fingerprint +
request ids; boot nonces kept), push per cell.

## RESULTS

### Cell 1 — D+H PRICING: WIN 1.0154x (x3 interleaved, spreads 0.112% / 0.034%)

| arm | dec tok/s (boot medians) | deep | pool TTFT | TTFT 0.4k/3.7k | vendor |
|---|---|---|---|---|---|
| dhoff (ship, D+H pinned `=0`) | **70.403** [70.475/70.403/70.396] | 58.98 | 0.969 | 1.052 / 3.445 | 58.36 |
| dhon (D+H ON) | **71.489** [71.508/71.489/71.484] | 59.86 | 0.960 | 1.043 / 3.429 | 61.82 (1 excl ct=107) |

dhoff reproduces the mv-battery 70.458 winner to 0.08%. dhon = **1.0154x**, +1.085
tok/s (> 2x pooled spread 0.158) — the -0.5 ms/round FLOOR of the moe-loc predicted
band (71.4-76.6): the 42 sync removals transferred at the bottom of the band, i.e. the
launch-submit wall reforms behind the removed drains (the diet-window class). Vendor
rows win too. Cross-arm tape identity 3x 14/14 (the D+H bit gates hold on the served
path). Loop-law 0/84. **Current best single-stream: 71.49 tok/s; the 100 bar needs
1.399x on it.**

### Cell 2 — THE DEDUP NUMBER: 21.96% repeat fraction (the lever is REAL)

One instrument boot (`MEMRA_MOE_VROWS_DEDUP_STAT=1 MEMRA_MOE_VROWS_DEV_TABLES=0`, ship
recipe), 2376 `[moe-vrows-dedup]` lines banked. Cumulative over 99,751 layer-calls /
2.55M visits: **repeat = 21.96%** (visits/call 25.6 vs the 26.7 arithmetic). Greedy
pools 22.27% vs vendor-default sampled 21.53% — mode-stable, so the number is a routing
property, not a decoding artifact. That is **6.9x the 3.2% independent-routing bound**:
the t~3-4 verify rows of one layer-call share a fifth of their expert-slab reads. Per
the banked §1.6 sensitivity table this is the **~-2.1 ms/round / +5.6-6% ship class** —
the dedup kernel campaign (expert-major pair ordering + weight-once shared-slab twin,
moe-loc follow-up #4) is justified by measurement.

### Cells 3+4 — traces + mints: the placement structure is real

Traces (greedy on real pools, tap pair armed, class purity by post-sample truncation;
full-file shas in `receipts/c3/trace-receipts.txt`, banked twins are the decode-filtered
mint inputs): agentic 63,462 / prose 43,008 / l3 43,008 t=1 rows; agentic-ship 24,528
verify rows (t=2: 7,224 / t=3: 3,696 / t=4: 13,608 — auto-K3's t=K+1 shape).

Mints (shared tool, ranks 2, entry-rank 0, expert-count 288, `--decode-only`):

| class | strategy | peer_touch mean | **single-rank fraction** | exp max-rank touch (vs even) |
|---|---|---|---|---|
| agentic-t1 | even | 0.9929 | 0.71% | 5.079 |
| agentic-t1 | **coactivation** | 0.6084 | **39.2%** | 6.894 (vs 5.079) |
| agentic-t1 | frequency | 0.9939 | 0.61% | 5.066 |
| prose-t1 | coactivation | 0.5133 | **48.7%** | 7.103 (vs 5.067) |
| l3-t1 | coactivation | 0.5524 | **44.8%** | 7.016 (vs 5.090) |
| agentic-ship t<=8 (sensitivity) | coactivation | 0.8591 | 14.1% | 22.19 (vs 15.49) |

- The even split's 99.3% peer-touch (tp2 SHARD-MAP §3 naive arithmetic) is REPRODUCED
  from real traffic — the arithmetic was right.
- glm5's sigmoid noaux_tc-balanced router did NOT flatten co-activation: greedy
  bundling turns 39-49% of t=1 layer-token events single-rank.
- frequency == even (balances load, bundles nothing) — the co-occurrence term is the
  whole lever.
- Coactivation loses to even on **0/42 layers in every class** (the fixture layer-1
  worry does not materialize on real traffic).
- The trade is named: expected max-rank touch RISES 5.08 -> 6.89 (locality bought with
  per-event balance) — which lever wins depends on the walk (v1 sequential dispatch:
  peer-touch should dominate; a parallel-rank walk: balance matters).
- Ship-shape verify rows (t<=8) keep only 14.1% single-rank — a verify row unions t
  tokens' experts, so placement pays less on the spec shape; stated before anyone
  extrapolates the t=1 A/B to the ship config.

### Cell 5 phase A — map identity on the real artifact: BYTE-IDENTICAL 56/56

Even reference tapes (5 prompts, 200 steps) vs the coactivation map TEACHER-FORCED on
them: **56/56 shared files byte-identical including every full-vocab f32 dump (prime +
first 8 decode steps), worst_f32_rel = 0.0** — STRONGER than the rig arm-M bar (prime
was band-gated there; here even prime is bit-exact because both arms are the same TP
walk with only placement moved). `ep-map armed` announce demanded+present on the map
arm (map sha 56dea5ca...), absent on even. Placement independence is proven on the
real geometry; the timed A/B prices a correctness-free lever.

### Cell 5 phase B — THE PLACEMENT A/B: the map LOSES on the v1 walk, and the
counters say exactly why (x3 interleaved, spreads 0.165% / 0.138%, gap 9.6x pooled)

| arm | decode-pool tok/s (boot medians) | prime median | peer-slot dispatches (x3, identical) |
|---|---|---|---|
| even (naive split) | **22.746** [22.759/22.722/22.746] | 4.614 s | 843,969 |
| coactivation map | **22.032** [22.051/22.020/22.032] | 5.104 s | **386,043** |

**RELATIVE DELTA map/even = 0.9686 (-3.14%)** — while the map cuts peer-slot
dispatches to **0.4574x** (deterministic across all 3 repetitions of both arms — the
greedy-determinism receipt, same class as tp2-battery's identical-x3).

The reconciliation the cell was designed to deliver:

- **peer_touch_fraction predicted the DISPATCH cut and the dispatch cut is real**
  (mint: peer_touch 0.6084 vs 0.9929; measured peer-visit share 0.457x — the
  visit-level number, stronger than the per-event fraction because entry-rank pinning
  moves the hot experts' visits to root).
- **It did NOT predict the wall, because v1's join cost is not proportional to
  peer-slot count.** The mint's own named trade wins the sign instead: expected
  max-rank touch 5.08 -> 6.89 puts ~36% more sequential per-expert work on the
  critical-path rank, and the v1 walk (host-canonical per-layer fan-out + per-slot
  sequential dispatch) pays per-layer join regardless of how few peer slots remain.
  Prime is +10.7% slower under the map for the same reason at larger t.
- Identity was byte-exact (phase A), so the -3.14% is pure data-movement/scheduling —
  exactly the composition statement the ep-place box plan pre-registered: **the
  placement lever prices INSIDE the EP-dispatch-diet follow-up** (batched per-rank
  chains + direct join), where per-event peer hops become the cost driver the map
  actually removes. Do not re-run this A/B on the v1 walk; re-run it on the dieted
  walk.

Loop-law 0/60. Vendor-twin note (receipted): the x3 rounds ran greedy-only at
max_new=200 — `probe_arm.sh` scrubs inherited `BOXP_*`, so env-prefix
`BOXP_SAMPLED=1 BOXP_MAX_NEW=256` was silently dropped (the tp2 RUNBOOK's spelling
carries the same trap; its own vendor rows came from trailing-arg invocations). Both
arms identical, relative delta unaffected; a t4v vendor-twin round with trailing-arg
extras was run as repair. **t4v (max_new 256, one boot per arm): even greedy 22.550 /
vendor 22.58; map greedy 22.075 / vendor 21.87 — vendor map/even = 0.9686, the exact
x3 greedy ratio: the sampled traffic shape confirms the verdict** (serving-law twin).
even-t4v peer dispatches 1,012,568 == the tp2-battery banked pool-boot number EXACTLY
(cross-window determinism receipt); map-t4v 445,157 = 0.4396x.

### Cell 6 — census spot on the D+H winner: the count claim lands on the served path

Duration-bounded nsys, c8-ship instrument config, 239p+192c, 73 rounds, acc 0.6959
(== the mv c5 winner census shape). vs the mv-battery winner census (no D+H):

- **pageable HtoD is GONE**: `cuMemcpyHtoD_v2` 8 calls in the WHOLE capture (the
  winner census read 71.6 HtoD/tok); what remains is `cuMemcpyHtoDAsync_v2` 16.0/tok
  = 42/round — ONE async pinned copy per MoE layer-call (+ its `cuMemsetD8Async`
  twin, also 3066), the residual staging class, named for a follow-up count.
- **syncs 16.6 -> 4.4/tok** (`cuStreamSynchronize` 844 calls): door D's 42
  router-admission drains/round are off the round, as designed.
- allocs 1448/tok (was ~1463): the D+H alloc bite is real but small, as predicted.
- GPU buckets UNCHANGED (moe 36.4% with the pair rows at 156.3/78.4 us avg — byte-same
  as mv c5; bf16-mmv 24.9%): the doors are host-side; the GPU program did not move.

## VERDICT AND RECOMMENDATIONS

1. **Door defaults: FLIP D+H ON** (`MEMRA_MOE_VROWS_DEV_TABLES=1` +
   `MEMRA_GLM5_HTOD_DIET=1`): 1.0154x composed on the ship shape (70.403 -> 71.489
   x3, gap 13.7x pooled spread), identity 3x14/14 on the served path, vendor rows win
   too, census receipts for the exact counts the doors claim. **Best single-stream
   today: 71.49 tok/s** (the moe-loc predicted band's floor). Receipts = this window;
   the FLAGS default flip rides the next engine PR, not this receipts branch.
2. **The dedup lever is REAL and now the biggest priced GPU-side item**: 21.96% repeat
   fraction (mode-stable 22.27% greedy / 21.53% sampled) = the ~-2.1 ms/round /
   **+5.6-6% ship** class per the banked sensitivity table. Build order per moe-loc
   follow-up #4: expert-major pair ordering first (L2 law: 378 MB working set vs
   ~128 MB L2), then the weight-once shared-slab twin. Door R (tcols reduce-tail,
   -1.0 to -2.0 ms) remains the other named GPU lever.
3. **Placement maps: DO NOT ADOPT on the v1 TP walk** — first-of-fleet A/B says the
   coactivation map is 0.9686x there despite halving peer-slot dispatches; the win
   condition is the EP-dispatch-diet walk (batched per-rank chains + direct join).
   Adopt-order: diet first, then re-run THIS A/B (maps + traces + tool receipts are
   banked and re-usable; the identity gate is already proven on the real geometry).
   The mint tool itself is validated: peer-touch/dispatch prediction confirmed at the
   counter level; per-class single-rank fractions 39-49% say glm5's balanced router
   left real co-activation structure on the table.
4. **The 100 bar arithmetic after this window**: 71.49 ship today; +dedup (~+5.6%)
   ~75.5; +door R (~+3-5%) ~78-79. The bar still needs ~1.27x beyond the named
   single-stream levers — it lives in the dieted TP composition (where the placement
   map re-enters), per the tp2-battery ladder.
5. Follow-up counts named: the residual 42/round async HtoD+memset pair on the vrows
   path; the two host embed gathers (moe-loc follow-up #1); the drafter tap round
   trip (#2).

## Status log

- 2026-08-31T11:52Z WINDOW START line queued (box was FREE after the mv-doors done-line;
  cards 0-3 at 1 MiB, no marker, no server). Clone + fetch + build launched nice-19
  (first attempt died on PATH — cargo not on the non-interactive PATH; relaunched with
  ~/.cargo/bin, receipt build-attempt1-nopath.log). Build GREEN 303s, strings probes
  6/6 + 3/3, drafter sha == pin.
- 12:10-12:26Z cell 1 (TIMED, marker held); banked 82f17406b. Marker down.
- 12:26-12:34Z cell 2 dedup boot; banked ce9b57bb5.
- 12:34-12:45Z cell 3 traces (4 boots) + cell 4 mints; banked 9ba1f74d4 (plain-text
  decode-filtered twins — gz binaries false-positived the public-boundary scanner and
  also blinded it; raw numerics pack equally small and stay scannable).
- 12:47-12:55Z cell 5 identity spot GREEN (56/56 byte-identical). ~12:56Z marker
  raised, timed A/B x3 interleaved launched.
- 12:56-13:47Z cell 5 timed x3 (marker held): map 0.9686x, peer dispatches 0.457x,
  x3 sufficient (gap 9.6x pooled). Banked.
- 13:49-14:04Z cell 6 census on the D+H winner (marker held): the door count receipts
  (pageable HtoD gone, syncs 16.6->4.4/tok), GPU buckets unchanged.
- 14:06-14:25Z cell 5 t4v vendor-twin repair round (the BOXP_* env-prefix scrub trap,
  found here, fixed in the banked c5_ab.sh): vendor rows confirm 0.9686. Marker DOWN.
- MID-WINDOW INCIDENT, receipted: the rig worktree /tmp/wt-ml was removed by the
  consolidation agent while this window was mid-flight (their moe-loc-merge lane
  closed and cleaned it; three commits — perf-ci rows + a clippy doc-line fix —
  landed on lane/glm5-struct-battery above this window's banks). Nothing was lost:
  every cell bank had been pushed before its close, un-banked receipts still lived on
  the box, and the window re-opened at /tmp/wt-struct off the pushed branch. All
  struct-battery commits verified ancestors of the new head before continuing.
- Window close: DONE line, /root/out-struct removed after final bank, marker down,
  cards 0-3 at 1 MiB, no processes. RETAINED: /root/memra-struct @ c7d936536 with the
  built memra-server (sha16 2a978b843d42746d) + glm5-tp2-box-probe (d4b2eb26a1d49393)
  — the D+H default-flip lane wants this binary; drafter + artifact stay.
