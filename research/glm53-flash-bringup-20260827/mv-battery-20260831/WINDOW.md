# glm5 MATVEC-DOORS box window (mv-battery, 2026-08-31)

Owner: mv-doors agent. Box B (glm53 lane, 4x RTX PRO 6000 Blackwell WS 96 GB),
cards 0/1/2 (CUDA_VISIBLE_DEVICES=0,1,2), port 18400, out=/root/out-mv/.
Queued PENDING 2026-08-31T07:03:22Z behind the tp2-battery window (TIMING-IN-FLIGHT
protocol; no build or inference before its done-line).

Charter: `../matvec-20260831/LANE.md` §7 — price the five matvec doors on the serving
shape and produce the current best single-stream number against the 100 bar.

## Pins

- Build: `146b13c33` = origin/lane/glm5-matvec head (consol-db 32dc957b8 = vrest +
  diet-battery merged, VERIFY_BATCH + HYPER_BATCH default ON, + the five matvec doors
  T/X/M/K/W, all default OFF). Own clone /root/memra-mv; build attribution receipt
  `build-146b13c33.log` (rebuild-after-checkout-attribution law; strings probes for the
  five door announces + the batched verify walk + the vrows pair announce).
  Coordinator note: bringup merge 94e6e4872 makes origin/lane/glm53-flash-bringup equal
  content; the window pins the matvec lane head.
- Artifact: /root/models/glm53-nvfp4 (the HF-verified 20-shard NVFP4 mint).
- Drafter: /root/models/glm53-dflash2, sha256 re-verified vs pin `b33c03475ba7322c`.
- Serving env: byte-identical to flip-battery/flip-reprice/diet-battery (3-card pinned
  recipe: PP_STAGES=3 SPLITS=15,30 BF16_MMV=1 PP_BF16=1 MOE_GROUPED_PREFILL=1
  MOE_RESIDENT_GB=98 MOE_SLOTS=16 CTX=131072 MAX_SESSIONS=4 PREFIX_CACHE_MB=0
  TF32_OVERRIDE=0 TIMEOUT_MS_MAX=600000). HYPER_BATCH + VERIFY_BATCH left at their
  ON defaults in every arm.
- Base arm (the ship config): DFlash2 + auto-K nopin + PMIN0.7, no diet doors —
  baseline **62.43 tok/s** decode-pool median (diet-battery V3 SHIP row, vrest head
  a3fc59aaf; deep 53.06, vendor 49.60, TTFT 3.53s@3.7k; K5 pin 63.56).
- Prediction (LANE.md §5, arithmetic not claim): composed doors ~69-76 tok/s.

## The five doors (all default OFF; announce demanded per ON boot, forbidden OFF)

| door | flag | announce |
|---|---|---|
| T drafter-head weight-once | MEMRA_BF16_TCOLS_WIDE=1 | `[bf16-tcols-wide] engaged` |
| X tcols x1-row grid | MEMRA_BF16_TCOLS_X1=1 | `[bf16-tcols-x1] engaged` |
| M moe verify-rows warp pack | MEMRA_MOE_VROWS_PACK=1 | `[moe-vrows-pack] engaged` |
| K topk shard split | MEMRA_TOPK_SHARDS=1 | `[topk-shards] engaged` |
| W verify-walk workspace | MEMRA_GLM5_VERIFY_WS=1 | `[glm5-verify-ws] engaged` |

Engagement scope (stated before the window): all five engage only on SPEC boots on
this recipe (T: drafter block head t=15; X: verify-walk tcols t=K+1; M/W: verify-rows
MoE pair + walk; K: DFlash2 selector, n_cols>=16384 gate). Every arm in this window is
a spec boot.

## Cells (per LANE.md §7; interleaved x3, x5 on anomaly; greedy instrument + one
vendor-default sampled row per boot; loop-law screen on every tape)

1. BYTE-IDENTITY SPOT (THE STOP BAR): composed-doors vs no-doors greedy tapes, ship
   spec shape pinned K=3, 4 prompts (d00/d02/d06/l3-A4630). All five doors carry rig
   bit gates — ANY divergence is a defect and STOPS the window.
2. THE COMPOSED RE-PRICE (the decision number, TIMED): ship config doors-OFF vs all
   five ON, interleaved x3 fresh boots. Decode/deep pools, TTFT 0.4k/3.7k, vendor row.
3. Per-door singles T/M/X/K/W (one boot each on the ship config, attribution evidence;
   own announce demanded, other four forbidden; identity vs the c2 off-1 tapes).
4. K-LADDER RE-PIN on the composed shape: pinned K3/K5/K7 + PMIN0.7 (door T removes
   the per-round 15x head re-read; X/M cheapen the verify row — the K economics moved).
5. CENSUS RE-RUN on the winner (duration-bounded nsys, c8-ship instrument config):
   per-kernel table vs LANE.md §1 — tcols GB/s, x4_rows gone from the round, moe pair
   us/call, topk us, allocs/token. Feeds the last-mile lane list.

## RESULTS (window run 2026-08-31T07:41-09:12Z, marker held; build GREEN 310s
sha16 bd0a973e19edcc6f, 7/7 strings probes; drafter sha re-verified == pin)

### Cell 1 — byte-identity STOP bar: GREEN

4/4 composed-doors tapes byte-identical to no-doors (ship spec K=3 pinned,
d00/d02/d06/l3-A4630, greedy 256, served path). All five announces demanded+present
ON / forbidden+absent OFF; batched-walk + vrows pair announces GREEN both spec boots;
loop-law 0/8. Announce shapes: `[bf16-tcols-wide] t=7 in_f=4096 out_f=154880` (the
drafter head's first call), `[topk-shards] rows=7 cols=154880 k=16 shards=16`.

### Cell 2 — THE COMPOSED RE-PRICE (x3 interleaved, spreads 0.088% / 0.007%)

| arm | dec tok/s (boot medians) | deep | pool TTFT | TTFT 0.4k/3.7k | vendor |
|---|---|---|---|---|---|
| off (ship, doors OFF) | **62.416** [62.468/62.413/62.416] | 53.02 | 1.043 | 1.127 / 3.533 | 57.98 |
| don (ALL FIVE ON) | **70.090** [70.095/70.090/70.090] | 58.73 | 0.972 | 1.057 / 3.447 | 65.52 |

off reproduces the banked 62.43 baseline to 0.02%. don = **1.1230x**, inside the §5
predicted 69-75 band. Vendor-default sampled rows win too (the serving law's shape).
Loop-law 0/84. X3 sufficient, no escalation rule fired.

### Cell 3 — per-door singles (one boot each, attribution; identity 5x 14/14 vs off-1)

| door | dec tok/s | ratio | verdict evidence |
|---|---|---|---|
| T tcols-wide | 67.320 | **1.0786** | the big lever, as §1 predicted |
| X tcols-x1 | 64.070 | **1.0265** | |
| K topk-shards | 63.378 | **1.0154** | |
| W verify-ws | 62.544 | **1.0021** | host lever; census receipt below |
| M vrows-pack | 62.160 | **0.9959 LOSS** | predicted -1.5..-3.1 ms did NOT transfer |

Singles additive: sum +7.39 vs composed +7.67. Two vendor rows excluded by the
128-floor (ct=85, ct=57 — the 3way estimator trap, guard working). Loop-law 0/70.

### Cell 4 — K-ladder re-pin + composed-minus-M (THE WINNER)

All-five composed + pin: k3p 70.156 (pin == auto policy, control clean) / k5p 70.419
(+0.47%) / k7p 66.218 (0.9448x LOSS). The banked vrest-head K5 gain (+1.8%) SHRANK to
+0.5% — the cheaper verify row moved the K economics DOWN, not up; smallest-
competitive-K law holds, auto-K3 stays on-peak.

**WINNER = composed-minus-M (T+X+K+W, M OFF), auto-K nopin, PMIN0.7:
70.458 tok/s x3** [70.458/70.482/70.397], spread 0.12%, +0.367 vs all-five don
(> 2x pooled spread 0.084) = **1.1288x vs doors-OFF**. Deep 59.01, pool TTFT 0.968,
TTFT@3.7k 3.441, vendor 73.10 (1 exclusion ct=88). Identity 14/14 vs off-1.
Winner + K5 pin (w5p, single boot): 70.965 (+0.72%), deep 60.66; its vendor row
excluded ct=20 (estimator collapsed to 45754 tok/s — the floor guard caught it, named).
Loop-law 0/98 across c4.

### Cell 5 — census re-run on the winner (duration-bounded nsys, CUPTI-flush trap fix;
239p+192c, 73 rounds, acc-rate 0.696 — the census prompt ran unwrapped vs c8-ship's
wrapped 189p/59-round form, so PER-ROUND is the comparable unit, stated)

- **x4_rows is GONE from the round**: 1 instance in the whole capture (was 1/round);
  the drafter head rides the weight-once tcols class. Door T's census receipt.
- Under door X, ALL tcols traffic runs `matvec_bf16_f32acc_x1_tcols` (10,877 inst,
  avg 63.9 us) — no x4_tcols, no tcols16 rows: X supersedes the grid form at every t.
- bf16-mmv bucket 695.9 ms = **24.7% of GPU (was 31.8%)**, ~9.5 ms/round after T+X.
- topk: 1.31 -> **0.57 ms/round** (shard 547 us + merge 21 us; below the predicted
  0.1-0.2 band but a clear -0.74 ms/round win).
- Door W: cuMemAllocAsync **4489 -> 3849/round (-14%)**, frees 4458 -> 3820 — about
  -1,280 driver calls/round of ~8,900, near the §4 ~17% arithmetic. Syncs flat/round.
- THE REMAINING WALL: **moe verify pair 9.86 ms/round = 36.4% of GPU** (gate_up 156.3
  us + down 78.4 us x42 layers). M-pack's occupancy fix did not transfer, so the pair
  is NOT occupancy-bound on this box — next levers are the expert-read locality /
  fused-epilogue class, not warp packing. Then: mla+indexer 14.2%, cublas-f32 7.6%
  (lt_ndep, vrest follow-up #2), mhc 7.3% (owned elsewhere), remaining host churn
  ~3,850 allocs/round (door-W extensions, LANE §6.1). HtoD 71.6 calls/tok (14.7s API
  wall in capture) is unattributed — folds prefill staging + drafter uploads; named
  for the follow-up.

## Verdict and recommendations

1. **THE NUMBER: best single-stream today = 70.46 tok/s** (x3; ship config + doors
   T/X/K/W, auto-K3 nopin, PMIN0.7) vs 62.43 baseline = **+12.9%**. 70.96 with a K5
   pin (single-boot row). Against the 100 bar: **1.42x still needed** (1.41x at K5) —
   this window is the matvec leg; the bar needs the TP-2/composition leg on top, and
   the tp2-battery window (same day) showed bare TP-2 does NOT pay (22.65 tok/s), so
   the remaining 1.42x lives in: moe-pair efficiency (36.4% GPU), door-W extensions,
   cublas-f32 batching, and a mature TP composition.
2. **Door defaults (FLAGS flip decisions, receipts = this window):**
   T `MEMRA_BF16_TCOLS_WIDE` FLIP ON (1.0786x single, identity everywhere).
   X `MEMRA_BF16_TCOLS_X1` FLIP ON (1.0265x).
   K `MEMRA_TOPK_SHARDS` FLIP ON (1.0154x).
   W `MEMRA_GLM5_VERIFY_WS` FLIP ON (+0.13 tok/s single + the -14% driver-call census
   receipt; no downside observed in any arm).
   M `MEMRA_MOE_VROWS_PACK` KEEP OFF (0.9959x single; all-five 70.090 < minus-M
   70.458 — negative in composition too).
3. **K policy: keep auto-K (nopin)** — pin==policy at K3 (70.156 vs 70.090), K5 buys
   only +0.5-0.7% single-stream (single-boot precision) and a blanket pin costs
   concurrency (flip-reprice c=4 receipts); re-visit K5 only with a c>=2 twin.
4. Window totals: 22 boots, 0 boot failures, identity 4/4 + 5x14/14 + 14/14 (STOP bar
   + singles + winner), loop-law 0 flagged of 350 screened tapes, engagement announces
   demanded/forbidden per arm every boot, 3 vendor rows excluded by the 128-floor
   (named above).

## Status log

- 2026-08-31T07:03Z PENDING line queued; prep (worktree, scripts, this doc) banked.
  Build deferred until the tp2-battery done-line.
- 07:23Z tp2 cell-4 done-line; 07:31:47Z nice-19 build launched under MARKER DOWN
  (disclosed); 07:31:58Z tp2 WINDOW DONE; build GREEN 310s.
- 07:41Z WINDOW START, marker raised and held. Cells 1-5 + w5p run; banked per cell
  to this lane (c1..c5 commits). 09:12Z window close, marker down, cards clean.
