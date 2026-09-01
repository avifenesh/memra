# glm5 DECODE-DIET box window (diet-battery, 2026-08-31)

Owner: decode-diet agent. Box B (glm53 lane, 4x RTX PRO 6000 Blackwell WS 96 GB),
cards 0/1/2 (CUDA_VISIBLE_DEVICES=0,1,2), port 18400, out=/root/out-diet/.
Queued PENDING 2026-08-30T22:35:13Z behind the flip-reprice window (TIMING-IN-FLIGHT
protocol; no build or inference before its done-line).

## Pins

- Build: `28cbc1af6` = origin/lane/glm5-decode-diet head (bringup 34e0c0bf2 +
  verify-batch merge + the four diet doors + post-merge re-proof). Own clone
  /root/memra-diet; build attribution receipt `build-28cbc1af6.log`
  (rebuild-after-checkout-attribution law; strings probes for all four door announces
  + the batched verify walk).
- Artifact: /root/models/glm53-nvfp4 (the HF-verified 20-shard NVFP4 mint).
- Drafter: /root/models/glm53-dflash2, sha256 re-verified `b33c03475ba7322c` == pin.
- Serving env: byte-identical to flip-battery-20260830/flip-reprice (3-card pinned
  recipe: PP_STAGES=3 SPLITS=15,30 BF16_MMV=1 PP_BF16=1 MOE_GROUPED_PREFILL=1
  MOE_RESIDENT_GB=98 MOE_SLOTS=16 CTX=131072 MAX_SESSIONS=4 PREFIX_CACHE_MB=0
  TF32_OVERRIDE=0 TIMEOUT_MS_MAX=600000). NOTE: `MEMRA_HYPER_BATCH` is DEFAULT ON at
  this head (hbatch-battery flip 2026-08-31) and is left at default in EVERY arm —
  c=1 rows unaffected per the ladder receipts (B=1 cost -0.30%).
- `MEMRA_GLM5_VERIFY_BATCH` default ON at this head — every spec arm runs the batched
  verify walk (BATCHED announce demanded, PER-ROW forbidden).
- Baselines: plain 35.408 tok/s decode / 30.00 deep / TTFT 0.422s@0.4k / 2.208s@3.7k
  (flip-battery cell 3, reproduced to the hundredth across windows). Predicted diet
  arms (LANE.md arithmetic, box prices them): lever 1 -1.8-2.8 ms, lever 2 -1.0-1.5,
  lever 3 -0.5-1.0, lever 4 -0.8-1.8; composed ~21.1-24.1 ms/token = 41-47 tok/s.

## Doors (all read PER CALL; default OFF at this head)

| door | flag | announce demanded in ON boots |
|---|---|---|
| 1 mHC pre-chain fusion | MEMRA_HC_FUSED_PRE=1 | `[hc-fused-pre] engaged` |
| 2 persistent decode workspace | MEMRA_HC_DECODE_WS=1 | `[hc-decode-ws] engaged` |
| 3 KDA fused-6 bf16 arm | MEMRA_KDA_FUSED_PROJ=1 | `[kda-fused6] engaged arm=bf16` (q8 form forbidden on this recipe) |
| 4 MLA decode-split | MEMRA_MLA_DECODE_SPLIT=1 | `[mla-decode-split] engaged` |

OFF arms demand ZERO door announce lines. Announces are once-per-boot at first
engagement, so the door gate runs after the fresh-boot sample. The dispatch counters
(`HC_FUSED_PRE_DISPATCHES` etc.) are engine-internal atomics not surfaced over HTTP:
the box engagement receipt is the announce line (with its shape params) + the A/B
identity + the timing delta; the 0->100 counter receipts are the rig gates'.

## Cells (interleaved x3 per the amended law, x5 on anomaly; greedy instrument + one
vendor-default sampled row per boot; loop-law screen on every tape)

1. COMPOSED-FIRST (the decision number): all four doors ON (`don`) vs all OFF (`off`),
   interleaved x3, TIMED under the marker. Decode tok/s both pools, TTFT 0.4k/3.7k.
   ON-vs-OFF greedy tapes byte-compared every round — ANY divergence STOPS the window.
2-5. Per-door single-flag rows (one boot each, only if composed wins cleanly):
   tok/s delta per door vs the cell-1 OFF baseline, matched prompts, isolation door
   gate (own announce demanded, other three forbidden). Single-boot rows = attribution
   evidence; the composed number is the claim.
6. Byte-identity spot on the composed shape: doors-ON plain vs doors-ON + DFlash2 K=3,
   greedy 256, 4 prompts (d00-code, d02-code rejection-heavy, d06-prose, l3-A4630).
7. THE COMPOSED SPEC RE-PRICE (the 100-bar number): composed plain (`don`) vs composed
   + DFlash2 K=1 (`dfk1`) / K=2 (`dfk2`) / K=3 (`dfk3`), interleaved x3, TIMED. Best
   single-stream number this shape produces today, reported against the owner's
   100 tok/s bar explicitly. Chain to doors-OFF plain = cell 1. K=2 is a NAMED
   DEVIATION from the pre-registered {1,3}: flip-reprice cell 3 (banked
   2026-08-31T00:05Z box clock, FIRST FLIP — K1 41.221 / K2 44.245 / K3 43.420 vs
   plain 35.423) measured K2 as the peak K on the batched walk, and this cell's
   deliverable is the best single-stream number, so the known peak is priced.
8. Census re-run on the winner (nsys 2026.1.3 via the launch-diet census script if the
   install lands, else the launch-econ constant): launches/allocs/syncs per token —
   feeds the matvec-pass lane decision.

Wall estimate: ~1.5-2h. Banking: per cell to this dir on lane/glm5-diet-battery
(worktree off origin/lane/glm5-decode-diet), scrub_bank.py over every payload
(system_fingerprint + request id; boot nonces KEPT — arm-identity receipt).

## Identity-bar correction (cell 1, receipted)

The cell-1 script treated composed ON-vs-OFF greedy byte identity as a STOP bar. That
bar was OVER-STRICT as written: door 3 (`MEMRA_KDA_FUSED_PROJ` bf16 arm) replaces
cuBLASLt with a deterministic warp tree on the three f32 rows per KDA layer — a
BAND-GATED numeric class (FLAGS.md: f32-row band worst 2.420e-7, bar 5e-5; ref-delta
ON==OFF at the bf16 operand floor), NOT a bit-identity claim. Doors 1, 2 and 4 do carry
whole-model byte-identity gates. So the composed ON-vs-OFF divergence (14/14 tapes, all
3 rounds, first-diff offsets 79-737) is the expected door-3 class unless cells 2-5 show
otherwise; the hard bit bars of this window are (a) each of doors 1/2/4 ALONE
byte-identical to OFF (cells 2-5) and (b) spec-vs-plain WITHIN the composed shape
(cell 6). The `C1_IDENTITY_DIVERGENCE ... STOP` console line is therefore reclassified
as a receipt, not a stop; the window continued by this note.

## Checkout-upgrade seam (coordinator order, mid-window)

Cells 1-7 ran on `28cbc1af6` (lane/glm5-decode-diet). Mid-cell-7 the coordinator
ordered the vrest head: `origin/lane/glm5-vrest @ a3fc59aaf` (ancestry VERIFIED:
28cbc1af6 is an ancestor; adds the MoE-batched-across-verify-rows port riding
`MEMRA_GLM5_VERIFY_BATCH`, no new flag). Per the order, cell 7 FINISHED on 28cbc1af6
(the seam is this note); the vrest phase (cells V1-V3 + the census) runs on a3fc59aaf:

- V1 identity-first (flip-battery c1 shape K1/K3 x 6 prompts, any divergence STOPS)
  + d34-composed K3 spot;
- V2 trace=2 receipt: `[glm5-phase-v] ... vrest=(vffn=)` banked (the 9.46 ->
  ~3.5-3.9 ms/row slope prediction's real-artifact receipt) + the one-flag
  `MEMRA_GLM5_VERIFY_BATCH=0` seam arm (must land near the 91.1 ms zctl K=3 wall);
- V3 re-price (TIMED x3 interleaved): plain vs the deployable config (DFlash2 +
  auto-K nopin + PMIN0.7; 45.65 baseline, predicted ~55-57) vs deployable + doors 3+4;
  K=4/5 upward ladder if the wall lands near the predicted band;
- census on the new winner. c7x (the old-head d34 deployable twin) is SUPERSEDED by
  V3's shipd34 arm and was not run.
- Door-gate extension on spec boots of this phase (`DIET_PHASE=vrest` pin): demand
  the BATCHED line ending `moe=pairs rows-call` AND `[glm5-vrows] verify MoE batched
  across rows`; both forbidden on the =0 seam arm.

## RESULTS (banked per cell; box timing = receipts)

### Old head 28cbc1af6 (cells 1-7)

| arm | decode tok/s | vs off 35.409 | ms/token delta |
|---|---|---|---|
| off (plain) | 35.409 | 1.0000x (== the flip-battery 35.408) | — |
| composed 4 doors (don) | 36.302 | 1.0252x | -0.69 |
| hcpre alone | 35.014 | 0.9888x | +0.318 (LOSS) |
| hcws alone | 35.375 | 0.9991x | +0.027 (nil) |
| kda6 alone | 36.218 | 1.0229x | -0.631 |
| mlasplit alone | 35.932 | 1.0148x | -0.411 |
| d34 (kda6+mlasplit) | 36.777 | 1.0386x | -1.050 (additive: singles sum -1.042) |

- THE DIET ARITHMETIC DID NOT TRANSFER: predicted 21.1-24.1 ms/token (41-47 tok/s),
  measured 27.55 ms (36.30). The launch-count levers (doors 1-2, predicted -2.8 to
  -4.3 ms together) bought ~zero or lost: the wall is sync/pipeline-bound, so removing
  launches and allocs BETWEEN the 42 router syncs does not shorten it. The real-GPU-work
  restructures (doors 3-4) landed roughly on their predicted bands.
- Identity attribution EXACT: doors 1/2/4 alone are 14/14 byte-identical to OFF on the
  real artifact (their bit gates hold); door 3 alone reproduces the composed divergence
  14/14 (its f32 trio is a band-gated numeric class per FLAGS.md, not a bit claim).
- hc-decode-ws is STRUCTURALLY ABSENT on spec boots (it owns only the t=1
  hyper_range_decode walk; spec decodes through the t=K+1 verify walk) — lever 2
  contributes nothing to the spec serving shape by construction.
- Composed spec re-price (cell 7, x3, spreads <=0.077%): don 36.309 | dfk1 40.313
  (1.1103x) | dfk2 42.292 (1.1648x peak) | dfk3 41.435. vs the flip-reprice no-doors
  arms (41.221/44.245/43.420): THE DOORS COST ~2 tok/s ON THE SPEC SHAPE (net-negative
  on the verify walk) while buying +0.9 on plain.

### vrest head a3fc59aaf (coordinator seam; cells V1-V3 + census)

- V1 identity-first: 6/6 (K1) + 6/6 (K3) spec-vs-plain byte-identical on the served
  path; d34-composed K3 spot 4/4. Loop-law 0/26.
- V2 trace=2 (K=3, t=4): vkda 17.59 / vmla 6.62 / vrest 24.41 (vffn 16.38) — vrest
  45.61 -> 24.41 ms/round (-21.2); slope ~4.15 ms/row vs predicted 3.5-3.9; verify
  ~48.6 from 69.72. The =0 seam arm logs the whole per-row walk under vrest=97.88
  (~= the banked 96.5 per-row verify), tapes 4/4 identical across the seam.
- V3 THE RE-PRICE (x3 interleaved, spreads <=0.076%, loop-law 0/126):
  plain 35.337 | SHIP (DFlash2 + auto-K nopin + PMIN0.7, no doors) **62.426 =
  1.7666x** (deep 53.06 = 1.770x, vendor-default sampled 49.60, TTFT 1.125s@0.4k /
  3.525s@3.7k) | shipd34 59.229 = 1.6761x (doors net-negative on spec, re-confirmed).
- K ladder upward (single boots, PMIN0.7): K4 62.78 / **K5 63.56 peak (1.799x)** /
  K6 62.58 / K7 60.95 — the ladder re-opened upward as the vrest lane predicted;
  curve is shallow (auto-K3 leaves ~1.8% on the table). K4 vendor row ct=80 and K7
  ct=127 excluded by the 128-token floor by name.
- THE 100-BAR NUMBER: best single-stream today = **62.4 tok/s ship config**
  (63.6 at a pinned K=5); the bar still needs **1.60x** (1.57x at K5).

### Census (cell 8, duration-bounded nsys 2026.1.3, this box's launch-econ constant
2.049 us/launch eager / 1.165 graph)

- SHIP shape (189p+192c in 3.86s, 59 rounds, acc-rate 0.888 on the census prompt):
  GPU total 2725.6 ms — moe 914.3 (33.5%) + bf16-mmv 866.6 (31.8%) + mla+indexer
  328.5 (12.1%) + cublas-f32 192.3 (7.1%) + mhc-sites 167.2 (6.1%). Host: 264,860
  cuMemAllocAsync + 263,022 frees (~1380/tok) and ~1293 launches/tok; 16.6 syncs/tok.
- PLAIN shape (same request in 5.33s): GPU total 4153.9 ms — bf16-mmv 1405.5 (33.8%)
  + moe 1056.0 (25.4%) + mla 706.8 (17.0%) + mhc 523.0 (12.6%); 2190 launches/tok
  (reproduces box A's ~2125 census).
- CENSUS VERDICT: on the winner the remaining ms sit in the MATVEC CLASSES — moe +
  bf16-mmv are 65.3% of GPU time (the matvec-efficiency pass the attribution named:
  moe epilogue at 57-64% and kda/tcols bf16 at ~70% vs q38's proven 87% on this card
  class), and in the spec loop's own host churn (~1380 allocs+frees/tok — hc-decode-ws
  does not reach the verify walk; a verify-walk workspace is a NEW named lever).
  Counts caveat: per-tok figures fold prefill + drafter shares (census's own note).

### Instrument traps caught (receipted)

- memra-server dies on TERM without CUPTI's atexit flush and ignores INT, so
  `nsys profile` + signal-stop yields a kernel-less trace (the c8-ship-noflush dir).
  Fix: `--duration=<n> --kill=none` (collection detaches and flushes while the server
  keeps running), then a scoped TERM. `--cuda-flush-interval` alone did NOT rescue it.
- nsys stats 2026.1.3 writes preamble lines INTO the csv before the header —
  c8_buckets.py skips to the `Time (%)` header (the stock census script's DictReader
  reads the preamble as the header and reports zero rows).

## Recommendation (door defaults + what the bar still needs)

1. All four diet doors STAY DEFAULT OFF. On the spec serving shape (the deployable
   config) the composed doors LOSE ~2-3 tok/s; hcpre loses even on plain; hcws is nil
   on plain and unreachable on spec. If a plain-only SKU ever matters, doors 3+4
   (d34, 1.0386x plain, additive) are the only pair worth a flip receipt.
2. The serving recommendation stands with the flip-reprice one, now repriced ON the
   vrest head: MEMRA_GLM5_SPEC=1 + MEMRA_GLM5_DFLASH (pinned b33c0347) +
   MEMRA_SPEC_PMIN=0.7, auto-K policy, VERIFY_BATCH default ON, NO diet doors —
   62.43 tok/s single-stream (1.767x plain), vendor-shape 49.6.
3. What the 100 bar still needs (1.60x): (a) the matvec-efficiency pass — moe +
   bf16-mmv = 65% of winner-shape GPU time, the largest sized lever (~87% vs 57-70%
   efficiency headroom); (b) a verify-walk allocation workspace (the ~1380
   allocs+frees/tok class — lever-2's pattern applied where the serving shape actually
   decodes); (c) K-policy re-pin toward K=5 (+1.8% free); (d) the named vrest
   follow-ups (B-session pairs port, dense L0-2, glue GEMV batch probe).

## Timeline

- 2026-08-30T22:35:13Z PENDING line on /root/BOX-QUEUE.md; prep done under the
  flip-reprice marker: clone+fetch (28cbc1af6 verified), drafter sha re-verified,
  scripts staged to /root/out-diet/. Build deferred to the done-line.
- 2026-08-31T01:29:39Z WINDOW START (box clock): build GREEN 308s sha16 c6588a952f,
  nsys 2026.1.3 installed. Cells 1-7 on 28cbc1af6; coordinator seam mid-cell-7;
  vrest build GREEN 295s sha16 9f11052616; V1-V3 + K ladder + censuses on a3fc59aaf.
- Boot count: 46 battery boots + the census boots, 0 boot failures; one door-GATE
  misfire (c6 spec arm demanded the hc-decode-ws announce that spec boots cannot
  print — gate semantics fixed, arm re-run green). Loop-law 0 flagged on every cell
  screen (c1 84, c7 168, V1 26, V2 8, V3 126, K-ladder 84, plus the c2-6 screens);
  every ON boot carried its engagement announces, every OFF boot zero.
