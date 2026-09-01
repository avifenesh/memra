# glm5 MATVEC-EFFICIENCY pass — the 65%-of-decode-GPU lever (lane/glm5-matvec, 2026-08-31)

Base: `origin/lane/glm53-flash-bringup` @ 32dc957b8 (consol-db: vrest + diet-battery
merged; `MEMRA_GLM5_VERIFY_BATCH` + `MEMRA_HYPER_BATCH` default ON). Worktree
`~/projects/wt-glm5-matvec`, branch `lane/glm5-matvec`.

Charter: the diet-battery census verdict (`../diet-battery-20260831/WINDOW.md` cell 8,
nsys 2026.1.3, box B 4x RTX PRO 6000 Blackwell WS, launch constant 2.049 us) — on the
SHIP winner (DFlash2 + auto-K nopin + PMIN0.7, 62.43 tok/s) "moe + bf16-mmv are 65.3% of
GPU time ... at 57-70% of the weight-read bound vs q38's proven 87%", plus the spec
loop's ~1380 cuMemAllocAsync+Free pairs/token (the verify walk has no workspace door —
`MEMRA_HC_DECODE_WS` owns only the t=1 walk, structurally absent on spec boots).

Every number measured in this lane is a RIG receipt (5090, exactness + counters only —
rig law). Every ms/GB/s number below is receipt-derived or PREDICTED arithmetic against
banked box constants, labeled as such; a separate box window prices the doors.

## 1. Per-kernel attribution (written BEFORE any code)

Source: `../diet-battery-20260831/receipts/c8-ship/diet-census2-kernsum.csv` (the SHIP
shape: 189p+192c, 59 spec rounds, acc-rate 0.888, drafted/round ≈ 2.5 ⇒ verify rows
t = K+1 ≈ 3.5 avg; GPU total 2725.6 ms). Bound constant: 1.79 TB/s GDDR7 (every banked
roofline row); "efficiency bound" = the 87% of peak q38 proved reachable on this card
class (1.56 TB/s effective). Model geometry (glm-config.json text_config): hidden 4096,
45 layers = 34 KDA + 11 MLA, first 3 dense, 42 MoE layers x 288 experts top-8 + 1
shared, moe_ff 2048, vocab 154880 (lm head bf16 = 1.269 GB), kda q/k/v/out 33.5M elem
bf16 = 67.1 MB each, indexer attn_q_b 6.3M = 12.6 MB.

### Prefill-vs-decode split first (the census's own caveat, now quantified)

Prefill-only kernels in the capture (instance counts NOT multiples of 59; the 189-token
prime): `moe_kq_sktail<7>` 126 inst = 42 layers x 3 projections, 251.2 ms;
`mul_mat_q_nvfp4_w4a8` 179 inst 15.8 ms; cutlass bf16 GEMMs 102 inst 14.3 ms;
`moe_kq_sk128v` 66 inst 5.6 ms; `kpool_score_tiled` 11 inst 6.5 ms; + small sgemm/rope
shares ≈ **~295 ms ≈ 10.8% of the capture is prefill**. The WINDOW's "moe 33.5%"
headline includes sktail: **decode-only moe is ~24%, bf16-mmv ~32% of a ~2430 ms decode
share (41.2 ms GPU / round; wall 65.4 ms/round)**. The lane's target set below is the
decode rounds only.

### The decode-round table (per round = one draft block + one t-row verify + accept)

| kernel | inst/rnd | ms/rnd | % dec GPU | bytes/call (model) | achieved | bound | gap class |
|---|---|---|---|---|---|---|---|
| `matvec_bf16_f32acc_x4_tcols` (trunk kda q/k/v/out x136 + indexer q_b x11) | 147 | ~8.7 | 21% | 67.1 MB (kda), 12.6 MB (idx), weight read ONCE for t rows | med 63.8 us on 67.1 MB = **1.05 TB/s (59% peak)** | 1.56 TB/s | launch shape: out_f/4 = 512..2048 blocks x 128 thr is ~ONE resident wave on 188 SMs — load phases and the t bit-pinned tree reduces phase-lock across the whole grid, DRAM idles in the reduce phases. Proof on the SAME kernel: the lm-head call (38720 blocks) runs 886 us / 1.27 GB = **1.43 TB/s (80%)** |
| `matvec_bf16_f32acc_x4_tcols` (lm head, 1/rnd via `matmul_rows_exact`) | 1 | ~0.9 | 2% | 1.269 GB once/round | **1.43 TB/s (80%)** | 1.56 | near-bound already; NOT a target |
| `matvec_bf16_f32acc_x4_rows` (DFlash2 drafter head) | 1 | **5.31** | 13% | the drafter block forward (block_size 16) reuses the TARGET's 1.27 GB lm head over nd = 15 mask-fill rows via `eh.matmul` → `matmul_decode_exact` → `matvec_bf16_rows_into` grid.y=15: **15 x 1.269 GB logical re-read** | 19 GB / 5.31 ms = 3.6 TB/s apparent (L2-served rereads); vs read-once-at-1.43 TB/s = 0.89 ms | 0.89 ms | **weight re-read x15**: t=15 > `MEMRA_BF16_TCOLS_MAX`=8 so the tcols weight-once twin refuses. The vrest lane's head check ("t=9..15 ... outside every measured serving shape") missed THIS site — the drafter head is t=15 EVERY round of the ship shape. Largest single kernel lever in the census |
| `moe_gate_up_preclamp8_q8_rows` | 42 | 7.05 | 17% | t*8 pairs x 9.44 MB NVFP4 gate+up per routed expert (near-disjoint across rows; repeated experts L2-served) | plain-shape receipt: **57-64% of the expert-read bound** (decode-gap ATTRIBUTION §3a: 4.70 ms vs 3.0 roofline) | 87% | occupancy/launch shape: `block=(32,1,1)` — ONE warp per block. Blocks/SM cap ⇒ ≤32 of 48 warp slots resident (≤67% occ) + 65k tiny-block scheduling per launch. The q38 mmvq family packs 4 warps/block (`MEMRA_MMVQ_ROWS`); this pair never got the packing. Bonus: 4 packed warps share one pair's activation row + read 4 ADJACENT expert rows (contiguous bytes) |
| `moe_down8_fma_q8_rows` | 42 | 3.54 | 9% | t tokens x 8 experts x 4.72 MB down | same class | 87% | same 1-warp block; plus the slot-ordered 8-expert FMA chain serializes per warp (bit bar — chain order stays) |
| cuBLASLt f32 m=1 pairs (`dot_kernel`+`reduce_1Block`+`gemv2T`) | ~1852 | 2.9 | 7% | tiny f32 rows: mixes GEMMs (90 sites x t rows) + KDA f32 trio x34 x t + head glue | — | — | launch fuel; per-row m=1 is the lt_ndep law's exact form. vrest follow-up #2 (cuBLASLt batch probe) — NOT this lane |
| `qmatvec_nvfp4_mmvq_b4_rp` + kin (MLA projections + shexp) | ~200 | 2.1 | 5% | ~10-50 MB shapes, already the q38-pattern batched-MMVQ kernels | ~10.4 us avg | — | already the proven family; not first-order |
| mhc `dsv4_*` | ~92 sites | 2.83 | 7% | GPU-heavy sinkhorn/rowsq at t rows | — | — | `MEMRA_HC_FUSED_PRE`'s territory (measured net-negative composed on spec — diet WINDOW) — not mine |
| MLA decompress/absorb/gathered | 11 | 2.78 | 7% | latent decompress/absorb ~4 ms class | — | — | `MEMRA_MLA_DECODE_SPLIT` (landed, OFF; d34 composed loses on spec) — not mine |
| `topk_rows_f32` (DFlash2 candidate selector, k=16 over vocab) | 1 | **1.31** | 3% | 15 x 154880 x 4 B = 9.3 MB read | **7 GB/s** | any | grid = n_rows = **15 blocks on 188 SMs** (8% of the machine) + a SINGLE-THREAD 256-list k-way merge tail per row. Top-k is exact discrete selection (total order: value desc, index asc) ⇒ restructurable BIT-identically |
| kda scan+conv+norms, quantize, rms, drafter q4_0/sdpa, argmax, glue | rest | ~5.0 | 12% | tiny-kernel launch fuel | — | — | diet doors / launch-econ territory |

Host side, ship shape (apisum): **264,860 cuMemAllocAsync + 263,022 cuMemFreeAsync
(~1380+1370 per token, ~4,490 alloc calls per round at 4.0 us avg = ~18-20 ms/round of
host alloc time, partially hidden under GPU)**; 16.6 `cuStreamSynchronize`/tok; 1293
launches/tok. The t=1 workspace door (`MEMRA_HC_DECODE_WS`) never reaches these — spec
decodes through the verify walk (WINDOW.md, structural). Per-site decomposition of the
round's churn (source census): KDA rows arm ~20 allocs/layer x 34 (6 projection y's,
ring snapshot, 3 conv, 2 l2, gate/beta/core/gated + 9-buffer stash replaced each
round); hyper glue pre/post at m=t x ~90 sites; `moe_vrows_pairs_q8` 7 device allocs +
2 htod x 42; MLA rows x11; drafter block forward + selector; accept/rollback replays.

## 2. The doors (all new flags DEFAULT OFF by design — no box timing receipt exists
yet; the rig is exactness-only. The box window flips them; FLAGS.md rows in this PR)

| # | door | flag (default OFF) | mechanism | bit bar | predicted Δms/round (box prices) |
|---|---|---|---|---|---|
| T | drafter-head weight-once (tcols t≤16) | `MEMRA_BF16_TCOLS_WIDE` | `matvec_bf16_rows_into` routes t=2..8 → the existing tcols kernel and t=9..16 → NEW `matvec_bf16_f32acc_x4_tcols16` (acc[16] twin — separate kernel so the priced t≤8 class keeps its register footprint and SASS untouched, the `_tw32` acc-sizing lesson); after the W8-mirror intercepts, before the x4_rows launch | tcols class is per-(row,token) BIT-identical to the t=1 program by construction (order-pinned chains + identical red[256] tree — the standing tcols gate's claim, extended to 16) | **-4.2 to -4.4** (5.31 → 0.9-1.1 read-once at the head call's own demonstrated 1.43 TB/s) |
| M | moe verify-rows warp packing | `MEMRA_MOE_VROWS_PACK` | `_w4` twins of the pair: `o = blockIdx.x*4 + threadIdx.y`, block (32,4), grid ((rows+3)/4, pairs/t) — per-warp body VERBATIM (no syncthreads in either kernel, warp-only reduces) | per (o,pair) arithmetic unchanged ⇒ bit-identical; gate twins vs the unpacked pair + swapped-pair/dropped-macro reds re-bitten | **-1.5 to -3.1** (10.59 ms at 57-64% → 80-85% eff class; occupancy 67% → 100% warp slots + 4x fewer blocks) |
| X | tcols x1-row grid (trunk wave fix) | `MEMRA_BF16_TCOLS_X1` | NEW `matvec_bf16_f32acc_x1_tcols`: one row per block (grid = out_f), p-loop dropped, per-row body/tree verbatim; taken in the tcols route when armed | per-row program unchanged ⇒ bit-identical | **-1.3 to -2.2** (trunk 8.7 ms at 1.05 → 1.3-1.4 TB/s: 4x wave count on 512-2048-block grids; the head call's 80% is the same-kernel existence proof) |
| K | topk shard split | `MEMRA_TOPK_SHARDS` | two-launch exact top-k: per-(row,shard) partial top-k (same insertion/tie rules) + per-row shard merge (same k-way merge rules); ties break to lower column index in both stages ⇒ the SAME selected (value, index) list | discrete selection with a total order ⇒ output-identical; gate vs the standing kernel incl. planted-tie fixtures | **-1.1 to -1.2** (1.31 → ~0.1-0.2 ms; 15 → 240+ blocks) |
| W | verify-walk workspace | `MEMRA_GLM5_VERIFY_WS` | `Glm5VerifyWs` pooled on the Engine (the `MEMRA_HC_DECODE_WS` pattern extended to the walk): size-keyed free-lists (f32/i8/u64) + stash recycling — the round's dead KDA stash buffers (9/layer) restock the pool; `moe_vrows_pairs_q8` staging (7 allocs + 2 tables/layer-call) and the KDA rows-arm scratch draw from it | pooled buffers carry the SAME uninit contract (every consumer fully writes before read — unchanged kernels); gate = multi-round byte identity ON vs OFF + `SCRATCH_ALLOC_CALLS` delta receipt + reds | host-side: **~-800 to -1700 driver calls/round** of ~4,490 (alloc+free pairs at ~4.5 us, sync-exposed share unknown → wall -0.5 to -2, labeled UNPRICED) |

Composed (T+M+X+K GPU arithmetic): decode GPU 41.2 → ~31.6-33.9 ms/round; round wall
65.4 → ~55.8-58.1 ms (assuming GPU-critical-path transfer of the saved ms — the diet
window's caution: sync/pipeline-bound walls do NOT always follow kernel savings, stated
here before the box prices it). **Ship-config prediction: 62.43 → ~70-75 tok/s
single-stream** (x1.13-1.20); with the K5 re-pin (+1.8% banked) ~72-77. The 100-bar
still needs TP-2 on top (this lane + TP-2 is the sized composition; 100/62.4 = 1.60x).

NOT in scope, named: mhc/MLA doors (owned, measured net-negative on spec), cuBLASLt
glue batch probe (vrest follow-up #2), dense L0-2 re-plumb (vrest #3), device-router
consumer (decode-diet follow-up #1), NVFP4/q8 dequant arithmetic (bit bar — never
changes), bf16 f32-acc class (stays its class), box re-price window (separate).

## 3. Gate table (rig 5090, flock held, NVIDIA_TF32_OVERRIDE=0, exactness only, 2026-08-31;
logs in `receipts/`)

| gate | result |
|---|---|
| `glm5_matvec_doors_gpu` door T: tcols16 vs the per-row t=1 program, t=9..=16 ALL bitwise (1197..2128 outputs/t); shifted-activation-row red bites (1596 diffs); route A/B — `matvec_bf16_rows_into` t=15 ON == OFF bytes, `BF16_TCOLS_WIDE_DISPATCHES` >0 ON / flat OFF | PASS |
| `glm5_matvec_doors_gpu` door X: x1-grid twin vs x4, t=2..=8 ALL bitwise; swapped-weight-row red bites (8 diffs = the 2 swapped rows x 4 tokens); `BF16_TCOLS_X1_DISPATCHES` anchored | PASS |
| `glm5_matvec_doors_gpu` door M: `_w4` packed pair vs unpacked, t=2..=8 ALL bitwise on minted NVFP4 banks + LIVE macro plane; the vrest gate-4 reds re-bitten THROUGH the packed door (swapped-pair 256 diffs, dropped-macro 256 diffs); `MOE_VROWS_PACK_DISPATCHES` anchored | PASS |
| `glm5_matvec_doors_gpu` door K: sharded top-k vs standing kernel — values bitwise + indices equal on planted-tie fixtures (cross-shard duplicated max resolves to idx 900; k+4 scattered ties select ascending-lowest; all-equal row selects 0..k-1; -inf-heavy row fills slot 16 with the col-0 -inf tie); below-threshold fall-through counter-flat; rotated-row red bites | PASS |
| door W (`glm5_spec_session_gpu::gpu_verify_ws_bursts_byte_identical_with_alloc_receipt`): 24 served tokens byte-identical ON vs OFF, **408 pool hits**, `SCRATCH_ALLOC_CALLS` 22431 -> 22023 on the mini fixture (-1.8% — the fixture is f32-class dominated: its trunk is under the 2M bf16-residency threshold so NO rows-exact y pools there, and prime + drafter allocs dominate the count; the real-shape share is §4 arithmetic, the box census prices it) | PASS |
| standing batteries, DEFAULT arm (all doors OFF — shipped program untouched): verify_batch 4/4 (incl. vrest gate 4), tparallel 9/9, spec_session 10/10 (the new W gate included), dflash_session 10/10, moe_epilogue 9/9, mtp_head 5/5, kpool_indexer 14/14, hyper 6/6, hc_fused_pre 4/4, hc_decode_ws 2/2, mla_decode_split 3/3, kda_fixture 3/3, kda_fused_proj 5/5, kda_fused_proj_bf16 5/5, kda_quant_operand 4/4, mla_gpu_forward 5/5 | ALL PASS |
| standing batteries, COMPOSE arm (ALL FIVE doors ON): verify_batch 4/4, tparallel 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 9/9. Engagement scope stated: doors W+M engage on these fixtures (pool hits + NVFP4 vrows pair); doors T/X need >=2M-element bf16 tensors and door K needs n_cols>=16384 — their exactness is carried by the doors gate's own fixtures | ALL PASS |
| `glm5-spec-ppn-gate` matrix (8 arms, stages 2+3) + compose arm (n3-even, all five doors ON) · `glm5-hyper-ppn-gate` matrix · `glm5-hyper-batch-gate` matrix | see receipts/ (run log below) |
| `memra-server` suite · `tools/check-flags.sh` (728 reads covered) · clippy zero · fmt | see status log |

## 4. Door W real-shape coverage (arithmetic, box census prices it)

Per ship round (t~3.5): MoE staging 7 x 42 = 294 pooled draws + 294 recycles; KDA
rows-arm scratch 9 x 34 = 306 draws + 306 recycles at last-consumer; the previous
round's stash 9 x 34 = 306 recycles (restock); rows-exact y ~165 draws (wq/wk/wv + wo
x34, MLA wrapper calls x11, verify head). ≈ **1,500 avoided driver calls/round of the
~8,940 measured (alloc 4,490 + free 4,450) ≈ 17%** — i.e. ~235 of the ~1,380
allocs/token plus their frees. The remaining churn is named, not owned here: hyper glue
at m=t (~90 sites), MLA rows internals, the f32 per-column linear class, the drafter
forward, accept/rollback replays — each a mechanical extension of the same pool.

## 5. Predicted ship numbers (arithmetic against the c8-ship census; the box window
prices every door — nothing here is a claim)

Ship today: 62.43 tok/s (V3), 65.4 ms wall/round, decode GPU 41.2 ms/round.

| door | ms/round Δ (predicted) | basis |
|---|---|---|
| T (drafter head weight-once) | **-3.8 to -4.4** | 5.31 -> 0.9-1.5 ms: 1.269 GB read once at the head-tcols call's own 1.43 TB/s, + the t=15 reduce-tree overhead band; the drafter phase is serial in the round so GPU ≈ wall here |
| M (moe warp pack) | -1.5 to -3.1 | 10.59 ms at the 57-64% receipt -> the 80-87% occupancy class (q38 precedent) |
| X (tcols x1 grid) | -1.3 to -2.3 | trunk 8.7 ms at 1.05 TB/s -> 1.3-1.43 (the same kernel's demonstrated 80% at wave depth) |
| K (topk shards) | -1.1 to -1.2 | 1.31 -> ~0.1-0.2 ms (15 -> 240 blocks + parallel merge) |
| W (verify ws) | -0.3 to -1.5 (host, UNPRICED) | ~1,500 driver calls/round at the box's 1.06-4.0 us/call, sync-exposed share unknown |

Composed GPU: -7.7 to -11.0 ms/round ⇒ wall 65.4 -> ~54.4-57.7 (full-transfer
assumption, stated: these are serial-phase GPU savings, the class that DID transfer for
the diet's doors 3/4, unlike its launch-count levers) ⇒ **ship-config predicted ~69-75
tok/s single-stream (x1.10-1.20)**; the banked K=5 re-pin (+1.8%) stacks to ~70-76.
Against the 100-bar: this lane alone is NOT the 1.60x — it is the matvec leg of the
lane+TP-2 composition the WINDOW named; at ~72 tok/s the bar needs ~1.39x from TP-2 +
the remaining vrest follow-ups.

## 6. Named follow-ups (not built here)

1. Door-W extensions, mechanical on the same pool: hyper glue at m=t (~90 sites/round),
   MLA rows internals, the f32 per-column linear class, the drafter forward + selector,
   accept/rollback replay buffers — together the remaining ~83% of the alloc churn.
2. The t=1 fused-epilogue MoE pair (`moe_gate_up_preclamp8_q8` / `moe_down8_fma_q8`,
   `MEMRA_MOE_FUSED_EPI`) has the same 1-warp blocks — a `_w4` twin is the same edit if a
   plain-only SKU ever matters (the ship shape never runs them).
3. tcols t=17..=32 (no serving shape needs it; the drafter block is 16).
4. The trunk tcols reduce-phase overlap beyond x1 (double-buffered loads) — only if the
   box prices door X below its band.
5. `memra_mla_attn_gathered` / mhc / cublas-glue: owned elsewhere (decode-diet doors,
   vrest follow-up #2).

## 7. What the box window carries (separate window, NOT this lane's)

- V3-shape re-price on this head: ship config (DFlash2 + auto-K nopin + PMIN0.7,
  VERIFY_BATCH on) plain vs +each door alone (T, M, X, K, W singles — one boot each,
  attribution evidence) vs ALL FIVE composed (the decision number), interleaved x3,
  greedy + vendor-default sampled twin, loop-law screen.
- Engagement receipts demanded per ON boot: `[bf16-tcols-wide] engaged` (t=15 drafter
  head), `[bf16-tcols-x1] engaged`, `[moe-vrows-pack] engaged`, `[topk-shards] engaged`,
  `[glm5-verify-ws] engaged` — each with zero lines on its OFF arm.
- Byte-identity spot first (the c6 shape): composed-doors greedy tapes vs no-doors,
  4 prompts — ANY divergence STOPS the window (all five doors carry bit gates, so a
  divergence is a defect, not a numeric class).
- Census re-run on the winner (the c8 duration-bounded nsys form): per-kernel table vs
  §1 — tcols GB/s, x4_rows gone from the round, moe pair us/call, topk us, alloc calls
  /token. The census IS the door-X/M efficiency receipt.
- K-ladder re-pin (K=4/5) on the composed winner — the banked +1.8% at K5 moves with a
  faster verify row.
- If composed lands in the §5 band: flip decision per door (FLAGS defaults are OFF
  awaiting exactly this), then the TP-2 composition window.

## 8. Status log

- Lane open 2026-08-31, worktree @ 32dc957b8. §1 attribution written from the banked
  c8-ship receipts BEFORE any code (commit 4000b864f).
- Doors T/X/M/K/W landed (b556fcf78): kernels (qmatvec.cu +4 symbols, kernels.cu +2),
  launchers + doors + counters (lib.rs), MoE staging ws (hybrid_forward.rs), KDA rows-arm
  ws + stash recycle (kda.rs), FLAGS.md 5 rows + KERNELS.md rows/count corrections in the
  same commit. `tools/check-flags.sh` green (728 reads).
- Gate battery same day: doors gate 4/4 with reds biting; default + compose batteries all
  green (receipts/run-battery.sh, per-suite logs).
- Split matrices: `glm5-spec-ppn-gate` 8/8 arms + the all-five-doors-ON n3-even compose
  arm PASS (engagement announces `[glm5-verify-ws]` + `[moe-vrows-pack]` in the compose
  log; T/X/K structurally silent on the mini fixture, stated); `glm5-hyper-ppn-gate`
  and `glm5-hyper-batch-gate` matrices ALL ARMS PASS (receipts/{ppn,hppn,hbatch}-gate/).
- `memra-server` suite 481/481. `tools/local-ci.sh --perf` exit 0 TWICE (correctness
  GREEN, perf 0 fail 0 warn, qwen9b cell 138.85 / 138.79 tok/s [OK] vs rolling median;
  receipts/local-ci-perf-run{1,2}.log). clippy all-targets zero lints; fmt clean;
  check-flags 728 reads covered.
- PUSHED to `origin/lane/glm5-matvec`; no self-merge. The box window (§7) prices the
  doors and owns every flip decision.
