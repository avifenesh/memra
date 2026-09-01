# glm5 decode diet — levers 2-5 of the decode-gap attribution

Lane: `lane/glm5-decode-diet` (2026-08-31), base `origin/lane/glm53-flash-bringup` @
34e0c0bf2. Parent evidence: darklanes `research/glm5-decode-gap-20260830/ATTRIBUTION.md`
(the census table and each lever's arithmetic; census source =
`../launch-diet-20260830/WINDOW-20260830.md`, nsys 2026.1.3, 2-card arm C + fused epi,
2,125 launches/token, 25.1 ms GPU of ~29.5 ms wall, 2,358 `cuMemAllocAsync+Free`
calls/token, 43 `cuStreamSynchronize`/token). Target: plain decode 35.4 tok/s
(28.24 ms/token on the 3-card serving shape) toward the diet's ~50-55, compounding with
the verify-batch lane's spec win.

Every number measured in this lane is a RIG receipt (5090, exactness + counters only —
rig law). Every ms/token number below is PREDICTED arithmetic against the box's banked
census constants (2.216 us/launch eager, ~1.06 us/driver alloc call), labeled as such;
the box A/B window prices them.

## Per-lever state

| # | lever | state | flag (default) | gate | predicted Δms/token (box prices it) |
|---|---|---|---|---|---|
| 1 | mHC pre-chain fusion (rowsq+Sinkhorn+collapse, one launch/site; bit-preserving Sinkhorn stationarity exit) | **LANDED, flagged** | `MEMRA_HC_FUSED_PRE` (OFF) | `hc_fused_pre_gpu.rs` 4/4 | ~1.8-2.8: census mhc-sites 3.306 ms GPU + 362 launches/tok; 90 sites x (15.5+18.3+collapse) us serial -> one ~18-22 us kernel/site, + ~180 fewer launches x 2.216 us. The attribution's EV row said ~2.5-3; the fused kernel's floor is the rowsq reduction, so the honest band is 1.8-2.8 |
| 2 | Persistent decode workspace + host-sync audit | **LANDED, flagged** (workspace); sync diet = AUDIT BANKED, device-router consumer named, not built | `MEMRA_HC_DECODE_WS` (OFF) | `hc_decode_ws_gpu.rs` 2/2 | ~1.0-1.5: measured -13.8 allocs/layer/step (rig counter receipt) -> ~622 allocs + 622 frees/token on the 45-layer trunk = ~53% of the census's 2,358 calls (~1.3 ms of the 2.5 ms alloc host time) + an unpriced share of the 4.4 ms drain-gap the allocs feed |
| 3 | BF16 operand arm of the fused-6 KDA projection door | **LANDED, flagged** | `MEMRA_KDA_FUSED_PROJ` (OFF, same door as the q8 arm) | `kda_fused_proj_bf16_gpu.rs` 5/5 | ~0.5-1: the WINDOW §5 class — ~8 launches + ~6 allocs per KDA layer per token (x34) off the sync-serialized path |
| 4 | MLA decode craft — decode-split absorb/decompress (pure restructure, BIT-gated; the bf16-MMA rework would be band-gated and is deliberately NOT this arm) | **LANDED, flagged** | `MEMRA_MLA_DECODE_SPLIT` (OFF) | `mla_decode_split_gpu.rs` 3/3 | ~0.8-1.8: absorb+decompress = 211 us/layer x 11 = ~2.3 ms at 64 blocks (single-digit occupancy); split takes absorb to ~1024 blocks (S=16) / decompress to 256 (S=4). Latency-bound scaling predicts 3-8x on the pair; the gathered-attention kernel (195 us/layer, softmax-serialized) is NOT restructurable bit-safely and stays |
| 5 | cublas-f32-gemv launch fuel | **FOLDED via lever 3** (not chased standalone, per the brief) | rides `MEMRA_KDA_FUSED_PROJ` | covered by lever-3 gate | the 3 cuBLASLt GEMV pairs per KDA layer (f_a/g_a/b_proj, ~204 of the 516 f32-gemv launches/token) die inside the fused-6 launch when the door is on; the mHC mixes GEMM stays `Engine::linear` (pre_exact law, untouched); head glue not chased |

**Stacked prediction** (3-card shape, all four doors ON, greedy instrument): 28.24 ms −
(1.8-2.8 + 1.0-1.5 + 0.5-1.0 + 0.8-1.8) ≈ **21.1-24.1 ms/token ≈ 41-47 tok/s plain**.
The attribution's ~50-55 diet ceiling additionally requires the matvec-efficiency pass
(kda-proj at ~70% and moe at ~57-64% vs q38's proven 87% — ~2-2.5 ms, NOT in this lane's
scope) and the remaining drain-gap diet (device router). Compounding with the
verify-batch lane's spec flip these are the ~90-110 path's plain-step inputs.

## Gate table (rig 5090, TF32 off, flock, debug builds, 2026-08-31)

| gate | OFF arm | ON arm | all-4-doors arm |
|---|---|---|---|
| `hc_fused_pre_gpu` (bit identity per site + stationarity receipt + isolation reds + 24-step ON/OFF, counter 0->100) | n/a (is the A/B) | 4/4 | 4/4 |
| `hc_decode_ws_gpu` (24-step byte identity + alloc receipt 76.0->48.3/step (-36.4%) + compose arm) | n/a (is the A/B) | 2/2 | 2/2 |
| `kda_fused_proj_bf16_gpu` (bf16 rows bitwise t=1..15; f32 band worst 2.420e-7; mixer 2.755e-7; ref-delta ON==OFF at the 4.2e-3 bf16 operand floor; 12 reds; refusals) | n/a | 5/5 | — (own binary, bf16_mmv latch) |
| `mla_decode_split_gpu` (bit identity at splits {2,3,5,16,out} incl. non-dividing; reds; 24-step ON/OFF counter 0->100) | n/a | 3/3 | 3/3 |
| `kda_fused_proj_gpu` (q8 arm, re-bitten after the door restructure) | 5/5 | — | 5/5 |
| `kda_fixture_gpu` / `kda_quant_operand_gpu` | 3/3, 4/4 | — | — |
| `hyper_connections_gpu` | 6/6 | 6/6 | 6/6 |
| `glm5_tparallel_verify_gpu` | 7/7 | 7/7 | 7/7 |
| `glm5_spec_session_gpu` | 9/9 | 9/9 | 9/9 |
| `glm5_dflash_session_gpu` | 10/10 | 10/10 | 10/10 |
| `glm5_mtp_head_gpu` | 5/5 | 5/5 | 5/5 |
| `glm5_kpool_indexer_gpu` | 14/14 | — | 14/14 |
| `glm5-hyper-ppn-gate` n=2 and n=3 (stages 2+3; prime-twin, prefill-twin, overlay-serial, overlay-ppn, overlay-ppn-windowed — all BIT-IDENTICAL vs the unsplit walk) | 6/6 PASS each | 6/6 PASS each | 6/6 PASS each |
| `memra-server` suite | 481/481 | — | — |
| `tools/check-flags.sh` | green (718 reads covered, all four new/updated rows in `docs/FLAGS.md`) | | |

Notable receipts:
- **Sinkhorn stationarity** (lever 1): on the gate fixture the bitwise fixed point lands
  at min=6 / mean=10.09 / max=20 of the 20-iteration cap — the serial chain roughly
  halves ON TOP of the fusion, with zero bit movement by construction (exit only when a
  full (row,col) application leaves every comb bit unchanged).
- **Workspace alloc receipt** (lever 2): `SCRATCH_ALLOC_CALLS` (new instrument counting
  every `alloc_uninit`/`zeros` call — the census's host-call axis) reads 76.0 -> 48.3
  allocs/step on the 2-layer mini fixture = -13.8 allocs/layer/step, byte-identical
  logits over 24 steps.
- **bf16 residency binding** (lever 3): the gate fixture is the REAL glm5 KDA geometry
  (64 heads x 128; 256x8192 = 2,097,152 elements/projection over the loader's 2M
  threshold) and every test carries the `[bf16-mmv] RESIDENT ... admit=bf16_mmv` loader
  receipt, so the arm is gated on the residency class the serving recipe actually runs.

## The host-sync audit (lever 2's second half, banked)

The census's 43 `cuStreamSynchronize`/token decompose from source as: 42 = one per MoE
layer, 1 = the final logits dtoh. The per-layer sync is ALREADY the thin form: with
`MEMRA_SIG_ROUTER` (default ON) the sigmoid `noaux_tc` router runs its top-k ON DEVICE
(`moe_router_sigmoid_topk`) and reads back only the `t*n_used` sel/w pair through the
persistent pinned `router_stage` — one async DtoH pair + ONE sync per layer, not a
logits-plane dtoh. The sync survives because the HOST consumes sel: it computes the
expert pointers (slab base + `ex*stride` on the resident arm) and drives admission on
the SLRU arm. At full residency admission is static, so the named follow-up is the
DEVICE-ROUTER CONSUMER: pass `sel_d` straight into pointer-computing expert kernels
(the `MEMRA_STEP_TP_DEV_ROUTER` pattern) — that kills the remaining 42 syncs AND is the
prerequisite for whole-token CUDA-graph capture, but it is a numeric-class door
(tie-break provenance) needing the run-gen argmax gate + boot battery, a lane of its
own. None of the 43 is alloc-driven per se; what the workspace kills is the alloc CALLS
that pad the drain-refill cycles between syncs.

## Coordination note for the lane/glm5-verify-batch merge

- `glm_spec.rs`: NOT touched.
- `hyper.rs`: `pre` / `pre_exact` / `post` / `contract_mean` / `collapse` signatures
  UNCHANGED. `pre_finish` was split into `pre_finish` (allocating, same behavior) +
  `pre_finish_into` (caller-owned outputs) — internal, private. Additive publics:
  `HyperDecodeWs`, `pre_t1_ws`, `post_t1_ws`, `HC_FUSED_PRE_DISPATCHES`.
- `kda.rs`: `kda_proj_fused6` signature unchanged; operand classification reordered
  (f32 trio first, then bf16 arm, then the q8 arm behind its unchanged env checks —
  q8-operand behavior identical). Additive: `kda_proj_fused6_bf16_raw`,
  `KDA_FUSED6_BF16_DISPATCHES`.
- `mla_ffi.rs`: `mla_absorb_q` / `mla_decompress_v` signatures unchanged; the split
  door lives inside them, so the verify walk is covered without touching it.
- `lib.rs`: Engine gains the private `hyper_decode_ws` pool slot + `SCRATCH_ALLOC_CALLS`.
- `hybrid_forward.rs`: `hyper_range_decode` signature unchanged (flag door at its top);
  new private `hyper_range_decode_ws{,_body}`; `HC_DECODE_WS_DISPATCHES` public.
- New kernels only in `dsv4_gpu.cu` / `qmatvec.cu` / `mla_attn.cu`; no existing kernel
  body was edited (the OFF arms are the shipped bytes).

## Box A/B window (separate window — NOT run by this lane)

Per flag, then composed (4 doors): interleaved x3 fresh boots per arm (x5 on anomaly,
per the amended law), serving card class, real artifact, 3-card serving shape
(`MEMRA_PP_STAGES=3` recipe pins), greedy (instrument) + the vendor-default sampled twin
(never-serve-greedy law), engagement announces demanded in every ON boot
(`[hc-fused-pre]`, `[hc-decode-ws]`, `[kda-fused6] engaged arm=bf16`,
`[mla-decode-split]`) and counter deltas in the receipts. Re-run the launch-econ census
(`../launch-diet-20260830/census-decode-phases.sh`) on the winning arm to bank the new
launches/allocs/syncs per token; the 3-card decode census the attribution named (its §9
cell 1) can ride the same boots. If the diet lands its band, re-run flip-battery cells
3+4 per the pre-staged form once the verify-batch lane's walk is in.

## Post-merge re-proof (verify-batch composition, 2026-08-31)

`origin/lane/glm53-flash-bringup` took the verify-batch lane mid-window (3f4accf13,
`MEMRA_GLM5_VERIFY_BATCH` default ON; glm_spec.rs restructure + additive
`mla_attn_cached_rows_exact` / `kda_verify_rows_cached`). Merged here clean (no
conflicts — the two lanes touched disjoint code, per the coordination contract) and the
WHOLE battery re-ran on the merged tree, both arms:

- default arm: tparallel 9/9 (their two new tests included), spec_session 9/9,
  dflash_session 10/10, mtp_head 5/5, hyper 6/6, all four diet gates green, kda gates
  re-bitten, ppn n2/n3 6/6 PASS each, server 481/481.
- all-4-doors-ON arm (composed WITH their default-ON verify walk): same set, all green
  — the batched verify rows funnel through the same `mla_absorb_q`/`mla_decompress_v`
  wrappers and `hyper::pre_exact`, so the doors cover the new walk by construction.

## Follow-ups named (not built here)

1. Device-router consumer at full residency (kills the 42 syncs; graph prerequisite;
   numeric-class door — argmax gate + boot battery).
2. Matvec-efficiency pass (kda-proj-bf16 at ~70% of peak, moe epilogue at ~57-64% vs the
   87% q38 proved on this card class) — the last ~2-2.5 ms of the diet's ~50-55 ceiling.
3. `memra_mla_attn_gathered` decode craft (195 us/layer): softmax-serialized, not
   bit-safely splittable; needs its own banded design.
4. `MEMRA_RMS_BLOCK` probe (attribution lever 9) — one cheap cell, ride any window.
