# qwen4_exp decode PROFILE-4 — round 4: the 90 tok/s crossing attempt (2026-08-30)

Same box (2× RTX PRO 6000 Blackwell 96 GB, sm_120a, PCIe — no NVLink), same artifact
(`~/data/q48fn-nvfp4`, the per-expert modelopt NVFP4 mint), same goldens prompts as
PROFILE-0..3 (read those first — especially PROFILE-3 §Residual, whose two named
crossing levers this round builds: W4A4 expert kernels and a hyper-gate diet). Greedy is
the instrument. Every change: interleaved ×5 same-run A/B + tiny gate (now TEN permanent
arms) + real-checkpoint gate re-runs, receipts banked before the next item.

## Round-4 baseline re-profile (perf18 — the round's opening measurement)

Round 3 reproduced exactly: TP2 14.2 ms/token (70.53 tok/s), single card 15.6 (63.99).

Single-card section profile (`profile-nvfp4-single-base.tsv`, sync-bounded shares):

| section | ms/token | % of attributed |
|---|---|---|
| hyper.read | 3.69 | 18.9 |
| moe.sel_grouped | 3.12 | 15.9 |
| gdn.proj | 2.84 | 14.5 |
| gdn.norm_gate_out | 1.40 | 7.2 |
| moe.router | 1.37 | 7.0 |
| moe.shared | 1.34 | 6.8 |
| hyper.write | 0.99 | 5.1 |
| qsa.* | 2.56 | 13.1 |
| lm_head | 0.85 | 4.3 |

The bytes-vs-latency question (work item 1's required pre-measurement): the sel slice
moves 48 layers × 10 experts × ~2.76 MB ≈ 1.32 GB/token in 3.12 ms ≈ **~440 GB/s on a
~1.6 TB/s card (27%)** — the slice is LATENCY/ISSUE-bound (not bytes-bound): grid
(out_f/4, n_sel) puts only ~800-3,200 one-warp blocks in flight with ~2.5 iterations per
thread, nowhere near enough outstanding loads to cover VRAM latency, and the v3 chain
spends ~80 instructions per 32 codes per row on LUT extract + f32 FMA. The honest W4A4
lever is therefore COMPUTE+chain-depth (integer dp4a via a byte-perm codebook + fusing
gate+up+silu into one launch), not bytes — exactly the PROFILE-3 projection.

TP2 nsys (graphs live, kernels inside replays not attributed): 360 graph replays +
~650 eager kernel launches + ~160 eager memcpys per token; cuGraphLaunch alone = 2.4
ms/token of host issue; `cuMemcpyDtoHAsync` (router/indexer/logits boundaries) absorbs
the serial waits. The eager remainder is mostly the 12 QSA layers + PLE — work item 3's
target (QSA phase-1 graphs).

## Per-change interleaved A/B (each row = its own box battery, both arms in ONE run)

| item | seam | change | A/B (mean of 5 means, single card) | win | rep-0 chains | receipt |
|---|---|---|---|---|---|---|
| 1 | `set_proj_stack` | GDN qkv/z/beta/alpha (4→1), QSA wq/wk/wv (3→1), shared gate/up (2→1) trunk launches collapse into one `qmatvec_bf16w_multi4_f32` over load-time row-stacked bf16 twins, outputs routed to the original slot buffers by row range. Per-row math bf16w-VERBATIM ⇒ BIT-IDENTICAL. Residency is DEDUPED: the stack REPLACES the per-mat twins (the first cut duplicated them and OOM'd the TP2 load at card-0's 92.5 GiB baseline); the per-mat OFF arm reads row-offset views of the same stack | 15.72 → 15.25 (perf20; perf19 pre-dedup: 15.65 → 15.14) | 1.031× | identical (−1) | `perf20/ab-projstack-nvfp4.tsv` |
| 2 | `set_hc_diet` | the read gate's 7-launch serial chain re-fuses into THREE (stage 1: per-stream RMS recompute + smem normed row + down/inject rows; stage 2: silu mean + inject sigmoid; stage 3: up dots + mix epilogue off the stage-1 inv scalars). Accumulation class; real-geometry oracle arm 0e worst rel 2.369e-6 vs the classic fused chain | 15.69 → 15.32 | 1.024× | identical (−1) | `perf20/ab-hcdiet-nvfp4.tsv` |
| 3 | `set_sel_w4a4` | W4A4 expert path: the mint's `input_scale` CONSUMED (per-projection static activation scale = max over experts, the TRT-LLM fused-MoE precedent), activations quantized once per projection to the calibrated W4A4 operand (dynamic per-16 UE4M3 block scales via a u32-integer RNE cast BIT-EXACT vs the host twin + e2m1 RNE codes, int8 lanes in table16 order), integer dp4a dots, gate+up+silu fused into ONE launch. REAL-ERROR seam — see the envelope battery below | 15.65 → 15.05 | 1.040× | fork at step 13 (expected: real-error seam under greedy) | `perf21/ab-w4a4-nvfp4.tsv` |
| 4 | `set_qsa_graphs` | QSA phase-1 segments join the capture set on both drivers (attn gate_read + proj/split/norms/rope/KV-append); position device-driven (per-step `qsa.posd` slot + `q4e_append_row_pos_f32` verbatim append twin); indexer/mask/SDPA stay eager | 15.64 → 15.71 — **single-card LOSS** (the host issue thread is NOT the single-card wall; graph-replay overhead ≈ the ~10 launches it removes); TP2-route A/B = perf23 | 0.996× (single) | identical (−1) | `perf22/ab-qsagraph-nvfp4.tsv` |

Correctness per change: tiny gate (10 arms — round 4 added 0e `gate_hc_diet_kernels`
and 0f `gate_w4a4_kernels`) green with the seams forced ON via `MEMRA_Q4E_SEAMS`, and
the seam-ON tiny receipts are BYTE-IDENTICAL to the defaults receipts for the exact
seams (projstack/qsagraph at tiny geometry; hcdiet/w4a4 fall back at tiny geometry by
%-guards and are covered by their real-geometry oracle arms). Real-checkpoint gate with
seams 1+2 ON (perf20 `nvfp4-r4-on`): goldens argmax 10/10, greedy divergences
none/8/none/48 — IDENTICAL to the round-3 banked baseline — tp2-gate 24/24 argmax,
worst rel 3.248e-5, **TP2 decode 13.6 ms/token (73.66 tok/s)**.

## W4A4 numeric envelope (the REAL-ERROR seam's acceptance battery)

Stated up front: activation quantization is the mint's own calibrated W4A4 arithmetic
(static amax clip + e2m1 RNE), NOT an exactness rewrite. Acceptance = the real gate
stays in the mint-error class (the NVFP4-vs-BF16 cross-arm class: argmax 10/10,
KL row-9 ≈ 0.17); if argmax drops, the seam stays OFF.

- Prefill rows are UNTOUCHED by construction (the seam is t==1-only): the goldens
  battery ON-arm reproduces argmax 10/10 and the `--compare-logits` rows vs the OFF-arm
  probe are all-zero (identical bytes) — which is why the round added the
  `--seam-gate` DECODE-ROW instrument (OFF vs ON per-step logits envelope + KL +
  argmax on the same fed chain).
- Greedy 4-prompt forks (ON arm): none / 8 / none / 26 vs baseline none / 8 / none /
  48 — one prompt forks earlier (step 26 vs 48), the others identical.
- Decode-row envelope (`perf22/seam-gate-w4a4-nvfp4.tsv`, 24 steps, OFF-chain fed):
  **argmax 22/24** (forks at steps 17 and 22), per-row KL vs the W4A16 arm 0.00-1.65
  (median ≈ 0.05; worst 1.65 at step 9), worst abs 9.18 on rows whose |logit| scale is
  ~15-36 — a measurable QUALITY change, not an accumulation reorder.

**Verdict: W4A4 is OWNER-RETIRED as a lever (owner order, 2026-08-30) — do not
re-propose it.** The measured kill first: the stated criterion — "if argmax drops, the
seam stays OFF" — tripped on the decode rows (22/24, KL to 1.65). The owner order then
retired the whole lever class: activation quantization in the serving path has hurt
correctness across many past attempts and models, and the owner has dropped it every
time. Per the flags doctrine (negative experiments leave receipts, not dead code) the
flag, dispatch arms, kernels (`q4e_act_quant_f32`, `qmatvec_nvfp4_w4a4_*`), host twins,
and oracle arm 0f were DELETED in-lane; `input_scale` is RECORDED-ONLY on the bank
source (validated + max-reduced, no compute consumer — the recorded-only comment in
`BankTensorSrc::Nvfp4` cites the order). The weight-only NVFP4 dequant-matvec shape
(f32/bf16 activations) is the correct serving shape and stays. Historical receipts kept
for the record: single-card A/B 15.65 → 15.05 (perf21), TP2-route A/B 13.61 → 12.95 /
77.2 tok/s (perf23) — the perf was real; the correctness class was not ours to serve.

## qsagraph: negative/flat, deleted (same doctrine)

Work item 3's QSA phase-1 segment-graphs experiment (built, gated green end-to-end on
the tiny plan with byte-identical receipts): single-card A/B **LOST** (15.64 → 15.71,
`perf22/ab-qsagraph-nvfp4.tsv` — the host issue thread is not the single-card wall, and
graph-replay overhead ≈ the ~10 launches it removes) and the TP2-route A/B was **FLAT**
(13.61 → 13.61, `perf23/ab-qsagraph-nvfp4-tp2.tsv`). Flag, kernels
(`q4e_append_row_pos_f32`), and the device-pos plumbing deleted; these receipts are the
record. Lesson for the corpus: after the round-3 segment graphs, the TP2 host-issue
budget is no longer the wall — the DtoH boundary waits and the GPU serial chain are.

## Round 4b (post-retirement, activation-precision-neutral levers)

| item | seam | change | A/B | receipt |
|---|---|---|---|---|
| 5 | `set_sel_gufuse` | fused gate+up+silu sel matvec — 4 gate + 4 up rows per warp off shared f32 activation registers, v3-VERBATIM per-row arithmetic + silu_mul_f32-VERBATIM epilogue ⇒ BIT-IDENTICAL to the 3-launch chain (oracle gufuse mode asserts byte identity incl. the count-gated pack twin); cuts the sel serial chain 5 → 3 launches and doubles outstanding code loads per warp | single 14.75 → 14.58 (1.012×), TP2 route 13.43 → 13.10 (1.025×), rep-0 chains IDENTICAL both configs | `perf24/ab-gufuse-nvfp4{,-tp2}.tsv` |
| 6 | `set_router_bf16` | the router GEMV was the last dense trunk mat on f32 cuBLASLt — bf16 residency twin (trunk accumulation class; routing near-ties gated by the real gate + the decode-row seam-gate: 24/24 argmax, worst KL 0.00116, worst abs 0.548) | single 14.75 → 14.68 (1.005×), TP2 route 13.47 → 13.36 (1.008×), rep-0 chains IDENTICAL | `perf24/ab-routerb16-nvfp4{,-tp2}.tsv`, `perf24/seam-gate-routerb16-nvfp4.tsv` |

All four surviving seams are **default ON** (receipts in the `*_DEFAULT` doc comments;
per-mat / chain / f32 arms stay resident as the OFF twins and geometry fallbacks).

## Headline (perf25, final combined battery at the shipped defaults)

| | ms/token (warm 40) | tok/s | 256-token warm | vs PROFILE-3 | vs PROFILE-0 |
|---|---|---|---|---|---|
| round 4 single card | 14.5 | 69.16 | 15.1 / 66.14 | 1.076× | 5.41× |
| **round 4 TP2** | **12.9** | **77.27** | **13.7 / 73.19** | **1.102×** | **6.09×** |

Correctness at the shipped defaults (one battery, `nvfp4-final`): goldens argmax
**10/10**, greedy divergences **none/8/none/48 — identical to the round-3 banked
baseline**, tp2-gate **24/24** argmax (worst abs 3.040e-5, rel 3.018e-5). Receipts:
`hidden-gate-nvfp4-{final,single-final,tp2-256,single-256}.tsv`,
`tp2-gate-nvfp4-final.tsv`, `greedy-gate-nvfp4-final.tsv`,
`profile-nvfp4-single-final.tsv`, `nsys-tp2final_*.csv`; box `~/realgate/perf18..25`.

## Verdict vs 90 tok/s: NOT crossed — 12.9 vs the 11.1 needed (1.16× gap, from 1.28×)

What the round moved: 14.22 → 12.9 ms TP2 (+9.9%), 15.6 → 14.5 single (+7.6%), with
four seams default-ON, all four rep-0-chain-identical, and zero movement in any
correctness receipt. What it could NOT use: the PROFILE-3 projection's biggest single
lever (W4A4 experts, measured 12.95 TP2 before retirement) is **owner-retired** —
activation precision is not a lever this product will pull (owner order 2026-08-30,
recorded above and in the loader; future lanes must not re-propose it).

The honest residual (final measurements, not guesses):

- Final single-card section profile: hyper.read 3.36 ms (still #1 — the diet's three
  kernels sit well above the gate's ~0.9 ms bandwidth floor; the stage-1/3 chunk knobs
  ROWS_PB=4/DIMS_PB=8 were picked blind and never laddered — an open micro-lever),
  moe.sel_grouped 2.91 (the fused kernel still runs ~1600 one-warp blocks; deeper
  restructure = split-K + two-stage reduce, untried), gdn.proj 2.60 (bytes-floored:
  ~52 MB/layer × 36 at ~1.6 TB/s ≈ 1.2 ms + sync distortion — only weight-side
  re-quantization could shrink it), lm_head 0.85 (weight-side-only quantization is
  owner-permitted if indicted, but it moves the logits the argmax gates read — untried).
- Final TP2 nsys API side: `cuMemcpyDtoHAsync` 70/token still absorbs the serial waits
  (48 router + 12 indexer boundaries — the host-routing doctrine's fixed price);
  360 graph replays = 2.3 ms + ~460 eager launches ≈ 1.6 ms of single-thread host
  issue. Two-thread per-card issue would halve the host side (~0.5-1.0 ms est) but
  needs the Engine borrow structure split across OS threads — the largest untried
  in-scope lever.
- Even the full stack of the untried micro-levers above projects to ~11.5-12.2 ms
  (78-85 tok/s). **Crossing 90 tok/s on this artifact with single-request greedy-shape
  decode is not reachable from this structure without the retired lever**; the honest
  paths to 90+ are the deferred MTP/spec-decode lane (SEMANTICS.md §MTP — a >1×
  multiplier, not a shave) and batching, both already scoped as separate lanes.

## Flags (round-4 seams, flags law)

| seam | default | why | rollback |
|---|---|---|---|
| `set_proj_stack` | **ON** | ab-projstack receipts, chains identical; VRAM-neutral (stack REPLACES per-mat twins) | `set_proj_stack(false)` (row-offset-view per-mat launches) |
| `set_hc_diet` | **ON** | ab-hcdiet receipts, chains identical; oracle arm 0e | `set_hc_diet(false)` (fused-gate chain) |
| `set_sel_gufuse` | **ON** | ab-gufuse receipts both configs, chains identical; oracle bit-identity | `set_sel_gufuse(false)` (v3 + silu chain) |
| `set_router_bf16` | **ON** | ab-routerb16 receipts both configs, chains identical; seam-gate 24/24 | `set_router_bf16(false)` (f32 cuBLASLt router) |
| `set_sel_w4a4` | **DELETED** (owner retirement 2026-08-30) | decode argmax 22/24, KL to 1.65 — receipts above | n/a — do not re-propose |
| `set_qsa_graphs` | **DELETED** (negative single / flat TP2) | receipts above | n/a |

New instruments kept: `--seam-gate <n>` (decode-row OFF-vs-ON envelope + KL + argmax —
the instrument that caught W4A4; prefill-shaped goldens cannot see t==1-only seams),
`--tp2 --ab-seam <name>` (interleaved A/B on the TP2 route), `MEMRA_Q4E_SEAMS`
(gates force not-yet-default seams for their correctness receipts — flags-law
ordering).


