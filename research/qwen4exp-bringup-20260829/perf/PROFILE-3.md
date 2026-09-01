# qwen4_exp decode PROFILE-3 — round 3: sel v3, GDN step twin, norm fusion, TP2 (2026-08-29)

Same box (2× RTX PRO 6000 Blackwell 96 GB, sm_120a, PCIe — no NVLink), same artifact
(`~/data/q48fn-nvfp4`, the per-expert modelopt NVFP4 mint), same goldens prompts as
PROFILE-0/1/2 (read those first). Greedy is the instrument. Every change: interleaved
×5 same-run A/B (fresh state + probe prefill + 4 warmup steps per arm) + tiny gate
(now EIGHT arms) + real-checkpoint gate re-runs, receipts banked before the next item.

## Headline

| | ms/token (warm) | tok/s | vs PROFILE-2 | vs PROFILE-0 |
|---|---|---|---|---|
| PROFILE-2 (single card) | 17.2 | 58.18 | — | 4.56× |
| **round 3 single card** (sel v3 + GDN step twin + norm fusion) | **15.6** | **64.13** | 1.10× | 5.03× |
| **round 3 TP2** (both cards, segment graphs) | **14.22** | **70.34** | **1.21×** | **5.52×** |

**The 90 tok/s owner target (≤11.1 ms/token) was NOT crossed.** The honest gap after
TP2 + every in-scope lever is 1.28× (14.22 vs 11.1), and §Residual below shows why the
remaining distance is not reachable without the out-of-scope levers (W4A4 experts
foremost). 256-token warm receipts at both configs are banked (§Long-run).

## Per-change interleaved A/B (each row = its own box battery, both arms in ONE run)

| item | change | A/B (mean of 5 means) | win | rep-0 chains | receipt |
|---|---|---|---|---|---|
| a | **sel matvec v3** — 4 output rows/warp sharing the activation registers + u16 scale loads (v2 ran ≤1 iteration/thread at the down geometry: no memory-level parallelism per warp) | 17.06 → 16.57 | 1.030× | identical (−1) | `ab-selv3-nvfp4.tsv` |
| b | **GDN decode-step scan twin** — `gdn_scan_step_f32`: grid (nv, hv) with one state ELEMENT per thread (the naive kernel ran nv=48 blocks on a 188-SM card with the state row per thread — latency-bound) | 16.59 → 15.60 | **1.063×** | identical (−1) | `ab-gdnstep-nvfp4.tsv` |
| c | **GDN norm+gate fusion** — `rms_sigmul_f32`: rms_norm + sigmoid + mul in one launch, rms_norm_f32-VERBATIM reduction, BIT-IDENTICAL to the chain (asserted by the new tiny-gate arm 0d) | 16.65 → 16.52 | 1.008× | identical (−1) | `ab-gdnfuse-nvfp4.tsv` |
| 2 | **TP2** (the ladder below) | 15.57 → 14.22 | **1.095×** | identical (−1) | `ab-tp2graphs2-nvfp4.tsv` |

The chain telescopes: each battery's off-arm reproduces the previous on-arm
(17.06 ↔ PROFILE-2's 17.07; 16.59/16.65 ↔ 16.57; 15.55-15.57 ↔ the combined 15.6).
All three single-card seams are **default ON** (receipts above; `set_sel_v3`,
`set_gdn_step`, `set_gdn_fuse`); v2/naive/chain stay resident as geometry fallbacks and
A/B twins. Combined-defaults full battery: `run-perf11-combined-nvfp4.log` — logits
argmax 10/10, greedy divergences none/8/none/48 (identical to the banked baseline),
cross-arm KL row 9 = 0.17303 top-1 true.

New permanent tiny-gate arm **0d** (`gate_gdn_step_kernels`): the scan step twin vs the
naive scan at the ARTIFACT geometry (nk 16, nv 48, hk/hv 128 — the tiny plan's hk=4
cannot reach the twin's warp guard) worst rel 1.192e-7, plus the fused norm's
bit-identity assert. Kernel oracle arm 0 gained the v3 modes (worst rel 3.099e-6,
including the out_f%4!=0 fallback shape).

## TP2 — what was built, and its ladder

Structure (tp2-join-diet playbook; full doc in `qwen4exp_gpu.rs` §TP2):

- **Replicated residual**: both cards hold the wide planes and run entry/PLE/hyper-gates/
  exit with bit-identical weights on bit-identical inputs — replicated deterministic
  compute kills every broadcast except the 2 joins/layer. Requires the bf16-trunk +
  fused-gate seams (deterministic kernels only).
- **Split**: GDN by key-head blocks (compact per-card head order keeps kh = h % nk_h;
  24/24 value heads), QSA 12/12 query + 1/1 KV heads, MoE routed experts by expert-id
  halves (top-10 splits ~5/5 on average; card 1 holds the bank's upper half, 40.2 GiB
  post-load), shared expert by ff halves, lm_head by vocab halves.
- **Joins**: 2/layer — each card's [hidden] partial pushed as a P2P kernel store
  (`q4e_push_f32`, direct-join) into the peer's staging (ping-pong ×2, proven safe at
  depth 2), one cross-device event wait each way, then BOTH cards add in the SAME rank
  order, so the replicated residual stays bit-identical across cards.
- **Host twins unchanged**: MoE routing (router GEMV + dtoh on card 0, top-k once,
  filtered selection to both), QSA indexer (card 0 + mask H2D to both), PLE hashing
  (host; the 102 GB table stays host-resident and shared).
- **Segment graphs** (the decisive piece): per card per layer, seg A (attn gate + GDN
  half + join push), seg B (join add + write + mlp gate + card-1 shared prestage),
  seg C (count-gated MoE tail: `qmatvec_nvfp4_modelopt_sel_f32_v3c` +
  `axpy_rows_seq_pack_f32` read the live expert split from a device pack blob, so the
  VARIABLE per-card slot count no longer forces eager shapes), seg D (MoE join add +
  write), exit segs. QSA/PLE phase-1 and the router boundary stay eager. Prefill stays
  single-card; the first TP2 decode migrates the mixer state (host bounce, one-way
  latch).

| TP2 iteration | vs single card (interleaved ×5) | receipt |
|---|---|---|
| v1, fully eager | 15.57 vs **16.45** (TP2 LOSES) | `ab-tp2v1-nvfp4.tsv` |
| + card0 shared off the router path | 15.55 vs 16.30 | `run-perf14-tp2sharedfix-nvfp4.log` |
| + seg A/B/exit graphs | 15.55 vs 15.28 | `ab-tp2graphs-nvfp4.tsv` |
| + seg C/D (count-gated MoE tail) | 15.57 vs **14.22** | `ab-tp2graphs2-nvfp4.tsv` |

Why eager TP2 lost: `nsys-tp2eager_cuda_api_sum.csv` — **3,908 kernel launches/token**
at 3.4 µs avg ≈ 13 ms of host issue under the profiler, against only ~9.4 ms/card of
GPU kernel time (`nsys-tp2eager_cuda_gpu_kern_sum.csv`: 188.3 ms over 10 steps, both
devices). The host was the wall; graphs removed it (final shape: **360 graph replays +
~500 eager launches/token**, `nsys-tp2g_cuda_api_sum.csv`).

### TP2 exactness gate (the tolerance statement)

TP2 matches single-card to TOLERANCE, not bit: the split out-projections sum row halves
in a different association than the full GEMV, the expert combine becomes
(Σ card-0 slots) + (Σ card-1 slots) instead of the slot-sequential chain, and the join
add reorders those partial sums — the same accumulation class as every banked seam.
Gate: `--tp2-gate` feeds BOTH a single-card twin and the TP2 state the SAME token
sequence (the single-card argmax chain) and compares logits row by row. **24/24 argmax
matches, worst abs 4.578e-5, worst rel 3.296e-5** (policy max_rel ≤ 0.01 + argmax) —
~300× inside tolerance, and the envelope is BYTE-IDENTICAL across the eager, seg-A/B,
and seg-C/D TP2 implementations (`tp2-gate-nvfp4.tsv`, `tp2-gate-graphs-nvfp4.tsv`,
`tp2-gate-graphs2-nvfp4.tsv`) — graph replay reproduces eager exactly, and the A/B
rep-0 chains are identical between TP2 and single-card in every battery.

## Long-run receipts (256 warm tokens, greedy-as-instrument, self-fed argmax)

| config | mean ms/token | tok/s | p90 | receipt |
|---|---|---|---|---|
| TP2 (winning config) | 14.9 | **67.30** | 15.5 | `hidden-gate-nvfp4-tp2-256.tsv` |
| single card | 16.2 | 61.68 | 16.8 | `hidden-gate-nvfp4-single-256.tsv` |

Both arms slow slightly with length (the QSA t_kv growth: dense masked SDPA, host
indexer over the raw-key cache, growing mask H2D) — the 40-step warm numbers above are
the standing comparison points; these are the long-run truth.

## Residual: why 90 is out of reach this round (measured, not guessed)

From the TP2-eager nsys (the only run where graph kernels are attributed):

- **GPU kernel time ≈ 9.4 ms/card/token** — the physics floor of THIS structure before
  any host/sync overhead. Its composition:
  - `qmatvec_bf16w_f32`: 8.9 ms/token across both cards (1,134 calls at 7.9 µs avg).
    The half-width GEMVs of the split mixers DO NOT run 2× faster — they are
    latency-floored small GEMVs at t=1, and the replicated hyper-gates deliberately burn
    on both cards (the broadcast-free trade).
  - sel v3 halves: 288 calls at 12.8 µs — a ~5-expert half sits near the launch/latency
    floor (the full 10-expert launch ran 21 µs; the split saves only ~40%, not 50%).
  - The decode step is a DEEP SERIAL CHAIN of ~500 dependent small kernels per card;
    TP2 splits width, not depth.
- **48 router host boundaries/token** (dtoh ~13 µs each clean) — the host-routing
  doctrine's price, identical on single card; plus 96 joins × 2 event hops.
- Remaining host: ~500 eager launches/token (QSA phase-1 + PLE + boundaries).

What the remaining levers would buy (projection, not receipts):

| lever | est. saving | lands at |
|---|---|---|
| QSA phase-1 segment graphs + device-t_kv SDPA twin | ~0.4-0.7 ms | ~13.6 ms / 74 tok/s |
| two-thread per-card issue | ~0.3-0.8 ms | ~13 ms / 77 tok/s |
| quantized lm_head (already vocab-split: 0.34 ms/card) | ~0.2 ms | ~12.8 ms / 78 tok/s |
| **W4A4 experts** (the parked `input_scale` consumer; halves sel bytes AND the sel latency floor) | ~0.6-1.0 ms TP2, ~1.2 ms single | **~11.8-12.2 ms / 82-85 tok/s** |

Even the full stack of these projects to ~82-85 tok/s. **Crossing 90 needs the W4A4
expert lane plus a hyper-gate diet** (the replicated read gates are 3.7 ms/card of which
only ~0.9 ms is bandwidth floor — a fused/wider gate program or activation-quantized
gate GEMVs is the untouched depth lever). Per the round scope (no W4A4 kernels this
round), that is the next lane's opening move, with PROFILE-2's TP2 projection now
receipted at its top end: projected 79-86, landed 70.3 with the enumerated in-scope
structure, ceiling ~85 with the out-of-scope levers above.

## Flags (round-3 seams, flags law)

| seam | default | why | rollback |
|---|---|---|---|
| `set_sel_v3` | **ON** | ab-selv3 receipts, chains identical | v2 (per-call fallback for out_f%4!=0 stays live) |
| `set_gdn_step` | **ON** | ab-gdnstep receipts; naive stays the prefill executor + tiny-geometry fallback | `set_gdn_step(false)` |
| `set_gdn_fuse` | **ON** | ab-gdnfuse receipts; kernel bit-identical to the chain | `set_gdn_fuse(false)` |
| TP2 | **OFF** (deployment opt-in: `--tp2`, needs 2 cards + P2P) | serving-topology choice, not a code default; its graphs ride `set_decode_graphs` | drop `--tp2` |

Receipts: `ab-{selv3,gdnstep,gdnfuse}-nvfp4.tsv`, `ab-tp2{v1,graphs,graphs2}-nvfp4.tsv`,
`tp2-gate{,-graphs,-graphs2}-nvfp4.tsv`, `*-perf11.tsv` (combined battery + profile),
`hidden-gate-nvfp4-{tp2,single}-256.tsv`, `nsys-tp2eager_*.csv`, `nsys-tp2g_*.csv`,
`run-perf{9,10,11,12,14,15,16,17}-*.log`, box `~/realgate/perf9..17`.
