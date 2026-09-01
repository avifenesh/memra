# glm5 VREST — the verify round's remaining wall (lane/glm5-vrest, 2026-08-31)

Base: origin/lane/glm53-flash-bringup @ cda7cceb1 (verify-batch + decode-diet + TP-2
fail-closed + hyper-batch ON). Worktree `~/projects/wt-glm5-vrest`.

The flip landed (`../flip-reprice-20260831/`, VERDICT: FLIP, 45.65 tok/s ship = 1.289x
plain) and its cell-2 trace split the round: at K=3 the verify phase is 69.72 ms of
which vkda 17.57 + vmla 6.26 + **vrest 45.61** — the classes the verify-batch lane
deliberately left per-row-shaped ("UNCHANGED named: MoE per-(token,expert) inner loop
..., hc glue/norms"). Round-wall fit today: 33.5 + 11.2K ms. Every ms out of vrest
raises the spec multiplier toward the 100-tok/s serving bar.

## 1. Attribution: where the 45.61 ms lives (written BEFORE any change)

Derived from banked receipts + a source census of THIS head on the pinned serving shape
(3-card recipe, `MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1`, full expert residency; every diet
door and `MEMRA_MOE_FUSED_EPI` at their defaults = OFF — verified against
`../flip-reprice-20260831/box/serve.sh`, which sets none of them). Box constants:
2.216 us/launch eager, ~1.06 us/driver alloc-or-free call (launch-diet census).

**The two banked vrest points fit a line** (cell-2 medians, t = K+1 rows):

    vrest(t) ~= 7.8 + 9.46 * t   ms/round      (t=2: 26.70 measured; t=4: 45.61 measured)

**The per-row marginal (9.46 ms/row) is the MoE per-(token,expert) inner loop.**
Source census of the dispatch the serving artifact actually takes (glm5 = sigmoid
router + PRE-clamped SwiGLU + per-expert NVFP4 `weight_scale_2` macros → denied every
fused/dev/pairs/gdec arm by predicate → `moe_ffn_sequential_zq8`'s per-expert SLAB
loop; `MEMRA_MOE_FUSED_EPI` default OFF):

| class | per verify ROW (42 sparse layers) | ms/row (arith) |
|---|---|---|
| expert weight bytes, NVFP4 (near-disjoint sets across rows) | ~4.8 GB re-read | ~2.7 @1.79TB/s |
| launch train: 8 experts x (gate+up qmatvec_expert_q8 + preclamp + act-quant + down + axpy) + z-quant = 49/token-layer | 42 x 49 = 2,058 launches | ~4.6 |
| driver alloc churn: ~6 alloc+free pairs per (token,expert) + z-quant pair | ~4,200 calls | ~2.2-4.4 (partially hidden in drain gaps) |
| hc-glue mixes GEMVs (per-row m=1 cuBLASLt by the lt_ndep law, 2 sites x 45 layers) | 90 launches | ~0.4-0.7 |
| dense L0-2 FFN per-row loop (`hyper_ffn_branch_batch` Dense arm) | ~12-15 launches + ~0.2 GB | ~0.2-0.4 |
| **sum** | | **~10.1-12.8 vs 9.46 measured** (host terms partially overlap device time) |

**The t-flat ~7.8 ms**: lm head (0.7 — BATCHED, verified below) + glue block-per-token
kernels (~3.3 ms GPU at t=1 per the diet census, t-parallel) + 42 per-MoE-layer router
readback syncs (t-shared) + shexp trio (m=t decode-exact) + the trace-2 instrument's own
per-layer drains.

**lm-head verification (task's bug-class check): CLEAN.** `glm5_verify_head` (the only
head site, shared by the ppN twin) rides `matmul_rows_exact` at t>1: FloatBf16 +
`MEMRA_BF16_MMV` + t 2..=8 → the tcols twin, weight read ONCE per round. Serving K<=7
⇒ t<=8 always covered. Named cliff, not a bug: t=9..15 falls to `matmul_decode_exact`'s
bf16 rows kernel (weight re-read per row) — outside every measured serving shape.

**hc glue verification (task port 3): ALREADY THE DECODE-BATCH PROGRAM.** The verify
walk calls the SAME `hyper::pre_exact` / `post` entries at m=t that
`hyper_batch_range_decode` (decode_step_batch_hyper's trunk) calls at m=B — the glue's
block-per-token kernels batch over rows, and the ONE width-dependent reduction (the
mixes GEMM) runs per-row m=1 cuBLASLt in BOTH walks because that is the lt_ndep law's
exact form (cuBLASLt's reduction split is n-dependent; plain decode runs m=1, so any
m=t batching of it breaks the byte bar by construction). Nothing to port; the glue's
per-row residue is the 90 tiny GEMV launches/row (~0.4-0.7 ms) named above. A
cuBLASLt batch_size=t (n=1 per entry) probe is a named follow-up, refused here: its
per-entry bit-equality would be an empirical property of the current library version,
not structural.

## 2. What this lane changes: batch the MoE across the K+1 rows

The existence proof (task charter): `decode_step_batch_hyper` runs B sessions' MoE at
m=B through the SAME call seam (`hyper_ffn_branch_batch` → `moe_ffn_il_zq8`) the verify
walk uses at t=K+1 — per-row bit identity through that seam at m>1 is the
glm5-hyper-batch gate's proven class. What stays per-row-shaped inside the seam is the
per-(token,expert) inner loop. This lane collapses it, for the VERIFY walk only:

**The pairs-shaped batched dispatch** (grouped over the pair union across rows — the
grouped-prefill pattern at decode scale, with the GEMM class it must NOT take, refused):

- ROUTER + ROUTING: unchanged invocations (`moe_router_logits` t<16 per-row program +
  `moe_route_sigmoid_cfg` host sel/w) — selection bit-identical by construction.
- ONE `moe_gate_up_preclamp8_q8_rows` launch covers ALL t x n_used routed pairs: per
  pair the body is `moe_gate_up_preclamp8_q8` (the fused-epilogue kernel, already the
  gated program for this arch's PRE-clamp + macro fold) VERBATIM — same
  `expert_dot_g` g-strided order per (pair,row) == `qmatvec_expert_q8`'s chain, same
  warp tree, same `swiglu_preclamped_mul_scaled_f32` expression on the exact dot
  values with per-pair gs/us.
- ONE `quantize_q8_1` over the [n_pairs, n_ff] activations (per-row program).
- ONE `moe_down8_fma_q8_rows` launch: per (token, out-row) the SLOT-ORDERED
  `__fmaf_rn` chain of `moe_down8_fma_q8` verbatim (== the sequential axpy chain,
  the gdec-gated class), down macro folded into the pair weight exactly where
  `axpy_into`'s `w[j] * macro_scale(ex)` folds it. Full row overwrite.
- Pointers/scales: host-built per-pair tables from the resident slab base + ex*stride
  (the sequential slab arm's exact pointer arithmetic), 2 small htods per layer-call.
- Expert-set UNION note: pairs are the union; near-disjoint sets across verify rows
  make expert-major weight dedup (grouped tensor-core GEMM, `moe_f16_grouped`) worth
  ~nothing here AND that class is measured non-bit-stable — REFUSED at this seam by
  the byte bar. The launch/alloc train is the win (49/token-layer → ~5/layer-call,
  ~flat in t); the expert bytes term stays ~linear in rows (physically irreducible
  under per-row bit identity).

Engagement predicates (fail-closed to the unchanged sequential loop): verify-walk
caller only (`MEMRA_GLM5_VERIFY_BATCH` arm, t>=2) + sigmoid router + PRE clamp
(l>1e-6) + q8-supported uniform expert layout + local resident slab (`!gu_il`,
`MEMRA_MOE_SLAB` on) + n_used<=8 + no cpu-expert hybrid. The hyper-batch decode walk
(`decode_step_batch_hyper`) is NOT rewired — its priced class stays byte-stable; the
same port for B-session decode is a named follow-up with its own re-price.

Flag: NO new flag — the port is the same batched-walk program, riding
`MEMRA_GLM5_VERIFY_BATCH`'s arm (=0 restores the per-row FFN class along with the
per-row mixer walk; one seam, both classes). FLAGS.md row amended in this PR.

## 3. Gates (rig 5090, flock, TF32 off, exactness only)

- Kernel bit-gates (`glm5_verify_batch_gpu`): pairs twins vs the sequential chain
  (quantize_q8_1_view + qmatvec_expert_q8 + swiglu_preclamped + quantize + qmatvec +
  axpy) on minted NVFP4 expert rows with a LIVE macro plane, t=2..8; red arms:
  swapped pair rows (row isolation) and dropped macro scales — both must bite.
- Walk gates: `glm5_tparallel_verify_gpu` fixture expert banks flipped Q8_0 → NVFP4 +
  live `weight_scale_2` macro planes (the SERVING expert class) so the standing 9/9
  — walk-vs-plain bit identity, accept-j byte identity, flag A/B, reds, e2e K=1..7 —
  now exercises the new arm end-to-end; engagement anchored on the new dispatch
  counter (>0 on the batched arm, ==0 on the =0 arm), never on liveness.
- Standing batteries re-run: tparallel, spec/dflash sessions, verify_batch, ppn arms,
  hyper-batch gate, epilogue gate, server suite, diet gates, local-ci --perf.

## 4. Predicted round wall (arithmetic AGAINST BANKED CONSTANTS — the box re-price
window prices it; cells 2+3 shape re-run against this head as a separate window)

Post-port vrest slope: bytes ~2.7 + glue ~0.5-0.7 + dense ~0.2-0.4 + MoE launch train
~0.1 (42 x ~5-6 launches, t-shared) ≈ **~3.5-3.9 ms/row** (from 9.46). Traced verify
marginal 13.03 → ~7.1-7.5 ms/row; scaling the wall fit by the traced ratio:

    round(K) ~= 33.5 + ~6.1-6.5 K ms      (from 33.5 + 11.2K)

⇒ K=3 greedy ~52-53 ms ≈ 55-56 tok/s (from 43.4); ship shape (auto-K3 + PMIN 0.7,
drafted/rnd 2.31, tok/cyc 2.71) ~47-49 ms ≈ **~55-57 tok/s predicted** (from 45.65,
1.289x → ~1.55-1.62x plain). "vrest ~flat in K" is reached for the launch/alloc terms;
the expert-bytes term stays ~linear (named, irreducible under the byte bar) — flatter
vrest also re-opens the K ladder upward (K=4/5 re-price on the box).

### What the box re-price window carries (separate window, NOT this lane's)

- Cells 2+3 shape of `../flip-reprice-20260831/` against THIS head, serving recipe
  byte-identical (BF16_MMV=1 load-bearing, 3-card pins, port 18400 scoping).
- `MEMRA_GLM5_SPEC_TRACE=2`: the `[glm5-phase-v]` line now carries `vrest=... (vffn=...)`
  — bank the vffn share; vkda/vmla/vrest fields unchanged and comparable with cell 2.
- Engagement receipts to grep per spec boot: the BATCHED walk line (now ends
  `moe=pairs rows-call where qualified`) AND `[glm5-vrows] verify MoE batched across
  rows: pairs=...` at the first spec round; PER-ROW line absent.
- The A/B seam is still ONE flag: `MEMRA_GLM5_VERIFY_BATCH=0` restores the per-row
  mixer walk AND the per-(token,expert) MoE class together (the zctl arm should
  reproduce ~91.1 ms @ K=3 as before).
- Byte-identity re-gate first (cell-1 shape, 12 tapes incl. d02/d04) — ANY divergence
  stops the window.
- If the wall lands near the §4 band, re-run the K ladder upward (K=4/5) — flat-ish
  vrest moves the peak.

## 5. What landed (per port)

| port | state | receipt |
|---|---|---|
| MoE across the K+1 rows (pairs-shaped: `moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows`, dispatch arm in `moe_ffn_sequential_zq8`, verify-walk-scoped via `hyper_ffn_branch_batch(vrows)`) | **LANDED**, rides `MEMRA_GLM5_VERIFY_BATCH` (no new flag — same batched-walk program; `=0` restores the per-(token,expert) class with the per-row mixer walk, one seam) | gate table below; engagement `[glm5-vrows] verify MoE batched across rows: pairs=...` |
| hc glue across rows | **VERIFIED ALREADY-BATCHED** (no code): the verify walk runs the SAME `pre_exact`/`post` m=t program as `decode_step_batch_hyper`'s glue; the mixes GEMM stays per-row m=1 cuBLASLt by the lt_ndep law in BOTH walks (batching it breaks the byte bar vs plain decode by construction) | §1; `hyper.rs pre_exact` doc |
| lm head | **VERIFIED BATCHED** (no code): `glm5_verify_head` → `matmul_rows_exact` tcols at t=2..8 under `MEMRA_BF16_MMV=1`; no per-row head site exists (ppN twin shares it). Named cliff: t=9..15 falls to the per-row bf16 rows kernel — outside every serving shape | §1 |
| dense L0-2 per-row FFN | **NAMED UNCHANGED** (~0.2-0.4 ms/row, 3 layers; `matmul_group` carries no cross-width per-row contract for every class this walk serves — a port is a decode-exact re-plumb worth <5% of the vrest slope) | §1 |
| trace: `vffn` sub-bucket at `MEMRA_GLM5_SPEC_TRACE=2` | LANDED — `[glm5-phase-v]` line gains `(vffn=...)` inside the UNCHANGED vrest field, so cell-2 receipts stay comparable and the re-price window gets the MoE share directly | glm_spec.rs |

Also in this PR: `glm5_tparallel_verify_gpu` + `glm5-spec-ppn-gate` fixture expert banks
flipped Q8_0 → NVFP4 + LIVE `weight_scale_2` macro planes (Q8_0 is not
`q8_expert_supported`, so the standing walk gates could never engage the new arm — the
vacuous-green class); dispatch-counter engagement anchors added to the flag A/B.

## 6. Gate table (rig 5090, flock held, NVIDIA_TF32_OVERRIDE=0, exactness only, 2026-08-31)

| suite | result |
|---|---|
| `glm5_verify_batch_gpu` (gates 1-3 standing + NEW gate 4: pairs twins vs the sequential chain, minted NVFP4 banks + live macro plane, t=2..8 all bit-identical; swapped-pair red bites 256/256, dropped-macro red bites 256/256) | 4/4 PASS |
| `glm5_tparallel_verify_gpu` on the NVFP4+macro fixture (walk rows vs plain bitwise; flag A/B with kda-stash AND `moe_vrows_dispatches` anchors — >0 batched / ==0 per-row; accept-j byte identity; corrupted-ring, stale-KDA, pool-key, rollback-disabled reds; e2e K=1..7; FR-Spec 7-arm) | 9/9 PASS |
| `glm5_spec_session_gpu` / `glm5_dflash_session_gpu` | 9/9, 10/10 PASS |
| `glm5_moe_epilogue_gpu` (the fused-epi reference gate, both provenances) | 9/9 PASS |
| `glm5-spec-ppn-gate` matrix on the NVFP4+macro fixture (stages 2: even/split1/split3/streams0/overlap0; stages 3: even/asym/streams0) — `[glm5-vrows]` engagement receipt in ALL 8 logs | 8/8 arms PASS |
| `glm5-hyper-batch-gate` matrix (decode walk NOT rewired — class-stability re-proof) + `glm5-hyper-ppn-gate` matrix | ALL ARMS PASS |
| adjacent seams: `kda_fixture_gpu` 3/3, `kda_fused_proj_gpu` 5/5, `kda_fused_proj_bf16_gpu` 5/5, `kda_quant_operand_gpu` 4/4, `mla_gpu_forward` 5/5, `mla_decode_split_gpu` 3/3, `hc_fused_pre_gpu` 4/4, `hc_decode_ws_gpu` 2/2, `hyper_connections_gpu` 6/6, `glm5_mtp_head_gpu` 5/5, `glm5_kpool_indexer_gpu` 14/14 | PASS |
| `memra-server` suite | 481/481 |
| `tools/check-flags.sh` (723 reads covered, no new flag) · clippy all-targets zero warnings · `cargo fmt --check` clean | green |
| `tools/local-ci.sh --perf` | exit 0 — correctness ALL GREEN, perf 0 fail 0 warn (absent-model cells SKIP, the rig's standing shape; qwen9b cell 138.45 tok/s [OK] vs rolling median); run TWICE green on this tree (`receipts/local-ci-perf-run{1,2}.log`) |

Logs: `receipts/` (batteries), `receipts/ppn-gate/`, `receipts/hbatch-gate/`, `receipts/hppn-gate/`.

## 7. Named follow-ups (not built here)

1. Port the same pairs arm to `decode_step_batch_hyper`'s B-session MoE (the hbatch
   battery named "the low-overlap routed-expert unions at B<=12 dominate" — same class,
   own re-price window; the flag seam would be `MEMRA_HYPER_BATCH`'s arm).
2. cuBLASLt batch_size=t (n=1/entry) probe for the 90/row glue mixes GEMVs — capture-
   then-gate, but per-entry equality would be a library-version property, not structural;
   only worth it if the box shows the glue share still matters post-port.
3. Dense L0-2 decode-exact re-plumb (~0.2-0.4 ms/row).
4. `docs/KERNELS.md` inventory recount lane: the per-file symbol counts date from the
   original inventory commit (qmatvec corrected to 318 here since this lane touched it)
   and `dsv4_gpu.cu` (65), `mla_attn.cu` (14), `kda.cu` (6) and others have no sections.

## 8. Status log

- Lane open 2026-08-31. Worktree `~/projects/wt-glm5-vrest` @ cda7cceb1. Attribution
  (§1) written from banked receipts + source census BEFORE any code change.
- BUILT (§2 + §5): kernels (qmatvec.cu), launchers + `MOE_VROWS_DISPATCHES` (lib.rs),
  dispatch arm + `moe_vrows_pairs_q8` (hybrid_forward.rs), walk plumbing + vffn trace
  (glm_spec.rs), fixture flips + gate 4 + anchors (tests, glm5_spec_ppn_gate.rs),
  FLAGS.md row amended + KERNELS.md rows (same PR).
- Gate table §6 all green same day; `tools/local-ci.sh --perf` exit 0 twice.
- PUSHED to `origin/lane/glm5-vrest` (code head `29ac174df` + perf-ci-row commits
  `e30ecc711`/`134064f5c`; base `cda7cceb1`); no self-merge. The box re-price window
  (cells 2+3 shape) runs against this head separately — carry list above §5.
