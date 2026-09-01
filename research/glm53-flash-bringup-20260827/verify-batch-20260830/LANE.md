# glm5 verify-row batching (lane/glm5-verify-batch, 2026-08-30)

The make-or-break lane for glm5 spec: BATCH THE VERIFY ROWS' MATH. The flip re-battery
(`../flip-battery-20260830/`, cells 2+3) proved the round is verify-row-bound: ~24-26 ms
per verify row = one full plain step per row, K+1 rows per round, so no spec arm beats
plain (35.41 tok/s) while every row re-pays the whole trunk. This lane restructures the
per-layer walk so the K+1 rows share their heavy math, keeping the sequential contract
ONLY where the math demands it (the KDA state chain), under the non-negotiable per-row
byte-identity bar vs the PLAIN decode tape.

Base: origin/lane/glm53-flash-bringup @ 34e0c0bf2. Branch: lane/glm5-verify-batch.

## 1. Attribution: where the 24-26 ms/row lives (written BEFORE any change)

Derived from banked receipts, not re-measured — the box re-battery re-verifies it with
the new trace instrument (§5):

- Verify per-row marginal, measured (flip-battery cell 2, box B, 3-card PRO 6000,
  serving recipe `MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1`, full expert residency):
  K=1 51.65 ms / 2 rows = 25.8; K=3 96.47 / 4 = 24.1; marginal 22.4 ms/K.
  Plain step = 28.24 ms (35.41 tok/s).
- Resident-traffic roofline per row (`../decode-attribution-receipts/ATTRIBUTION.txt`
  bytes, halved where BF16_MMV applies, at ~1.79 TB/s; geometry CENSUS.md):

| class | in the CURRENT walk | bytes/row | ms/row @1.79TB/s | launches/row (est) |
|---|---|---|---|---|
| KDA mixer, 34 layers (wq/wk/wv/wo 4x[4096,8192] + f/g/b + conv + scan + norms) | PER ROW (`kda_decode_cached` in the mixer loop) | ~9.4 GB bf16 | ~5.2 | ~15 x 34 = 510 |
| MLA mixer, 11 layers (q_a/q_b/kv_a/o NVFP4 + kv_b f32 absorb/decompress + kpool indexer + attend) | PER ROW (`mla_attn_cached` t=1 in the mixer loop) | ~1.9 GB | ~1.1 | ~22 x 11 = 242 |
| MoE routed experts, 42 layers x 8/row (near-disjoint expert sets across rows) | batched at the CALL seam (`moe_ffn_il_zq8` t=K+1) but per-(token,expert) inside | ~4.8 GB NVFP4 | ~2.7 | ~1750 (the ATTRIBUTION.txt census, dominant launch term) |
| shexp + dense L0-2 + hc glue + norms | batched (rows-kernel classes re-read per row) | ~0.7 GB | ~0.4 | flat-ish |
| lm head [4096,154880] bf16 | batched call, rows kernel re-reads PER ROW | 1.27 GB | 0.7 | flat |
| **bytes total** | | **~18.1 GB** | **~10.1** | |
| launch/latency/sync structure (remainder vs measured 24-26; matches the ATTRIBUTION.txt X-term 17.1 ms invariant, scaled to this box) | ~linear in rows today | — | **~14-16** | ~3200/row-equiv |

Reading: the row marginal is ~40% resident bytes re-read per row and ~60% launch
structure, BOTH ~linear in rows in the current walk, because the mixers run per row and
the biggest weight classes (KDA projections, lm head) ride grid.y=t rows kernels that
re-read weights per row. The MoE per-(token,expert) inner loop is the one class that is
irreducibly ~linear in rows (near-disjoint expert sets) — named out of scope here (the
residency/fused-epilogue arc owns it); everything else batches.

## 2. What this lane changes (per LAYER, not per round)

Inside `glm5_verify_range` (shared by the unsplit walk and every ppN stage — the change
is structural for both):

- **KDA (34 layers): ONE t=K+1 `kda_core` call per layer** (`kda::kda_verify_rows_cached`).
  Projections/gates/norms batch m=K+1 through the decode-exact matmul classes
  (`matmul_rows_exact`); the conv takes the PREFILL arm (per-token taps in the same
  ascending order as the decode arm — bit-identical per row by construction, gated); the
  **recurrence stays sequential INSIDE `memra_kda_scan_s128`** — the kernel already loops
  T steps over register-resident state, so scan(T=K+1) is the chained t=1 program by
  construction (gated). Rollback: pre-round ring snapshot + stolen RAW projection rows
  (ring re-roll at T=keep) + pre-round ssm snapshot + stolen batched scan inputs (ONE
  scan replay at T=keep) — the loop-port-3 ReplaySSM diet, batched.
- **MLA (11 layers): ONE t=K+1 `mla_attn_cached_rows_exact` call per layer** — the same
  core the per-row walk runs, at t rows, with every internal matmul routed through the
  decode-exact classes. Selection is per-query by construction (`mla_kpool_score/select`
  take per-query visibility `first_pos + t + 1`; masked pools are non-finite and never
  selected); attention is per-(query,head) blocks walking that query's OWN idx list, and
  -1 padding is arithmetic-invariant (exp(0)=1 rescale). Rollback unchanged (latent
  truncate + pool-key clamp).
- **Head**: `glm5_verify_head`'s lm-head matmul rides `matmul_rows_exact` (the tcols
  class below), so the 1.27 GB bf16 head is read ONCE per round instead of once per row.
- **NEW kernel, the house `_vl`-twin discipline** (PATTERN:varlen-batched-cores +
  LAW:vl-bit-identity-order-pinning): `matvec_bf16_f32acc_x4_tcols` — the t-column twin
  of `matvec_bf16_f32acc_x4_rows`. One block owns 4 output rows for ALL t tokens: the
  weight pack is loaded once and each token keeps its OWN single-chain f32 accumulator
  fed in the exact per-token order of the rows kernel, then the identical shared-tree
  reduce runs once per token. Per-(row,token) bit-identity is structural and bit-gated.
  This converts the KDA-projection and lm-head byte terms from (K+1)x to 1x.
- **MoE / shexp / dense / hc glue: unchanged** — already batched at the call seam;
  the per-(token,expert) inner loop stays (named, out of scope).

Exactness routing (`Engine::matmul_rows_exact`, verify-batch walk only):
FloatBf16 + `MEMRA_BF16_MMV` + t 2..=8 -> tcols twin; everything else ->
`matmul_decode_exact` (Quant q8-fast -> batched MMVQ, bit-identical per (token,row),
weight read once; FloatBf16 -> rows kernel; Float f32 -> per-token m=1 cuBLASLt — the
lt_ndep law's exact form). Every class is per-row bit-exact vs the t=1 chain by existing
contract; the fixture (Float f32) and the served artifact (NVFP4 + FloatBf16) both
engage the batched arm.

Flag: `MEMRA_GLM5_VERIFY_BATCH` (default ON, read per call), the rollback seam — `0`
restores the per-row mixer walk byte-for-byte. Deliberate default (new-flags law): the
walk only exists behind `MEMRA_GLM5_SPEC` (default OFF in prod), per-row byte identity
is bit-gated on the rig before merge, and the box re-battery A/Bs the seam in one build.

## 3. Per-class decisions (batched bit-proven / kept-sequential named)

| class | decision | why |
|---|---|---|
| KDA projections + gates + conv + norms | BATCHED (bit-gated) | decode-exact matmul classes at m=t; conv prefill arm == decode arm per token (ascending taps, same window values); grid.y=t per-row kernels |
| KDA delta-rule recurrence | KEPT-SEQUENTIAL (inside ONE kernel launch) | true state chain; `memra_kda_scan_s128` loops T in-kernel over register state — the sequential floor, one launch instead of t |
| MLA projections + indexer + select | BATCHED (bit-gated) | per-query kernels; decode-exact matmul classes |
| MLA attention (gathered) | BATCHED (bit-gated) | per-(query,head) blocks over per-query causal idx lists; -1 pad arithmetic-invariant |
| MLA attention (absorbed, no indexer) | KEPT PER-ROW | glm5_next always carries the DSA indexer; the absorbed t>1 arm is unproven at this seam — refused into the per-row loop by name |
| lm head | BATCHED (bit-gated) | tcols twin, weight read once |
| MoE routed experts | UNCHANGED (already call-seam batched) | per-(token,expert) inner loop scales in rows: near-disjoint expert sets — the residency/fused-epilogue arc owns it |
| hc glue (pre_exact/post/expand/collapse), rms norms | UNCHANGED | already t-parallel and per-row exact (batched-decode gate); mHC residual/Sinkhorn/collapse chain is strictly PER TOKEN (hybrid_forward prime-split doc) so it batches freely |
| Full/Linear mixers | REFUSED at walk entry (pre-existing) | not a glm5_next class |

## 4. Expected shape of the win (rig-relative; the box re-battery prices it)

verify_row_marginal drops for every batched class: the KDA projection and lm-head byte
terms go flat in K (read once per round), the KDA/MLA launch trains collapse ~(K+1)-fold,
and the sequential floor becomes the KDA scan chain (34 in-kernel T-step launches' walls)
plus the MoE per-(token,expert) term. Predicted round wall:

    round(K) ~= fixed(draft ~8.6 + accept/roll/maint ~0.3)
              + KDA_chain(K+1)                  (scan in-kernel steps, small — box census
                                                 priced the whole kda family at 3.4% of a
                                                 prime; the projections are the cost, not
                                                 the scan)
              + batched_classes (~flat in K: KDA proj/conv ~5.2 + MLA ~2.3 + head 0.7
                                 + glue, plus their now-flat launch trains)
              + MoE (~linear: ~3.0 ms/row bytes + its launch share)

Target arithmetic (task charter): verify collapsing from (K+1) plain steps toward ~1.3
lands K=3 near ~16 ms/token (~63 tok/s, 1.8x plain). The honest post-restructure range
before box receipts: verify K=3 96.5 -> ~40-55 ms (bytes flat-side ~19 ms + MoE linear
~12 ms + residual launch structure), round ~50-64 ms at acc/cyc 2.907 -> ~45-58 tok/s.
K=1: verify 51.6 -> ~28-33 ms. The trace=2 receipts decide, not this table.

## 5. Instruments

`MEMRA_GLM5_SPEC_TRACE=2` (trace level extension, instrument-only): everything `=1`
emits, PLUS per-burst verify sub-shares — batched-class vs sequential-class time:
`[glm5-phase-v] rounds=N | vkda=… (scan=…) vmla=… vrest=…` where `scan` is the
sequential KDA chain inside the batched call, `vrest` = glue+FFN+head. Level 2 adds
per-layer stream syncs on top of level 1's phase syncs — shares, never walls (the
standing law). Level 1 output is unchanged and comparable with the cell-2 receipts.

## 6. Gates

Standing battery (all re-run on the restructured walk — `glm5_verify_rows` is the same
entry point, so gates 1-7 gate the new arm directly):
- `glm5_tparallel_verify_gpu`: walk bit-identity vs plain rows, accept-j rollback
  byte-identity j=0..K, stale-KDA red, pool-key red, e2e K=1..7 + forced rounds,
  rollback-disabled red, FR-Spec trim 7-arm.
- `glm5_spec_session_gpu`: served bursts, sampled twin, EOS, receipt red/green.
- `glm5-spec-ppn-gate` matrix: stages 2+3, splits/streams/overlap arms (run script in
  `../ppn-verify-20260830/run-spec-ppn-gate.sh`).

New in this lane:
- Flag A/B: walk with `MEMRA_GLM5_VERIFY_BATCH=0` vs `1` vs plain — all bit-identical.
- Kernel bit-gates (`glm5_verify_batch_gpu`): tcols twin vs t x rows-kernel calls (all
  t 2..8, tail rows); scan(T=t) vs chained scan(T=1); conv prefill-arm-vs-decode-chain
  at t>1; ring-roll replay vs sequential rolls.
- Red arms that break row isolation and must bite: pre-rolled conv ring (walk rows
  diverge from plain), post-round ssm state reinstated (continuation diverges — the
  existing gate-3 shape now covering the batched rollback), tcols with a shifted
  activation row (bit-gate fails).

Rig law: exactness only, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, never a
timing number.

## 7. Status log

- Lane open. Worktree `~/projects/wt-glm5-verify-batch` @ 34e0c0bf2. Attribution table
  written from banked receipts (§1) before any code change.
- BUILT (all of §2): `matvec_bf16_f32acc_x4_tcols` + launcher + `matmul_rows_exact`
  (qmatvec.cu, lib.rs); `KdaRowsStash`/`KdaStash` + the rows arm of `kda_core` +
  `kda_verify_rows_cached` + `kda_verify_rollback_rows` (kda.rs); `rows_exact` threading
  through `mla_attn_core`/`mla_kpool_indices_ex` + `mla_attn_cached_rows_exact`
  (hybrid_forward.rs); the walk restructure + `MEMRA_GLM5_VERIFY_BATCH` (per-call read)
  + batched-arm rollback dispatch + rows-exact head + trace level 2 `[glm5-phase-v]`
  (glm_spec.rs). ppN twin inherits the restructure structurally (`glm5_verify_range`
  shared); per-stage `Glm5VerifyPos` keeps the per-stage pos_d law.

## 8. Gate table (rig 5090, flock held, NVIDIA_TF32_OVERRIDE=0, exactness only)

| suite | result |
|---|---|
| `glm5_verify_batch_gpu` (NEW kernel bit-gates: tcols twin t=2..8 + ragged tail + shifted-row red; scan T=t vs chained T=1, readouts+state + swapped-row red; conv prefill-arm vs decode chain + re-roll every keep + corrupted-snapshot red) | 3/3 PASS, all reds bite |
| `glm5_tparallel_verify_gpu` (walk rows vs plain bitwise; accept-j j=0..7 byte identity; NEW flag A/B both arms + stash-kind wiring anchors; NEW corrupted-ring red on the batched arm; stale-KDA red; pool-key tripwire red; e2e K=1..7 + forced full-accept + forced-rejection sweep; rollback-disabled red; FR-Spec trim 7-arm) | 9/9 PASS |
| `glm5_spec_session_gpu` (served bursts: greedy K=1..7 across burst boundaries, forced-rejection j-sweep, sampled determinism + burst-split invariance, EOS, demote handoff, PMIN, receipt red/green) | 9/9 PASS |
| `glm5_dflash_session_gpu` (DFlash2 source over the batched walk: tape identity K=1..7, tap-shift red, selection matrix, K>block refusal, sampled continuity) | 10/10 PASS |
| `glm5-spec-ppn-gate` matrix (stages 2: even/split1/split3/streams0/overlap0; stages 3: even/asym/streams0) — batched arm ENGAGED under every split (receipt line in each log, `ppn-gate/*.log`) | 8/8 arms PASS |
| adjacent seams: `kda_fixture_gpu` 3/3, `mla_gpu_forward` 5/5, `glm5_kpool_indexer_gpu` 14/14, `hyper_connections_gpu` 6/6, `glm5_mtp_head_gpu` 5/5 | PASS |
| clippy (all targets) zero warnings, `cargo fmt` clean | PASS |
| `tools/local-ci.sh --perf` (kernel-check, run-gen argmax, run-spec K=1..8, gemma stream, verify-gate depth, decode-batch gate, spec-on-cache-hit qwen full matrix; perf cells vs rolling medians) | exit 0 — ALL GREEN, perf 0 fail 0 warn (absent-model cells SKIP, the rig's standing shape) |

Per-class decisions confirmed by the gates (per §3): KDA proj/conv/gates/norms BATCHED
(bit-proven); KDA recurrence KEPT-SEQUENTIAL inside one launch (scan-chain bit-gate); MLA
proj/select/attend BATCHED (bit-proven, indexer arm); no-indexer MLA KEPT PER-ROW (named);
lm head BATCHED (tcols bit-gate); MoE call-seam unchanged (per-(token,expert) loop named
out of scope); hc glue/norms unchanged (mHC per-token contract).

## 9. PUSHED — handoff to the box flip re-battery

Lane head `41fe867cc` (restructure `3f4accf13` + perf-ci rows) pushed to
`origin/lane/glm5-verify-batch`, 2026-08-31. No self-merge. `tools/local-ci.sh --perf`
ran TWICE green on this tree (correctness ALL GREEN both runs; qwen9b cell 139.10 /
138.68 tok/s [OK] vs rolling median).

The box flip re-battery (flip-battery cells 3+4 shape) re-runs against this head as a
separate window — NOT this lane's. What that window should carry:
- The pinned serving recipe unchanged (`MEMRA_BF16_MMV=1` is load-bearing: it is what
  puts the KDA projections and lm head on the tcols class).
- The `MEMRA_GLM5_VERIFY_BATCH` seam for the A/B (one build, both arms) and
  `MEMRA_GLM5_SPEC_TRACE=2` for a cell-2-shaped attribution with the
  `[glm5-phase-v]` sub-split (batched-class vs the in-kernel scan share vs vrest).
- The engagement receipt to grep at boot/first-round:
  `[glm5-spec] verify walk BATCHED per layer: ...`.
- Byte-identity re-gate first (cell-1 shape) — ANY divergence stops the window.
