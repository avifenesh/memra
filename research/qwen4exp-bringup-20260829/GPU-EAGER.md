# qwen4_exp GPU EAGER arm — phase 7 (2026-08-29)

Branch `qwen4exp-bringup-20260829`; code `crates/memra-engine/src/qwen4exp_gpu.rs`,
kernels appended to `crates/memra-engine/cu/kernels.cu` (KERNELS.md rows in the same
commit), gate `crates/memra-engine/src/bin/qwen4exp_gpu_gate.rs`. Semantics source:
SEMANTICS.md; geometry: ARCH.md. Correctness class only — this rig's 5090 laptop ran the
gates serialized under `flock /tmp/memra-gpu.lock`; no timing numbers exist or are valid
from here (rig-GPU exactness-only law).

## What runs

Text-only single-request prefill + INCREMENTAL decode (no prompt recompute) of the full
qwen4_exp layer program, on GPU, gated against memra-reference:

- **Wide-stream entry/exit**: embed ×hc_count stream-major planes; exit through the
  global `hyper_connection_mixer` read gate (no final norm — census truth).
- **Gated residual** read/write around every mixer: grouped RMSNorm per stream slice,
  rank-320 sigmoid mix, per-stream `2*sigmoid(inject/S)` write gates. The `/streams` folds
  are pow2-exact. The read gate's 12 GEMVs are cuBLASLt f32 linears; its reduce/elementwise
  work is three fused kernels (`hc_lowrank_reduce_f32`, `hc_mix_epilogue_f32`,
  `hc_inject_gates_f32` — decode perf phase below), with the unfused composition from
  existing engine ops kept as `gate_read_legacy`, the A/B twin.
- **GDN layers**: fused qkv/z/beta/alpha projections, causal depthwise conv
  (`dwconv_causal_f32`, dilation 1, raw-qkv history rows), geometry-generic sequential
  delta-rule scan (`gdn_scan_naive_f32` — the s128 kernel cannot take the tiny 4/4
  geometry), gated RMSNorm with the family's **sigmoid** z-gate (`rms_norm` +
  `sigmoid`/`mul`; the silu arm stays available for qwen35-class plans).
- **QSA layers**: fused [q|gate] split (`q_gate_split`), q/k RMSNorm, partial rope
  (`rope_neox`, positions array), KV cache append (post-norm+rope keys), **dense
  attention under causal∧selection** (`sdpa_naive_mask_f32`) with the indexer selection
  computed exactly per reference as a HOST twin over the host-resident raw-key cache
  (pooling, k_layernorm, rope at block starts, relu-sum fp32 scores, pinned tie rule
  score-desc/index-asc, tail always visible), sigmoid fused output gate.
- **MoE**: device router GEMM, HOST softmax-top-10 routing twin (renorm floor
  6.1035156e-5, tie rule), sigmoid-gated shared expert (`sigmoid_dot_rows` +
  `add_scaled_rows`). Routed experts: per-expert gathered GEMMs +
  `scatter_slot`/`reduce_slots` for PREFILL (and any non-NVFP4 bank); single-token DECODE on
  NVFP4 banks runs the grouped path — `qmatvec_nvfp4_modelopt_sel_f32` once per projection
  over all selected experts, straight off the as-stored bank (perf phase below).
- **PLE block** (its GDN layer only): host n-gram hashing over the full token history
  (EOS-segment resets, checkpoint I64 buffers loaded never re-derived), host gather from
  the HOST-resident table → H2D [T, 16·160] → device key/value projections, grouped
  norms, signed-sqrt sigmoid gate scalars (host twin), dilated depthwise conv
  (`dwconv_causal_f32`, dilation = max_ngram, mode silu-add), per-stream normed-conv
  history cache.
- **State** (`Qwen4ExpState`): GDN raw-qkv conv rows [pad, conv_dim] + recurrent matrix
  [nv, hv, hk] (reference layout); QSA KV [cap, 2·256] + host indexer raw-key cache
  (128/token/QSA layer); PLE normed-conv history [(K−1)·dilation, H] per stream + full
  token history (host). Decode is one-token incremental; the indexer selection is
  recomputed per decode token over the cached raw keys.

## Loader wiring

`Qwen4ExpGpu::load_from_dir` / `read_checkpoint`: config.json → `ModelConfig::from_hf` →
qwen4_exp pack → plan → `compile_tensor_contract(HfSafetensors)` → per-requirement walk
over `StModel` bytes. Per row: name binding, shape/dtype refusal, transform application
through the shared `hf_mapping::TransformKind` implementations (GDN V-head reorders,
−exp(A_log), conv squeeze), and the **(1+w) fold** on every norm row EXCEPT
`linear_attn.norm` (qwen35 receipt, qwen.py:302-303; VERIFY marker on the indexer
layernorms — goldens lane pins them against transformers). Residency:

- n-gram table: HOST-resident bf16 (pure gather source; plain `Vec` for the eager arm —
  see deferred: pinned + prefetch).
- expert banks, per HALF (fused gate_up [E,2ff,H] / down [E,H,ff]): BF16 → dequantized
  f32 device-resident; **modelopt-NVFP4 stacked** → as-stored device residency (codes +
  e4m3 scales + pow2 macros, `find_nvfp4_stacked_native`'s validation math + the dsv4
  pow2-macro refusal), dequantized per ROUTED expert at forward time through the existing
  `memra_dsv4_nvfp4_deq_bf16` kernel + `bf16_to_f32` (exact chain for pow2 macros).
- MTP + vision tensors: contract-declared owners, NOT materialized (eager arm executes
  neither).
- `LoadedCheckpoint::into_reference_weights()` expands banks/table to host f32 so
  memra-reference executes the same checkpoint — tiny/sibling scale only.

### Expert dialects (RESOLVED 2026-08-29 — the mint answered the sibling VERIFY)

The real mint (Avifenesh/Qwen3.8-Flash-Next-NVFP4; census banked at
raw/nvfp4-census-names.tsv, 296,347 tensors incl. the BF16 mtp graft shard) does NOT
keep the fused suffix-less banks: modelopt UN-FUSES to per-expert 2D projections with
gate/up SPLIT — `model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.
{weight, weight_scale, weight_scale_2, input_scale}` (U8 codes [out,in/2], F8_E4M3
per-16 weight_scale, F32 scalar pow2 weight_scale_2, F32 scalar input_scale) — and
re-exports the n-gram table UNSHARDED as one `.weight` tensor [320001536, 160] (the
128→1 reshard was the 127-tensor delta in every pre-fixture count derivation).

Both dialects bind (pack `ExpertDialect { FusedBanks, PerExpertModelopt }` +
`tensor_contract_for`; per-expert rows are `TensorId::Expert` requirements with
`QuantConstraint::Nvfp4` and `[weight_scale, weight_scale_2, input_scale]` auxiliaries —
the dsv4 pattern). The loader PROBES the artifact (`experts.0.gate_proj.weight` present
⇒ per-expert) and assembles stacked device banks in numeric expert order (the census
map iterates `experts.10` before `experts.2` — an E=512-only trap the tiny gate cannot
see, guarded by keyed assembly). Census gate: the pack test
`nvfp4_census_matches_the_banked_mint_and_the_dialect_contract_binds_it` reproduces the
mint census EXACTLY from config math and binds the dialect contract against it.

### input_scale decision (owner-facing — SETTLED by owner order 2026-08-30)

`input_scale` is modelopt's STATIC ACTIVATION scale (W4A4/W4A8 operand). Representation:
a contract AUXILIARY of its expert weight row (the dsv4 precedent — "W4A8 activation
scale, unused for decode"), NOT a TensorId of its own. The loader VALIDATES it (F32
scalar, finite positive) and RECORDS the per-projection max on the bank source
(`BankTensorSrc::Nvfp4::act_scale`) — and that is where it ENDS. Round 4 built and
measured the W4A4 consumer; it moved decode argmax (22/24, KL to 1.65 — PROFILE-4
§W4A4) and the owner then RETIRED activation quantization as a serving lever entirely
(order 2026-08-30: it has hurt correctness across many past attempts and models). The
weight-only NVFP4 dequant-matvec shape with f32/bf16 activations is the correct serving
shape. No compute path consumes `input_scale`; no future lane re-proposes consuming it
without a fresh owner ruling.

## Gates (banked: gpu-eager/tiny-fixture-gate.tsv)

Policy: `max_abs<=0.01 max_rel<=0.01` + per-row argmax (the modelplan_reference_gate
class), f32 GPU vs f32 memra-reference. Token program: 18-token prompt with mid-prompt
EOS + 7 decode steps with a second EOS (PLE segment resets both modes); tiny indexer
budget (2 blocks) drops blocks at every position past 11; tie-free fixture weights
(dsv4-lane lesson). Every arm runs full prefill row-compare AND the cache-vs-full decode
invariance (prefill N, then M single-token steps vs the full-sequence reference rows).

```
binary: target/debug/qwen4exp_gpu_gate  (sha256 in the receipt header)
invocation: flock -w 600 /tmp/memra-gpu.lock ./target/debug/qwen4exp_gpu_gate \
    research/qwen4exp-bringup-20260829/gpu-eager/tiny-fixture-gate.tsv

qwen4exp-gpu-gate PASS [nvfp4-sel-matvec kernel oracle: worst abs 5.722e-6 rel 7.153e-7 over gate_up+down modes, NaN scales + non-pow2 macros + duplicate slots]
qwen4exp-gpu-gate PASS [fixture: prefill worst abs 5.083e-4 rel 5.083e-4; decode worst abs 1.158e-4 rel 1.158e-4]
qwen4exp-gpu-gate PASS [dir-bf16: prefill worst abs 3.478e-4 rel 3.478e-4; decode worst abs 3.262e-4 rel 3.262e-4]
qwen4exp-gpu-gate PASS [dir-nvfp4-stacked: prefill worst abs 5.236e-4 rel 5.236e-4; decode worst abs 2.485e-4 rel 2.485e-4]
qwen4exp-gpu-gate PASS [dir-nvfp4-perexpert: prefill worst abs 5.219e-4 rel 5.219e-4; decode worst abs 7.111e-5 rel 7.111e-5]
```

100 compared rows, 0 failures, argmax match on every row; reference row |logit| ≈ 0.47
(`ref_absmax` column), so the worst diffs sit ≈1e-3 of the logit scale — the
accumulation-order class (decode tighter than prefill, the opposite signature of a cache
bug). Arms:

- `fixture` — the pack tiny fixture through `from_reference_weights`.
- `dir-bf16` — a synthesized census-complete tiny BF16 safetensors dir through the FULL
  pack/plan/contract loader; the reference consumes the loader's own materialization, so
  the arm proves loader↔forward consistency end to end (transform/fold truth vs upstream
  = the tinyparity/goldens lane, running concurrently).
- `dir-nvfp4-stacked` — same dir with NVFP4 STACKED trunk gate_up banks (random valid
  codes/scales, 2^-5 macros): device kernel dequant vs the host `dequant_nvfp4_expert`
  decoder. The tiny down bank (ff=8) cannot carry per-16 scale groups — geometry, not
  policy; on the artifact (ff=640) every projection takes the same code path.
- `dir-nvfp4-perexpert` — the REAL mint's tensor-name shape from the PerExpertModelopt
  dialect contract: per-expert modelopt triplets + `input_scale` siblings + the
  UNSHARDED n-gram table (tiny down_proj derives BF16 per geometry). The name-set truth
  for the FULL config is separately census-gated in the pack against
  raw/nvfp4-census-names.tsv.

Real-checkpoint gates (Qwen/Qwen3.8-Flash-Next @ de4b8e4d, 360 GB) run on a fleet box —
this rig's 24 GB cannot hold the artifact; the tiny gate is the rig deliverable.

## Deliberately deferred (one line each, where it resumes)

- **Vision tower** — plan/census rows exist; resumes in a qwen4_exp vision lane (needs
  the 3-axis mrope answer, REUSE-MAP row "mrope").
- ~~**MTP execution** (draft/spec)~~ — DONE in the mtp-spec round (2026-08-30, below):
  loader + draft forward + K>1 chaining + the exact verify/accept loop + measurement,
  all receipts in spec/ + perf/PROFILE-5.md.
- **Batching > 1** — eager arm is single-request by scope; resumes in the serving-perf
  lane after checkpoint parity. ~~CUDA graphs~~ — DONE in perf round 2 (PROFILE-2.md):
  per-layer interior + MoE-tail + exit graphs over the step workspace; the host routing
  twin keeps a whole-step graph structurally impossible, so segments meet at the
  per-layer router dtoh.
- **Gather/compact QSA kernel** — eager runs dense masked attention (`sdpa_naive_mask`);
  the ≤2051-token gather form is the QSA perf lane. Long-ctx also needs the gmem-scores
  twin of the masked kernel (smem bound at T_kv>12288, same fix as sdpa_naive_gmem).
- **Device indexer selection** — the host twin is exact but O(T²) host math on long
  prefills; device pooling/scoring + host top-k resumes with the QSA perf lane. It now
  SKIPS scoring entirely while `n_complete_blocks <= budget_blocks` (a structural no-op:
  the top-k then keeps every complete block whatever the scores say), which covers every
  position below 2051 on real geometry — so the device kernel is a long-context question
  only. Measured 0.47 → 0.020 ms/token (PROFILE-1).
- **ngram table pinned-host + async prefetch** — eager gather is synchronous host math
  from a plain Vec; pinned allocation + prefetch-behind-decode is the PLE perf lane
  (SGLang serves it this way per SEMANTICS.md §MTP notes).
- **MoE grouped PREFILL** — still the per-expert loop with scatter/reduce slots (the DECODE
  path is now grouped, see the perf phase below); prefill's grouped executor resumes in the
  serving lane, where the operand is a token-batched GEMM rather than a matvec.
- ~~**W4A4/W4A8 expert kernels**~~ — CLOSED NEGATIVE, owner-retired 2026-08-30 (built
  and measured in round 4, moved decode argmax, then retired as a lever class — see the
  input_scale decision above and PROFILE-4 §W4A4). The arm stays W4A16-class dequant
  permanently; this line is a tombstone, not a deferral.
- ~~**Real-mint end-to-end load gate**~~ — DONE 2026-08-29 on the fleet box: BOTH real
  artifacts (BF16 export AND the NVFP4 mint) loaded through `load_from_dir` and gated
  against the transformers goldens (per-layer envelopes, greedy divergence, cross-arm
  KL). Receipts + findings (non-pow2 mint macros fixed; single-device + residency
  reality): REAL-CHECKPOINT-GATE.md.

## Decode perf phase (2026-08-29, perf/) — 12.74 → 34.67 tok/s

The eager arm's untuned decode was **78.5 ms/token**; it is now **28.8 ms/token
(34.67 tok/s)** on the same box, artifact and prompts, with every correctness gate
unchanged. Full tables: `perf/PROFILE-0.md` (before) and `perf/PROFILE-1.md` (after,
including the residual analysis and the next lane's ordering).

What the profile said, and it was NOT one hotspot: **15,308 kernel launches and 11,366
pooled allocations per token** (nsys, 8-step warm window) — the 27B launch-boundary lesson,
an order worse for this family. MoE was 54.6% of the attributed token and the single largest
slice was `moe.dequant` at 28.8%: 1,440 launches/token materializing f32 copies of
already-resident NVFP4 weights, per token, then discarding them.

Landed, each with an interleaved-×5 same-run A/B (ranges non-overlapping, identical rep-0
greedy chains) and both gate suites re-run:

| change | A/B (mean of 5 means) | win |
|---|---|---|
| (a) `qmatvec_nvfp4_modelopt_sel_f32` — one launch per projection over ALL routed experts, reading the as-stored modelopt bank (no repack, no dequant, no transient f32 weights); decode-only (t==1), NVFP4-only | 78.10 → 40.91 ms | **1.91×** |
| (c) fused read gate — `hc_lowrank_reduce_f32` + `hc_mix_epilogue_f32` (both bit-identical) + `hc_inject_gates_f32` (accumulation class), ~71 → 15 launches per gate | 40.41 → 28.84 ms | **1.40×** |
| (b) indexer structural no-op fast path | 0.47 → 0.020 ms/token on that section | not a rock; kept for long-context flatness |

Launches/token fell 15,308 → 2,932 (5.22×), allocations 11,366 → 2,234, memsets 1,685 → 0.

Both optimizations are **default ON** with the unoptimized path retained as the A/B twin and
the prefill/non-NVFP4 executor: `set_moe_sel_path` and `set_hc_fused_gate` (gate binary
`--ab-seam moe|hc`). Gates: the tiny four-arm gate gained **arm 0**, the grouped kernel's own
oracle against the host `dequant_nvfp4_expert` decoder chain (NaN scale bytes, non-pow2
macros, duplicate slots, both stride modes; worst rel 7.153e-7) — necessary because the tiny
geometry cannot reach that kernel (tiny down_proj is BF16 by geometry). The real-checkpoint
gate re-ran identical to its banked baseline: argmax 10/10, greedy divergences none/8/none/48,
envelope and KL matching to the 4th digit (the inject reduction-tree ULP class).

Instrumentation kept: `qwen4exp_gpu::prof` section timers (zero-cost off, sync-bounded on)
behind `--profile <n>`, and `--profiler-window` (cuProfilerStart/Stop around the warm decode
steps) for `nsys --capture-range=cudaProfilerApi` launch censuses.

## Decode perf round 2 (2026-08-29, perf/PROFILE-2.md) — 34.67 → 58.18 tok/s

Five seams, each default ON with its interleaved-×5 A/B banked and both gate suites
re-run against the banked baseline (argmax 10/10, greedy divergences unchanged):

| seam | flag | A/B |
|---|---|---|
| bf16 trunk residency (`qmatvec_bf16w_f32`, exact-widening + in_f%8 guards; f32 stays resident as fallback/twin) | `set_trunk_bf16` | 28.84 → 22.58 ms |
| named-slot step workspace (`StepPool` — address-stable transients, alloc churn gone) | `set_step_ws` | 22.59 → 22.29 |
| decode CUDA graphs (35 GDN-interior + 48 MoE-tail + 1 exit; host routing = per-layer boundary; QSA/PLE eager) | `set_decode_graphs` | 22.24 → 21.96 |
| grouped sel matvec v2 (uint4 loads, 2 rows/warp; v1 fallback per geometry) | `set_sel_v2` | 21.95 → 20.37 |
| read/write-gate micro bundle (batched plane norms over a device ptr table, two-stage inject, slab write, shared-expert bf16) | `set_hc_micro` | 20.38 → 17.07 |

Host launches/token 2,932 → 531 (84 graph replays); pooled allocs 2,234 → 39. The lane
also banked a NEW permanent tiny-gate arm, `gate_hc_micro_kernels` (real-geometry
micro-vs-classic oracle, streams 4 × hidden 2560) — born from the perf7 incident where a
block-reduce store bug was invisible at tiny geometry and corrupted the real model from
layer 0 (details + lesson in PROFILE-2.md). Residual physics and the TP2 projection
(~79-86 tok/s projected; NOT implemented) live in PROFILE-2.md §Residual/§TP2.

## Decode perf round 3 (2026-08-29, perf/PROFILE-3.md) — 58.18 → 70.34 tok/s (TP2)

Three single-card seams, each default ON with its interleaved-×5 A/B banked and both
gate suites re-run (combined battery identical to the banked baseline: argmax 10/10,
greedy divergences none/8/none/48):

| seam | flag | default | A/B |
|---|---|---|---|
| grouped sel matvec v3 (4 rows/warp sharing activation registers + u16 scale loads; v2/v1 fallbacks per geometry) | `set_sel_v3` | ON | 17.06 → 16.57 ms |
| GDN decode-step scan twin (`gdn_scan_step_f32` — grid (nv, hv), one state element/thread; naive stays prefill + tiny fallback) | `set_gdn_step` | ON | 16.59 → 15.60 |
| GDN norm+gate fusion (`rms_sigmul_f32` — BIT-IDENTICAL to the rms/sigmoid/mul chain) | `set_gdn_fuse` | ON | 16.65 → 16.52 |

Then **TP2** — two-card tensor-parallel decode over PCIe P2P (replicated residual,
GDN key-head-block / QSA head / MoE expert-id / shared-ff / vocab splits, 2 direct
joins per layer, per-rank segment graphs incl. a count-gated MoE tail): single 15.57 vs
TP2 **14.22 ms (70.34 tok/s)**, interleaved ×5, rep-0 chains identical; TP2-vs-single
logits gate 24/24 argmax, worst rel 3.296e-5, envelope byte-identical across all three
TP2 iterations. TP2 is a DEPLOYMENT OPT-IN (`--tp2` on the real gate; needs 2 cards +
P2P), not a lib default; its graphs ride `set_decode_graphs`. The eager-TP2 ladder,
the host-issue diagnosis (3,908 launches/token before graphs), the 256-token warm
receipts (TP2 67.30 tok/s, single 61.68), and the honest 90-tok/s gap analysis (W4A4
experts are the next real lever) live in perf/PROFILE-3.md. Two new permanent tiny-gate
arms landed on the way: arm 0d `gate_gdn_step_kernels` (real-geometry scan twin +
norm-fusion bit-assert) and the v3 modes in the sel-matvec oracle. **The 90 tok/s owner
target was NOT crossed: 14.22 vs the 11.1 needed (1.28× gap), receipts + residual math
in PROFILE-3.md §Residual.**

## Decode perf round 4 (2026-08-30, perf/PROFILE-4.md) — the 90 crossing attempt

Re-profile first (perf18): round 3 reproduced (TP2 14.2 / single 15.6); the sel slice
measured LATENCY-bound at ~27% of card bandwidth (not bytes-bound), the read gates a
7-launch serial chain. Four seams built + measured (interleaved ×5 per seam, tiny gate
per change, real gate re-runs), two more after the owner order:

| seam | default | A/B (single / TP2 route) | class |
|---|---|---|---|
| `set_proj_stack` (GDN 4→1, QSA 3→1, shared 2→1 stacked trunk launches; VRAM-neutral row-offset-view residency) | **ON** | 15.72→15.25 / rides | bit-identical |
| `set_hc_diet` (read gate 7→3 launches; oracle arm 0e) | **ON** | 15.69→15.32 / rides | accumulation |
| `set_sel_gufuse` (fused gate+up+silu sel matvec, f32 activations; oracle gufuse mode) | **ON** | 14.75→14.58 / 13.43→13.10 | bit-identical |
| `set_router_bf16` (router GEMV bf16 residency twin) | **ON** | 14.75→14.68 / 13.47→13.36 | accumulation (seam-gate 24/24 argmax, worst KL 0.00116) |
| W4A4 expert path | **OWNER-RETIRED 2026-08-30** | (measured 1.040× single / 12.95 TP2 before retirement) | REAL ERROR — decode argmax 22/24, KL to 1.65; kernels DELETED, `input_scale` recorded-only |
| QSA phase-1 graphs | **deleted** (negative/flat: single −0.4%, TP2 flat) | 15.64→15.71 / 13.61→13.61 | graph replay (chains identical) |

The round also added the `--seam-gate` decode-row instrument (OFF vs ON per-step logits
envelope + KL + argmax — prefill-shaped goldens cannot see t==1-only seams) and
`MEMRA_Q4E_SEAMS` (gates force not-yet-default seams for their correctness receipts).
Final numbers, 256-token receipts, the honest 90-verdict and residual: perf/PROFILE-4.md.

## MTP spec-decode round (2026-08-30, spec/MTP-SPEC.md + perf/PROFILE-5.md) — 69 → 119.5 tok/s

The deferred MTP execution landed as a full speculative-decode loop on the eager arm:

- **Load**: `LoadOptions::load_mtp` materializes mtp.* through the contract; the BF16
  graft's 512-expert bank goes DEVICE-resident bf16 (`BankHalf::DeviceBf16`, ~5 GB) —
  trunk NVFP4 + draft co-resident on ONE card at 95,283/97,887 MiB, plain decode
  unchanged (14.3-14.4 ms with the draft resident).
- **Draft**: `mtp_draft_forward` — fusion (fc_embedding(norm(embed)) broadcast +
  per-stream fc_hidden(FLAT norm_10240(wide))) + the one QSA+MoE layer on the draft's
  own cache/indexer (`pos_off=1`: draft row i ↔ target position i+1) + own mixer exit
  + shared lm_head; returns the post-layer wide CARRIER (the K>1 seed). Parity: tiny
  arms vs reference.mtp + the real `--draft-gate` vs the host MTP twin (20/20 argmax,
  KL 0.00000).
- **Verify**: exact chunks (`spec_arm` stash, 1<t<=k_cap) run EVERY row bit-identical
  to the t==1 decode program — the qwen38 t-parallel lesson re-derived per kernel:
  `qmatvec_bf16w_mt_f32` + hc-diet MT stages + the MoE verify-column merge
  (`set_verify_mt`, default ON, receipts in spec/) read each weight tile ONCE per
  chunk with per-(row,token) chains VERBATIM; GDN scans per column with the decode
  dispatch + per-column state snapshots; replay-free partial REWIND rebuilds GDN
  conv/state and PLE history from the stash. `--verify-bit-gate`: 24/24 rows
  bit-identical on the real checkpoint.
- **Loop**: `spec_generate` — batched accepted-token replay carries the next tip row,
  K-1 carrier-chained steps, one t=K+1 verify with device per-row argmax, greedy
  accept walk. `--spec-gate`: spec output BYTE-IDENTICAL to plain greedy on all 4 real
  prompts; the vendor-default SAMPLED probe engaged 54/58 rounds (serving law).
- **Headline** (mtp7, interleaved ×5, 256 tokens/arm, real prompt): plain 14.86 →
  **spec K=5 8.37 ms/token (119.50 tok/s, 1.78×)**; ladder knee K=5 (accept 0.84, mean
  accept len 5.12). The 90 tok/s line PROFILE-4 could not cross is crossed; **200 NOT
  crossed (1.6× gap)** — residual order (FR-Spec draft-head trim, draft/verify segment
  graphs, sel restructure, TP2 t-generic verify) in PROFILE-5 §Verdict. Negative
  receipts kept: sel warp packing (decode +0.75 ms, verify flat; spec/mtp6).

## The mtp9 round — the residual's top two levers, both NEGATIVE (perf/PROFILE-6.md)

- **FR-Spec draft-head trim** (`build_draft_trim`, default OFF): the draft scores a top-N
  own-gen rank subset while the target verify stays full-vocab, so exactness is untouched by
  construction. Ranks from the owner SXC pools + a composed real-shaped pack (chat template
  on, 97 prompts / 18 classes / 291 generations / 93,152 counted tokens → 5,538 distinct ids).
  Head cut **44.8×** (248,320 → 5,538 rows) and **e2e LOST 16.6%**: full 8.26 ms = 121.03
  tok/s accept 0.840 len 5.12 vs trim 9.91 ms = 100.93 accept 0.561 len 3.82. Every width
  1,024..5,538 loses, monotone in coverage; in-class held-out narrows it to 2.6% without
  crossing zero. Binding constraint is corpus SCALE — 5,538 ids is 2.2% of this vocab, and
  law 1's ≥4×-topN floor was met 4.2× over without catching it.
- **Verify scan-chain segment graphs** (`set_verify_graphs`, default OFF): the per-GDN-layer
  dwconv + t×(scan step + state snapshot) + conv roll, 576 launches/round and the only
  serially DEPENDENT all-device chain in the chunk, captured once per width. **FLAT** —
  eager 9.96 vs vgraph 9.97 ms, 0.9992. Launch issue is not this model's decode bottleneck at
  t=1 or t=K+1; PROFILE-5's "est. 1-3 ms/round" is retired with a receipt.
- **Exactness held everywhere**, which is the design claim confirmed: rep0 chains
  byte-identical across both A/Bs, five different d2t maps each reproduced the control chain,
  verify-bit 24/24, spec-gate 4/4, tiny arms byte-identical with `vgraph` on and off.
- **Shipped defaults unchanged and perf-neutral**: spec K=5 8.34 ms = 119.97 tok/s (mtp7:
  119.50), ladder knee still K=5 (121.55), sampled probe ENGAGED 54/58, tp2-gate 24/24 worst
  rel 3.018e-5, TP2 plain 12.6 ms (79.67).
- **Two findings that outrank the verdicts**: (1) every mtp2..mtp8 perf row shared one prompt
  file whose accept is 0.840 — chat-template renders accept 0.290-0.588, i.e. **55-96 tok/s,
  not 121**; (2) spec **cannot run prompts past ~400 tokens** at this residency (held-out
  spec-gate OOM at 495; corpus skipped 6 prompts of 502-724), so moving the draft bank to the
  idle card 1 is a PREREQUISITE for agentic-length spec, not an optimization.

## The mtp10 round — card-1 draft, the thinking-shape regression fixed (perf/PROFILE-7.md)

- **Card-1 draft placement** (`load_from_dir_dev1`, gate `--mtp-dev1`): draft block +
  5 GB bank + a same-bytes lm-head copy on card 1; wide seed rows cross P2P per round at
  **0.020-0.037 ms/round** (~0.05% of a round); draft-gate 20/20 ON CARD 1; the mtp9
  OOM set (502-724-token prompts) passes spec-gate 6/6 — the placement is the
  PREREQUISITE finding 2 named, and card 0 frees 5.2 GB.
- **The thinkon decay diagnosed** (`--spec-trace`, per-round fork margins + carrier
  drift + accept-vs-position, plain-twin byte identity hard-asserted): content class —
  think prose is branchy (target entropy 1.59 vs 0.91 nats at forks, 71% of misses are
  word starts, draft rank of the missed token median 2), transitioning 3.3 → ~2.0
  accept by position ~100 and plateauing. Carrier drift REFUTED (identical between
  accepting/rejecting rounds); indexer divergence dead by construction below pos 2051.
- **The fix — bounded admission, ported from the prior families** (MEMRA_SPEC_PMIN
  semantics incl. zero-draft rounds; the dflash accepted+1 window): at
  `adapt k_lo=1 + pmin 0.3`, interleaved ×5 over 256 tokens — thinkon (the DEFAULT
  render) **0.87× → 1.18×** (75.5 tok/s), efflow 0.93× → 1.22×, 724-token agentic
  0.97× → 1.22×, thinkoff 1.50× → 1.56×, raw 1.78× → 1.73× (bench-only shape). NO
  shape regresses vs plain; spec-gate green under every arm; sampled probe engaged per
  shape. Dyn-K decay built as the last-resort bound and UNUSED.
- **Round-cost identity** (owner direction): a K=5 round = 2.87 plain steps (verify
  t=6 36.5 ms = 24.5 GPU + ~12 per-layer host-twin bubbles; chain 6.68 head-dominated);
  the policy attacks the waste directly (chain → 2.09, verify → 22.7 ms/round mean on
  thinkon). Named follow-ups: device-side router/indexer twins (≤ ~1/3 of verify wall),
  mt-kernel bandwidth on the weight-shared dense sections.
- **FR-Spec retry at corpus scale (owner lever 2): trim stays OFF.** 405k own-gen
  tokens (355 prompts incl. 300 SXC, long prompts now included) discovered only 11,854
  distinct ids (4.8% of the vocab); trim A/B at that width: raw 0.882 (mtp9's −16.6%
  → −11.8%), thinkoff-ship 0.905, thinkon-ship 1.014. Discovery binds; a 32k set
  prices at ~4M tokens (~28 GPU-hours) — the revival condition, stated not hoped.

## YaRN long-context cell (2026-08-30, yarn/YARN-CELL.md)

The owner's affordability question ("full context on two cards?") answered with receipts,
plus the long-context execution the eager arm was missing:

- **YaRN wires up as a CONFIG FIELD** (`rope_parameters.rope_type=yarn` + factor +
  original) -> `RopeFactors::Yarn` on trunk QSA + MTP draft; the indexer shares the table.
  Default OFF (shipped config = `default` -> `RopeFactors::None`). Gates:
  factor-1.0 vs shipped **byte-identical on the real checkpoint** (max_abs 0.000e0, KL 0),
  fresh yarn-1M transformers goldens **argmax 10/10**, tiny yarn arms green. Divisor +
  mscale math pinned against the box's own transformers (yarn/transformers-yarn-params.tsv).
- **Verdict: 1M on two cards is not today, and YaRN is not the blocker** — residency is.
  1M f32 KV = 47.8 GiB vs 7.7 GiB free on card 0 after the trunk; KV parked on card 1
  MEASURED 462 ms/token at a 4k fill (18x the 25.7 ms local — PCIe gather). The route that
  fits is TP2 residency (trunk halves + LOCAL KV halves, ~68 GiB/card), and **TP2 prefill
  is not implemented** — the one named piece.
- **Measured today: 100,000 tokens at 35.6 tok/s**, decode near-flat with depth (22.3 ms at
  4k -> 28.1 ms at 100k; x3 rounds, spreads <= 0.40%, no escalation), prefill 8.4 min to
  100k, single-card ceiling ~100-130k on f32 KV.
- **QSA bounded attention receipted**: `qsa.sdpa` 7.6 ms/token at 4k AND at 100k — flat,
  once attention reads only the <= 2052 selected rows (`sdpa_blocklist_f32`; the dense mask
  READ every t_kv row and refused past 12288).
- **Deferred item "Device indexer selection" CLOSED**: the host twin was 52% of the token
  at a 32k fill (29.3 ms) and quadratic across a long prefill; `qsa_index_score_f32` (BIT
  identical scores — explicit non-FMA ops after a 1-ULP catch) took 32k decode 52.8 -> 24.1
  ms and leaves 2.2 ms at 100k.
- Also landed: chunked prefill (`prefill_extend`, head-skip + last-row head) riding the
  GROUPED MoE program (a 4096-token fill went 1007 s -> 18.3 s), grouped-MoE slot
  sub-batching (grid.y caps at 65,535), pooled indexer-key cache + parallel selection,
  chunk-bounded state reserve, optional KV card placement, and the spec long-context
  machinery (ring-bounded wide stash + chunked co-prefill, `mtp-spec-ring` byte-identity;
  spec AT DEPTH not measured — the draft wants the card the KV needs, so it belongs with
  the TP2 work, stated).
- New permanent tiny-gate arms: 0f `gate_sdpa_blocklist`, 0g `gate_qsa_index_score`,
  `fixture-yarn`, `yarn-identity`, `fixture-longatt`, `prefill-extend`, `mtp-spec-ring`
  (battery: 24 summaries, failures=0). FLAGS rows: `set_longatt` (AUTO), `set_idx_dev` (ON).

## Phase log

- 2026-08-31 mtp11 round (owner-ordered spec.rs slice-2 loop port): the deferred round
  readback built behind default-OFF seams (device-chained draft + card-1 chain-embed
  table proven bf16 bit-clean + one chain drain + both guard arms + the defer-ab
  x3+escalation harness) and measured FLAT-TO-NEGATIVE on this family — the 0.67 ms
  K=1-class win does not reproduce (~0.02 ms/round) because the router/indexer host
  twins inside each draft step already serialize the host; both seams stay OFF with
  the re-measure baseline banked. THE HEADLINE WAS A CORRECTNESS CATCH: the round's
  256-token spec-gate exposed a LATENT mtp10-era byte-identity defect (a prompt
  shorter than k_cap prefilled through the per-row decode programs instead of FUSED;
  gen-157 thin-margin flip; reproduces at the mtp10-close commit whose gates ran 64
  tokens) — fixed (exact chunks require base_pos > 0), plus the graphs-tail
  wide-capture skip fixed, three new gates grown (rewind-bit, rewind-bit-replay,
  armed-prefill-bit), and all identity gates green at 256 tokens. Receipts:
  spec/mtp11/, perf/PROFILE-8.md; box ~/realgate/mtp11.
- 2026-08-30 mtp10 round: the mtp9 thinkon regression (0.87×) closed at **1.18×** with
  a self-keying bounded admission policy (p-min guard + accepted+1 window, both ported
  from prior-family receipts), the draft moved to the idle card 1 (crossing ~22
  µs/round; agentic-length prompts unlocked, 6/6 byte identity), the decay attributed
  to think-prose branchiness with trace receipts (carrier/indexer hypotheses refuted),
  the verify decomposed (24.5 GPU + 12 host-twin ms/round), and the FR-Spec trim
  re-refuted at 4.3× corpus scale with its revival priced. All rule gates + spec-gate
  byte identity green at the merged tip (35a0b4c98). Receipts: spec/mtp10,
  perf/PROFILE-7.md; box ~/realgate/mtp10.
- 2026-08-30 mtp9 round: both PROFILE-5 residual levers built, gated and measured NEGATIVE
  (trim −16.6%, verify graphs 0.999×), both defaults confirmed OFF by their own receipts; all
  three rule gates green at HEAD; the two findings above reframe the lane's headline and its
  residency answer. Receipts: spec/mtp9, perf/PROFILE-6.md; box ~/realgate/mtp9.
- 2026-08-30 mtp-spec round: the deferred MTP execution shipped end to end (section
  above): 15-arm tiny gate (draft parity, spec byte-identity, rewind invariance, mt
  bit-identity oracles), real-checkpoint draft gate + verify-bit gate + spec gate +
  sampled engagement probe all GREEN; plain 14.86 → spec K=5 8.37 ms/token (119.5
  tok/s) interleaved ×5 over 256 tokens/arm. Receipts: spec/mtp1..7,
  perf/PROFILE-5.md; box ~/realgate/mtp1..7.
- 2026-08-29 phase 7: eager kernels (sdpa_naive_mask_f32, gdn_scan_naive_f32,
  dwconv_causal_f32) + qwen4exp_gpu module + dir loader + three-arm gate GREEN on the
  rig (serialized). Receipt: gpu-eager/tiny-fixture-gate.tsv.
- 2026-08-29 phase 7b (mint dialect): NVFP4 mint census banked
  (raw/nvfp4-census-names.tsv) and answered the sibling VERIFY — per-expert un-fused
  layout + unsharded n-gram table. Pack ExpertDialect + dialect contract census-gated
  on the real name set; loader probes and binds both dialects; gate arm
  dir-nvfp4-perexpert GREEN (receipt re-banked, 100 rows).
- 2026-08-29 real-checkpoint gate (fleet box, 2× RTX PRO 6000 96 GB): the engine's
  FIRST run of the real 360 GB artifacts. Both dialects loaded and gated vs
  transformers goldens: logits argmax 10/10 on both arms, greedy 64/64 on 2 chains,
  smooth per-layer envelopes; NVFP4 fits one card (81.9 GiB), BF16 gated via
  host-resident banks (LoadOptions::host_bf16_banks). Finding fixed in-lane: the mint's
  amax-derived NON-pow2 `weight_scale_2` → macro folds post-upcast in f32
  (dequant chain now host-decoder-exact for any finite macro; tiny arm D re-gated on
  the non-pow2 class). Full receipts: REAL-CHECKPOINT-GATE.md +
  gpu-eager/real-checkpoint/.
- 2026-08-29 decode perf phase (same fleet box, single card): profiled the eager decode
  step per section + nsys launch census, then attacked in measured order. 78.5 → 28.8
  ms/token (12.74 → 34.67 tok/s, 2.72×) via the grouped NVFP4 selected-experts matvec
  (1.91×) and the fused hyper-connection read gate (1.40×); launches/token 15,308 →
  2,932. Every gate re-run green and matching the banked baseline; tiny gate gained the
  grouped kernel's own oracle arm. Receipts: perf/PROFILE-0.md, perf/PROFILE-1.md,
  perf/ab-{moe,hc}-nvfp4.tsv, perf/profile{0,0a,1}-nvfp4.tsv, perf/*nsys*.csv.
  Residual for the next lane (PROFILE-1 §Residual): CUDA graphs (the read gate is still
  ~49% of remaining launches), then bf16/quantized trunk weights (gdn.proj is at ~1.3
  TB/s — memory bound, no launch fix helps), then the sel-kernel dp4a port, then TP2.
- 2026-08-29 decode perf round 2 (same fleet box, single card): 28.8 → 17.2 ms/token
  (34.67 → 58.18 tok/s, cumulative 4.56× from PROFILE-0) via bf16 trunk residency
  (1.28×), the step workspace + decode CUDA graphs (address-stable slots, 84 replays/
  token, host routing keeps the per-layer boundary), the sel matvec v2 (1.08×), and the
  read/write-gate micro bundle + shared-expert bf16 (1.19×). Every seam default ON with
  interleaved-×5 A/B receipts; real gate re-run matches the banked baseline (argmax
  10/10, greedy divergences none/8/none/48). One incident: the micro bundle's stage-1
  block-reduce store bug shipped tiny-green and broke real prefill from layer 0 —
  caught by the new real-geometry oracle arm (gate_hc_micro_kernels), fixed, re-gated.
  Receipts: perf/PROFILE-2.md + ab-*/profile*/perf8 rows. Residual: sel v3, gdn scan
  step twin, small-norm batching, quantized lm_head, W4A4 experts, TP2 (projection
  banked, not implemented).
- 2026-08-30 decode perf round 4 (same fleet box, both cards): 15.6 → 14.5 ms/token
  single-card and 14.22 → **12.9 ms/token TP2 (77.27 tok/s; 256-token warm 13.7 /
  73.19)** via four default-ON seams (proj stack 1.031×, hc diet 1.024×, gufuse
  1.012×/1.025×, router bf16 1.005×/1.008× — every A/B rep-0-chain-identical; tiny gate
  grew arm 0e gate_hc_diet_kernels + the sel oracle's gufuse bit-identity mode).
  90 tok/s NOT crossed (1.16× gap). W4A4 activation quantization measured 22/24 decode
  argmax with KL to 1.65 and was then OWNER-RETIRED as a lever class (order 2026-08-30):
  kernels deleted, `input_scale` recorded-only, never re-propose. QSA phase-1 graphs
  measured negative (single) / flat (TP2) and were deleted per the flags doctrine.
  New instruments: --seam-gate (decode-row OFF-vs-ON envelope — the W4A4 catch),
  --tp2 --ab-seam (TP2-route interleaved A/B), MEMRA_Q4E_SEAMS. Receipts:
  perf/PROFILE-4.md + ab-{projstack,hcdiet,w4a4,qsagraph,gufuse,routerb16}-*.tsv +
  seam-gate-*.tsv + hidden-gate-nvfp4-*{final,256}*.tsv + nsys-tp2final_*.csv, box
  ~/realgate/perf18..25.
- 2026-08-29 decode perf round 3 (same fleet box, both cards): 17.2 → 15.6 ms/token
  single-card (sel v3 1.030×, gdn step twin 1.063×, gdn norm fusion 1.008× — all
  default ON with interleaved-×5 receipts, chains identical, full battery matching the
  banked baseline), then TP2 → **14.22 ms/token (70.34 tok/s)**: replicated-residual
  two-card split with 2 direct joins/layer and per-rank segment graphs (eager TP2
  LOST at 16.45 — 3,908 launches/token of host issue — the graphs flipped it; the
  variable expert split graphs via a device pack blob + count-gated kernel twins).
  TP2 gate: 24/24 argmax vs the single-card twin, worst rel 3.296e-5, byte-identical
  envelope across all three TP2 iterations; state migrates single→TP2 one-way at the
  first TP2 decode. New tiny-gate arm 0d (gdn step twin + norm-fusion bit-assert);
  sel oracle gained v1/v2/v3 modes. 90 tok/s NOT crossed (1.28× gap); the honest
  residual + what W4A4/quantized-lm_head/QSA-graphs would buy: perf/PROFILE-3.md.
  Receipts: perf/PROFILE-3.md + ab-{selv3,gdnstep,gdnfuse,tp2*}-nvfp4.tsv +
  tp2-gate*.tsv + hidden-gate-nvfp4-{tp2,single}-256.tsv + nsys-tp2*.csv, box
  ~/realgate/perf9..17.
