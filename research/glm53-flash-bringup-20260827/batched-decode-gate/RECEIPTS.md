# mHC batched decode — gate receipts (lane/glm53-batched-decode, 2026-08-28)

Lane goal: lift the batched-decode refusal for glm5_next. Every batched entry point in
`decode_batch.rs` called `refuse_hyper`, so GLM-5.3-Flash served SINGLE-STREAM ONLY at any
`MEMRA_MAX_SESSIONS` — the production blocker between the PP residency arc and concurrent
serving. `00-refusal-standing.log` is the gate run against that refusal: it stops before a
single comparison, which is what "the gate arrived before the code it holds down" looks like.

## What was lifted, and what the walk is

The serving chain `decode_step_batch` / `_sampled` / `_sampled_lean` / `_masked` now routes
`hyper.is_some()` to `decode_step_batch_hyper` (decode_batch.rs) over
`hyper_batch_range_decode` (hybrid_forward.rs):

- **Batched at m=B** where the arithmetic is row-independent: the hc glue (`expand`, the
  `pre_finish` kernel set, `post`, `collapse`) is block-per-token by construction; rms_norm
  is per-row; the MoE FFN batches whole (router = the fixed per-row program at
  t < PRIME_MIN_T, experts per-(token,expert), shexp on the per-column decode-exact arm) —
  the routed-expert weight stream amortizing across B rows is the win this walk exists for.
- **Per-session** where the state is: the KDA (conv ring + delta rule) and MLA+kpool
  (latent rows + indexer plane) mixers run each row through its OWN cache with the SAME t=1
  call its solo step makes, with a per-row single-position buffer. The dense FFN (layer 0)
  also runs per-row through the serial `hyper_ffn_branch`.
- **Decode-exact** where a reduction is width-dependent: the hc mixing GEMM runs per-row at
  m=1 (`hyper::pre_exact` — cuBLASLt's reduction split is n-dependent, the lt_ndep probe),
  and the lm_head is `matmul_decode_exact` at m=B.
- The tail is the SHARED `decode_batch_epilogue` (masks, device sampling, lean park, pos
  bump) — one serving contract, not a copy.
- Its OWN serial PP-N split (`decode_step_batch_hyper_ppn`): per-stage engines, per-stage
  pos uploads, `[B, streams, n_embd]` boundary payload, head + epilogue on the LAST stage,
  the #87 entry fence — mirroring `decode_step_hyper_ppn` and `decode_step_batch_ppn`.

## The truth chain (GATE:pin-against-truth) — cite all three together

| half | what anchors it | status on this tree |
|---|---|---|
| the serial hc walk vs an independent host executor | `tests/hyper_connections_gpu.rs` vs `memra_reference` | 6/6 PASS (`21-adjacent-hyper-reference.log`) |
| the SPLIT serial walk vs the unsplit walk | `glm5-hyper-ppn-gate` | 3/3 arms PASS (`20-adjacent-ppn-n2.log`; full matrix in `../ppn-hyper-gate/`) |
| the BATCHED walk vs the serial walk, per session, bit for bit | `glm5-hyper-batch-gate` | 7 knob arms x 3 sub-arms PASS (below) |

Batched-vs-serial alone is ARM-EQUALITY; the first row is what anchors the family to truth.
The two adjacent logs were RE-RUN on this tree because the lift refactored shared serial
code (`hyper::pre` -> `pre_finish` extraction).

## GREEN — the knob matrix

Rig 5090 under `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`. Rig law: exactness
only; no timing number is read out of any of these runs. Driver: `run-hyper-batch-gate.sh`.
Fixture: the `glm5-hyper-ppn-gate` mini glm5_next (4 mHC streams, mean collapse, KDA +
DSA(MLA+kpool) alternating, dense L0 + sigmoid noaux_tc MoE, F32 weights + Q8_0 expert
banks). Every arm: three sub-arms, all BIT-IDENTICAL on the full-logit bar —

1. **staggered-batch** — B sessions, DIFFERENT streams, DIFFERENT depths (prefix P+bi;
   the gate asserts pairwise-distinct positions), N concurrent ticks, each row compared
   against that session's ISOLATED serial tape;
2. **b1-class-pin** — B=1 through the batched body vs serial (one numeric class at every
   live width, the step35/Q35 class-crossing law);
3. **devsample-greedy** — the serving tick's shape; rows bit-identical AND every device
   token equals the reference argmax.

| log | knobs |
|---|---|
| `10-b3-default.log` | B=3, P=5, N=8 |
| `11-b8-wide.log` | B=8 |
| `12-b2-longer.log` | B=2, P=12, N=24 |
| `13-b3-ppn2.log` | stages=2 — reference AND batched both split |
| `14-b3-ppn2-streams0.log` | stages=2, `MEMRA_PP_STREAMS=0` (same-stream seam) |
| `15-b3-ppn4.log` | stages=4 (every fixture layer its own stage) |
| `16-b8-ppn2.log` | B=8 under the 2-stage split |
| `17-b12.log` | B=12 |
| `18-b15-cap.log` | B=15 — the DERIVED cap width |
| `19-b15-ppn2.log` | B=15 under the 2-stage split |

The `stages>1` arms open the door for the WHOLE process, so the only axis under test is
batching; split-vs-unsplit is the ppn gate's axis. All 10 arms exit 0; matrix verdict
"ALL ARMS PASS".

## The cap, DERIVED (owner challenge 2026-08-28: "why 8? with 4 cards we have headroom for 32-64")

The first cut inherited gemma4's 8. The owner was right to challenge it; the audit of every
term that grows with B (`hyper_batch_cap()`'s doc carries it verbatim) found exactly ONE
numeric-class knee, and it is neither memory nor the mixer loop:

| term | behavior as B grows | wall below 64? |
|---|---|---|
| per-session mixer loop (34 KDA + 11 MLA/kpool per session per layer) | step latency ~linear in B on that segment | NO — throughput term only; state is ~104 MB MLA latent/session at 8k ctx + trivial KDA, so memory never binds |
| hc glue (expand / pre_finish kernels / post / collapse) | block-per-token, grid.y chunked at 65535 (B=64 = 256 rows on hc_post) | NO |
| hc-mix GEMM | per-row m=1 by construction (`pre_exact`) | NO — width never enters |
| lm_head (`matmul_decode_exact`) | float: per-token m=1; quant: b-tier to m=16, grid.y=m mmvq above (per-row bit-identical; costs weight re-reads, not rounding) | NO |
| MoE router (`router_gemv`) + sigmoid top-k + routed experts | m-invariant router at every t under defaults; per-token selection and per-(token,expert) execution | NO |
| **MoE SHARED EXPERT trio** | `verify_t = t > 1 && t < PRIME_MIN_T` (hybrid_forward.rs): at t >= 16 gate/up/down cross from `matmul_decode_exact` onto the plain prefill matmul (cuBLASLt n-dependent for float; m>16 MMQ/GEMM block-scale for quant) | **YES — t=16 exactly** |
| PP boundary `[B, streams, n_embd]` + per-row pos H2Ds | 6.3 MB at B=64 on the real width; B tiny uploads/stage/step | NO |
| B-sized scratch | none fixed; all allocations b_n-scaled; slots lazily sized | NO |

So the exact tier is 1..=15 and the cap is `PRIME_MIN_T - 1 = 15`, one source of truth for
the engine refusal and the worker chunk policy (`HybridModel::hyper_batch_cap()`), env
narrows only. MEASURED, not argued:

| log | what it shows |
|---|---|
| `18-b15-cap.log`, `19-b15-ppn2.log` | B=15 bit-identical, unsplit and split |
| `30-overcap-b16-refused-EXPECTED-REFUSAL.log` | B=16 on the clean tree stops on the engine's named refusal (exit 1, zero batched comparisons) |
| `31-KNEE-b16-forced.log` | cap forced to 16 by a banked temporary edit: EVERY batched tick mismatches (128/128 devsample rows), first diff `ref=-0.901187` vs `batch=-0.90118694` — the low-order-bit cuBLASLt reduction class — while the greedy tape MATCHES even so. This is the silent class the refusal exists for, and it is why the full-logit bar is load-bearing |

The knee's signature (tape matches, logits differ in low bits) is cleanly distinguishable
from contamination (`92-`: tape diverges, 62/120 device tokens wrong) — the gate separates
the two failure classes it guards against.

Widening to 32/64 is a NAMED FOLLOW-UP, not a tuning flip: it needs a decode-exact shexp
arm for t >= 16, and that must not be flipped inside the shared `!prefill` branch — step35's
MoESD target forward (t up to 256) rides the same branch and its banked spec receipts pin
the current bytes. The follow-up needs its own seam plus a re-run of this gate at the new
widths.

## RED — the mutation check

A gate that has never failed is not a gate. Cross-session contamination is THE failure mode
of a batch walk (silent, fluent, per-customer corruption), so both sabotages break exactly
that seam. Each is a temporary uncommitted edit whose diff is banked at the head of its own
log, reverted immediately after; weight corruption is deliberately NOT in the set (it breaks
both arms equally and proves nothing about row/state routing).

| log | sabotage | result |
|---|---|---|
| `90-RED-m1-swapped-row.log` | SWAPPED H-ROW: row bi's mixer consumes session (bi+1)%B's normed branch input (B=3) | **staggered FAIL** (24/42 mismatched, tape diverged), **devsample FAIL** (24/24 + 14 device tokens wrong), **b1-class-pin PASS** — the mutation only exists at B>=2, so the multi-session compare is what binds, not a shared artifact |
| `91-RED-m2-wrong-cache-slot.log` | WRONG CACHE SLOT: row bi's mixer reads and WRITES session (bi+1)%B's recurrent/latent state (B=3) | same shape: **staggered FAIL / devsample FAIL / b1 PASS** |
| `92-RED-m1-swapped-row-b15.log` | M1 RE-PROVEN AT THE CAP WIDTH (B=15) after the cap derivation — contamination risk grows with B and must not be assumed from B=3 | **staggered FAIL** (120/300, tape diverged), **devsample FAIL** (120/120 + 62 device tokens wrong), **b1 PASS** |

Finding the mutations produced (the argument for running them): the first M1 run printed
"38/24 comparisons mismatched" — device-token mismatches were folded into the logit
counter, the SAME self-arithmetic class the ppn gate's mutations caught ("15/14"). Fixed
(`token_bad` its own field and verdict fragment) before the banked runs; the green matrix
was re-run on the clean tree after both reverts.

## Refusals left standing, each naming its missing piece (in the message itself)

- `decode_step_batch_sampled_lean_masked_pending` — the overlap scheduler's
  deferred-readback epilogue is unwired for mHC; the hyper walk keeps the synchronous
  readback contract. `decode_step_overlap_eligible` reports false for hyper trunks, so the
  worker never routes there.
- `decode_step_batch_sampled_lean_masked_scheduled` (dual-wave PP-2) — no dual-wave twin of
  `hyper_batch_range_decode`; mHC chunks are serial ticks, and the worker's chunk policy
  pins `dual=false` for this topology.
- `moesd_target_forward` — no `[B*gamma, streams, n_embd]` hyper rows-walk with causal
  per-session verify appends; mHC speculative verify is a separate lane.

Neither the pending nor the dual entry is on the glm5 serving path today.

## Serving wiring, and why the default is OFF

Worker carve-out mirrors gemma4's: `MEMRA_HYPER_BATCH=1` routes mHC sessions into batched
chunks (cap = `hyper_batch_cap()` = 15, the derived knee; `MEMRA_DECODE_BATCH_CAP` narrows
only; serial ticks only); unset/0 keeps the eager per-session decode. **DEFAULT OFF,
deliberately**: the rig is exactness-only by law, so the ON arm has NO serving-box
throughput receipt. Flip condition (FLAGS.md row): an interleaved x5 A/B sweeping c=1..32+
(the scheduler chunks c>15 into <=15 waves; high-concurrency aggregate is what the 4-card
box buys) on the box with the real GLM-5.3-Flash artifact under the deployed PP placement,
plus the vendor-default sampled twin and the 8-turn cache-on twin.

The `DECODE_BATCH` manifest table was left conservative ON PURPOSE: admitting this
topology's mixer ops (LatentMlaAttention/KimiDeltaNet/SparseIndex/LatentKvState) would
falsely flip serial-MLA plans (glm_dsa) to "batch supported" while the generic body has no
arm for their mixers. Consequence, stated: under a SEALED rewrite bundle the decode-batch
surface stays unqualified for hyper plans and `_schedule` falls back to receipt-backed
eager rows (correct output, no batching win). Qualifying a decode-batch surface for hyper
plans in the manifest is named follow-up work.

## Scope — what this does NOT establish

- F32 fixture weights, tiny widths, no quantized-expert throughput class, no timing claim.
- Same-device only: cross-device batched-hyper arms (`MEMRA_PP_DEVICES=0,1`) need a
  two-card box — ask the owner for box time; the 4-card vast box is running another lane.
- The lean logits park and grammar masks ride the shared `decode_batch_epilogue` (proven by
  the generic batteries); this gate pins the hyper TRUNK.
- `MEMRA_MOE_FUSED_EPI` stayed at its default (OFF) in every arm; a fused-epi-ON batched
  twin needs its own gate arm before that flag and this one may ship ON together.
- No real-artifact evidence: box validation (below) is the remaining bill.

## Remaining box validation (the real-artifact bill)

1. Real GLM-5.3-Flash NVFP4 artifact, deployed PP placement, `MEMRA_HYPER_BATCH=1`: served
   identity gate (batched c=2..15 vs isolated, byte-exact, real prompts) — the
   quantized-kernel twin of this fixture gate, INCLUDING the quantized knee probe: the
   fixture knee is the float cuBLASLt class, and the real artifact's t=16 crossing lands on
   the m>16 MMQ/GEMM class instead, which deserves its own banked first-diff.
2. Interleaved x5 aggregate A/B sweeping c=1..32+ (flag on vs off) + vendor-default sampled
   twin + the 8-turn larger-prompt cache-on twin — the flip evidence for the FLAGS.md
   default, at the concurrency the 4-card box is bought for.
3. Cross-device gate arms (`MEMRA_PP_DEVICES=0,1`, stages=2; shard on and off).
