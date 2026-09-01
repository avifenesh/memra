# Tensor-core MLA prefill for glm5_next (lever: 75.8% of the cold-prime wall)

Lane: `lane/glm5-mla-tc-prefill` (2026-08-30). Parent evidence:
`../launch-diet-20260830/WINDOW-20260830.md` §4 (the census that named the owner:
`memra_mla_attn_gathered_kernel` 3060 ms + `memra_mla_absorb_q_kernel` 979 ms +
`memra_mla_decompress_v_kernel` 960 ms = 4999 ms of a 6598 ms GPU-busy cold
4626-token prime; 139.1 / 44.5 / 43.6 ms per layer-chunk, 3 kernels x 11 MLA
layers x 2 chunks; ~2.5 TF/s effective on a card class carrying hundreds),
`../prefill-gap-20260829/PREFILL-GAP.md` §2.5 (the dual-form MLA law),
`../engine-survey-20260829/ENGINE-SURVEY.md` (how vLLM/SGLang serve this family).

Flag: `MEMRA_MLA_TC_PREFILL`, DEFAULT OFF, FLAGS.md row in the same commit.

## 1. The dense/selected split decision, and why it is not the sketch

The brief sketched the dual-form law's dense answer: DECOMPRESS/materialize K and V
from the latent per chunk, run Q@K^T and P@V as batched per-head GEMMs, keep the
gathered form for selected-sparse spans. Reading the kernel's actual gather pattern
changed the design, and the reasons are structural, not preferential:

1. **The DSA selection is per-query, SHARED ACROSS ALL 64 HEADS** (the indexer mixes
   heads with `weights_proj` BEFORE top-k; one `idx[t][width]` list per query,
   `mla_attn_gathered_kernel` reads `idx + i * n_slots` for every head of query i).
   In the ABSORBED form the latent row is one operand for all heads, so per query the
   score is `Q_lat[64, 512] @ gathered_rows[512, TK]` — a real MMA with m = 64 heads
   and ONE B operand per tile. MATERIALIZING K gives every head its OWN K plane
   (`K[t, h, 256]`), so the same gathered walk decomposes into per-(query, head)
   matvecs again — materialization DESTROYS the m axis the shared selection provides.
2. **There is no t x t dense attention above ~2051 tokens in this model.** DSA caps
   every query at `topk + tail` (2048 + 0..3) rows. At the census shape (4626 = 2313
   + 2313), chunk 2 is 100% budget-limited and chunk 1 is 2051 trivial + 262 limited.
   The dual-form law's premise (dense score FLOPs grow t x t, the 256-dim
   materialized form halves them) is cut off by the indexer: the absorbed form's 2.2x
   score-width penalty applies only to a topk-capped term, and it buys the m=64 MMA
   axis and the deletion of BOTH per-position absorb/decompress kernels from the
   attention loop.
3. **Trivial-selection queries need no separate dense program.** When
   `visible_pools <= select_k` the selector emits exactly the full causal prefix
   (`n_fin < select_k` clamps the rank; tail always appended), so the gathered TC
   kernel's walk IS the dense causal walk there — identity gather, same kernel, no
   split, no second program to gate. The -1 padding is TRAILING by the select
   kernels' emit order, so a dead first slot ends a query's walk (load-bearing for
   early prime queries whose lists are mostly pad).
4. **This is upstream's own answer for SPARSE prefill.** The "nobody runs absorbed
   MLA at prefill" law (PREFILL-GAP.md §2.5) is about DENSE prefill; FlashMLA's
   sparse (DSA) prefill kernels are the absorbed geometry (q 576/512 over a gathered
   kv plane, 640 TF/s H800 / 1450 TF/s B200) — the head axis as m, the index list as
   the gather. Design copied, no kernel code.

**So the split is: NO split.** One TC kernel (`fa_mla_gathered_bf16`) serves every
prefill query through its selection list; dense spans are the identity-gather case
of the same program. The gathered f32 kernel remains the fallback arm (flag off,
decline, d_rope != 0, kv_rank != 512, t < 16, portable-MMA archs) and the ONLY
decode arm.

Gather traffic note: per (query, head-band) CTA the gathered rows are re-read
(grid.y = 4 bands at 64 heads), but the bf16 latent plane is L2-class at serving
contexts — 4.7 MB at 4.6k, ~61 MB at 60k tokens (GB202 carries 128 MB L2) — and
adjacent queries' top-2048 lists overlap heavily.

## 2. What ships (all in one commit on this lane)

| piece | where | what |
|---|---|---|
| `fa_mla_gathered_bf16` | `cu/flash_attn.cu` | `fa_prefill_bf16_hd512_sp` body (validated m16n8k16 bf16 MMA primitives, split-K GEMM0, online exp2f softmax, f32 accumulate) with the axes recast: CTA = (query, 16-head band), K tiles gathered via `idx`, V aliases K (NoPE: the latent row is both), mask = `idx < 0` (causality and selection live in the list; the kernel must not re-derive either), trailing-pad early exit. smem ~51.3 KB dynamic. |
| `memra_bf16_gemm_sb` | `cu/f16_prefill.cu` | strided-batched bf16 cuBLASLt TN GEMM (batch = heads; per-head activation views expressed via ld + `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`; y f32 or bf16). Absorb: `q_lat[:,h,:] = q_nope[:,h,:] @ W_uk[h]^T` (m=t, n=512, k=256, ONE launch for all 64 heads). Decompress: `attn[:,h,:] = o_lat[:,h,:] @ W_uv[h]^T` (m=t, n=256, k=512). Plans cached per (shape, stride, dtype) tuple. |
| wrappers | `src/mla_ffi.rs`, `src/lib.rs` | `mla_bf16_gemm_sb_{bf16out,f32out}` (2xxxx no-heuristic = decline, else Err); `Engine::mla_attn_gathered_tc` launcher (refuses kv_rank != 512). |
| the door | `src/hybrid_forward.rs` `mla_attn_core` | flag read PER CALL; engages iff gathered selection && d_rope==0 && kv_rank==512 && t>=16 && !portable_mma; chain = f32→bf16 converts (weights per call, ~50 us class) + absorb sb-GEMM (bf16 out) + latent-window convert + TC attention + o_lat convert + decompress sb-GEMM (f32 out); cuBLASLt decline announces once per shape and falls back to the f32 kernels. Engagement: `[mla-tc-prefill] engaged` once per boot + `MLA_TC_PREFILL_DISPATCHES` counter at the invocation. |
| gate | `tests/mla_tc_prefill_gpu.rs` | six gates, below. |
| flag row | `docs/FLAGS.md` | same commit, per the new-flag law. |

Memory, stated: transients per (layer, chunk) at the census shape (t=2313,
t_kv=4626): bf16 q_lat 152 MB, bf16 latent window 4.7 MB, bf16 q_nope + o_lat
copies ~230 MB — freed with the call. The brief's materialized-K/V sketch would
have been 4096 x 64 x (256+256) x 2B = **256 MB/layer-chunk of K/V alone**, plus
the per-head-K program cost of §1.1.

## 3. Gate receipts — RUN GREEN 6/6 (rig 5090, TF32 off, debug, 2026-08-30)

Invocation: `NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock cargo test -p
memra-engine --test mla_tc_prefill_gpu -- --ignored --test-threads=1 --nocapture`
→ `test result: ok. 6 passed; 0 failed` (52.93 s). Band pinned **8e-3** (bf16
operands, f32 accumulate — the brief's "bf16 8e-3 class"; operand choice: bf16 is
the fa_prefill/MEMRA_PP_BF16 class already gated in-tree, and true-f32 has no
tensor cores on this card class).

**Kernel level, full GLM5_NEXT geometry (64 heads x 256 nope, rank 512), vs a CPU
gathered oracle AND the shipped f32 chain:**

| shape | budget-limited | tc-vs-cpu | tc-vs-f32 | f32-vs-cpu |
|---|---|---|---|---|
| t16 fresh, trivial (identity gather) | 0/16 | 2.289e-3 | 2.289e-3 | 3.429e-7 |
| t16 chunked (slot>0), sparse | 16/16 | 2.802e-3 | 2.802e-3 | 5.987e-7 |
| t64 fresh, sparse past q35 | 29/64 | 2.240e-3 | 2.240e-3 | 2.932e-7 |
| t64 chunked, all sparse | 64/64 | 3.228e-3 | 3.228e-3 | 6.308e-7 |
| **t4096, REAL 2048/4 budget, width 2051** | 2045/4096 | spot q1024: 3.229e-3, q4095: 3.045e-3 | **2.280e-3** | (f32 arm is the reference-anchored one at this size) |

**RED, four mutations** — each loads, runs, stays finite, silently wrong; the SAME
comparator that passes green at 3.314e-3 catches each (t=64 chunked, all-sparse
fixture, asserted non-vacuous):

| red | relative | vs band 8e-3 |
|---|---|---|
| W_uk/W_uv swapped (equal element counts — the checkpoint-mixup shape) | 1.388e0 | 173x |
| per-head TRANSPOSED W_uk (the Q@K GEMM's Q operand corrupted) | 3.392e-1 | 42x |
| **selection mask dropped** (full causal lists where DSA selected) | 8.002e-1 | 100x |
| **causal off by one** (one future row appended per list) | 2.247e-1 | 28x |

**Mixer level, the REAL door** (established kpool mini fixture family with
kv_lora_rank raised to the 512 stamp; `memra_reference::execute` truth; sparse from
~40 tokens by the kpool gate's own regime proof):

| T | OFF vs ref | ON: rows above 8e-3 | worst row | engagement |
|---|---|---|---|---|
| 16 | 4.453e-7 | 0/16 | 5.404e-3 | 2 dispatches (2 MLA layers) |
| 64 | 6.211e-7 | 1/64 | 4.061e-2 | 2 |
| 4096 | 9.647e-7 | 63/4096 (1.5%) | 2.181e-1 | 2 |

The above-band rows are the DISCRETE near-tie re-selection class (layer-2 indexer
pools + sigmoid router top-k are discrete functions of the bf16-perturbed layer-1
output — the MEMRA_PP_BF16 near-tie precedent), asserted by SIGNATURE: isolated
(<= t/16 rows, vs red c's 0.8-relative across the WHOLE batch) and bounded (3e-1
pinned, measured worst 2.181e-1 on this 2-layer hidden-128 micro model where one
flipped top-2-of-4 expert legitimately rewrites tenths of a token's row).
Correctness with pinned selections is carried by the kernel gates, every query
inside 8e-3.

**Decode byte-identity, gated not assumed:** two identical caches primed flag-OFF,
then 24 decode steps flag-ON vs flag-OFF: logits BIT-IDENTICAL at every step,
`MLA_TC_PREFILL_DISPATCHES` FLAT. Also t=8 (below the t>=16 door): bit-identical,
counter flat. The t=1 program never enters the door by construction AND by gate.

**OFF arm untouched:** pre-existing `mla_gpu_forward` (5/5) and
`glm5_kpool_indexer_gpu` (12/12) re-run green on the same checkout.

Public boundary from repo root: `677 matches (677 grandfathered, 0 new)`.

**Post-merge re-run:** the bringup lane advanced 22 commits (vision, T-parallel
verify, batched-decode consolidations) during this lane's work;
`origin/lane/glm53-flash-bringup` was merged in (413fc5b0a) and the FULL suite
re-ran green on the merged head: `mla_tc_prefill_gpu` 6/6 (53.96 s),
`mla_gpu_forward` 5/5, `glm5_kpool_indexer_gpu` 12/12, boundary still 0 new.

## 4. The box A/B plan (the flip condition; NOT this lane's call)

Hardware: the replacement 2-card box (windows through the coordinator). Recipe =
the adopted arm C (`MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_MOE_RESIDENT_GB=98
MEMRA_MOE_SLOTS=16 MEMRA_CTX=8192 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0
MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_FUSED_EPI=1`)
with the real NVFP4 artifact.

1. **Arms**: OFF = arm C as adopted; ON = arm C + `MEMRA_MLA_TC_PREFILL=1`.
   Interleaved x5 FRESH BOOTS per arm (interleaved-A/B protocol law; boot nonce +
   pgrep-clear arm identity per the 30k-cell lesson).
2. **Workload**: the banked real pool (`l3-ab/prompts.json`), cold primes at the
   census shape (~4.6k tokens) plus one longer prompt (~6.5k, two-to-three chunks).
   Metrics: TTFD (streamed), prefill tok/s, decode ms/token unchanged (decode does
   not enter the door — assert its rows match across arms within noise).
3. **Engagement receipts in BOTH arms**: ON must show `[mla-tc-prefill] engaged` +
   a dispatch count of 22/prime (11 layers x 2 chunks); OFF must show zero. A
   cuBLASLt DECLINE line in the ON arm invalidates the cell (fell back silently).
4. **Correctness on the box**: first-token argmax gate on the real prompts; any
   flip gets the 8-draw census (the near-tie adjudication shape from the
   MEMRA_PP_BF16 row) — owner accepts or holds.
5. **Serving law**: greedy is the instrument; the cell carries the vendor-default
   sampled twin (no sampling params request shape) with spec-engagement receipts
   before any serving-decision claim.
6. Owner accepts/holds the default flip on those receipts; the FLAGS row is
   updated in the same PR as any flip.

Prize arithmetic (unchanged from the census, restated): the 4999 ms census term at
GEMM class (~350 GFLOP/layer-chunk; 50-100+ TF/s effective vs today's 2.5) is a
~4-10 ms/layer-chunk attention term → TTFD at 4.6k from ~6.7 s toward ~2 s,
prefill from ~700 toward the 1,500-3,500 band, and the 90 s context ceiling from
~60k toward 250k+ tokens. Numbers become claims ONLY through the A/B above.

## 5. Named follow-ups (not blockers)

- Resident bf16 mirrors for wk_b/wv_b and an incrementally-appended bf16 latent
  twin (removes the per-call converts, ~50 us + ~250 us class per layer-chunk).
- An f16-P/V arm of `fa_mla_gathered_bf16` (the `MEMRA_FA_F16PV` class, ~2x MMA
  rate on this card family) — own numeric config, own battery.
- GLM-5.2 (d_rope 64) is OUT OF SCOPE by the door's `d_rope == 0` guard; a rope
  arm needs the k_pe plane threaded through the gather and its own gate.
