# P1a — gemma4 text prime rides the generic chunked driver (memra#535, #534)

Lane `lane/exec-p1a-gemma-prime-20260919`. Rig: one rented RTX 5090 (32 GB, driver 595.71,
CUDA 13.1, sm_120a build), `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`. Exactness
only — no timing claims in this lane. Raw logs: `raw/`.

## What changed (engine)

- `prime_cache` (`hybrid_forward.rs`): text-only gemma no longer early-returns to the fresh-only
  `gemma4_prime`; it rides the generic chunked, continuation-capable driver. E4B (PLE) and the
  vision-overlay arm keep their fresh-only graphs, bytes unchanged.
- `prime_chunk`: gemma prologue (√n_embd embed scale), per-chunk layer stack `prime_layers_gemma`
  (attn_norm → attention → post_attention_norm → shared `gemma4_layer_tail_add`), epilogue
  softcap + suppress.
- `gemma4_attn_prime(.., view_attend)`: the one-program arm — every row, chunk 0 included,
  attends over the quantized planes it just appended (`sdpa_naive[_w]_quantized_view_fmt`; new
  format-named twins read e4m3 planes through the kf8vf8 module). Naive dequant-once kernels;
  the tuned view twins (hd256 windowed, hd512) are the follow-up before any default flip.

## Gates

| gate | artifact | result | raw |
|---|---|---|---|
| `tests/gemma4_chunked_prime_gpu.rs` (fixture: 1 sliding hd256 + 1 global hd512, window 64) | deterministic fixture, F32 weights | 4/4 GPU arms PASS | `raw/fixture-gate-default.log`, `raw/fixture-gate-nofa.log` |
| `tools/chunk-invariance-gate.sh --chunks 4096,1024,513,256,64 --steps 24` | gemma-4-12B QAT Q4_0, T=4882 | **CHUNK-INVARIANT**: prefill logits bit-identical at every chunk size, 24-step streams identical | `raw/12b/chunkinv-pp6257.log` |
| `concat-prime-probe tickinv --budgets 0,1024,256,64 --splits 64,256,512,1000` | same | **TICK-INVARIANT**: 5/20/77-call primes and off-grid resumes at 64/256/512/1000 all EXACT vs one call | `raw/12b/tickinv-pp6257.log` |
| `tools/argmax-margin-gate.sh` (prefill vs decode argmax, board-2048, window 12) | same, NEW binary | flips=0 bad=0 PASS | `raw/12b/argmax-new/` |
| same, OLD binary (P0 tip `c30b509b`, `gemma4_prime` path) | same | flips=0 bad=0 PASS — both programs decode-consistent on this prompt | `raw/12b/argmax-old/` |
| `chunk-invariance-gate.sh` (same chunks) | gemma-4-31B QAT Q4_0 (dense), T=4882 | **CHUNK-INVARIANT** | `raw/31b/chunkinv-pp6257.log` |
| `tickinv` (same budgets/splits) | gemma-4-31B | **TICK-INVARIANT**, 7/7 EXACT | `raw/31b/tickinv-pp6257.log` |
| `argmax-margin-gate.sh` | gemma-4-31B, NEW binary | flips=1 bad=0 PASS (within the calibrated 31B budget; every flip margin-explained) | `raw/31b/argmax-new/` |
| `chunk-invariance-gate.sh` (same chunks), FIRST pass (cuBLAS router) | gemma-4-26B-A4B QAT Q4_0 (parallel MoE), T=4882 | **CHUNK-DEPENDENT**: logits differ at EVERY chunk size, first divergence row 0, maxdiff 3.6–8.2 (O(1)); streams part at step 4–8 → root-caused and fixed below (#562) | `raw/26b-a4b/chunkinv-pp6257.log` |
| `chunk-invariance-gate.sh`, AFTER the router fix | gemma-4-26B-A4B | **CHUNK-INVARIANT**, EXACT at 4096/1024/513/256/64 | `raw/26b-a4b/routerfix/chunkinv.log` |
| `tickinv`, AFTER the router fix | gemma-4-26B-A4B | **TICK-INVARIANT**, 7/7 EXACT | `raw/26b-a4b/routerfix/tickinv.log` |
| `tickinv`, FIRST pass (cuBLAS router) | gemma-4-26B-A4B | **TICK-DEPENDENT**: 0/7 EXACT, row 0, maxdiff 4.9–8.2 | `raw/26b-a4b/tickinv-pp6257.log` |
| `argmax-margin-gate.sh` | gemma-4-26B-A4B, cuBLAS-router binary | flips=1 bad=0 (default budget 1) | `raw/26b-a4b/argmax-new/` |
| `argmax-margin-gate.sh`, AFTER the router fix | gemma-4-26B-A4B | flips=2 bad=0, both margin-explained; PASS under the calibrated 26B row (3/12), see below | `raw/26b-a4b/routerfix/argmax/`, `raw/fin/` |
| greedy A/B old vs new, 96 tokens, chat template, 3 prompts (60 / 60 / 4882 tokens) | same | p1, p2 (60 tokens, under the window): tokens and text IDENTICAL. p3 (4882 tokens, window live): identical for 10 generated tokens, then a near-tie flip ("technical description regarding the optimization of" vs "technical sentence regarding"); both continuations coherent and on-topic | `raw/12b/greedy/` |

## Finding → root cause → fix: the gemma MoE router (26B-A4B), memra#562

Row 0 diverging rules out a boundary-carry defect (row 0 sees no cross-chunk state); O(1)
movement at every chunk size means a **discrete decision** changed — expert selection. Attribution
(`raw/26b-a4b/diag/`): the dependence is unchanged with `MEMRA_GEMMA_MOE_MMA=0` (dp4a expert pairs
instead of the int8-MMA expert GEMM), so the expert kernels are not the mover. The mover is the
router: `gemma4_moe` computed prefill router logits with `e.matmul(&m.gate_inp, router_in, t)` —
cuBLAS, whose reduction order depends on m — so a token's top-k over 128 experts depended on the
chunk it was primed in; near-tie flips then amplified through 48 MoE layers. The serial trunk
closed exactly this class in lane/concat-prime-exact (`moe_router_logits`: "cuBLASLt's reduction
changes with m") by routing every t through the m-invariant `router_gemv`; the gemma arm never
needed it while its prime was monolithic.

**Fix** (this lane): `gemma4_moe` routes through `router_gemv` at every t. **Receipt**
(`raw/26b-a4b/routerfix/`, chunking allowed): chunkinv **CHUNK-INVARIANT** (4096/1024/513/256/64
EXACT, 24-step streams identical); tickinv **TICK-INVARIANT** (5/20/77 calls + off-grid resumes
64/256/512/1000 all EXACT). The registry row `GemmaParallelMoeResidual.chunked_prime` flips to
yes on that receipt; the 26B leaves the monolithic-prime class with the dense gemmas.

argmax-margin on the fixed 26B: flips=2, both margin-explained (margins 0.014 / 0.0135 vs config
spreads 0.64 / 3.02), decision position agreed — over the UNCALIBRATED default budget of 1. The
26B had no calibration row; its decode margins are identical across the two binaries (min 0.014,
p10 0.342, p50 0.695) and 6 of 12 / 5 of 12 positions are arithmetically flippable, the 31B's
coin distribution. A `gemma-4-26B*` row (budget 3 per 12-window, from those margins, the 31B
rule) is added to `tools/argmax-margin-gate.sh`; the pre-fix run (1 flip) was luck inside the
same distribution, not a better program.

Same class, not fixed here: `moe_ffn_lockstep` routes all lockstep-decode streams through one
cuBLAS call with m = stream count (CLI `run_lockstep` only; no serving caller) — tracked in #562.

## Measured numeric classes (fixture, `2394110f`)

- Semantics vs `memra_reference::execute`: the pre-change f32 attention path under
  `MEMRA_NOFA=1 MEMRA_FA_EMIT=0` matches at 1.2e-6 (T=40..200, windows 64 and 1024) — window
  rule, rope factors, residual order and softcap agree with the reference exactly.
- One-program arm vs reference = KV-plane quantization class: e4m3 planes (default) 1.1e-2 /
  4.2e-2 (T=200, window live) / 1.3e-2 (window inactive); q8_0 K + q5_1 V 1.9e-2 / 1.3e-2 /
  1.2e-2. The two rows move together (no ~1.5e-2 step) — plane error, not a window defect.
- Splits on the F32 fixture: many bitwise; the rest ≤3.6e-7 (logits) / ≤8.9e-6 (stack)
  relative — cuBLAS f32 GEMM m-selection, absent on MMQ weights (see the 12B rows: EXACT).

## Stated cost, for the owner's decision

Fresh-prompt gemma prefill now attends quantized K/V (as the serial trunk has since
2026-08-05) instead of exact bf16/f32 K/V. First-token logits move by the KV-quant class.
Measured on the 12B: prefill/decode argmax consistency unchanged (0 flips both binaries);
short-prompt greedy text identical; on the 4882-token prompt the two programs part at a near-tie
after 10 identical tokens, both coherent. In exchange the prime is chunkable and tick-splittable
with bit identity, which is what removes the whole-prompt tick. The default flips only on this
receipt plus the tuned view twins (naked prefill speed must not regress).

## Not done in this lane

- `moe_ffn_lockstep`'s cuBLAS router (lockstep decode, CLI-only) → `router_gemv` (#562).
- Tuned view twins for hd256-windowed and hd512; TTFT A/B before the default flip.
- Vision-overlay continuation (islands through the view path); E4B PLE prime (P3).
