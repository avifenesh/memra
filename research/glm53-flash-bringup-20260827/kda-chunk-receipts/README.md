# Chunked KDA prefill scan — gate receipts (L3 of the prefill-gap plan)

Lane: `lane/glm5-kda-chunk-scan`, branched from `lane/glm53-flash-bringup` @ `9e4b197bf4`.
Plan term: `../prefill-gap-20260829/PREFILL-GAP.md` 1.3 and L3 — the KDA prefill scan was
sequential over tokens by declared increment (`crates/memra-engine/src/kda.rs`); this lane
lands the chunked per-channel-Gcum twin behind `MEMRA_KDA_CHUNKED` (DEFAULT OFF; FLAGS.md row
in the same commit).

## What was built

- `cu/kda.cu` `memra_kda_chunk_{cumgate,attn,solve{,32,64},state,output}_f32`: the WY /
  per-channel cumulative-gate form of the KDA delta rule, derived from the banked
  `chunk_kimi_delta_attention` reference (`../modular_glm5_next-ref.py`), NOT a transcription
  of the GDN K1-K5 chain — KDA's decay is per channel, so Gcum is a `[T, qkv]` tensor, the
  pair-matrix decay sits INSIDE the d-reduction (the factored `k*exp(-Gcum)` form GDN uses
  would overflow at C >= 18), beta rides the ROW token in A and both K3 right-hand sides, and
  the inter-chunk output gate folds into the K5 q staging. Every exp() argument is <= 0.
  5 launches per layer call; K1-K3+K5 chunk-parallel, K4 sequential over chunks with the
  state in shared memory. Algebra verified symbolically against the sequential recurrence at
  C=1 and on the C=2 cross terms before any kernel ran.
- `crates/memra-engine/src/kda.rs`: `Engine::kda_scan_chunked` (orchestration),
  `Engine::kda_scan_prefill` (the dispatch seam — Prefill conv arm only, engages at
  `t >= MEMRA_KDA_CHUNK_MIN_T`, default 256), `MEMRA_KDA_DIFF=1` both-forms band oracle
  (prints error stats, keeps the sequential result). The Decode conv arm routes to
  `memra_kda_scan_s128` directly: decode and the spec verify are byte-untouched.

## Numeric class (measured, 5090, NVIDIA_TF32_OVERRIDE=0, 2026-08-29)

NOT bit-identical to the sequential scan (chunked FP accumulation order — the GDN A4
precedent). The bar is the calibrated scale-relative band `maxdiff <= 5e-5 * scale`
(kda_fixture_gpu.rs constant). Measured (this file's `gate-run.txt`):

- full mixer, chunked vs `memra_reference::kimi_delta_net_layer`, T in
  {63, 64, 65, 128, 130, 192}: worst rel **9.9e-7**
- scan level, chunked vs sequential on identical inputs with a NONZERO carried-in state,
  T in {63, 64, 65, 128, 145, 192}: worst out rel **5.0e-7**, worst state rel **1.4e-6**
- `MEMRA_KDA_DIFF` oracle through the full fixture mixer: out max_rel <= 5.8e-7,
  state max_rel <= 5.2e-6 (`diff-oracle-smoke.txt`)

All comfortably inside the 2e-5 class the chunked-prime precedent set; ~40x headroom under
the 5e-5 bar, an order of magnitude below the ~7e-4 TF32-on class.

The one BIT-identity claim the chunked form makes and proves: calls split at multiples of the
chunk size are bit-identical to one call (out + final state), which keeps the 4096-token prime
schedule bit-stable. Decode after a chunked prime is BYTE-identical between flag arms
(out + state + conv ring).

## Gate (crates/memra-engine/tests/kda_chunked_gpu.rs, 8 tests, all green)

Boundary crossings: 1 chunk (63, 64), 1 + remainder (65), exactly 2 (128), 2 + remainder
(130, 145), exactly 3 (192); stateful two-call primes (64+66, 65+63, 128+64) + decode
continuation vs full reference recompute.

RED ARMS (mutants that MUST exceed the band; all did, by 5+ orders):

| mutant | out rel | state rel |
|---|---|---|
| state not carried across a chunk boundary | 8.2e-1 | 3.9e-1 |
| gate cumulative product off by one (exclusive-cumsum K1) | 2.9e-1 | 1.8e-1 |
| decay applied twice at the boundary | 3.9e-1 | 2.4e-1 |

Green + red outputs banked in `gate-run.txt`. Flag-off regression: `kda_fixture_gpu.rs`
3/3 green (sequential path untouched); `glm5_chunked_prime_gpu` green (2 pass, 4 ignored
needing artifacts); `kda_quant_operand_gpu` unaffected (ignored without artifact).

## Threshold (MEMRA_KDA_CHUNK_MIN_T = 256, derived)

ARITHMETIC from receipts, not a guess: the chunked chain replaces one launch with five
(~4 x 6-8 us extra launch+gap, the serving box family's measured class,
`../decode-attribution-receipts/ATTRIBUTION.txt`), against the sequential scan's ~0.1-0.2
us/token serial dependent chain (expf + two warp reductions + state FMAs per token) —
break-even near t ~ 150-300 — and the chunk algebra needs t to span several chunks (serial
depth C + T/C vs T). 256 = 4 chunks at the default C=64. Spec-verify widths (K+1) and small
primes stay sequential by construction. The box knee sweep owns the final number
(`MEMRA_KDA_CHUNK_MIN_T` env exists for it).

## Cost note for the flip

Transient VRAM at t=4096, C=64, H=64 (ARITHMETIC): gcum 128 MB + A/P 134 MB + U/W/Y 402 MB +
Ssnap 268 MB ~= 0.93 GB per layer call, stream-pool transient. The 1M-context config ran
81.9-94.9 GB/96 GB per-card peaks (1m-demo lane) — re-check headroom there before flipping
this flag on that config, or drop C / sub-window the chain if it binds.

## Box A/B plan (the flip condition; flag stays OFF until this is green)

Same battery as the grouped-prefill flip, on the serving card class with the real NVFP4
artifact (NOT the rig — rig is exactness-only by law):

1. `MEMRA_KDA_DIFF=1` one cold prime: band stats on real weights at t=4096 (band receipt).
2. Interleaved x5 A/B, `MEMRA_KDA_CHUNKED=0` vs `=1`, cold 4.6k/6.5k-token primes:
   TTFD per arm; `../prefill-gap-20260829/profile-prime-phases.sh` first if the window allows,
   so the KDA-scan share of the wall is attributed before and after.
3. Vendor-default sampled twin (NO sampling params) with spec-engagement receipt from the
   server log — never a 200 alone (serving law).
4. First-token argmax gate on real prompts, both arms.
5. 8-draw census on the flip candidate.
6. Knee sweep over `MEMRA_KDA_CHUNK_MIN_T` (64/128/256/512) and `MEMRA_KDA_CHUNK` (32/64/128)
   if step 2 shows the lever pays.
Box window is owner-scheduled; this lane requests it rather than touching a busy box.
