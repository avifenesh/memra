# Lane: prefill-gap (2026-08-29)

**Branch**: `lane/prefill-gap` off `origin/lane/glm53-flash-bringup` (`9e4b197bf4`).
**Deliverable**: attribution + plan, not a patch. `PREFILL-GAP.md` is the document;
`profile-prime-phases.sh` is the first action of the next box window. This lane feeds
the next engineering arc; it changes no engine code.

## Question

glm5_next prefill measures ~80-83 tok/s on RTX PRO 6000 Blackwell (TTFD 57-79 s at
4.6-6.5k prompt tokens, ring-sizing ctxprobe 2026-08-29) against thousands to tens of
thousands tok/s for vLLM/SGLang/TRT-LLM/llama.cpp on comparable hardware and model
class. Attribute the 100x-class gap from source + banked receipts + what other engines
do, and rank the levers.

## Method

- Source audit of the actual prime path at `9e4b197bf4`: `prime_cache_hyper` ->
  `prime_chunk_hyper` -> per-layer mixers + `hyper_ffn_branch` -> `moe_ffn_inner`
  dispatch predicates, `Engine::matmul` prefill classes, `kda.rs` scan dispatch,
  `mla_attn_core`/kpool, `hyper.rs`. Every claim carries file:line.
- Receipts: `../moe-epilogue-receipts/nsys-launch-counts.md` (49 launches/token-layer),
  `../decode-attribution-receipts/ATTRIBUTION.txt` (the 17.1 ms invariant launch term,
  4.76 GB/token expert VRAM traffic), `../ring-sizing-20260828/box-ctxprobe/`
  (TTFD ladder), `../kpool-bench-Frankfurt-crossover.txt` (MLA/DSA is ms-class),
  `docs/FLAGS.md` rows `MEMRA_MOE_FUSED_EPI`, `MEMRA_PP_BF16`, `MEMRA_BF16_MMV`.
- Engine literature survey with per-mechanism citations (PREFILL-GAP.md §2).
- No box access this lane (the 4-card box runs the 1M cell). Where a receipt is
  missing, the number is marked ARITHMETIC and the profile script is named instead of
  guessing.

## Verdict (three causes, ranked)

1. **Per-token MoE dispatch at prefill** (~75-90% of the wall): every batched/grouped
   MoE arm is predicate-denied for sigmoid-router glm5_next, so a 4,096-token chunk
   runs 4,096 sequential per-expert matvec programs per MoE layer - ~8.4M launches and
   4.76 GB/token of expert weight re-reads per chunk. Prefill per-token cost equals
   decode's launch-structure term; measured TTFD is flat ~12.3 ms/token.
2. **BF16 trunk projections ride f32 non-tensor-core GEMMs at prefill** (all 34 KDA
   layers): the `MEMRA_PP_BF16` cliff, 15-20 TFLOP/s on a 250+ TFLOP/s card.
3. **KDA prefill scan is sequential over tokens** (deliberate increment; the chunked
   per-channel-decay twin is a named follow-up with its reference already banked).

MLA/DSA and mHC are chunk-wide and second-order (receipts in the doc).

## Plan handle

L1 grouped MoE prefill for sigmoid-router archs (the lever; ingredients exist: step37
NVFP4 grouped GEMM class + the fused-epilogue lane's sigmoid/pre-clamp/macro-fold
solutions + A2 grouping). L2 tensor-core BF16 trunk prefill. L3 chunked KDA scan.
L4 host-sync diet. L5 prefix-cache re-enable (product lever). Gates and agent-time in
PREFILL-GAP.md §3; the L1-vs-L3 sequencing check is the profile question in §4.

## State

- 2026-08-29: attribution complete, plan banked, profile script written. Lane closes
  with the document; the engineering arc it feeds gets its own lanes (L1 first).
