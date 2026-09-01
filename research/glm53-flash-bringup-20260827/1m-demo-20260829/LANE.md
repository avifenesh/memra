# lane/glm53-1m-demo: the 1,048,576-token context demonstration, and the prefill gap

Owner directive: "we dont need 262 we need 1m" / "the 1m is a must be served."
Box: rented 4x RTX PRO 6000 Blackwell 96 GB (vast, exclusive; IP stays out of this repo),
1007 GB RAM. Artifact: the bit-verified GLM-5.3-Flash NVFP4 mint. Binary: built in a
detached worktree from the bringup head 9e4b197bf4 plus this lane's commits, identified by
strings census + newer-than-sources assertion every boot (00-verify-binary.txt; two
wrong-binary handoffs that week made provenance inadmissible).

Raw logs live in the box lane dir `lane-1mdemo-vast-20260829/` and are mirrored into this
directory as numbered receipts (a spot box closes and dangling pointers die with it).

## What was demonstrated

RESULT ROWS (all on the serving surface, streamed, MEMRA_PREFIX_CACHE_MB=0,
reasoning_effort pinned "low", real Gutenberg prose corpus sha-banked, token counts from
the server's own usage):

| rung | prompt_tokens | prefill | prefill tok/s | decode tok/s (greedy) | receipt |
|---|---|---|---|---|---|
| 6.5k | 6,466 | 45.2 s | 143.1 | 24.8 | phase3-C-6k.json |
| 16k | 15,766 | 91.3 s | 172.7 | 24.5 | phase3-R16K.json |
| 131k | 128,566 | 749.7 s | 171.5 | 22.8 | phase3-R131K.json |
| 262k | 257,775 | 1,519.9 s | 169.6 | 21.1 | phase3-R262K.json |
| 524k | 525,616 | 3,169.8 s | 165.8 | 18.9 | phase4-R524K.json |
| **1M greedy** | **1,035,357** | **6,419.8 s (107.0 min)** | **161.28** | **16.0 (steady p50 15.67)** | phase7-R1M.json |
| **1M vendor-default sampled** | **1,035,357** | **6,420.3 s** | **161.26** | **15.9 (steady p50 15.58)** | phase7-R1M-V.json |

THE DEMONSTRATION LANDED: a real 1,035,357-token prompt (Gutenberg prose, sha-banked,
token count from the server's own usage, cached_tokens=0) primed through the serving
surface on the PP4 placement inside the model's 1,048,576 window, then decoded greedy to
EOS (88 tokens) with a coherent CROSS-BOOK answer (it names War and Peace AND The Count of
Monte Cristo and draws the Pierre/Dantes parallel — content spanning the whole 4.2 MB
corpus). The vendor-default sampled twin (a request with NO sampling params) primed the
same prompt at 161.26 tok/s — the two independent full primes agree to 0.01%, a clean
clock — and answered coherently (fate/vengeance across both novels). Error census of the
successful boot's serve log: 0.

Chunked prefill throughput is DEPTH-FLAT to 1M (172.7 at 16k -> 161.3 tok/s at 1.035M,
-6.6%); greedy decode decays smoothly with plane depth (24.5 at 16k -> 15.7 at 1.035M),
which is the kpool-indexer-plus-latent-attention cost curve the serving story needs:

  decode tok/s vs depth (greedy, steady): 27.5 @1k | 24.5 @16k | 22.8 @129k | 21.1 @258k
  | 18.9 @526k | 15.7 @1.035M — a 1.75x decay over three orders of magnitude of depth,
  no cliff, sampled twin within 0.6% of greedy at 1M (15.58).

Output sanity held at every depth: at 131k the greedy answer cited Rousseau's Contrat
Social from the salon debate inside War and Peace, a genuinely deep in-context retrieval,
not a template answer.

## The walls found and removed on the way (each its own receipt)

1. **90 s platform deadline** (TIMEOUT_MS_MAX, main.rs): binds the SERVING surface, and in
   streaming it bounds first-token time, so every rung past ~13k tokens was cancelled
   mid-prime in the ring lane's ladder. Cell bypass: `MEMRA_TIMEOUT_MS_MAX` measurement
   override (FLAGS.md row in the same commit; never for the fronted product route).
2. **Monolithic ppN prime** (the door PP4 opens): the single-engine walk chunks via
   `hyper_prime_ranges` since 08-28, but the ppN twin primed the WHOLE prompt as one
   staged call. Two independent caps measured on this box: per-call transients
   CUDA_ERROR_OUT_OF_MEMORY from ~32k prompt tokens (phase2b-32k/64k/66k.json), and the
   CUDA gridDim.y = 65,535 ceiling — every launch placing t in grid.y (kda_conv_silu,
   kda_gate, per-row rms_norm, the router) — CUDA_ERROR_INVALID_VALUE, instantly, at a
   128,566-token prime (04-phase2.txt). Fix in this lane: the ppN prime walks the SAME
   chunk schedule (93927b1fac), gated by glm5-hyper-ppn-gate at stages=4 cross-device with
   the chunk loop engaged (07-phase3.txt: BIT-IDENTICAL, all three arms, both chunkings).
3. **Placement identity closed**: phase 1's door-off vs PP4 6.5k divergence was the
   monolithic-vs-chunked schedule mismatch (the documented cuBLASLt m-shape near-tie
   class), NOT a PP seam: after the fix, door-off-chunked vs PP4-chunked greedy output is
   BYTE-IDENTICAL at 6.5k (07-phase3.txt).
4. **The expert SLRU arena auto-cap**: the arena grows on demand toward 0.85 of free VRAM
   per device, on top of host-pinned staging. It is also what made phases 3-5 fast: at
   4096-token chunks each stage's full expert working set gets admitted once and stays,
   i.e. the arena IS the de-facto device residency. But the auto cap left the last-stage
   card (which always carries the worker's primary engine, the f32 output head, the
   whole-prime hidden-stack aggregation — 17.1 GB at 1M — and its stage's planes) with no
   room for the 1M request's upfront allocations: instant OOM at 1,030,761 prompt tokens
   with 97,241 MiB peak on that card, three boots, three placements (07/08/09 receipts).
   Phase-6 negative arm: MEMRA_MOE_SLOTS=256 starves the fused-epi SLRU arm below
   3*n_used, it fails closed to the sequential loop and prefill halves (~40 tok/s) — the
   floor is not a lever. Demonstrated equilibrium: MEMRA_MOE_SLOTS=12000 (arena cap ~52 GB,
   working sets 10/13/13/6 expert layers under MEMRA_PP_SPLITS=13,26,39) + the uneven
   splits keeping the tail stage light.

## Per-card VRAM at 1M

Peaks over the whole 1M phase (boot -> greedy 1M -> sampled 1M, phase7-vram.csv, 20 s
sampling; cards are 97,887 MiB):

  gpu0 (stage0, 13 layers) 81,945 | gpu1 (stage1, 13) 80,121 | gpu2 (stage2, 13) 80,089 |
  gpu3 (stage3, 7 layers + primary engine + f32 output head + 17.1 GB whole-prime hidden
  stack) 94,905 MiB

The second 1M prime ran with the first session PARKED (eos-terminated sessions park and
keep their planes): gpu3 carried TWO 1M sessions' planes plus the aggregation and stayed
under the card by ~3 GB. A third 1M session would not fit; repeated 1M cells on one boot
should set MEMRA_REUSE_POOL=0 or expect the pool eviction to decide.

## The prefill gap statement

Measured: a 1,035,357-token prime takes **6,420 s (1h47m) at 161.3 tok/s** on the PP4
placement with the fused MoE epilogue ON (both 1M arms agree to 0.01%; rate is depth-flat,
so the wall is per-token trunk work, not attention depth).

The multiple missing for 1M-to-first-token:
- (a) inside the 90 s platform deadline: needs 11,504 tok/s -> **71.3x** missing;
- (b) inside 10 min: needs 1,726 tok/s -> **10.7x** missing.

Capacity is no longer the blocker; prefill throughput is the whole distance between this
demonstration and serving it. The levers, with what is measured vs estimated:

1. **Grouped MoE prefill** — the attributed dominant share (prefill-gap lane: glm5 prefill
   runs PER-TOKEN MoE dispatch, ~8.4M launches per 4096-token chunk). The fix
   (`MEMRA_MOE_GROUPED_PREFILL`, grouped GEMM per projection) is landed on
   lane/glm5-grouped-prefill, default OFF, gated but not yet box-validated. Every number
   in this lane is the DEFAULT-OFF baseline arm, deliberately: this cell is the capacity
   receipt and the honest baseline; the grouped A/B belongs to its own lane on separate
   hardware. Its multiple is that lane's to measure, not this one's to guess.
2. **Fused MoE epilogue** (`MEMRA_MOE_FUSED_EPI=1`): already ON in every number here; its
   ~2x is banked (epilogue lane; phase-6's fail-closed arm re-measured the OFF class at
   ~40 tok/s, ~4x below the 161.3 baseline in the same boot family).
3. **PP4 chunk pipelining**: today the chunked ppN prime is SERIAL across stages — one
   stage computes while three idle (observed: single-GPU 99%/0%/0%/0% rotating). Door-off
   single-engine measured 143-151 tok/s at 6.5k vs PP4 171: PP4 currently buys placement,
   not speed. Overlapping chunks across stages is a up-to-4x-class lever (estimate; the
   chunk loop landed in this lane is where the overlap would go).
4. **KDA prefill scan**: sequential per token by construction (the chunked UT transform is
   not the shipped path); with MoE dispatch fixed, this is the next per-token term.
5. **BF16 prefill GEMM** (`MEMRA_PP_BF16`): exists but FAILED step37's first-token gate;
   NOT tested in this cell. Any future test gates max_tokens=1 first-token identity on
   real prompts first, per that receipt.

Serving statement: at the measured 161 tok/s the 90 s first-token deadline admits ~14.5k
prompt tokens; 262k admits nothing past the deadline either (27 min), so "1M served" is
gated on lever 1 (and likely 3) landing with receipts, not on any capacity work. The
demonstration itself needed the cell-scoped `MEMRA_TIMEOUT_MS_MAX` override, which must
never reach a fronted deploy.
