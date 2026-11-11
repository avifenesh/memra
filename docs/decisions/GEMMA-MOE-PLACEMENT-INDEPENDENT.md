# A gemma4_moe model runs one MoE program on both expert placements (2026-11-11)

**Chosen:** the resident program is the program for every gemma4_moe MoE layer call, whether
the routed experts sit in the fits-VRAM resident slab or in the SLRU cache (memra#1225, the
gemma follow-up of memra#1217). That is the q8_1/dp4a decode and verify-rows kernels (t < 16),
the pairs arm for prefill (t >= 16; the int8-MMA expert GEMM on the default), and the layer
tail's q8z fusion that feeds them. On the SLRU placement the same kernels read the same expert
bytes from cache slots through the same `[3 * n_expert]` pointer table. Under one hold of the
cache lock that also covers the launches, the table is:

1. the cache's full-layer row, when every block of the layer is resident (the one-shot prewarm
   admits a whole layer while free slots cover it);
2. one pinned admission (`MoeSlotCache::acquire_pinned`, memra#1217) of the call's active
   experts; a small-t call reads its device top-k selection back once to name them;
3. prefill only: contiguous CSR runs of experts that fit, one pinned admission and one kernel
   chain per run, concatenated in CSR order. This extends memra#1217's grouped-f16 argument to
   the int8-MMA and decode-once dp4a arms: their kernels launch one block row per expert segment
   and walk only that segment's rows over the full K, with a tile form fixed by the device and
   out_f, and their activations are quantized per row. The placement gate holds the runs' bytes
   to the single launch's.

The source is decided once per boot (`gemma_experts`): resident slab, else the SLRU arm when the
cache can pin the most experts a small-t call can select, `min(n_expert, 15 * top_k)` (360 slots
on gemma-4-26B-A4B), else the host-staged loop for every call. The floor check reads the cache's
slot count (fixed when the cache is built) once per device and the model keeps the answer
(`gemma_slru_admits`), so no request can switch programs between calls. The cache off
(`MEMRA_MOE_CACHE=0`), a frozen or banked cache, a mixed-layout bank or a cache below that floor
keep the staged program, and the boot log says which. If a call nevertheless finds fewer
pinnable slots than its set (only quarantined slots can do that; gemma issues no prefetch), it
fails closed with an error rather than run another program.

**The ragged-k tail.** The int8-MMA down GEMM pads gemma-4-26B-A4B's ragged k (704) to whole
256-value superblocks and so reads 36 B past an expert's last row (at most 126 B for any ragged
k), decoding a block scale from them that meets a zero activation. The scale must be finite.
The slab has carried a 144 B tail for its last expert since 2026-07-31, but `htod_bytes_padded`
left it as stale pool bytes: a stale f16 Inf or NaN there would turn column `n_embd - 1` of every
pair routed to the last expert into NaN on the resident placement. Not observed, and fixed here:
the pad is zeroed (byte-neutral whenever the stale bytes were finite, which the resident window
and served cells confirm). A gemma cache slot is built 144 B longer than the largest block
(`GEMMA_MMA_RAGGED_TAIL`); `alloc_u8` zeroes it and no block ever writes it, because the SLRU arm
requires the down block to be the layer's largest and every layer's down block has one size. A
cache built without that room refuses the SLRU arm at boot. The other reads past an expert's
bytes are harmless by construction: the wide-load q4_0 dots read up to 6 B past the final block
through an aligned window and discard them (`SLOT_TAIL_PAD_BYTES`), and the qwen-class MMA arm
requires `n_ff_exp % 256 == 0`.

**Rejected:**

- *Keep two programs and state them.* The served greedy output of one request then depends on
  free memory at load. On gemma-4-26B-A4B both the prefill and the decode logits move (decode
  too, unlike Ornith in memra#1217, because the staged loop is an f32 `qmatvec` chain, not the
  q8 kernels), and the staged program is far slower.
- *Make the staged program the reference.* It stages every (token, expert, projection) with a
  synchronous H2D: board-2048 takes 108.4 s to first token against 0.55 s resident, misses the server's
  default 90 s deadline, and decodes at 17.9 tok/s against 113.7.
- *Fall back to the staged program per call when a call's set does not fit.* Spec verify and
  decode of one request could then run different programs. The floor makes the choice per boot.
- *A transient staging slab* instead of pinned slots: extra VRAM and a copy of every hit;
  memra#1217 rejected it for the same reasons.

**Measurement that settled it** (RTX 5090 Laptop GPU, gemma-4-26B-A4B QAT Q4_0, board-2048;
receipts in `research/gemma-moe-placement-20261111/`):

- The cause, by run on base (memra#1229's head, whose gemma code is main's): the argmax window's
  prefill AND decode columns differ between the placements (prefill `7a5b4417` against
  `5edd0c6a`, eager decode `9e07f782` against `d97f6678`); the prefill top-1 flips at row 2096
  and the decode top-1 at row 2091. A 256-token greedy decode returns a different text from the
  first generated token (`9ad77520` resident, `ceefd021` staged); the board-2048 completion and
  chat requests happened to match.
- The placement GPU test (`tests/moe_slru_gemma_placement_gpu.rs`): on base every SLRU arm
  differs at every row; on the fix the roomy (full-layer row), mid (row and pinned) and tight
  (pinned and chunked) arms are bit-identical to the resident arm on 20 rows, on the int8-MMA
  arm and under `MEMRA_GEMMA_MOE_MMA=0`; a cache below the floor runs no SLRU mode.
- After the fix, the argmax window's whole table is identical across resident, SLRU at the auto
  cache (LayerRow), 2,000 slots (Pinned), 400 slots (LayerRow, Pinned) and 360 slots (Pinned and
  Chunked, two runs), eager and batch, and the resident columns equal base's.
- Served, N=3: every placement returns the same texts (board-2048 `3eb1a659`, chat `442ccca2`,
  decode `9ad77520`). Board-2048 end to end: resident 1.865 s (base 1.868), SLRU 2.370 s, SLRU at
  2,000 slots 3.332 s, base staged 115.7 s (N=1). Decode: resident 113.7 tok/s (base 113.6), SLRU
  auto cache 113.7, 6,000 slots 96.6, 2,000 slots 56.6, staged 17.9 (N=1). A layer that is not
  fully cache resident reads its top-k back once per token (one stream sync) and admits its
  misses; that is what spill costs.
- Peak device memory: resident unchanged (17,104 against 17,168 MiB). The SLRU placement now
  builds the expert cache the qwen-class SLRU placement builds (85% of free memory by default):
  20,560 MiB on the idle card against the staged loop's 4,784; 7,024 at 2,000 slots.

**Scope.** gemma4_moe. The graph decode programs (`gemma4_generate_graph`,
`gemma4_generate_plain_graph`, bench and gate binaries; the server decodes eager) and the
`MEMRA_GEMMA_ROUND_GRAPH=1` speculative round capture would bake the cache's pointers into a
graph that later admissions invalidate, so they refuse a model whose experts are not resident
(the round falls back to the eager round). glm5_next's grouped prefill (memra#1226) is the
remaining slab-only MoE prefill arm.
