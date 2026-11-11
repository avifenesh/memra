# A softmax-routed MoE prefills with one program on both expert placements (2026-11-11)

**Chosen:** the pairs arm's routed-expert program (the grouped f16 lane on sm_120a and Hopper)
is the program for every softmax-routed MoE layer call wider than 16 rows, whether its experts
sit in the fits-VRAM resident slab or in the SLRU cache (memra#1217). That is prefill, and a
decode wave of 17 to 32 rows on the worker's exact-32 tier. On the SLRU placement the same kernels
read the same expert bytes from cache slots through the same `[3 * n_expert]` pointer table. The
table comes from one of three places, under one hold of the cache lock that also covers the
kernel launches:

1. the cache's full-layer row, when every block of the layer is resident;
2. one pinned admission (`MoeSlotCache::acquire_pinned`): the layer call's active blocks
   are admitted through the normal `dispatch_source`, and eviction skips the set's own members
   until admission ends;
3. grouped f16 on the sk visitor only (`MEMRA_MOE_F16G` >= 2, both GEMM shapes `% 32 == 0`,
   sk form not rolled back): when the cache cannot pin the active set at once, contiguous CSR
   runs of experts that fit, one pinned admission and one grouped-f16 chain per run,
   concatenated in CSR order. The cuBLAS grouped mode and the grid-scan rollback are not argued
   run-independent and need the whole set pinned.

Every step of the grouped f16 chain is per CSR row or per expert group (the activation's f16 row
scale, SiLU, and the sk visitor's per-group tiles over the full K, whose form depends on the
group's m_e and the device only). So the runs' bytes equal the single launch's. The placement gate
and the Ornith window at 500 slots both hold that bit for bit.

**Rejected:**

- *Keep two programs and state them* (the gate already keys on the placement since #1215).
  The served greedy text of one request then depends on free memory at load, across a fleet of
  identical cards. That is the "one numeric program per request" failure at fleet scope, and the
  fix costs no memory and makes the SLRU placement faster.
- *Make the per-token decode program the reference for both placements.* Resident prefill
  would go from the grouped f16 lane to a per-token loop: the served board-2048 request takes
  1.72 s resident against 6.53 s on that program.
- *A transient staging slab for the active experts* instead of pinned cache slots. It costs a
  layer's active expert bytes of extra VRAM (about 0.7 GB for Ornith) plus a copy of every hit.
  Pinned admission reuses the slots, and a hit copies nothing.

**Measurement that settled it** (RTX 5090 Laptop GPU, Ornith-1.5-35B-A3B NVFP4+Q5_K MTP,
board-2048, receipts in `research/slru-prefill-program-20261111/`):

- The cause, localized by run on main ede1b8119f: on an idle card the SLRU prewarm holds every
  layer, and SLRU prefill ran `moe_ffn_dev`, the per-token decode program, over all 2,048 tokens.
  With the experts resident, prefill ran the grouped f16 lane. Forcing the resident placement off
  the f16 lane (`MEMRA_MOE_F16G=0`, which gives the pairs q8 arm) or off the pairs arm
  (`MEMRA_MOE_PAIRS=0`) reproduces the SLRU window byte for byte, the layer 0 and 1 MoE outputs
  included. Resident against SLRU: the layer 0 MoE input is byte-equal, and its output differs in
  4,171,757 of 4,171,776 floats (max 5.6e-4). The only differing op is the f16 lane: f16
  activations with a row scale and f16 tensor-core MMA with f32 accumulation, against q8_1
  activations with dp4a.
- After the fix, the argmax window's whole table (prefill and decode columns) is identical
  between the resident and SLRU placements, eager and served batch, and at 500 slots (chunked:
  two runs per layer call) and 2,000 slots (pinned). The decode columns equal main's on both
  placements.
- The served greedy pair, now three-way: resident, SLRU and SLRU at 4,000 slots return the same
  text (`1408f53ea2349977`). Main's SLRU placement returned `a5f81b0cc03207a1`.
- SLRU served time for the same request, N=3 interleaved: main 6.533 s, fix 2.202 s (idle card).
  Under spill (4,000 slots): main 11.167 s, fix 4.966 s. Peak device memory is unchanged on the idle card
  (21,276 MiB both) and under spill (6,332 MiB both).

**Scope.** The softmax pairs arm only: the qwen35_moe family (Ornith, Qwen3.x MoE). Two other
MoE prefill arms are slab-only, with the same class of split, and are filed as follow-ups:
`gemma4_moe` (memra#1225: its pairs and rows arms are gated on the resident slab) and glm5_next's
grouped prefill (memra#1226: fails closed to the staged loop on the SLRU placement). A layer that
the SLRU arm cannot serve keeps the per-token program: the cache is off (`MEMRA_MOE_CACHE=0`) or
frozen (the Hy3 CPU-expert door), or the layer is on the MMA or q8 sub-arm and the cache cannot
pin its whole active set. On the sm_120a default, that needs a non-default `MEMRA_MOE_F16G` or a
qtype the f16 lane does not take, plus a cache smaller than one layer. For those sub-arms, a
cache right at that boundary can take one program or the other depending on how many slots
in-flight prefetches hold at that moment (review finding, kept as a nit). The pairs arm also feeds the host
observers (route, weight and input traces, MoESD capture, hidden trace) and routes with the
pruned-expert mask, on both placements.
