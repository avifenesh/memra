# Gathered MLA verify mechanism, 2026-09-09

Source base: `628521007` (origin/main at lane start). Source locations below
refer to this immutable base unless marked candidate.

`crates/memra-engine/src/mla_ffi.rs:2734` launches attention once per MLA
layer for the entire verify width. In `cu/mla_attn.cu:2702`, the DSA single-pass
kernel maps blockIdx.x to query*n_head+head. The launcher at :2801 uses
`t_q*n_head` CTAs, 256 threads (8 warps) each. PP1 has 64 heads: t2/4/7
therefore launch 128/256/448 CTAs, not t sequential launches. Each of the
11 MLA layers launches separately. Rows are concurrent in the grid, subject
to the scheduler's occupancy and wave count.

`src/hybrid.rs:2264` defines top_k=2048 raw tokens, pool=4; :2273 selects
512 pools and :2290 expands to 2048 slots plus up to 3 tail slots. These
are not 512 independent latent rows. Capture will record the actual slot
count. At 32k and 128k prompts the source cache and scorer pool grow, but
the selected attention budget is fixed. GLM5 has latent rank 512, absorbed
RoPE width 0, so the unique gathered payload is about 4 MiB per query
before cross-query overlap (16 MiB at t4). Heads share the index list.

Each CTA walks ceil(n_slots/8) tiles serially (:2740). One warp scores one
slot: coalesced float4 KV staging, warp-local synchronization, lane-strided
dot in ascending dimensions and shuffle-down 16,8,4,2,1. A block barrier
publishes eight scores. Every thread computes the tile max, rescale,
eight weights and denominator; it then updates its latent accumulator in
ascending slot order. A second block barrier precedes the next tile.
Thus about 512 barriers per CTA and dependent softmax/PV recurrences span
2048 slots. Staging already shares the score/PV load within a head; adding
that staging again repeats a measured losing arm. Bytes are mostly L2
traffic, not evidence of an HBM bottleneck (:2640 mechanism record).

The p32k tally reports t2 694.218 us/layer (7636.399 us/round), t4
656.218 (7218.395), t7 952.051 (10472.558). These are single-request
profile means, not warmed ABBA. t2 and t7 use the generic gathered kernel;
t4 uses the DSA single-pass twin. This is occupancy/dispatch scaling, not
linear t launches. tptrace9/10 report decode t1 warp-online at 49 us/layer
on TP2 32-head shards at 128k/1M; that excludes its small combine and is a
different head shape and numeric class. The t4/t1 kernel ratio is about
13.39, not a controlled scaling experiment.

`src/mla_ffi.rs:200` pins MLA_B200_ARM_HEADS=64; :211 multiplies existing
output-range split factors by 64/n_head for shards. Those splits repeat
the whole score/softmax walk and only divide output dimensions. They do
not partition selected KV. The current warp-online class is decode-only
(:366): the earlier t4 argmax failure must not be bypassed. PR #307's RP1
scorer and #308's cooperative select descent optimize different kernels
and remain untouched.

## Candidate and arithmetic, recorded before implementation

Use 8 slot partitions aligned to eight-slot tile boundaries. Each CTA
keeps the current per-slot dot and eight-slot online fold inside its
partition; a small kernel combines unnormalized (m,l,acc) partials in
ascending partition order. This is distinct from the killed warp-online
arm: no per-slot softmax and no xor dot tree. The partition combine is
still a new numeric class and any latent-row argmax flip is NEGATIVE.

At t4 this makes 2048 partial CTAs instead of 256. Optimistic serial-time
screen: 656.218/8 + 10 us combine = 92.02725 us/layer, saving 6.206098 ms
across 11 layers. Budget a further 100 us/layer for extra CTA waves and
partial traffic: 192.02725 us, saving 5.106098 ms/round. These are explicit
hypotheses, not predictions from measured HBM bandwidth. Even 610.763 us
per layer clears 0.5 ms/round at t4. Batching query rows would reduce the
working grid and is deferred unless measured geometry warrants it.

Preconditions: B200, 64 unsharded heads, rank512, d_rope0, t2..7, 512
four-token selected pools plus optional 0..3 tail slots. Odd selected-pool
counts and unsupported shard layouts refuse to the current path. Tail
slots go in the final partition without changing slot order.

Oracle before timing: real captured q_lat/q_pe/cache/idx at each of all
11 MLA layers for t2/4/7 on p32k and p128k. Byte-exact gather against CPU
copy of captured cache. Every output finite, latent-row argmax identical,
and max absolute error <= 1e-5 + 1e-4*max_abs(reference row). Threshold is
fixed before viewing candidate output. Warmed ABBA x5, per-layer and
rotating 11-layer rounds; conditional width weights 48/162,103/162,11/162.
Report both prompt totals, with equal prompt weights for the combined
component verdict. KEEP needs >=0.5 ms/round and zero oracle failures.

Algorithm reference: [Flash-Decoding, Dao et al.](https://crfm.stanford.edu/2023/10/12/flashdecoding.html),
steps 1..3: partition keys/values, compute partial attention, rescale and
combine. No external code or runtime is imported; the implementation is
derived from Memra's current tiled gathered kernel.

## Captured geometry and measured resource usage

All 66 captures have 64 heads, rank512, rope0, and 2051 slots. The MLA
ordinals 0..10 map to trunk layers 3,7,11,15,19,23,27,31,35,39,43 in the
pinned mint config. The original kernel therefore executes 257 tiles and
514 per-tile block barriers (plus initialization). Eight partitions use
264-slot spans: 33 tiles in each of the first seven partitions and 26 in
the last. The last partition owns the tail; no slot is dropped or padded
into a different reduction group.

The measured executable's cuobjdump resource receipt reports 54 registers
per thread for the DSA control, 50 for the candidate, and 32 for the generic
control; each has 10304 bytes static shared memory, zero stack/local bytes.
DSA control and candidate add 16384 bytes of dynamic KV staging. These are
compiled resource counts, not measured occupancy or hardware stall counters.
The t4 result is about 374 us/layer, rather than the 192 us screening
hypothesis. The larger grid still performs the same total tiled work;
serial-time division by eight is not a throughput prediction across its
additional CTA waves. The measured component clears the predeclared bar.

Both the current and candidate arithmetic live in `cu/mla_attn.cu`, whose
build uses normal FMA contraction (`build.rs:486`). This lane introduces
no cross-file twin in a `-fmad=false` translation unit. The per-dimension
dot and slot accumulate are copied from this same numerical class.
